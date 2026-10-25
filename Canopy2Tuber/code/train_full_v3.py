import copy,random,time,argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from model import DualStreamMSTNet

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)

def augment(x,c,g):
 if not c['noise'] and not c['mask']:return x
 z=x.clone();noise=c['noise']
 sel=torch.rand(len(z),1,generator=g)>.5
 z=torch.where(sel,z+torch.randn(z.shape,generator=g)*noise,z*(1+torch.randn(z.shape,generator=g)*noise*.5))
 if c['mask']:
  mask=torch.rand(len(z),generator=g)<c['mask'];cols=torch.randint(z.shape[1],(len(z),),generator=g)
  z[torch.arange(len(z))[mask],cols[mask]]=0
 return z

def fit(model,x,y,xv,yv,c,seed,epochs,logpath,selection=True):
 seed_all(seed);g=torch.Generator().manual_seed(seed)
 opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2,eta_min=1e-6)
 lossfn=nn.HuberLoss() if c['loss']=='huber' else nn.MSELoss()
 best=float('inf');best_epoch=0;state=None;rows=[];start=time.time()
 for epoch in range(1,epochs+1):
  model.train();order=torch.randperm(len(x),generator=g);total=0.;seen=0
  for ix in order.split(16):
   assert len(ix)>1
   opt.zero_grad();loss=lossfn(model(augment(x[ix],c,g)),y[ix]);assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);opt.step()
   total+=loss.item()*len(ix);seen+=len(ix)
  assert seen==len(x)
  scheduler.step();v=None
  if selection:
   model.eval()
   with torch.no_grad():v=float(torch.mean((model(xv)-yv)**2).sqrt())
   if v<best:best=v;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
  rows.append({'epoch':epoch,'training_loss':total/seen,'selection_rmse_standardized':v,'samples_seen':seen})
  if epoch%50==0:
   pd.DataFrame(rows).to_csv(logpath,index=False)
   print(logpath.stem,'epoch',epoch,'elapsed',round(time.time()-start,1),flush=True)
  if selection and epoch>=50 and epoch-best_epoch>50:break
 pd.DataFrame(rows).to_csv(logpath,index=False)
 if selection:model.load_state_dict(state)
 return {'best_epoch':best_epoch if selection else epochs,'selection_rmse_standardized':best if selection else None,'executed_epochs':epoch,'seconds':time.time()-start}

def main():
    p=argparse.ArgumentParser(description='Field selection and refitting from saved PROSAIL-pretrained V3 weights.')
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[3407,3408,3409])
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/'config/lai_protocol.json').read_text());n=protocol['normalization']
    d=pd.read_csv(a.calibration)
    if len(d)!=286 or d.Inner_split.value_counts().to_dict()!={'Training':228,'Selection':58}:
        raise ValueError('The paper protocol requires 286 calibration rows: 228 Training + 58 Selection.')
    x=torch.tensor((d[[str(b) for b in protocol['bands']]].to_numpy(float)-n['x_mean'])/n['x_scale'],dtype=torch.float32)
    y=torch.tensor((d[['lai']].to_numpy(float)-n['y_mean'])/n['y_scale'],dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('Nonfinite input.')
    ti=np.flatnonzero(d.Inner_split.eq('Training'));vi=np.flatnonzero(d.Inner_split.eq('Selection'))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        if seed not in protocol['seeds']: raise ValueError('No supplied pretraining checkpoint for this seed.')
        seed_all(seed);model=DualStreamMSTNet(5)
        initial=torch.load(root/f'weights/pretrained_{seed}.pth',map_location='cpu',weights_only=True)
        choices=[]
        for lr in protocol['deep_search']['transfer_learning_rates']:
            for regime in protocol['deep_search']['regimes']:
                c=dict(regime,lr=lr);model.load_state_dict(initial,strict=True)
                result=fit(model,x[ti],y[ti],x[vi],y[vi],c,seed+10000,protocol['deep_search']['max_epochs'],a.output/f'{seed}_{lr}_{c["name"]}.csv')
                result.update(config=c,selection_rmse=result['selection_rmse_standardized']*n['y_scale']);choices.append(result)
        winner=min(choices,key=lambda v:v['selection_rmse'])
        model.load_state_dict(initial,strict=True)
        fit(model,x,y,None,None,winner['config'],seed+20000,winner['best_epoch'],a.output/f'{seed}_refit.csv',False)
        torch.save({'Model':'Full_V3_TL','seed':seed,'normalization':n,'model_state':model.state_dict(),'selected':winner},a.output/f'MSTNet_V3_{seed}.pth')
        (a.output/f'{seed}_selection.json').write_text(json.dumps(choices,indent=2))
if __name__=='__main__': main()
