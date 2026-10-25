"""V3 per-run inference. No averaging of runs or evaluation-based seed selection."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import DualStreamMSTNet
ROOT=Path(__file__).resolve().parents[1]
BANDS=['450','560','650','730','840']

def predict(frame, checkpoint, batch_size=512):
    if batch_size<1 or frame.empty: raise ValueError('Nonempty input and positive batch size required.')
    x=frame[BANDS].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Reflectances must be finite.')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('Model')!='Full_V3_TL': raise ValueError('A full V3 checkpoint is required; legacy weights are incompatible.')
    n=saved['normalization'];model=DualStreamMSTNet(5)
    model.load_state_dict(saved['model_state'],strict=True);model.eval()
    # Match the verified pipeline: double-precision normalization, then float32 tensor.
    x=torch.tensor((x-n['x_mean'])/n['x_scale'],dtype=torch.float32)
    with torch.no_grad():
        z=torch.cat([model(b) for b in x.split(batch_size)]).numpy().ravel()
    return z*n['y_scale']+n['y_mean'],int(saved['seed'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,nargs='+')
    p.add_argument('--batch-size',type=int,default=512);a=p.parse_args()
    torch.set_num_threads(1)
    d=pd.read_csv(a.input);outputs=[]
    for w in a.weights or [ROOT/f'weights/MSTNet_V3_{s}.pth' for s in [3407,3408,3409]]:
        prediction,seed=predict(d,w,a.batch_size)
        q=d.copy();q['Seed']=seed;q['Predicted_LAI']=prediction;outputs.append(q)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(outputs,ignore_index=True).to_csv(a.output,index=False)
    print('Saved separate predictions for',len(outputs),'runs:',a.output)
if __name__=='__main__': main()
