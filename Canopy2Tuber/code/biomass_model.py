"""Frozen interval RUE and organ allocation prediction; no outcomes are read."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
PRED=['Total_pred','Tuber_pred','Root_pred','AGB_pred','Leaf_pred','Stem_pred']
INPUT=['Sample_ID','Date','N_Level (kg/ha)','D','GDD_After_Max_LAI','APAR_Delta']

def curve(r,module,t):
    if module=='HI':
        h,k,t50=r['HI'];return h/(1+np.exp(np.clip(-k*(t-t50),-700,700)))
    if module=='Root':
        a,b,c=r['Root'];return a+b*np.exp(np.clip(-c*(t-r['tmin']),-700,700))
    a,b=r['LVR'];return a+b*t

def transfer(records,n,d,t,module):
    densities=sorted({r['D'] for r in records});values=[]
    for density in densities:
        r=sorted([r for r in records if r['D']==density],key=lambda v:v['N'])
        values.append(np.interp(n,[v['N'] for v in r],[curve(v,module,t) for v in r]))
    return np.interp(d,densities,values)

def validate(d):
    if d.empty or d[INPUT].isna().any().any(): raise ValueError('Missing required inputs.')
    x=d[INPUT[1:]].to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError('Inputs must be finite.')
    if (d[['APAR_Delta','GDD_After_Max_LAI','N_Level (kg/ha)']]<0).any().any() or (d.D<=0).any():
        raise ValueError('Invalid radiation, post-peak age, N rate or density.')
    for _,g in d.groupby('Sample_ID',sort=False):
        if (np.diff(g.Date.to_numpy(float))<=0).any(): raise ValueError('Each trajectory must be in strictly increasing thermal-time order.')
        if g['D'].nunique()!=1 or g['N_Level (kg/ha)'].nunique()!=1: raise ValueError('Treatment changes within a trajectory.')

def donor_predict(d,bank,soil_N=80.):
    validate(d);q=d.copy()
    base,alpha,beta,gamma,delta=bank['RUE'];nt=q['N_Level (kg/ha)'].to_numpy(float)+soil_N
    q['RUE']=np.maximum((base+alpha*q.D)*(1-np.exp(-beta*nt))*np.exp(-delta*nt)-gamma*q.GDD_After_Max_LAI,0)
    q['DM_increment']=q.RUE*q.APAR_Delta
    q['Total_pred']=q.groupby('Sample_ID',sort=False).DM_increment.cumsum()
    for m in ['HI','Root','LVR']:
        q[m+'_sim']=[transfer(bank['organs'],n,d,t,m) for n,d,t in zip(q['N_Level (kg/ha)'],q.D,q.Date)]
    # Do not silently clip fractions: replay the fitted curves and flag extrapolation.
    q['Fraction_out_of_bounds']=((q[['HI_sim','Root_sim','LVR_sim']]<0)|(q[['HI_sim','Root_sim','LVR_sim']]>1)).any(axis=1)
    q['Tuber_pred']=q.Total_pred*q.HI_sim
    q['Root_pred']=q.Total_pred*(1-q.HI_sim)*q.Root_sim
    q['AGB_pred']=q.Total_pred*(1-q.HI_sim)*(1-q.Root_sim)
    q['Leaf_pred']=q.AGB_pred*q.LVR_sim;q['Stem_pred']=q.AGB_pred*(1-q.LVR_sim)
    return q

def predict(d,config,external=False):
    if config.get('mode')!='interval': raise ValueError('Interval parameter bank required.')
    if external:
        out=[]
        for key,b in config['donors'].items():
            q=donor_predict(d,b,config['soil_N']);q['Donor_experiment']=key;out.append(q)
        donors=pd.concat(out,ignore_index=True)
        means=donors.groupby(['Sample_ID','Date'],sort=False)[PRED].mean().reset_index()
        flags=donors.groupby(['Sample_ID','Date'],sort=False).Fraction_out_of_bounds.any().reset_index()
        return means.merge(flags,on=['Sample_ID','Date'],validate='one_to_one'),donors
    if 'Experiment' not in d: raise ValueError('Historical inputs require Experiment.')
    # Sample_ID must identify one complete trajectory, not a date-specific sample.
    if d.groupby('Sample_ID').Experiment.nunique().max()>1: raise ValueError('Sample_ID must be globally unique across experiments.')
    parts=[]
    for e,g in d.groupby('Experiment',sort=False):
        parts.append(donor_predict(g,config['donors'][str(int(e))],config['soil_N']))
    return pd.concat(parts).sort_index(),None

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external',action='store_true');a=p.parse_args()
    q,donors=predict(pd.read_csv(a.input),json.loads(a.parameters.read_text()),a.external)
    a.output.parent.mkdir(parents=True,exist_ok=True);q.to_csv(a.output,index=False)
    if donors is not None: donors.to_csv(a.output.with_name(a.output.stem+'_donors.csv'),index=False)
    print('Saved',len(q),'predictions to',a.output)
if __name__=='__main__': main()
