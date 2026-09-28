import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/verify/'
splits=['train','test_1','test_2','test_3','test_4']
res={}
for s in splits:
    d=pd.read_csv(f'{s}.csv'); d['bucket']=(d.frame//8)*8
    dyn=pd.read_csv(OUT+f'c5b_{s}_window_dynamics.csv')
    d=d.merge(dyn[['window','hand','touch','g_on','p_on','g_any','p_any','g_t0','g_touchfrac_t','start']].rename(columns={'start':'st'}),on=['window','hand'])
    t=d[d.touch>0]; nf=t[t.start!=8]
    prof=nf.groupby('bucket')[['contact_mm','motion_mm','contact_map_err_mm','contact_trans_mm','contact_resid_mm','gt_min_dist_mm']].mean().round(1)
    print(f'\n== {s} NONFIRST touching (n_wh={len(nf)//64})\n',prof.T.to_string())
    # windows touched at every frame ("held throughout")
    held=t[t.g_touchfrac_t>=0.999]
    print(f' held-throughout n_wh={len(held)//64}:', held.groupby('bucket').contact_mm.mean().round(1).values, '| cmap', held.groupby('bucket').contact_map_err_mm.mean().round(1).values, '| motion', held.groupby('bucket').motion_mm.mean().round(1).values)
    # onset bias by first vs nonfirst
    on=dyn[(dyn.touch>0)&dyn.g_any&(~dyn.g_t0)&dyn.p_any]; on['e']=on.p_on-on.g_on
    for nm,sub in [('first',on[on.start==8]),('nonfirst',on[on.start!=8])]:
        print(f' onset err {nm}: n={len(sub)} median {sub.e.median():+.0f} mean {sub.e.mean():+.1f} MAD {sub.e.abs().median():.0f} frac|e|>8 {(sub.e.abs()>8).mean():.2f}')
    # error around gt onset: align frames relative to gt onset (first windows, onset in window)
    o=t[t.g_any&(~t.g_t0)].copy(); o['rel']=o.frame-o.g_on
    ar=o[(o.rel>=-16)&(o.rel<=32)].groupby((o.rel//8)*8).contact_mm.mean().round(0)
    print(' contact_mm by frame rel. to gt onset (bins of 8 from -16):', ar.to_dict())
    res[s]=dict(nonfirst_profile=prof.to_dict(), held_profile=held.groupby('bucket').contact_mm.mean().round(1).to_dict(), rel_onset=ar.to_dict())
json.dump(res,open(OUT+'c5_extra_profiles.json','w'),indent=1,default=float)
