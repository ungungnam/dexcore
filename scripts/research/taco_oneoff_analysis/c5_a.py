import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/verify/'
splits=['train','test_1','test_2','test_3','test_4']
D={s:pd.read_csv(f'{s}.csv') for s in splits}
for s,d in D.items():
    d['split']=s
    d['bucket']=(d.frame//8)*8
    d['third']=np.minimum(d.frame//22,2)  # 0-21,22-43,44-63
    d['first']=d.start.eq(8)
    wt=d.groupby(['window','hand']).gt_touch_frac.mean().rename('touch')
    d=d.merge(wt,left_on=['window','hand'],right_index=True); D[s]=d
allf=pd.concat(D.values())
tch=allf[allf.touch>0]
print('window-hands touching:',tch.groupby('split').window.apply(lambda x: len(x)//64).to_dict())
# (a) frame-bucket profile, touching only: all / first / non-first
rows=[]
for s in splits:
    for name,m in [('all',tch.split.eq(s)),('first',tch.split.eq(s)&tch['first']),('nonfirst',tch.split.eq(s)&~tch['first'])]:
        sub=tch[m]; p=sub.groupby('bucket').contact_mm.mean()
        rows.append(dict(split=s,subset=name,n_wh=len(sub)//64,**{f'f{b}':round(v,1) for b,v in p.items()}))
a=pd.DataFrame(rows); print(a.to_string()); a.to_csv(OUT+'c5a_frame_profile_by_first_window.csv',index=False)
# also frame-level fine profile for first 16 frames
for s in splits:
    for name,m in [('first',tch.split.eq(s)&tch['first']),('nonfirst',tch.split.eq(s)&~tch['first'])]:
        sub=tch[m]; print(s,name,'frames0-15:',np.round(sub.groupby('frame').contact_mm.mean().values[:16],0))
# gap by bucket
piv=tch.groupby(['split','bucket']).contact_mm.mean().unstack(0)
piv['gap31']=piv.test_3-piv.test_1; piv['ratio31']=piv.test_3/piv.test_1; print(piv.round(1).to_string())
pivnf=tch[~tch['first']].groupby(['split','bucket']).contact_mm.mean().unstack(0); pivnf['gap31']=pivnf.test_3-pivnf.test_1; print('NONFIRST\n',pivnf.round(1).to_string())
pivf=tch[tch['first']].groupby(['split','bucket']).contact_mm.mean().unstack(0); pivf['gap31']=pivf.test_3-pivf.test_1; print('FIRST\n',pivf.round(1).to_string())
# (c) thirds
t=tch.groupby(['split','third']).contact_mm.mean().unstack(0); t['gap31']=t.test_3-t.test_1; t['gap21']=t.test_2-t.test_1; t['gap41']=t.test_4-t.test_1; print('THIRDS\n',t.round(1).to_string())
t.to_csv(OUT+'c5c_thirds.csv')
tnf=tch[~tch['first']].groupby(['split','third']).contact_mm.mean().unstack(0); tnf['gap31']=tnf.test_3-tnf.test_1; print('THIRDS nonfirst\n',tnf.round(1).to_string())
# window-level within-window slope for gap: per window-hand, compute mean(last third)-mean(first third), then compare between splits (bootstrap CI on gap difference)
wh=tch.groupby(['split','window','hand','first']).apply(lambda g: pd.Series(dict(c1=g[g.third==0].contact_mm.mean(),c3=g[g.third==2].contact_mm.mean(),cm=g.contact_mm.mean(),cmap1=g[g.third==0].contact_map_err_mm.mean(),cmap3=g[g.third==2].contact_map_err_mm.mean()))).reset_index()
wh['d31']=wh.c3-wh.c1
print(wh.groupby('split').d31.describe()[['mean','50%','std','count']].round(1))
rng=np.random.default_rng(0)
def boot(a,b,n=2000):
    v=[a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean() for _ in range(n)]; return np.percentile(v,[2.5,97.5])
for s in ['test_2','test_3','test_4']:
    a1=wh[wh.split==s].d31.values; b1=wh[wh.split=='test_1'].d31.values
    print(s,'- test_1: mean d31 diff',round(a1.mean()-b1.mean(),1),'CI',np.round(boot(a1,b1),1))
    # median-based
wh.to_csv(OUT+'c5_windowhand_thirds.csv',index=False)
# (d) contact_map_err profile
pm=tch.groupby(['split','bucket']).contact_map_err_mm.mean().unstack(0); print('CMAP profile\n',pm.round(1).to_string())
pm.to_csv(OUT+'c5d_cmap_err_frame_profile.csv')
pmf=tch[tch['first']].groupby(['split','bucket']).contact_map_err_mm.mean().unstack(0); print('CMAP first\n',pmf.round(1).to_string())
pmn=tch[~tch['first']].groupby(['split','bucket']).contact_map_err_mm.mean().unstack(0); print('CMAP nonfirst\n',pmn.round(1).to_string())
# iou profile & gt_min_dist profile
print('IOU\n',tch.groupby(['split','bucket']).touch_iou.mean().unstack(0).round(3).to_string())
print('gt_min_dist\n',tch.groupby(['split','bucket']).gt_min_dist_mm.mean().unstack(0).round(1).to_string())
print('gt_touch_frac\n',tch.groupby(['split','bucket']).gt_touch_frac.mean().unstack(0).round(3).to_string())
