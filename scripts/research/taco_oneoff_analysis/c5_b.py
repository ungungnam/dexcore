import pandas as pd, numpy as np, json
from scipy.stats import spearmanr, mannwhitneyu
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/verify/'
splits=['train','test_1','test_2','test_3','test_4']
TH=0.01
rows=[]; onset_rows=[]
for s in splits:
    d=pd.read_csv(f'{s}.csv')
    wh=d.groupby(['window','hand']).agg(contact=('contact_mm','mean'),cmap=('contact_map_err_mm','mean'),motion=('motion_mm','mean'),
        c_trans=('contact_trans_mm','mean'),touch=('gt_touch_frac','mean'),drift=('gt_contact_drift_mm','mean'),start=('start','first'),
        c_first=('contact_mm',lambda x: x.values[:22].mean()),c_last=('contact_mm',lambda x: x.values[44:].mean())).reset_index()
    z=np.load(f'{s}_contact_maps.npz'); gt=z['gt']; pr=z['pred']; idx=pd.read_csv(f'{s}_contact_maps_index.csv')
    assert len(idx)==gt.shape[0]
    wpos={w:i for i,w in enumerate(idx.window)}
    for hi,h in enumerate(['L','R']):
        sl=slice(hi*1024,(hi+1)*1024)
        g=gt[:,:,sl].astype(np.float32); p=pr[:,:,sl].astype(np.float32)
        # total change: sum_t mean_k |g[t]-g[t-1]|  (mm)
        tot=np.abs(np.diff(g,axis=1)).mean(axis=2).sum(axis=1)*1000
        tot_touch=np.abs(np.diff((g<TH).astype(np.float32),axis=1)).mean(axis=2).sum(axis=1)  # touched-set churn
        drift0=np.abs(g-g[:,:1]).mean(axis=(1,2))*1000
        # per-frame min distance -> onset/release
        gmin=g.min(axis=2); pmin=p.min(axis=2)
        gtouch=gmin<TH; ptouch=pmin<TH
        def first_true(a):
            r=np.where(a.any(axis=1),a.argmax(axis=1),-1); return r
        g_on=first_true(gtouch); p_on=first_true(ptouch)
        g_off=np.where(gtouch.any(axis=1),63-gtouch[:,::-1].argmax(axis=1),-1); p_off=np.where(ptouch.any(axis=1),63-ptouch[:,::-1].argmax(axis=1),-1)
        gfrac=gtouch.mean(axis=1); pfrac=ptouch.mean(axis=1)
        # per-frame error profile of pred touched-frac vs gt touched-frac, temporal correlation of touch time series
        tmp=pd.DataFrame(dict(window=idx.window.values,hand=h,tot_change=tot,touch_churn=tot_touch,drift0=drift0,g_on=g_on,p_on=p_on,g_off=g_off,p_off=p_off,
                              g_touchfrac_t=gfrac,p_touchfrac_t=pfrac,g_any=gtouch.any(axis=1),p_any=ptouch.any(axis=1),
                              g_t0=gtouch[:,0],p_t0=ptouch[:,0],gmin_mean=gmin.mean(axis=1)*1000,pmin_mean=pmin.mean(axis=1)*1000))
        rows.append(tmp.assign(split=s))
    m=pd.concat([r for r in rows if r.split.iloc[0]==s]).merge(wh,on=['window','hand'])
    m.to_csv(OUT+f'c5b_{s}_window_dynamics.csv',index=False)
    t=m[m.touch>0]
    def sp(a,b): r=spearmanr(a,b); return f'{r.statistic:+.3f} (p={r.pvalue:.1e})'
    print(f'\n== {s}: touching window-hands n={len(t)}')
    print(' Spearman(contact, drift0 recomputed)     ', sp(t.contact,t.drift0), '| vs probe drift col', sp(t.contact,t.drift))
    print(' Spearman(contact, total |dmap| change)    ', sp(t.contact,t.tot_change))
    print(' Spearman(contact, touched-set churn)      ', sp(t.contact,t.touch_churn))
    print(' Spearman(cmap_err, total change)          ', sp(t.cmap,t.tot_change))
    print(' Spearman(contact, mean gt min-dist)       ', sp(t.contact,t.gmin_mean))
    print(' Spearman(contact, gt touch frac of frames)', sp(t.contact,t.g_touchfrac_t))
    nf=t[t.start!=8]
    print(f' NONFIRST n={len(nf)}: Spearman(contact,total change)', sp(nf.contact,nf.tot_change), '| (contact, drift0)', sp(nf.contact,nf.drift0), '| (c_last-c_first, total change)', sp(nf.c_last-nf.c_first,nf.tot_change))
    ff=t[t.start==8]
    print(f' FIRST n={len(ff)}: Spearman(contact,total change)', sp(ff.contact,ff.tot_change), '| (contact, g_on onset frame)', sp(ff.contact,ff.g_on))
    # partial: within tertiles of gmin_mean
    t=t.copy(); t['gq']=pd.qcut(t.gmin_mean,3,labels=False)
    print(' Spearman(contact,total change) within gt-min-dist tertiles:', [f'{spearmanr(x.contact,x.tot_change).statistic:+.2f}' for _,x in t.groupby('gq')])
    # onset timing
    on=t[(t.g_any)&(~t.g_t0)]  # gt onset inside window
    print(f' gt onset-in-window window-hands: {len(on)} ({len(on)/len(t):.0%}); of these pred never touches: {(~on.p_any).mean():.0%}; pred touches at t0 already: {on.p_t0.mean():.0%}')
    ob=on[on.p_any]
    print(f'   pred onset - gt onset (frames), n={len(ob)}: mean {(ob.p_on-ob.g_on).mean():+.1f}, median {(ob.p_on-ob.g_on).median():+.1f}, MAD {np.median(np.abs(ob.p_on-ob.g_on)):.1f}, |err|>8 frames {(np.abs(ob.p_on-ob.g_on)>8).mean():.0%}')
    print(f'   gt onset frame mean {on.g_on.mean():.1f}; ')
    both=t[t.g_any&t.p_any]
    print(f'   touched-frame-fraction: gt {both.g_touchfrac_t.mean():.2f} pred {both.p_touchfrac_t.mean():.2f}; Spearman(gt,pred frac) {spearmanr(both.g_touchfrac_t,both.p_touchfrac_t).statistic:+.2f}')
    print(f'   pred touches anywhere in window: gt-touch {t.p_any.mean():.0%}; among gt NEVER-touch window-hands: {m[m.touch==0].p_any.mean():.0%} (n={len(m[m.touch==0])})')
    # does contact error depend on onset timing error?
    ob=ob.copy(); ob['on_err']=np.abs(ob.p_on-ob.g_on)
    print('   Spearman(contact, |onset err|)', sp(ob.contact,ob.on_err), ' Spearman(cmap, |onset err|)', sp(ob.cmap,ob.on_err))
    onset_rows.append(dict(split=s,n_touch=len(t),n_onset_in_window=len(on),pred_never=(~on.p_any).mean(),pred_early_t0=on.p_t0.mean(),
        onset_bias=(ob.p_on-ob.g_on).mean(),onset_median=(ob.p_on-ob.g_on).median(),onset_mad=np.median(ob.on_err),frac_gt8=(ob.on_err>8).mean(),
        gt_frac=both.g_touchfrac_t.mean(),pred_frac=both.p_touchfrac_t.mean(),false_touch_on_never=m[m.touch==0].p_any.mean()))
pd.DataFrame(onset_rows).round(3).to_csv(OUT+'c5e_onset_timing.csv',index=False)
print(pd.DataFrame(onset_rows).round(3).to_string())
