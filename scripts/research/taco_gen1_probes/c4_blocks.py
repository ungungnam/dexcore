import pandas as pd, numpy as np, json
P='/result/uhnam/dexcore/bimart_taco/contact_probe/'; OUT=P+'verify/'; S='/result/uhnam/dexcore/bimart_taco/train_store/'
pd.set_option('display.width',250)
w = pd.read_csv(P+'window_level_tagged.csv'); w=w[w.touch>0].copy()
w['grp']=np.where(w.target_cat=='bowl','bowl (novel cat)',np.where(w.target_mesh_seen,'seen mesh','novel mesh, seen cat'))
# ---- 1. composition (verb x target cat unseen) in seen categories
nb=w[w.target_cat!='bowl']
print('== non-bowl: verb_target_seen effect')
print(nb.groupby(['verb_target_seen']).agg(n=('contact','size'),nseq=('sequence_id','nunique'),contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean')).round(1))
print(nb.groupby(['target_mesh_seen','verb_target_seen']).agg(n=('contact','size'),nseq=('sequence_id','nunique'),contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean')).round(1))
print('== non-bowl: triplet_seen / mesh_pair_seen')
print(nb.groupby(['triplet_seen','mesh_pair_seen']).agg(n=('contact','size'),nseq=('sequence_id','nunique'),contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean')).round(1))
# ---- 4. within-bowl variance
b=w[w.target_cat=='bowl']
def eta2(df,col,y='contact'):
    g=df.groupby(col)[y]; return 1-(g.transform('mean')-df[y]).pow(2).sum()/((df[y]-df[y].mean())**2).sum()
print('== within-bowl eta^2 of contact by:', {c:round(eta2(b,c),3) for c in ['verb','target_mesh','tool_mesh','sequence_id','split','hand','tool_cat']})
print('== within non-bowl seen eta^2:', {c:round(eta2(nb,c),3) for c in ['verb','target_mesh','target_cat','tool_mesh','sequence_id','split','hand']})
ms=pd.read_csv(P+'mesh_size.csv').set_index('mesh')
bm=b.groupby('target_mesh').agg(n=('contact','size'),contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean'),touch=('touch','mean')); bm['radius']=ms.loc[bm.index,'radius_m'].round(3)
print('== bowl by mesh'); print(bm.round(1)); print('corr radius vs contact:', np.corrcoef(bm.radius,bm.contact)[0,1].round(2))
print('== bowl by verb'); print(b.groupby('verb').agg(n=('contact','size'),nseq=('sequence_id','nunique'),contact=('contact','mean'),c_trans=('c_trans','mean'),motion=('motion','mean'),cmap=('cmap','mean'),iou=('iou','mean'),touch=('touch','mean')).round(2))
# ---- 2/3. block decomposition + similarity
blk={'L/tool':slice(0,512),'L/target':slice(512,1024),'R/tool':slice(1024,1536),'R/target':slice(1536,2048)}
rows=[]; sim_rows=[]
# train templates: per (target_cat) and per (verb, target_cat) mean recorded TOUCH-probability map in target block, from train_store
off=pd.read_csv(S+'offsets.csv'); C=np.load(S+'contact.npy',mmap_mode='r')
tr=off[off.split=='train']
tmpl={}   # (hand, key) -> mean touch map (512,)
acc={}
for _,r in tr.iterrows():
    m=np.asarray(C[r.start:r.start+r.n_frames_store],dtype=np.float32)
    for hand,sl in [('L',blk['L/target']),('R',blk['R/target'])]:
        t=(m[:,sl]<0.01); 
        if t.any(axis=1).sum()==0: continue
        tm=t[t.any(axis=1)].mean(0)   # touch prob over touching frames
        for key in [('cat',r.target_cat),('verbcat',r.verb+'|'+r.target_cat),('verb',r.verb),('mesh',r.target_mesh)]:
            k=(hand,)+key; a=acc.setdefault(k,[np.zeros(512),0]); a[0]+=tm; a[1]+=1
tmpl={k:v[0]/v[1] for k,v in acc.items()}
cats=sorted(tr.target_cat.unique())
def corr(a,b):
    a=a-a.mean(); b=b-b.mean(); d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d>0 else np.nan
for split in ['test_1','test_2','test_3','test_4']:
    z=np.load(P+f'{split}_contact_maps.npz'); pred=z['pred']; gt=z['gt']; idx=pd.read_csv(P+f'{split}_contact_maps_index.csv')
    tags=w[w.split==split].set_index(['window','hand'])
    for i,r in idx.iterrows():
        p=pred[i].astype(np.float32); g=gt[i].astype(np.float32)
        for hand in ['L','R']:
            if (r.window,hand) not in tags.index: continue
            t=tags.loc[(r.window,hand)]
            d={'split':split,'window':r.window,'hand':hand,'verb':r.verb,'target_cat':r.target_cat,'target_mesh':r.target_mesh,'grp':t.grp,'contact':t.contact,'cmap':t.cmap}
            for bn in ['tool','target']:
                sl=blk[f'{hand}/{bn}']; e=np.abs(p[:,sl]-g[:,sl])
                d[f'err_{bn}_mm']=e.mean()*1000
                gt_t=g[:,sl]<0.01; pr_t=p[:,sl]<0.01
                d[f'gt_touch_{bn}']=gt_t.mean(); d[f'pred_touch_{bn}']=pr_t.mean()
                inter=(gt_t&pr_t).sum(); uni=(gt_t|pr_t).sum(); d[f'iou_{bn}']=inter/uni if uni>0 else np.nan
                # touching frames only: does the pred touch region sit in the right place? centroid over basis index is meaningless; use IoU
            # similarity of predicted target-block touch map (mean over frames) to templates
            sl=blk[f'{hand}/target']; pm=(p[:,sl]<0.01).mean(0); gm=(g[:,sl]<0.01).mean(0)
            if pm.sum()>0:
                d['sim_pred_vs_gt']=corr(pm,gm)
                d['sim_gt_vs_owncat']=corr(gm,tmpl[(hand,'cat',r.target_cat)]) if (hand,'cat',r.target_cat) in tmpl else np.nan
                best=None
                for c in cats:
                    k=(hand,'cat',c)
                    if k in tmpl:
                        s=corr(pm,tmpl[k]); d[f'simcat_{c}']=s
                        if best is None or s>best[1]: best=(c,s)
                d['pred_nearest_cat']=best[0]; d['pred_nearest_cat_sim']=best[1]
                # same-verb template of the seen cat that's nearest
                k=(hand,'verb',r.verb); d['sim_pred_vs_verbtmpl']=corr(pm,tmpl[k]) if k in tmpl else np.nan
                d['sim_gt_vs_verbtmpl']=corr(gm,tmpl[k]) if k in tmpl else np.nan
                k=(hand,'mesh',r.target_mesh); d['sim_pred_vs_meshtmpl']=corr(pm,tmpl[k]) if k in tmpl else np.nan
                d['sim_gt_vs_meshtmpl']=corr(gm,tmpl[k]) if k in tmpl else np.nan
            rows.append(d)
R=pd.DataFrame(rows); R.to_csv(OUT+'c4_block_decomposition.csv',index=False)
print('== per-block map error (mm) and touch stats, touching window-hands')
print(R.groupby('grp').agg(n=('contact','size'),contact=('contact','mean'),cmap=('cmap','mean'),err_tool=('err_tool_mm','mean'),err_target=('err_target_mm','mean'),gt_touch_tool=('gt_touch_tool','mean'),pred_touch_tool=('pred_touch_tool','mean'),gt_touch_target=('gt_touch_target','mean'),pred_touch_target=('pred_touch_target','mean'),iou_tool=('iou_tool','mean'),iou_target=('iou_target','mean')).round(3).to_string())
print(R.groupby(['split','grp']).agg(n=('contact','size'),contact=('contact','mean'),err_tool=('err_tool_mm','mean'),err_target=('err_target_mm','mean'),iou_tool=('iou_tool','mean'),iou_target=('iou_target','mean'),gt_touch_target=('gt_touch_target','mean'),pred_touch_target=('pred_touch_target','mean')).round(3).to_string())
print('== similarity (target block, touch-prob pattern over BPS index)')
print(R.groupby('grp')[['sim_pred_vs_gt','sim_gt_vs_owncat','sim_pred_vs_verbtmpl','sim_gt_vs_verbtmpl','sim_pred_vs_meshtmpl','sim_gt_vs_meshtmpl','pred_nearest_cat_sim']].mean().round(3).to_string())
print('== which train category does the PREDICTED bowl target-map look like?')
print(R[R.target_cat=='bowl'].pred_nearest_cat.value_counts(normalize=True).round(3))
print('   ...and the RECORDED bowl map (gt nearest cat):')
def gtnearest(row):
    pass
print(R[R.target_cat=='bowl'][[f'simcat_{c}' for c in cats]].mean().round(3))
print('== by-verb bowl target-block sim pred vs gt, and gt-vs-verbtmpl')
print(R[R.target_cat=='bowl'].groupby('verb')[['sim_pred_vs_gt','sim_gt_vs_verbtmpl','sim_pred_vs_verbtmpl','err_target_mm','err_tool_mm','contact']].mean().round(3))
