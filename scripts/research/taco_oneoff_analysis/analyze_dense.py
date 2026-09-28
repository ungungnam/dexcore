import pandas as pd, numpy as np, itertools, pickle, json
rng=np.random.default_rng(1)
meta=pd.read_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/train_seq_meta.csv')
D=pickle.load(open('/result/uhnam/dexcore/bimart_taco/contact_probe/train_dense_summaries.pkl','rb'))
NP=50
def bw(X,y):
    vs=sorted(set(y)); C={v:X[y==v].mean(0) for v in vs}
    return np.mean([np.linalg.norm(C[a]-C[b]) for a,b in itertools.combinations(vs,2)]), np.mean([np.linalg.norm(X[i]-C[y[i]]) for i in range(len(y))])
def loo_nc(X,y):
    vs=sorted(set(y)); c=0
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False; C={v:X[m&(y==v)].mean(0) for v in vs}
        c+=min(vs,key=lambda v:np.linalg.norm(X[i]-C[v]))==y[i]
    return c/len(y)
def loo_1nn(X,y):
    c=0
    for i in range(len(y)):
        d=np.linalg.norm(X-X[i],axis=1); d[i]=np.inf; c+=y[np.argmin(d)]==y[i]
    return c/len(y)
def feats(sid,kind,n_tool):
    d=D[sid]
    if kind=='dense_full': return d['soft'].reshape(-1)
    if kind=='dense_tool': return d['soft'][:,:n_tool].reshape(-1)
    if kind=='dense_target': return d['soft'][:,n_tool:].reshape(-1)
    if kind=='dense_left': return d['soft'][0]
    if kind=='dense_right': return d['soft'][1]
    if kind=='dense_hard_full': return d['hard'].reshape(-1)
    if kind=='dense_full_l2norm':
        x=d['soft'].reshape(-1); return x/ (np.linalg.norm(x)+1e-8)
    if kind=='dense_tool_l2norm':
        x=d['soft'][:,:n_tool].reshape(-1); return x/(np.linalg.norm(x)+1e-8)
    if kind=='prof16': return d['prof16'].reshape(-1)          # 64-d temporal fraction-in-contact
    if kind=='mind16': return d['mind16'].reshape(-1)          # 64-d temporal min distance
    if kind=='prof_mean': return d['prof_mean']                # 4-d hand/object usage
KINDS=['dense_full','dense_tool','dense_target','dense_left','dense_right','dense_hard_full','dense_full_l2norm','dense_tool_l2norm','prof16','mind16','prof_mean']
rows=[]
for level,keys in [('meshpair',['tool_mesh','target_mesh']),('catpair',['tool_cat','target_cat'])]:
  for key,g in meta.groupby(keys):
    cnt=g.verb.value_counts(); keep=cnt[cnt>=2].index; g=g[g.verb.isin(keep)]
    if g.verb.nunique()<2: continue
    y=g.verb.values
    n_tool=min(D[s]['n_tool_ub'] for s in g.sequence_id)  # conservative boundary (tool part surely tool)
    Vs={D[s]['V'] for s in g.sequence_id}
    for kind in KINDS:
        if level=='catpair' and kind.startswith('dense'): continue   # dense verts not comparable across meshes
        if kind.startswith('dense') and len(Vs)>1: continue
        X=np.stack([feats(s,kind,n_tool) for s in g.sequence_id])
        b,w=bw(X,y); acc=loo_nc(X,y); a1=loo_1nn(X,y)
        pr=[];pa=[];p1=[]
        for p in range(NP):
            yp=rng.permutation(y); b2,w2=bw(X,yp); pr.append(b2/w2); pa.append(loo_nc(X,yp)); p1.append(loo_1nn(X,yp))
        rows.append(dict(level=level,kind=kind,tool=key[0],target=key[1],tool_cat=g.tool_cat.iloc[0],target_cat=g.target_cat.iloc[0],
            n_seq=len(g),n_verbs=len(keep),verbs='|'.join(sorted(keep)),between=b,within=w,ratio=b/w,perm_ratio=np.mean(pr),perm_ratio3=np.mean(pr[:3]),
            perm_ratio_p975=np.quantile(pr,0.975),ratio_pctile=np.mean(b/w>np.array(pr)),acc=acc,acc_1nn=a1,chance=1/len(keep),perm_acc=np.mean(pa),perm_acc3=np.mean(pa[:3]),perm_acc_1nn=np.mean(p1)))
out=pd.DataFrame(rows); out.to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/gt_contact_by_verb_dense_temporal.csv',index=False)
print(f"{'level':9s}{'kind':19s}{'ngrp':>5s}{'nseq':>5s}{'ratio':>7s}{'perm':>7s}{'frac>p975':>10s}{'acc':>6s}{'chance':>7s}{'permacc':>8s}{'acc1nn':>7s}{'perm1nn':>8s}")
for (level,kind),d in out.groupby(['level','kind'],sort=False):
    w=d.n_seq.values
    print(f"{level:9s}{kind:19s}{len(d):5d}{w.sum():5d}{np.average(d.between,weights=w)/np.average(d.within,weights=w):7.3f}{np.average(d.perm_ratio,weights=w):7.3f}{np.mean(d.ratio>d.perm_ratio_p975):10.3f}{np.average(d.acc,weights=w):6.3f}{np.average(d.chance,weights=w):7.3f}{np.average(d.perm_acc,weights=w):8.3f}{np.average(d.acc_1nn,weights=w):7.3f}{np.average(d.perm_acc_1nn,weights=w):8.3f}")
