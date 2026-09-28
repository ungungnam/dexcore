import pandas as pd, numpy as np, itertools, json, sys
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/verify/'
rng=np.random.default_rng(0)
meta=pd.read_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/train_seq_meta.csv')
P=np.load(OUT+'train_phase_summaries.npz'); assert (P['sequence_id']==meta.sequence_id.values.astype(str)).all()
S=P['full']; SA=P['approach']; SH=P['hold']
FE={'full':np.arange(2048),'tool':np.r_[0:512,1024:1536],'target':np.r_[512:1024,1536:2048]}
NP=int(sys.argv[1]) if len(sys.argv)>1 else 500

def bw(X,y):
    vs=sorted(set(y)); C={v:X[y==v].mean(0) for v in vs}
    b=np.mean([np.linalg.norm(C[a]-C[b]) for a,b in itertools.combinations(vs,2)])
    w=np.mean([np.linalg.norm(X[i]-C[y[i]]) for i in range(len(y))]); return b,w
def loo(X,y):
    vs=sorted(set(y)); c=0
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False; C={v:X[m&(y==v)].mean(0) for v in vs}
        c+=min(vs,key=lambda v:np.linalg.norm(X[i]-C[v]))==y[i]
    return c/len(y)
def perm_within(y): return rng.permutation(y)

def analyse(groups,name,feats=('full',),nperm=NP):
    """groups: list of (gname, X[n,d], y[n], info). Same statistics as the original C3 script,
    plus a joint (all-groups) permutation p-value for the pooled statistics."""
    rows=[]; obs={}; 
    for f in feats:
        per=[]
        for gname,X,y,info in groups:
            Xf=X[:,FE[f]] if X.shape[1]==2048 else X
            b,w=bw(Xf,y); pr=[]; pa=[]
            for _ in range(50):
                yp=perm_within(y); b2,w2=bw(Xf,yp); pr.append(b2/w2); pa.append(loo(Xf,yp))
            per.append(dict(group=gname,feat=f,n=len(y),n_labels=len(set(y)),labels='|'.join(sorted(set(y))),
                sizes='|'.join(str((y==v).sum()) for v in sorted(set(y))),between=b,within=w,ratio=b/w,
                acc=loo(Xf,y),chance=1/len(set(y)),perm_ratio50=np.mean(pr),perm_p975=np.quantile(pr,.975),
                above_p975=b/w>np.quantile(pr,.975),perm_acc50=np.mean(pa),
                p_group=np.mean(np.array(pr)>=b/w),**info))
        rows+=per
    d=pd.DataFrame(rows)
    # joint permutation for pooled stats (feat full only)
    def pooled(gr,f):
        bs=[];ws=[];ns=[];accs=[]
        for gname,X,y,info in gr:
            Xf=X[:,FE[f]] if X.shape[1]==2048 else X
            b,w=bw(Xf,y); bs.append(b); ws.append(w); ns.append(len(y)); accs.append(loo(Xf,y))
        ns=np.array(ns); return np.average(bs,weights=ns)/np.average(ws,weights=ns), np.average(accs,weights=ns)
    summ={}
    for f in feats:
        r0,a0=pooled(groups,f); pr=[];pa=[]
        for _ in range(nperm):
            gp=[(g,X,perm_within(y),i) for g,X,y,i in groups]; r,a=pooled(gp,f); pr.append(r); pa.append(a)
        pr=np.array(pr); pa=np.array(pa); dd=d[d.feat==f]
        summ[f]=dict(n_groups=len(groups),n_items=int(sum(len(y) for _,_,y,_ in groups)),ratio=r0,perm_ratio_mean=pr.mean(),
            perm_ratio_p975=np.quantile(pr,.975),p_ratio=float(np.mean(pr>=r0)),acc=a0,chance=float(np.average(dd.chance,weights=dd.n)),
            perm_acc_mean=pa.mean(),p_acc=float(np.mean(pa>=a0)),frac_groups_above_p975=float(dd.above_p975.mean()),
            median_group_ratio=float(dd.ratio.median()),frac_groups_p_le_0333=float((dd.p_group<=0.34).mean()))
    print(f'\n### {name}'); print(pd.DataFrame(summ).T.round(3).to_string())
    return d,summ

# ---------- build the C3 groups (meshpair, verb) exactly as before ----------
def build(meta,S,keys,label,min_per=2):
    out=[]
    for key,g in meta.groupby(keys):
        cnt=g[label].value_counts(); keep=cnt[cnt>=min_per].index; g=g[g[label].isin(keep)]
        if g[label].nunique()<2: continue
        key=key if isinstance(key,tuple) else (key,)
        out.append(('/'.join(map(str,key)),S[g.index.values],g[label].values.astype(str),
                    dict(tool_cat=g.tool_cat.iloc[0] if 'tool_mesh' in keys else '',target_cat=g.target_cat.iloc[0])))
    return out
G=build(meta,S,['tool_mesh','target_mesh'],'verb')
results={}
d_verb,results['C3_meshpair_verb']=analyse(G,'C3 reproduction: verb within (tool_mesh,target_mesh)',feats=('full','tool','target'))
d_verb.to_csv(OUT+'c3_repro_groups.csv',index=False)
dv=d_verb[d_verb.feat=='full']
print('\nGroup sizes:',dv.n.value_counts().sort_index().to_dict(),' sizes pattern:',dv.sizes.value_counts().to_dict())
print('Fraction of groups where observed ratio == max over perms (can never exceed p97.5):',np.isclose(dv.ratio,dv.perm_p975).mean())

# ---------- JOB 1: verb-pair composition ----------
pairs=[]
for gname,X,y,info in G:
    vs=sorted(set(y))
    for a,b in itertools.combinations(vs,2):
        m=np.isin(y,[a,b]); Xs=X[m]; ys=y[m]
        bo,wo=bw(Xs,ys); r=bo/wo
        # exact enumeration of all distinct labelings with the same sizes
        idx=np.arange(len(ys)); na=(ys==a).sum(); rs=[]
        for comb in itertools.combinations(idx,na):
            yp=np.array([b]*len(ys),dtype=object); yp[list(comb)]=a; yp=yp.astype(str)
            b2,w2=bw(Xs,yp); rs.append(b2/w2)
        rs=np.array(rs); 
        pairs.append(dict(group=gname,**info,verb_a=a,verb_b=b,n_a=na,n_b=len(ys)-na,ratio=r,acc=loo(Xs,ys),
            n_partitions=len(rs),exact_p=float(np.mean(rs>=r-1e-9)),is_best=bool(r>=rs.max()-1e-9),
            ratio_target=bw(Xs[:,FE['target']],ys)[0]/bw(Xs[:,FE['target']],ys)[1],
            ratio_tool=bw(Xs[:,FE['tool']],ys)[0]/bw(Xs[:,FE['tool']],ys)[1]))
pairs=pd.DataFrame(pairs); pairs['pair']=pairs.verb_a+' | '+pairs.verb_b
pairs.to_csv(OUT+'c3_verb_pairs_within_meshpair.csv',index=False)
GRIP_EQ={'brush | dust','cut | scrape off'}
pairs['grip_equiv_per_task']=pairs.pair.isin(GRIP_EQ)
agg=pairs.groupby('pair').agg(n_groups=('ratio','size'),n_distinct_meshpairs=('group','nunique'),tool_cats=('tool_cat',lambda s:'|'.join(sorted(set(s)))),
    mean_ratio=('ratio','mean'),median_ratio=('ratio','median'),mean_ratio_target=('ratio_target','mean'),mean_ratio_tool=('ratio_tool','mean'),
    mean_acc=('acc','mean'),frac_best_partition=('is_best','mean'),mean_exact_p=('exact_p','mean'),grip_equiv=('grip_equiv_per_task','first')).sort_values('n_groups',ascending=False)
agg['expected_frac_best_null']=1/3
agg.to_csv(OUT+'c3_verb_pair_summary.csv')
print('\n### Verb-pair composition of the 80 mesh-pair groups (all pairs, 2 seqs per verb unless noted)')
print(agg.round(3).to_string())
print('\nShare of verb pairs that are grip-equivalent per task (brush/dust, cut/scrape off): %.3f (%d/%d)'%(pairs.grip_equiv_per_task.mean(),pairs.grip_equiv_per_task.sum(),len(pairs)))
print('Share of GROUPS whose ONLY pair is grip-equivalent: %.3f'%(dv.labels.isin(['brush|dust','cut|scrape off']).mean()))
# Fisher-style pooled test for "is_best" using joint permutation within groups (handles dependence among pairs in a group)
def frac_best(Gs):
    cnt=0;tot=0
    for gname,X,y,info in Gs:
        for a,b in itertools.combinations(sorted(set(y)),2):
            m=np.isin(y,[a,b]); Xs=X[m]; ys=y[m]; bo,wo=bw(Xs,ys); r=bo/wo
            na=(ys==a).sum(); best=True
            for comb in itertools.combinations(np.arange(len(ys)),na):
                yp=np.array([b]*len(ys),dtype=object); yp[list(comb)]=a; b2,w2=bw(Xs,yp.astype(str))
                if b2/w2>r+1e-9: best=False;break
            cnt+=best; tot+=1
    return cnt/tot
fb=frac_best(G); fbp=[frac_best([(g,X,perm_within(y),i) for g,X,y,i in G]) for _ in range(200)]
results['C3_pairs_frac_best_partition']=dict(observed=fb,null_mean=float(np.mean(fbp)),p=float(np.mean(np.array(fbp)>=fb)),n_pairs=len(pairs))
print('Pooled over all verb pairs: fraction where the true verb split is the best of all partitions = %.3f ; joint-perm null mean %.3f, p=%.3f'%(fb,np.mean(fbp),np.mean(np.array(fbp)>=fb)))
for cls,sub in [('grip-equivalent (brush/dust, cut/scrape off)',pairs.grip_equiv_per_task),('all other pairs',~pairs.grip_equiv_per_task)]:
    p=pairs[sub]; print(f'  {cls}: n_pairs={len(p)}, mean ratio={p.ratio.mean():.3f}, frac_best={p.is_best.mean():.3f}, mean LOO acc={p.acc.mean():.3f}, mean ratio_target={p.ratio_target.mean():.3f}')

# ---------- JOB 2: power checks ----------
# (i) tool_cat within fixed target_mesh
Gi=build(meta,S,['target_mesh'],'tool_cat')
d_i,results['P1_toolcat_within_targetmesh']=analyse(Gi,'(i) tool_cat within fixed target_mesh (full n)',feats=('full','tool','target'))
d_i.to_csv(OUT+'power_i_toolcat_groups.csv',index=False)
# (ii) left vs right hand half, items=(seq,hand), groups=meshpair
Gii=[]
for key,g in meta.groupby(['tool_mesh','target_mesh']):
    if len(g)<2: continue
    idx=g.index.values; X=np.vstack([S[idx][:,:1024],S[idx][:,1024:]]); y=np.array(['L']*len(idx)+['R']*len(idx))
    Gii.append(('/'.join(map(str,key)),X,y,dict(tool_cat=g.tool_cat.iloc[0],target_cat=g.target_cat.iloc[0])))
d_ii,results['P2_left_vs_right']=analyse(Gii,'(ii) left-hand half vs right-hand half, same sequences, groups=meshpair (full n)')
# (iii) approach (first 20%) vs hold (20-80%) frames
Giii=[]
for key,g in meta.groupby(['tool_mesh','target_mesh']):
    if len(g)<2: continue
    idx=g.index.values; X=np.vstack([SA[idx],SH[idx]]); y=np.array(['approach']*len(idx)+['hold']*len(idx))
    Giii.append(('/'.join(map(str,key)),X,y,dict(tool_cat=g.tool_cat.iloc[0],target_cat=g.target_cat.iloc[0])))
d_iii,results['P3_approach_vs_hold']=analyse(Giii,'(iii) approach vs hold frames, groups=meshpair (full n)',feats=('full','tool','target'))

# ---------- matched-n (2|2) versions: same design as 62/80 verb groups ----------
def match22(Gs,draws=3):
    out=[]
    for gname,X,y,info in Gs:
        vs=sorted(set(y))
        for a,b in itertools.combinations(vs,2):
            ia=np.where(y==a)[0]; ib=np.where(y==b)[0]
            if len(ia)<2 or len(ib)<2: continue
            for k in range(draws):
                sel=np.r_[rng.choice(ia,2,replace=False),rng.choice(ib,2,replace=False)]
                out.append((f'{gname}:{a}|{b}:{k}',X[sel],y[sel],info))
    return out
for nm,Gs in [('(i) tool_cat|target_mesh',Gi),('(ii) L|R',Gii),('(iii) approach|hold',Giii),('C3 verb pairs',G)]:
    Gm=match22(Gs,draws=1 if nm.startswith('C3') else 2)
    _,results['matched22_'+nm]=analyse(Gm,f'MATCHED 2|2 design: {nm}',nperm=200)
json.dump({k:{kk:(vv if not isinstance(vv,dict) else {a:float(b) for a,b in vv.items()}) for kk,vv in v.items()} for k,v in results.items()},open(OUT+'c3_verify_results.json','w'),indent=1,default=float)
