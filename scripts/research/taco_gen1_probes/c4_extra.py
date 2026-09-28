import pandas as pd, numpy as np, json
from scipy import stats
P='/result/uhnam/dexcore/bimart_taco/contact_probe/'; OUT=P+'verify/'
pd.set_option('display.width',250)
w=pd.read_csv(P+'window_level_tagged.csv'); w=w[w.touch>0].copy()
ms=pd.read_csv(P+'mesh_size.csv').set_index('mesh')
w['tgt_size']=ms.loc[w.target_mesh,'radius_m'].values; w['tool_size']=ms.loc[w.tool_mesh,'radius_m'].values
w['novel_tgt']=(~w.target_mesh_seen).astype(float); w['novel_tool']=(~w.tool_mesh_seen).astype(float); w['handR']=(w.hand=='R').astype(float)
w['bowl']=(w.target_cat=='bowl').astype(float); w['vt_unseen']=(~w.verb_target_seen).astype(float); w['pair_unseen']=(~w.mesh_pair_seen).astype(float); w['trip_unseen']=(~w.triplet_seen).astype(float)
def ols_cluster(y,X,cl):
    X=np.asarray(X,float); y=np.asarray(y,float); n,k=X.shape
    Xi=np.linalg.pinv(X.T@X); b=Xi@X.T@y; e=y-X@b; g=pd.factorize(cl)[0]; G=g.max()+1
    meat=sum(np.outer(X[g==i].T@e[g==i],X[g==i].T@e[g==i]) for i in range(G))
    V=Xi@meat@Xi*(G/(G-1))*((n-1)/(n-k)); se=np.sqrt(np.diag(V)); return b,se,2*stats.t.sf(np.abs(b/se),G-1),1-e.var()/y.var()
def fit(df,y,extra,verb=True,split=False,label=''):
    X=pd.DataFrame({'const':1.0,'handR':df.handR,'touch':df.touch,'tgt_size':df.tgt_size,'tool_size':df.tool_size},index=df.index)
    for c in extra: X[c]=df[c]
    if verb: X=pd.concat([X,pd.get_dummies(df.verb,prefix='v',drop_first=True).astype(float)],axis=1)
    if split: X=pd.concat([X,pd.get_dummies(df.split,prefix='s',drop_first=True).astype(float)],axis=1)
    X=X.loc[:,(X.std()>0)|(X.columns=='const')]
    b,se,p,r2=ols_cluster(df[y],X,df.sequence_id)
    out={c:f'{b[i]:.1f}±{se[i]:.1f} (p={p[i]:.2g})' for i,c in enumerate(X.columns) if c in extra}
    print(f'[{label}] y={y} n={len(df)} r2={r2:.3f}', out); return out
nb=w[w.target_cat!='bowl']
print('== non-bowl seen-category subset: geometry-pair vs action-composition novelty (verb dummies + covariates, cluster by seq)')
for y in ['contact','motion','cmap']:
    fit(nb,y,['novel_tgt','novel_tool','pair_unseen','vt_unseen'],label='non-bowl')
    fit(nb,y,['novel_tgt','novel_tool','pair_unseen','vt_unseen'],split=True,label='non-bowl +split')
    fit(nb[nb.split=='test_1'],y,['pair_unseen','vt_unseen'],label='test_1 only')
print('== full sample: bowl vs novel mesh separately')
for y in ['contact','motion','cmap']:
    fit(w,y,['bowl','novel_tgt','novel_tool','pair_unseen','vt_unseen'],label='all')
print('== verb x bowl interaction (contact): per-verb bowl minus seen-target, same verb, cluster SE')
rows=[]
for v,g in w.groupby('verb'):
    if g.bowl.sum()==0: continue
    X=pd.DataFrame({'const':1.0,'bowl':g.bowl,'handR':g.handR,'touch':g.touch,'tool_size':g.tool_size,'novel_tool':g.novel_tool},index=g.index); X=X.loc[:,(X.std()>0)|(X.columns=='const')]
    b,se,p,_=ols_cluster(g.contact,X,g.sequence_id); i=list(X.columns).index('bowl')
    rows.append(dict(verb=v,n_bowl=int(g.bowl.sum()),n_seen=int((1-g.bowl).sum()),bowl_effect=round(b[i],1),se=round(se[i],1),p=p[i],seen_mean=round(g[g.bowl==0].contact.mean(),1),bowl_mean=round(g[g.bowl==1].contact.mean(),1)))
ib=pd.DataFrame(rows).sort_values('bowl_effect'); ib.to_csv(OUT+'c4_bowl_effect_by_verb.csv',index=False); print(ib.to_string())
print('== bowl by hand x verb (contact, c_trans, touch)')
b=w[w.bowl==1]; print(b.pivot_table(index='verb',columns='hand',values=['contact','touch'],aggfunc='mean').round(2))
print('== bowl by tool_cat'); print(b.groupby('tool_cat').agg(n=('contact','size'),nseq=('sequence_id','nunique'),contact=('contact','mean'),motion=('motion','mean')).round(1).sort_values('contact'))
print('== seen-target small containers vs bowl (size-matched): target radius<0.13')
sm=w[w.tgt_size<0.13]; print(sm.groupby('target_cat').agg(n=('contact','size'),nseq=('sequence_id','nunique'),radius=('tgt_size','mean'),contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean')).round(3))
# same-verb same-tool-mesh matched pairs: bowl vs seen target
print('== matched (verb, tool_mesh) cells containing both bowl and seen-target windows')
cells=w.groupby(['verb','tool_mesh']).bowl.agg(['mean','size']); cells=cells[(cells['mean']>0)&(cells['mean']<1)]
m=w.set_index(['verb','tool_mesh']).loc[cells.index].reset_index()
mm=m.groupby(['verb','tool_mesh','bowl']).contact.mean().unstack(); mm['diff']=mm[1.0]-mm[0.0]
print('n cells',len(mm),'mean diff',round(mm['diff'].mean(),1),'median',round(mm['diff'].median(),1),'frac>0',round((mm['diff']>0).mean(),2))
wt=stats.wilcoxon(mm['diff']); print('wilcoxon p',wt.pvalue)
mm.to_csv(OUT+'c4_matched_verb_toolmesh_bowl_vs_seen.csv')
