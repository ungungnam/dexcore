import pandas as pd, numpy as np, json, sys
from scipy import stats
P='/result/uhnam/dexcore/bimart_taco/contact_probe/'; OUT=P+'verify/'
w = pd.read_csv(P+'window_level_tagged.csv'); w=w[w.touch>0].copy()
ms = pd.read_csv(P+'mesh_size.csv').set_index('mesh')
w['tgt_size']=ms.loc[w.target_mesh,'radius_m'].values; w['tool_size']=ms.loc[w.tool_mesh,'radius_m'].values
w['novel_tgt']=(~w.target_mesh_seen).astype(float); w['novel_tool']=(~w.tool_mesh_seen).astype(float)
w['handR']=(w.hand=='R').astype(float); w['bowl']=(w.target_cat=='bowl').astype(float)

def ols_cluster(y, X, cl):
    X=np.asarray(X,float); y=np.asarray(y,float); n,k=X.shape
    XtX_inv=np.linalg.pinv(X.T@X); b=XtX_inv@X.T@y; e=y-X@b
    groups=pd.factorize(cl)[0]; G=groups.max()+1
    meat=np.zeros((k,k))
    for g in range(G):
        idx=groups==g; s=X[idx].T@e[idx]; meat+=np.outer(s,s)
    V=XtX_inv@meat@XtX_inv*(G/(G-1))*((n-1)/(n-k))
    se=np.sqrt(np.diag(V)); t=b/se; p=2*stats.t.sf(np.abs(t),G-1)
    r2=1-e.var()/y.var()
    return b,se,p,r2,G

def build(df, verb=True, split=False, cat=False):
    cols={'const':1.0,'novel_tgt':df.novel_tgt,'novel_tool':df.novel_tool,'handR':df.handR,'touch':df.touch,'tgt_size':df.tgt_size,'tool_size':df.tool_size}
    X=pd.DataFrame(cols,index=df.index)
    if verb: X=pd.concat([X,pd.get_dummies(df.verb,prefix='v',drop_first=True).astype(float)],axis=1)
    if split: X=pd.concat([X,pd.get_dummies(df.split,prefix='s',drop_first=True).astype(float)],axis=1)
    if cat: X=pd.concat([X,pd.get_dummies(df.target_cat,prefix='tc',drop_first=True).astype(float)],axis=1)
    return X

rows=[]
for ycol in ['contact','motion','cmap','c_trans','c_resid']:
    for name,df,kw in [('base',w,dict(verb=False)),('+verb',w,dict(verb=True)),('+verb+split',w,dict(verb=True,split=True)),
                       ('+verb, non-bowl only',w[w.target_cat!='bowl'],dict(verb=True)),('+verb+split, non-bowl only',w[w.target_cat!='bowl'],dict(verb=True,split=True)),
                       ('+verb+target_cat dummies',w,dict(verb=True,cat=True)),
                       ('+verb, test_3 only',w[w.split=='test_3'],dict(verb=True)),('+verb, test_4 only',w[w.split=='test_4'],dict(verb=True))]:
        X=build(df,**kw); X=X.loc[:,X.std()>0] if 'const' not in X else pd.concat([X[['const']],X.drop(columns='const').loc[:,X.drop(columns='const').std()>0]],axis=1)
        b,se,p,r2,G=ols_cluster(df[ycol],X,df.sequence_id)
        d=dict(zip(X.columns,b)); s=dict(zip(X.columns,se)); pp=dict(zip(X.columns,p))
        rows.append(dict(y=ycol,model=name,n=len(df),n_clusters=G,r2=round(r2,3),
            novel_tgt=round(d.get('novel_tgt',np.nan),1),se_tgt=round(s.get('novel_tgt',np.nan),1),p_tgt=pp.get('novel_tgt',np.nan),
            novel_tool=round(d.get('novel_tool',np.nan),1),se_tool=round(s.get('novel_tool',np.nan),1),p_tool=pp.get('novel_tool',np.nan),
            tgt_size=round(d['tgt_size'],1),se_tgt_size=round(s['tgt_size'],1),tool_size=round(d['tool_size'],1),touch=round(d['touch'],1),handR=round(d['handR'],1)))
res=pd.DataFrame(rows); res.to_csv(OUT+'c4_ols_cluster_by_sequence.csv',index=False)
pd.set_option('display.width',250); print(res.to_string())
# full coefficient table for contact +verb model
X=build(w,verb=True); b,se,p,r2,G=ols_cluster(w.contact,X,w.sequence_id)
full=pd.DataFrame(dict(coef=b,se=se,p=p),index=X.columns).round(3); full.to_csv(OUT+'c4_ols_contact_verb_full_coefs.csv'); print(full)
