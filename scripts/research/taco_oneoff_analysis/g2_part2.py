import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, mannwhitneyu
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/gen/'
rng=np.random.default_rng(1)
si=pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
si['date']=si.seq.str[:8]
tr=si[si.split=='train']
ss=pd.read_csv(OUT+'g2_sequence_support.csv')
wl=pd.read_csv(OUT+'g2_window_level_support.csv')
wl['triplet']=wl.sequence_id.str.rsplit('/',n=1).str[0]
wl=wl.merge(si[['sequence_id','date']],on='sequence_id')
t=wl[wl.touch>0].copy(); t1=t[t.split=='test_1'].copy(); trw=t[t.split=='train'].copy()
SUPS=['n_verb','n_triplet','n_tool_mesh','n_target_mesh','n_mesh_pair','n_verb_tool']
print('--- support distribution: train (LOO) vs test_1 sequences ---')
for k in SUPS:
    a=ss[ss.split=='train'][k]; b=ss[ss.split=='test_1'][k]
    print(f'{k:14s} train median={a.median():.0f} frac0={np.mean(a==0):.2f} | test_1 median={b.median():.0f} frac0={np.mean(b==0):.2f}  | test_2 median={ss[ss.split=="test_2"][k].median():.0f} frac0={np.mean(ss[ss.split=="test_2"][k]==0):.2f}')
# date-based support
d_trip_date=tr.groupby(['triplet','date']).size(); d_pair_date=tr.groupby(['tool_mesh','target_mesh','date']).size(); d_date=tr.groupby('date').size()
def date_sup(df):
    df=df.copy()
    df['n_same_triplet_same_date']=d_trip_date.reindex(pd.MultiIndex.from_frame(df[['triplet','date']])).fillna(0).astype(int).values
    df['n_same_pair_same_date']=d_pair_date.reindex(pd.MultiIndex.from_frame(df[['tool_mesh','target_mesh','date']])).fillna(0).astype(int).values
    df['n_same_date']=d_date.reindex(df.date).fillna(0).astype(int).values
    return df
si2=date_sup(si)
for s in ['train','test_1','test_2']:
    x=si2[si2.split==s]
    print(s,'n_seq',len(x),'frac date-seen-in-train',np.mean(x.n_same_date>0).round(2),'frac same triplet+date',np.mean(x.n_same_triplet_same_date>0).round(2),'frac same pair+date',np.mean(x.n_same_pair_same_date>0).round(2))
# For test_1: is mesh_pair seen related to same date?
x=si2[si2.split=='test_1']
print(pd.crosstab(x.n_mesh_pair.gt(0) if 'n_mesh_pair' in x else ss[ss.split=='test_1'].n_mesh_pair.gt(0).values, x.n_same_date>0))
x=x.merge(ss[['sequence_id','n_mesh_pair']],on='sequence_id')
print(pd.crosstab(x.n_mesh_pair>0, x.n_same_date>0, rownames=['mesh_pair_seen'],colnames=['date_seen']))
# how many distinct dates per triplet in train, and per mesh pair
print('train: seqs per (triplet,date) median',d_trip_date.median(),' seqs per (pair,date) median',d_pair_date.median())
print('train: distinct mesh pairs per triplet median', tr.groupby('triplet').apply(lambda g: g[['tool_mesh','target_mesh']].drop_duplicates().shape[0]).median())
# --- sequence-level with mesh_pair seen / unseen in test_1, per verb ---
s1=t1.groupby('sequence_id').agg(contact=('contact','mean'),motion=('motion','mean'),verb=('verb','first'),n_mesh_pair=('n_mesh_pair','first'),n_triplet=('n_triplet','first'),triplet=('triplet','first'),tool_mesh=('tool_mesh','first'),target_mesh=('target_mesh','first')).reset_index()
s1['pair_seen']=s1.n_mesh_pair>0
print(s1.groupby('pair_seen')[['contact','motion']].agg(['mean','count']).round(2))
print(pd.crosstab(s1.verb,s1.pair_seen))
print('by verb, seq-level, seen vs unseen mesh pair:')
print(s1.groupby(['verb','pair_seen'])[['contact','motion']].mean().unstack().round(1))
# Mann-Whitney seen vs unseen within verbs with both
# matched mesh-pair comparison: test_1 vs train windows with the SAME mesh pair
trw['pair']=list(zip(trw.tool_mesh,trw.target_mesh)); t1['pair']=list(zip(t1.tool_mesh,t1.target_mesh))
both=set(t1.pair)&set(trw.pair)
rows=[]
for p in both:
    a=t1[t1.pair==p]; b=trw[trw.pair==p]
    rows.append(dict(pair=str(p),verbs_test=','.join(sorted(a.verb.unique())),n_train_seq_pair=int(a.n_mesh_pair.iloc[0]),n_test1_wh=len(a),n_test1_seq=a.sequence_id.nunique(),n_train_wh=len(b),n_train_seq_sampled=b.sequence_id.nunique(),
        test1_contact=a.contact.mean(),train_contact=b.contact.mean(),test1_motion=a.motion.mean(),train_motion=b.motion.mean()))
mp=pd.DataFrame(rows); mp['ratio_contact']=mp.test1_contact/mp.train_contact; mp['ratio_motion']=mp.test1_motion/mp.train_motion
mp=mp.sort_values('n_train_seq_pair',ascending=False); mp.to_csv(OUT+'g2_meshpair_matched.csv',index=False)
print(mp.round(2).to_string())
print('pooled over matched mesh pairs: test1 contact %.2f motion %.2f (n_wh=%d) vs train contact %.2f motion %.2f (n_wh=%d)'%(
    t1[t1.pair.isin(both)].contact.mean(),t1[t1.pair.isin(both)].motion.mean(),t1.pair.isin(both).sum(),trw[trw.pair.isin(both)].contact.mean(),trw[trw.pair.isin(both)].motion.mean(),trw.pair.isin(both).sum()))
# --- sequence-bootstrap CIs for restricted gap ratios ---
def boot_ratio(a,b,B=3000):
    def prep(d):
        g=d.groupby('sequence_id').agg(c=('contact','sum'),m=('motion','sum'),n=('contact','size'))
        return g.c.values,g.m.values,g.n.values
    ca,ma,na=prep(a); cb,mb,nb=prep(b)
    ia=rng.integers(len(na),size=(B,len(na))); ib=rng.integers(len(nb),size=(B,len(nb)))
    Ac=ca[ia].sum(1)/na[ia].sum(1); Am=ma[ia].sum(1)/na[ia].sum(1); Bc=cb[ib].sum(1)/nb[ib].sum(1); Bm=mb[ib].sum(1)/nb[ib].sum(1)
    out=np.column_stack([Ac,Bc,Ac/Bc,Am,Bm,Am/Bm]); return np.percentile(out,[2.5,97.5],axis=0)
rows=[]
def add(name,a,b):
    ci=boot_ratio(a,b)
    rows.append(dict(subset=name,n_test1_wh=len(a),n_test1_seq=a.sequence_id.nunique(),n_train_wh=len(b),n_train_seq=b.sequence_id.nunique(),
        test1_contact=a.contact.mean(),test1_contact_ci=f'[{ci[0,0]:.1f},{ci[1,0]:.1f}]',train_contact=b.contact.mean(),ratio_contact=a.contact.mean()/b.contact.mean(),ratio_contact_ci=f'[{ci[0,2]:.2f},{ci[1,2]:.2f}]',
        test1_motion=a.motion.mean(),test1_motion_ci=f'[{ci[0,3]:.1f},{ci[1,3]:.1f}]',train_motion=b.motion.mean(),ratio_motion=a.motion.mean()/b.motion.mean(),ratio_motion_ci=f'[{ci[0,5]:.2f},{ci[1,5]:.2f}]'))
add('all test_1 vs all train',t1,trw)
add('mesh pair seen (n_mesh_pair>=1) vs train',t1[t1.n_mesh_pair>=1],trw)
add('mesh pair seen (>=1) vs train same pairs',t1[t1.pair.isin(both)],trw[trw.pair.isin(both)])
add('mesh pair unseen (n_mesh_pair==0) vs train',t1[t1.n_mesh_pair==0],trw)
add('mesh_pair>=3 vs train same filter',t1[t1.n_mesh_pair>=3],trw[trw.n_mesh_pair>=3])
add('mesh_pair>=5 vs train same filter',t1[t1.n_mesh_pair>=5],trw[trw.n_mesh_pair>=5])
add('triplet>=10 & mesh_pair>=3 vs train same filter',t1[(t1.n_triplet>=10)&(t1.n_mesh_pair>=3)],trw[(trw.n_triplet>=10)&(trw.n_mesh_pair>=3)])
add('triplet>=30 (top-support triplets) vs train same filter',t1[t1.n_triplet>=30],trw[trw.n_triplet>=30])
add('triplet>=30 & verb_tool>=20 vs train same',t1[(t1.n_triplet>=30)&(t1.n_verb_tool>=20)],trw[(trw.n_triplet>=30)&(trw.n_verb_tool>=20)])
add('verb>=100 vs train same',t1[t1.n_verb>=100],trw[trw.n_verb>=100])
# best-supported everything: verb>=100, triplet>=30, tool_mesh>=median, target_mesh>=median
add('verb>=100 & triplet>=30 & tool_mesh>=10 & target_mesh>=20 vs train same',t1[(t1.n_verb>=100)&(t1.n_triplet>=30)&(t1.n_tool_mesh>=10)&(t1.n_target_mesh>=20)],trw[(trw.n_verb>=100)&(trw.n_triplet>=30)&(trw.n_tool_mesh>=10)&(trw.n_target_mesh>=20)])
# the "easy verb" subset for reference: spatula/pan verbs regardless of pair support
easy=['put in','put out','skim off','stir','stir-fry','pour in some','empty']
add('easy verbs (put in/out, skim, stir(-fry), pour, empty) vs train same verbs',t1[t1.verb.isin(easy)],trw[trw.verb.isin(easy)])
add('easy verbs & mesh pair unseen vs train same verbs',t1[t1.verb.isin(easy)&(t1.n_mesh_pair==0)],trw[trw.verb.isin(easy)])
add('easy verbs & mesh pair seen vs train same verbs',t1[t1.verb.isin(easy)&(t1.n_mesh_pair>=1)],trw[trw.verb.isin(easy)])
add('hard verbs (brush,dust,cut,scrape off) & mesh pair seen vs train same verbs',t1[t1.verb.isin(['brush','dust','cut','scrape off'])&(t1.n_mesh_pair>=1)],trw[trw.verb.isin(['brush','dust','cut','scrape off'])])
add('hard verbs & mesh pair unseen vs train same verbs',t1[t1.verb.isin(['brush','dust','cut','scrape off'])&(t1.n_mesh_pair==0)],trw[trw.verb.isin(['brush','dust','cut','scrape off'])])
g=pd.DataFrame(rows); g.to_csv(OUT+'g2_gap_restricted_bootstrap.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
print(g.round(2).to_string())
# OLS: seq-level contact/motion on verb dummies + pair_seen + log1p(n_mesh_pair) -> does pair_seen explain gap?
import numpy.linalg as la
def ols(X,y):
    X=np.column_stack([np.ones(len(y)),X]); b=la.lstsq(X,y,rcond=None)[0]; yhat=X@b; return b,1-((y-yhat)**2).sum()/((y-y.mean())**2).sum()
D=pd.get_dummies(s1.verb).values[:,1:].astype(float)
for y in ['contact','motion']:
    b,r2=ols(np.column_stack([D,s1.pair_seen.astype(float)]),s1[y].values)
    bs=[]
    for _ in range(2000):
        i=rng.integers(len(s1),size=len(s1)); bs.append(ols(np.column_stack([D[i],s1.pair_seen.values[i].astype(float)]),s1[y].values[i])[0][-1])
    print(f'{y}: seq-level effect of mesh-pair-seen with verb dummies: {b[-1]:.2f} mm, 95%CI [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}], R2={r2:.3f}')
