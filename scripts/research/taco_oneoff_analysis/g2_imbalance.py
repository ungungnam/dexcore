import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
OUT='/result/uhnam/dexcore/bimart_taco/contact_probe/gen/'
P='/result/uhnam/dexcore/bimart_taco/contact_probe/'
rng=np.random.default_rng(0)
si=pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
tr=si[si.split=='train']
# ---------- 1. support counts ----------
def cnt(keys):
    return tr.groupby(keys).size()
sup={}
sup['n_verb']=cnt(['verb']); sup['n_triplet']=cnt(['triplet']); sup['n_tool_mesh']=cnt(['tool_mesh'])
sup['n_target_mesh']=cnt(['target_mesh']); sup['n_mesh_pair']=cnt(['tool_mesh','target_mesh']); sup['n_verb_tool']=cnt(['verb','tool_mesh'])
keymap={'n_verb':['verb'],'n_triplet':['triplet'],'n_tool_mesh':['tool_mesh'],'n_target_mesh':['target_mesh'],'n_mesh_pair':['tool_mesh','target_mesh'],'n_verb_tool':['verb','tool_mesh']}
seq=si.copy()
for k,keys in keymap.items():
    idx=pd.MultiIndex.from_frame(seq[keys]) if len(keys)>1 else seq[keys[0]]
    seq[k]=sup[k].reindex(idx).fillna(0).astype(int).values
# for train sequences, exclude itself (leave-one-out support)
for k in keymap: seq.loc[seq.split=='train',k]-=1
seq_sup=seq[['sequence_id','split','triplet','verb','tool_mesh','target_mesh','seq']+list(keymap)]
seq_sup.to_csv(OUT+'g2_sequence_support.csv',index=False)
# window-level join
wl=pd.read_csv(P+'window_level.csv')
parts=[]
for s in ['train','test_1','test_2','test_3','test_4']:
    m=pd.read_csv(P+f'{s}.csv',usecols=['split','window','sequence_id']).drop_duplicates(['split','window'])
    parts.append(m)
m=pd.concat(parts)
wl=wl.merge(m,on=['split','window'],how='left')
assert wl.sequence_id.notna().all()
wl=wl.merge(seq_sup.drop(columns=['split','triplet','verb','tool_mesh','target_mesh']),on='sequence_id',how='left')
wl.to_csv(OUT+'g2_window_level_support.csv',index=False)
SUPS=list(keymap)
t=wl[wl.touch>0].copy()
t1=t[t.split=='test_1'].copy()
trw=t[t.split=='train'].copy()
print('touching window-hands: test_1',len(t1),'train',len(trw),'test_1 seqs',t1.sequence_id.nunique(),'train seqs sampled',trw.sequence_id.nunique())
res={}
# sequence-level table (mean over touching window-hands of a sequence)
def seqlevel(df):
    g=df.groupby('sequence_id').agg(contact=('contact','mean'),motion=('motion','mean'),n=('contact','size'),
        verb=('verb','first'),triplet=('triplet','first') if 'triplet' in df else ('verb','first'),
        **{k:(k,'first') for k in SUPS}).reset_index()
    return g
t1['triplet']=t1.sequence_id.str.rsplit('/',n=1).str[0]
trw['triplet']=trw.sequence_id.str.rsplit('/',n=1).str[0]
s1=seqlevel(t1); str_=seqlevel(trw)
print('test_1 sequence-level n=',len(s1))
# ---------- 2. Spearman + OLS ----------
def ols(X,y):
    X=np.column_stack([np.ones(len(y)),X]); b,*_=np.linalg.lstsq(X,y,rcond=None); yhat=X@b
    r2=1-((y-yhat)**2).sum()/((y-y.mean())**2).sum(); return b,r2
def boot_ols(df,cols,ycol,dummies=None,B=2000):
    def design(d):
        X=np.log1p(d[cols].values.astype(float))
        if dummies is not None:
            D=pd.get_dummies(d['verb']).reindex(columns=dummies,fill_value=0).values[:,1:].astype(float)  # drop first
            X=np.column_stack([X,D])
        return X
    b0,r2=ols(design(df),df[ycol].values)
    bs=[]
    for _ in range(B):
        d=df.sample(len(df),replace=True,random_state=rng.integers(1e9))
        try: bs.append(ols(design(d),d[ycol].values)[0])
        except Exception: pass
    bs=np.array(bs); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    return b0,lo,hi,r2
rows=[]
verbs_t1=sorted(s1.verb.unique())
for ycol in ['contact','motion']:
    for k in SUPS:
        rw,pw=spearmanr(t1[k],t1[ycol]); rs,ps=spearmanr(s1[k],s1[ycol])
        b,lo,hi,r2=boot_ols(s1,[k],ycol)
        bd,lod,hid,r2d=boot_ols(s1,[k],ycol,dummies=verbs_t1)
        rows.append(dict(y=ycol,support=k,model='single',spearman_window=rw,p_window=pw,spearman_seq=rs,p_seq=ps,
            n_seq=len(s1),coef_per_log1p=b[1],ci_lo=lo[1],ci_hi=hi[1],r2=r2,
            coef_with_verb_dummies=bd[1],ci_lo_vd=lod[1],ci_hi_vd=hid[1],r2_vd=r2d))
    b,lo,hi,r2=boot_ols(s1,SUPS,ycol); bd,lod,hid,r2d=boot_ols(s1,SUPS,ycol,dummies=verbs_t1)
    for i,k in enumerate(SUPS):
        rows.append(dict(y=ycol,support=k,model='joint',n_seq=len(s1),coef_per_log1p=b[i+1],ci_lo=lo[i+1],ci_hi=hi[i+1],r2=r2,
            coef_with_verb_dummies=bd[i+1],ci_lo_vd=lod[i+1],ci_hi_vd=hid[i+1],r2_vd=r2d))
    # verb dummies only
    bv,r2v=ols(pd.get_dummies(s1['verb']).values[:,1:].astype(float),s1[ycol].values)
    rows.append(dict(y=ycol,support='verb_dummies_only',model='dummies',n_seq=len(s1),r2=r2v))
reg=pd.DataFrame(rows); reg.to_csv(OUT+'g2_test1_support_regressions.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(reg.round(3).to_string())
# ---------- 3. verb-level ----------
nv=tr.verb.value_counts()
vt=[]
for v in sorted(si.verb.unique()):
    a=t1[t1.verb==v]; b_=trw[trw.verb==v]; sa=s1[s1.verb==v]; sb=str_[str_.verb==v]
    vt.append(dict(verb=v,n_train_seq=int(nv.get(v,0)),n_test1_seq=si[(si.split=='test_1')&(si.verb==v)].shape[0],
        n_test1_wh=len(a),n_train_wh=len(b_),
        test1_contact=a.contact.mean(),train_contact=b_.contact.mean(),ratio_contact=a.contact.mean()/b_.contact.mean() if len(b_) and len(a) else np.nan,
        test1_motion=a.motion.mean(),train_motion=b_.motion.mean(),ratio_motion=a.motion.mean()/b_.motion.mean() if len(b_) and len(a) else np.nan,
        test1_contact_seqmean=sa.contact.mean(),train_contact_seqmean=sb.contact.mean(),test1_motion_seqmean=sa.motion.mean(),train_motion_seqmean=sb.motion.mean(),
        n_test1_seq_touch=len(sa),n_train_seq_sampled=len(sb)))
vt=pd.DataFrame(vt); vt.to_csv(OUT+'g2_verb_level.csv',index=False)
print(vt.round(2).to_string())
vv=vt.dropna(subset=['ratio_contact'])
vv=vv[(vv.n_test1_wh>=10)&(vv.n_train_wh>=10)]
res['verb_ratio_spearman']={}
for c in ['ratio_contact','ratio_motion','test1_contact','test1_motion','train_contact','train_motion']:
    r,p=spearmanr(vv.n_train_seq,vv[c]); res['verb_ratio_spearman'][c]=dict(rho=r,p=p,n_verbs=len(vv))
    print(f'verb-level spearman(n_train_seq, {c}) rho={r:.3f} p={p:.3f} n={len(vv)}')
# weighted by n windows
# ---------- 4. gap under restriction / reweighting ----------
gap=[]
def add(name,d,dtr=None):
    dtr=trw if dtr is None else dtr
    gs=d.groupby('sequence_id')[['contact','motion']].mean(); gtr=dtr.groupby('sequence_id')[['contact','motion']].mean()
    gap.append(dict(subset=name,n_wh=len(d),n_seq=d.sequence_id.nunique(),contact=d.contact.mean(),motion=d.motion.mean(),
        contact_seqmean=gs.contact.mean(),motion_seqmean=gs.motion.mean(),
        train_contact=dtr.contact.mean(),train_motion=dtr.motion.mean(),train_n_wh=len(dtr),
        ratio_contact=d.contact.mean()/dtr.contact.mean(),ratio_motion=d.motion.mean()/dtr.motion.mean()))
add('test_1 all touching',t1)
add('triplet>=20 & mesh_pair>=3',t1[(t1.n_triplet>=20)&(t1.n_mesh_pair>=3)])
add('triplet>=20 & mesh_pair>=3 (train same filter)',t1[(t1.n_triplet>=20)&(t1.n_mesh_pair>=3)],trw[(trw.n_triplet>=20)&(trw.n_mesh_pair>=3)])
add('triplet>=10 & mesh_pair>=3',t1[(t1.n_triplet>=10)&(t1.n_mesh_pair>=3)])
add('triplet>=10 & mesh_pair>=3 (train same filter)',t1[(t1.n_triplet>=10)&(t1.n_mesh_pair>=3)],trw[(trw.n_triplet>=10)&(trw.n_mesh_pair>=3)])
add('verb>=100',t1[t1.n_verb>=100]); add('verb>=100 (train same filter)',t1[t1.n_verb>=100],trw[trw.n_verb>=100])
add('verb<30',t1[t1.n_verb<30]); add('verb<30 (train same filter)',t1[t1.n_verb<30],trw[trw.n_verb<30])
add('mesh_pair>=5',t1[t1.n_mesh_pair>=5]); add('mesh_pair>=5 (train same)',t1[t1.n_mesh_pair>=5],trw[trw.n_mesh_pair>=5])
add('mesh_pair>=5 & verb_tool>=5',t1[(t1.n_mesh_pair>=5)&(t1.n_verb_tool>=5)]); add('mesh_pair>=5 & verb_tool>=5 (train same)',t1[(t1.n_mesh_pair>=5)&(t1.n_verb_tool>=5)],trw[(trw.n_mesh_pair>=5)&(trw.n_verb_tool>=5)])
add('all supports >= median of test_1',t1[np.all([t1[k]>=t1[k].median() for k in SUPS],axis=0)])
add('all supports >= median (train same)',t1[np.all([t1[k]>=t1[k].median() for k in SUPS],axis=0)],trw[np.all([trw[k]>=t1[k].median() for k in SUPS],axis=0)])
# top-quartile triplet support
q=s1.n_triplet.quantile(0.75)
add(f'triplet support top quartile (>={q:.0f})',t1[t1.n_triplet>=q]); add(f'triplet top quartile (train same)',t1[t1.n_triplet>=q],trw[trw.n_triplet>=q])
# reweight test_1 windows to the train verb distribution
wtr=trw.verb.value_counts(normalize=True); wt1=t1.verb.value_counts(normalize=True)
w=t1.verb.map(wtr/wt1).fillna(0)
gap.append(dict(subset='test_1 reweighted to train verb distribution',n_wh=len(t1),n_seq=t1.sequence_id.nunique(),contact=np.average(t1.contact,weights=w),motion=np.average(t1.motion,weights=w),
    train_contact=trw.contact.mean(),train_motion=trw.motion.mean(),train_n_wh=len(trw),ratio_contact=np.average(t1.contact,weights=w)/trw.contact.mean(),ratio_motion=np.average(t1.motion,weights=w)/trw.motion.mean()))
# reweight to train triplet distribution (only triplets in both)
wtr=trw.triplet.value_counts(normalize=True); wt1=t1.triplet.value_counts(normalize=True)
w=t1.triplet.map(wtr/wt1).fillna(0)
gap.append(dict(subset='test_1 reweighted to train(sampled) triplet distribution',n_wh=int((w>0).sum()),n_seq=t1[w>0].sequence_id.nunique(),contact=np.average(t1.contact,weights=w),motion=np.average(t1.motion,weights=w),
    train_contact=trw.contact.mean(),train_motion=trw.motion.mean(),train_n_wh=len(trw),ratio_contact=np.average(t1.contact,weights=w)/trw.contact.mean(),ratio_motion=np.average(t1.motion,weights=w)/trw.motion.mean()))
# matched-triplet comparison: for each triplet present in both, ratio
both=set(t1.triplet)&set(trw.triplet)
mt=[]
for tp in both:
    a=t1[t1.triplet==tp]; b_=trw[trw.triplet==tp]
    mt.append(dict(triplet=tp,n_train_seq=int(seq_sup[(seq_sup.split=='train')&(seq_sup.triplet==tp)].shape[0]),n_test1_wh=len(a),n_train_wh=len(b_),
        test1_contact=a.contact.mean(),train_contact=b_.contact.mean(),test1_motion=a.motion.mean(),train_motion=b_.motion.mean()))
mt=pd.DataFrame(mt); mt['ratio_contact']=mt.test1_contact/mt.train_contact; mt['ratio_motion']=mt.test1_motion/mt.train_motion
mt=mt.sort_values('n_train_seq',ascending=False); mt.to_csv(OUT+'g2_triplet_matched.csv',index=False)
print(mt.round(2).to_string())
mm=mt[(mt.n_test1_wh>=10)&(mt.n_train_wh>=10)]
for c in ['ratio_contact','ratio_motion']:
    r,p=spearmanr(mm.n_train_seq,mm[c]); res.setdefault('triplet_matched_spearman',{})[c]=dict(rho=r,p=p,n_triplets=len(mm))
    print(f'triplet-matched spearman(n_train_seq,{c}) rho={r:.3f} p={p:.3f} n={len(mm)}')
gap.append(dict(subset='matched triplets (>=10 wh each side), unweighted mean of per-triplet ratios',n_wh=int(mm.n_test1_wh.sum()),n_seq=np.nan,contact=np.nan,motion=np.nan,train_contact=np.nan,train_motion=np.nan,train_n_wh=int(mm.n_train_wh.sum()),ratio_contact=mm.ratio_contact.mean(),ratio_motion=mm.ratio_motion.mean()))
gap=pd.DataFrame(gap); gap.to_csv(OUT+'g2_gap_under_support_restriction.csv',index=False)
print(gap.round(2).to_string())
# support distribution summaries
sd={k:dict(test1_seq_median=float(s1[k].median()),test1_seq_min=int(s1[k].min()),test1_seq_max=int(s1[k].max()),
           train_seq_median=float(seq_sup[seq_sup.split=='train'][k].median())) for k in SUPS}
res['support_distribution']=sd
# quartile bins of triplet support with seq-level error
for k in ['n_triplet','n_mesh_pair','n_verb','n_verb_tool']:
    s1['bin']=pd.qcut(s1[k].rank(method='first'),4,labels=['Q1 low','Q2','Q3','Q4 high'])
    b=s1.groupby('bin',observed=True).agg(n_seq=('contact','size'),support_min=(k,'min'),support_max=(k,'max'),contact=('contact','mean'),motion=('motion','mean')).reset_index()
    b.insert(0,'support',k); print(b.round(2).to_string())
    b.to_csv(OUT+f'g2_test1_seqlevel_bins_{k}.csv',index=False)
json.dump(res,open(OUT+'g2_summary.json','w'),indent=1,default=float)
