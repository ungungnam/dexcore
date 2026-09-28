import numpy as np, pandas as pd, json
from scipy import stats
from scipy.spatial.distance import cdist
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',50)
G='/result/uhnam/dexcore/bimart_taco/contact_probe/gen/'; P='/result/uhnam/dexcore/bimart_taco/contact_probe/'
rng=np.random.default_rng(0)
sub=pd.read_csv(G+'betas_per_sequence_subjects.csv',dtype={'date':str})
Lc=[f'L{i}' for i in range(10)]; Rc=[f'R{i}' for i in range(10)]
wl=pd.read_csv(P+'window_level.csv')
keys=pd.concat([pd.read_csv(P+f'{s}.csv',usecols=['split','window','sequence_id','start']).drop_duplicates(['split','window']) for s in ['train','test_1','test_2','test_3','test_4']])
wl=wl.merge(keys,on=['split','window'],how='left'); assert wl.sequence_id.notna().all()
wl=wl.merge(sub[['sequence_id','subject','date','take','seq']],on='sequence_id',how='left'); assert wl.subject.notna().all()
si=pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
tr=sub[sub.split=='train']
n_train_subj=tr.groupby('subject').size()
sub['subj_in_train']=sub.subject.isin(tr.subject)
sub['n_train_seq_subject']=sub.subject.map(n_train_subj).fillna(0).astype(int)
# distance to nearest train beta (excluding own sequence irrelevant: exact duplicates give 0 if subject in train)
TL=tr.groupby('subject')[Lc].first(); TR=tr.groupby('subject')[Rc].first()
dL=cdist(sub[Lc].values,TL.values); dR=cdist(sub[Rc].values,TR.values)
sub['dist_train_L']=dL.min(1); sub['dist_train_R']=dR.min(1)
sub['dist_train_LR']=cdist(sub[Lc+Rc].values, pd.concat([TL,TR],axis=1).values).min(1)
# nearest DIFFERENT train subject (for context)
for i,s in enumerate(sub.subject):
    if s in TL.index:
        j=list(TL.index).index(s); dL[i,j]=np.inf; dR[i,j]=np.inf
sub['dist_nearest_other_train_subj_L']=dL.min(1); sub['dist_nearest_other_train_subj_R']=dR.min(1)
# session flags for test seqs
trset_trip_date=set(zip(tr.triplet,tr.date))
si['date']=si.seq.str[:8]; tri=si[si.split=='train']
trset_mesh_date=set(zip(tri.tool_mesh,tri.target_mesh,tri.date)); trset_date=set(tri.date)
sub=sub.merge(si[['sequence_id','tool_mesh','target_mesh','verb']],on='sequence_id')
sub['same_triplet_date_in_train']=[(t,d) in trset_trip_date for t,d in zip(sub.triplet,sub.date)]
sub['same_meshpair_date_in_train']=[(a,b,d) in trset_mesh_date for a,b,d in zip(sub.tool_mesh,sub.target_mesh,sub.date)]
sub['same_date_in_train']=sub.date.isin(trset_date)
sub['same_subject_triplet_in_train']=[ (s,t) in set(zip(tr.subject,tr.triplet)) for s,t in zip(sub.subject,sub.triplet)]
sub['n_train_seq_same_triplet_date']=[sum((tr.triplet==t)&(tr.date==d)) for t,d in zip(sub.triplet,sub.date)]
# sequence-level errors (touching windows)
t=wl[wl.touch>0]
seq=t.groupby(['split','sequence_id']).agg(contact=('contact','mean'),motion=('motion','mean'),cmap=('cmap','mean'),n_wh=('contact','size')).reset_index()
seq=seq.merge(sub.drop(columns=['split']),on='sequence_id')
seq.to_csv(G+'sequence_level_errors_with_subject.csv',index=False)
sub.to_csv(G+'test_sequence_subject_session_flags.csv',index=False)
out={}
def boot_mean(x,n=5000):
    x=np.asarray(x); 
    if len(x)==0: return (np.nan,np.nan,np.nan)
    m=rng.choice(x,(n,len(x))).mean(1); return (x.mean(),*np.quantile(m,[0.025,0.975]))
def boot_diff(a,b,n=5000):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)==0 or len(b)==0: return (np.nan,np.nan,np.nan)
    d=rng.choice(a,(n,len(a))).mean(1)-rng.choice(b,(n,len(b))).mean(1); return (a.mean()-b.mean(),*np.quantile(d,[0.025,0.975]))
print('='*30,'baseline: sequence-level (touching windows) by split'); 
for s,g in seq.groupby('split'):
    print(s, 'n_seq',len(g),'contact %.1f [%.1f,%.1f]'%boot_mean(g.contact),'motion %.1f [%.1f,%.1f]'%boot_mean(g.motion))
print('window-hand level by split:'); print(t.groupby('split')[['contact','motion']].agg(['mean','size']).round(1))

print('\n','='*30,'TASK 2: subject seen vs unseen')
tst=seq[seq.split!='train']
print(pd.crosstab(tst.split,tst.subj_in_train))
print('test_1: all subjects in train ->', tst[tst.split=='test_1'].subj_in_train.all())
for s,g in tst.groupby('split'):
    a=g[g.subj_in_train]; b=g[~g.subj_in_train]
    if len(b): print(s,'seen n=%d contact %.1f motion %.1f | unseen n=%d contact %.1f motion %.1f | diff contact %.1f [%.1f,%.1f] motion %.1f [%.1f,%.1f]'%(len(a),a.contact.mean(),a.motion.mean(),len(b),b.contact.mean(),b.motion.mean(),*boot_diff(b.contact,a.contact),*boot_diff(b.motion,a.motion)))
# pooled OLS with split dummies (sequence-level)
import numpy.linalg as la
def ols(y,X,names):
    X=np.column_stack([np.ones(len(y)),X]); beta,res,_,_=la.lstsq(X,y,rcond=None)
    r=y-X@beta; s2=r@r/(len(y)-X.shape[1]); cov=s2*la.inv(X.T@X); se=np.sqrt(np.diag(cov))
    return {n:(round(b,2),round(sd,2),round(2*stats.t.sf(abs(b/sd),len(y)-X.shape[1]),4)) for n,b,sd in zip(['const']+names,beta,se)}
D=pd.get_dummies(tst.split,drop_first=True).astype(float)
for y in ['contact','motion']:
    print('pooled OLS',y,'~ split dummies + subj_unseen:',ols(tst[y].values,np.column_stack([D.values,(~tst.subj_in_train).astype(float)]),list(D.columns)+['subj_unseen']))
# also within target_cat != bowl? use all; add bowl dummy
tst=tst.assign(bowl=(tst.sequence_id.map(si.set_index('sequence_id').target_cat)=='bowl').astype(float)); print('bowl seqs per split', tst.groupby('split').bowl.sum().to_dict())
for y in ['contact','motion']:
    print('pooled OLS',y,'~ split + bowl + subj_unseen:',ols(tst[y].values,np.column_stack([D.values,tst.bowl.values,(~tst.subj_in_train).astype(float)]),list(D.columns)+['bowl','subj_unseen']))

print('\n--- test_1 per-subject error vs subject train count')
t1=seq[seq.split=='test_1']
ps=t1.groupby('subject').agg(n_test1_seq=('sequence_id','size'),contact=('contact','mean'),motion=('motion','mean'),n_train_seq=('n_train_seq_subject','first'))
# add train error for same subject (from sampled train windows)
trs=seq[seq.split=='train'].groupby('subject').agg(train_contact=('contact','mean'),train_motion=('motion','mean'),n_train_seq_sampled=('sequence_id','size'))
ps=ps.join(trs)
ps['gap_contact']=ps.contact-ps.train_contact; ps['gap_motion']=ps.motion-ps.train_motion
print(ps.round(1).to_string())
ps.to_csv(G+'test1_per_subject.csv')
for y in ['contact','motion']:
    r=stats.spearmanr(ps.n_train_seq,ps[y]); r2=stats.pearsonr(np.log(ps.n_train_seq),ps[y])
    print(f'test_1 per-subject {y} vs n_train_seq: spearman rho={r[0]:.2f} p={r[1]:.3f}; pearson(log n)={r2[0]:.2f} p={r2[1]:.3f}; n_subjects={len(ps)}')
    # sequence-level weighted
    r=stats.spearmanr(t1.n_train_seq_subject,t1[y]); print(f'   sequence-level (n={len(t1)}): spearman rho={r[0]:.2f} p={r[1]:.3f}')
# within-subject gap: for subjects with both train and test1
w=ps.dropna(subset=['train_contact'])
print('subjects with both train(sampled) and test_1 windows:',len(w),'; mean within-subject gap contact %.1f [%.1f,%.1f], motion %.1f [%.1f,%.1f]'%(*boot_mean(w.gap_contact),*boot_mean(w.gap_motion)))
print('   fraction of subjects where test_1 > train: contact %.2f motion %.2f'%((w.gap_contact>0).mean(),(w.gap_motion>0).mean()))
# train error vs subject count (does model fit big subjects better even in train?)
for y in ['train_contact','train_motion']:
    r=stats.spearmanr(trs.n_train_seq_sampled.index.map(n_train_subj),trs[y]); print(f'train per-subject {y} vs n_train_seq: rho={r[0]:.2f} p={r[1]:.3f} n={len(trs)}')
# variance decomposition: how much of test_1 seq-level variance is between-subject?
for y in ['contact','motion']:
    g=t1.groupby('subject')[y]; ssb=sum(len(v)*(v.mean()-t1[y].mean())**2 for _,v in g); sst=((t1[y]-t1[y].mean())**2).sum()
    F=stats.f_oneway(*[v.values for _,v in g if len(v)>0]); print(f'test_1 {y}: between-subject share of seq-level variance = {ssb/sst:.2f}; one-way ANOVA F={F[0]:.2f} p={F[1]:.4f}')

print('\n','='*30,'TASK 3: session')
t1=seq[seq.split=='test_1'].copy()
for f in ['same_triplet_date_in_train','same_meshpair_date_in_train','same_date_in_train','same_subject_triplet_in_train']:
    a=t1[t1[f]]; b=t1[~t1[f]]
    print(f'{f}: yes n={len(a)} contact {a.contact.mean():.1f} motion {a.motion.mean():.1f} | no n={len(b)} contact {b.contact.mean() if len(b) else np.nan:.1f} motion {b.motion.mean() if len(b) else np.nan:.1f} | diff(no-yes) contact %.1f [%.1f,%.1f] motion %.1f [%.1f,%.1f]'%(*boot_diff(b.contact,a.contact),*boot_diff(b.motion,a.motion)))
r=stats.spearmanr(t1.n_train_seq_same_triplet_date,t1.contact); print('test_1 contact vs n_train_seq_same_triplet_date: rho=%.2f p=%.3f'%r)
r=stats.spearmanr(t1.n_train_seq_same_triplet_date,t1.motion); print('test_1 motion vs n_train_seq_same_triplet_date: rho=%.2f p=%.3f'%r)
# all test_1 sequences (not just those with touching windows) session flags
a1=sub[sub.split=='test_1']
print('all test_1 seqs (n=%d): same_triplet_date %d, same_meshpair_date %d, same_date %d, same_subject_triplet %d'%(len(a1),a1.same_triplet_date_in_train.sum(),a1.same_meshpair_date_in_train.sum(),a1.same_date_in_train.sum(),a1.same_subject_triplet_in_train.sum()))
# take order: within triplet+date, rank of test takes vs train takes
rows=[]
for (trip,d),g in sub[sub.split.isin(['train','test_1'])].groupby(['triplet','date']):
    g=g.sort_values('take'); tt=g[g.split=='train']['take'].values; te=g[g.split=='test_1']['take'].values
    if len(tt)==0 or len(te)==0: continue
    for x in te:
        rows.append(dict(triplet=trip,date=d,take=x,n_train=len(tt),frac_train_earlier=(tt<x).mean(),rank_pct=stats.percentileofscore(np.concatenate([tt,te]),x,'mean')/100, is_last=x>tt.max(), is_first=x<tt.min()))
tk=pd.DataFrame(rows)
print('test_1 takes within triplet+date groups having train takes: n=%d; mean frac of train takes earlier=%.2f (0.5=random); rank pct mean=%.2f; is_last %.2f is_first %.2f'%(len(tk),tk.frac_train_earlier.mean(),tk.rank_pct.mean(),tk.is_last.mean(),tk.is_first.mean()))
print('   wilcoxon frac_train_earlier vs 0.5: p=%.3f'%stats.wilcoxon(tk.frac_train_earlier-0.5).pvalue)
# does take order relate to error? within-sequence positions: correlate error with frac_train_earlier and raw take number
t1=t1.merge(tk[['triplet','date','take','frac_train_earlier','rank_pct']],on=['triplet','date','take'],how='left')
for y in ['contact','motion']:
    m=t1.dropna(subset=['rank_pct']); r=stats.spearmanr(m.rank_pct,m[y]); print(f'test_1 {y} vs within-session rank pct (n={len(m)}): rho={r[0]:.2f} p={r[1]:.3f}')
    r=stats.spearmanr(t1['take'],t1[y]); print(f'test_1 {y} vs raw take number (n={len(t1)}): rho={r[0]:.2f} p={r[1]:.3f}')
# Same in train: does error depend on take number in train?
trq=seq[seq.split=='train']
for y in ['contact','motion']:
    r=stats.spearmanr(trq['take'],trq[y]); print(f'train {y} vs raw take number (n={len(trq)}): rho={r[0]:.2f} p={r[1]:.3f}')
# date-level comparison: mean error per date train vs test_1
dd=seq[seq.split.isin(['train','test_1'])].groupby(['date','split'])[['contact','motion']].mean().unstack('split')
dd.columns=['_'.join(c) for c in dd.columns]; dd['n_test1']=seq[seq.split=='test_1'].groupby('date').size(); dd['n_train']=seq[seq.split=='train'].groupby('date').size()
print('\nper-date train vs test_1 (sequence-level):'); print(dd.round(1).to_string())
w=dd.dropna(); print('dates with both: %d; mean gap contact %.1f motion %.1f; test1>train on %.2f / %.2f of dates'%(len(w),(w.contact_test_1-w.contact_train).mean(),(w.motion_test_1-w.motion_train).mean(),(w.contact_test_1>w.contact_train).mean(),(w.motion_test_1>w.motion_train).mean()))
dd.to_csv(G+'per_date_train_vs_test1.csv')

print('\n','='*30,'TASK 4: hand size')
trm_L=tr[Lc].mean().values; trm_R=tr[Rc].mean().values
seq['devL']=np.linalg.norm(seq[Lc].values-trm_L,axis=1); seq['devR']=np.linalg.norm(seq[Rc].values-trm_R,axis=1)
# sequence-weighted train mean (as in training exposure)
for sp in ['test_1','train']:
    g=seq[seq.split==sp]
    for y in ['contact','motion']:
        for x in ['L0','R0','devL','devR']:
            r=stats.spearmanr(g[x],g[y]); print(f'{sp} seq-level {y} vs {x}: rho={r[0]:.2f} p={r[1]:.3f} n={len(g)}')
# hand-specific: L windows with L beta, R windows with R beta, window-hand level
th=t.merge(sub[['sequence_id']+Lc+Rc],on='sequence_id')
th['b0']=np.where(th.hand=='L',th.L0,th.R0)
th['dev']=np.where(th.hand=='L',np.linalg.norm(th[Lc].values-trm_L,axis=1),np.linalg.norm(th[Rc].values-trm_R,axis=1))
for sp in ['test_1','train']:
    g=th[th.split==sp]
    for y in ['contact','motion']:
        for x in ['b0','dev']:
            r=stats.spearmanr(g[x],g[y]); print(f'{sp} window-hand level ({len(g)}) {y} vs hand-matched {x}: rho={r[0]:.2f} p={r[1]:.3f}')
# subject-level: 31 points, test_1 subjects
ps2=seq[seq.split=='test_1'].groupby('subject').agg(contact=('contact','mean'),motion=('motion','mean'),L0=('L0','first'),R0=('R0','first'),devL=('devL','first'),devR=('devR','first'))
for y in ['contact','motion']:
    for x in ['L0','R0','devL','devR']:
        r=stats.spearmanr(ps2[x],ps2[y]); print(f'test_1 subject-level (n={len(ps2)}) {y} vs {x}: rho={r[0]:.2f} p={r[1]:.3f}')
# Do unusual hands get higher error?  compare seq-level test_1 error for top/bottom tercile of dev
g=seq[seq.split=='test_1']
for x in ['devL','devR']:
    q=g[x].quantile([1/3,2/3]); lo=g[g[x]<=q.iloc[0]]; hi=g[g[x]>=q.iloc[1]]
    print(f'test_1 {x} bottom tercile (n={len(lo)}) contact {lo.contact.mean():.1f} motion {lo.motion.mean():.1f} | top tercile (n={len(hi)}) contact {hi.contact.mean():.1f} motion {hi.motion.mean():.1f}; diff contact %.1f [%.1f,%.1f]'%boot_diff(hi.contact,lo.contact))

print('\n','='*30,'EXTRA A: mesh-pair-same-date vs mesh-pair-seen-anywhere (test_1)')
trset_mesh=set(zip(tri.tool_mesh,tri.target_mesh))
t1=seq[seq.split=='test_1'].copy()
t1['meshpair_in_train_any']=[(a,b) in trset_mesh for a,b in zip(t1.tool_mesh,t1.target_mesh)]
print(pd.crosstab(t1.meshpair_in_train_any,t1.same_meshpair_date_in_train,rownames=['meshpair_any'],colnames=['meshpair_same_date']))
for (a,b),g in t1.groupby(['meshpair_in_train_any','same_meshpair_date_in_train']):
    print(f'meshpair_any={a} same_date={b}: n={len(g)} contact {g.contact.mean():.1f} motion {g.motion.mean():.1f}')
m=t1[t1.meshpair_in_train_any]; a=m[m.same_meshpair_date_in_train]; b=m[~m.same_meshpair_date_in_train]
print('within meshpair-seen: same-date n=%d vs other-date n=%d; diff(other-same) contact %.1f [%.1f,%.1f] motion %.1f [%.1f,%.1f]'%(len(a),len(b),*boot_diff(b.contact,a.contact),*boot_diff(b.motion,a.motion)))
# same subject + same mesh pair in train (regardless of date)
trset_subj_mesh=set(zip(tr.merge(si[['sequence_id','tool_mesh','target_mesh']],on='sequence_id').subject, tr.merge(si[['sequence_id','tool_mesh','target_mesh']],on='sequence_id').tool_mesh, tr.merge(si[['sequence_id','tool_mesh','target_mesh']],on='sequence_id').target_mesh))
t1['subj_meshpair_in_train']=[(s,a,b) in trset_subj_mesh for s,a,b in zip(t1.subject,t1.tool_mesh,t1.target_mesh)]
print(pd.crosstab(t1.subj_meshpair_in_train,t1.same_meshpair_date_in_train,rownames=['subj+meshpair'],colnames=['meshpair_same_date']))
print('\n','='*30,'EXTRA B: within-session take rank vs error in TRAIN (later takes worse?)')
rows=[]
for (trip,d),g in sub[sub.split=='train'].groupby(['triplet','date']):
    tk_=g['take'].values
    if len(tk_)<2: continue
    for x,sid in zip(tk_,g.sequence_id):
        rows.append(dict(sequence_id=sid,rank_pct_train=stats.percentileofscore(tk_,x,'mean')/100,n_in_session=len(tk_)))
rk=pd.DataFrame(rows); trq=seq[seq.split=='train'].merge(rk,on='sequence_id')
for y in ['contact','motion']:
    r=stats.spearmanr(trq.rank_pct_train,trq[y]); print(f'train {y} vs within-session rank pct (n={len(trq)}): rho={r[0]:.2f} p={r[1]:.3f}')
    lo=trq[trq.rank_pct_train<=0.33]; hi=trq[trq.rank_pct_train>=0.67]; print(f'   early-third n={len(lo)} {y} {lo[y].mean():.1f} | late-third n={len(hi)} {y} {hi[y].mean():.1f}')
# same for test_1 within full session (train+test takes)
rows=[]
for (trip,d),g in sub[sub.split.isin(['train','test_1'])].groupby(['triplet','date']):
    tk_=g['take'].values
    for x,sid in zip(tk_,g.sequence_id): rows.append(dict(sequence_id=sid,rank_all=stats.percentileofscore(tk_,x,'mean')/100))
rk=pd.DataFrame(rows); q=seq[seq.split.isin(['train','test_1'])].merge(rk,on='sequence_id')
for y in ['contact','motion']:
    X=np.column_stack([(q.split=='test_1').astype(float),q.rank_all]); print('OLS',y,'~ is_test1 + rank_in_session (train+test1 seqs, n=%d):'%len(q),ols(q[y].values,X,['is_test1','rank_in_session']))
print('\n','='*30,'EXTRA C: test_1 vs train error within subject AND within triplet+date session (matched)')
q=seq[seq.split.isin(['train','test_1'])]
grp=q.groupby(['subject','triplet','date'])
rows=[]
for k,g in grp:
    a=g[g.split=='test_1']; b=g[g.split=='train']
    if len(a) and len(b): rows.append(dict(key=k,n_test=len(a),n_train=len(b),gap_contact=a.contact.mean()-b.contact.mean(),gap_motion=a.motion.mean()-b.motion.mean(),test_contact=a.contact.mean(),train_contact=b.contact.mean(),test_motion=a.motion.mean(),train_motion=b.motion.mean()))
mp=pd.DataFrame(rows); print('matched (subject,triplet,date) cells with both train(sampled) and test_1: %d; test_1 seqs covered %d'%(len(mp),mp.n_test.sum()))
print('mean gap contact %.1f [%.1f,%.1f], motion %.1f [%.1f,%.1f]; test>train in %.2f/%.2f of cells'%(*boot_mean(mp.gap_contact),*boot_mean(mp.gap_motion),(mp.gap_contact>0).mean(),(mp.gap_motion>0).mean()))
print('matched cell means: test contact %.1f train %.1f; test motion %.1f train %.1f'%(mp.test_contact.mean(),mp.train_contact.mean(),mp.test_motion.mean(),mp.train_motion.mean()))
mp.to_csv(G+'matched_subject_session_cells.csv',index=False)
