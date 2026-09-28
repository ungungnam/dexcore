import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
G='/result/uhnam/dexcore/bimart_taco/contact_probe/gen/'
df = pd.read_csv(G+'betas_per_sequence.csv')
Lc=[f'L{i}' for i in range(10)]; Rc=[f'R{i}' for i in range(10)]
df['Lkey']=[tuple(np.round(b,4)) for b in df[Lc].values]
df['Rkey']=[tuple(np.round(b,4)) for b in df[Rc].values]
df['date']=df.seq.str[:8]; df['take']=df.seq.str.split('_').str[1].astype(int)
# assign ids ordered by first appearance sorted by date
df=df.sort_values(['date','take']).reset_index(drop=True)
df['subjL']=pd.factorize(df.Lkey)[0]; df['subjR']=pd.factorize(df.Rkey)[0]
ct=pd.crosstab(df.subjL,df.subjR)
print('L/R consistent (each L cluster maps to exactly one R cluster and vice versa):', (ct>0).sum(1).max()==1 and (ct>0).sum(0).max()==1)
print('n subjects L', df.subjL.nunique(), 'R', df.subjR.nunique())
df['subject']=df.subjL
# min distance between distinct clusters
UL=df.groupby('subject')[Lc].first().values; UR=df.groupby('subject')[Rc].first().values
DL=cdist(UL,UL); np.fill_diagonal(DL,np.inf); DR=cdist(UR,UR); np.fill_diagonal(DR,np.inf)
print('min inter-cluster dist L', DL.min().round(3),'R', DR.min().round(3))
# subject x split table
tab=pd.crosstab(df.subject, df.split)[['train','test_1','test_2','test_3','test_4']]
tab['total']=tab.sum(1)
sd=df.groupby('subject').agg(n_dates=('date','nunique'), dates=('date',lambda s: ','.join(sorted(set(s)))), n_triplets=('triplet','nunique'))
tab=tab.join(sd)
print(tab.to_string())
print('\nsubjects in test_1 but not train:', sorted(set(df[df.split=='test_1'].subject)-set(df[df.split=='train'].subject)))
for s in ['test_1','test_2','test_3','test_4']:
    print(s, 'subjects missing from train:', sorted(set(df[df.split==s].subject)-set(df[df.split=='train'].subject)))
# date x subject
dt=pd.crosstab(df.date, df.subject); print('\nsubjects per date:'); print((dt>0).sum(1).to_string())
print('dates per subject:'); print((dt>0).sum(0).to_string())
df.to_csv(G+'betas_per_sequence_subjects.csv', index=False)
tab.to_csv(G+'subject_by_split.csv')
