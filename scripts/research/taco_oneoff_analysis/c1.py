import pandas as pd, json
df = pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
print(df.split.value_counts().to_dict(), len(df))
tr = df[df.split=='train']
def S(cols):
    return set(map(tuple, tr[cols].astype(str).values.tolist()))
seen = {k:S(list(k)) for k in [('verb',),('tool_cat',),('target_cat',),('tool_mesh',),('target_mesh',),('verb','tool_cat'),('verb','target_cat'),('tool_cat','target_cat'),('tool_mesh','target_mesh'),('triplet',)]}
out={}
for sp in ['test_1','test_2','test_3','test_4']:
    d=df[df.split==sp]; row={'n_seq':len(d)}
    for k,s in seen.items():
        nov = ~d[list(k)].astype(str).apply(tuple,axis=1).isin(s)
        row['novel_'+'_'.join(k)] = int(nov.sum()); row['novel_'+'_'.join(k)+'_pct']=round(100*nov.mean(),1)
    out[sp]=row
res=pd.DataFrame(out).T
pd.set_option('display.width',250); print(res)
# test_3 target details
t3=df[df.split=='test_3']
print('test_3 target_cat:',t3.target_cat.value_counts().to_dict())
print('bowl in train:', (tr.target_cat=='bowl').sum(), 'bowl as tool_cat in train:', (tr.tool_cat=='bowl').sum())
print('bowl per split (target):', df[df.target_cat=='bowl'].split.value_counts().to_dict())
print('train verbs:', sorted(tr.verb.unique()))
for sp in ['test_1','test_2','test_3','test_4']:
    d=df[df.split==sp]
    nvt = d[~d[['verb','target_cat']].astype(str).apply(tuple,axis=1).isin(seen[('verb','target_cat')])]
    print(sp,'novel (verb,target_cat) pairs:', nvt.groupby(['verb','target_cat']).size().to_dict())
    nvtool = d[~d[['verb','tool_cat']].astype(str).apply(tuple,axis=1).isin(seen[('verb','tool_cat')])]
    print(sp,'novel (verb,tool_cat) pairs:', nvtool.groupby(['verb','tool_cat']).size().to_dict())
    nv = d[~d.verb.isin(tr.verb)]
    print(sp,'novel verbs:', nv.verb.value_counts().to_dict())
# does a novel (verb,target_cat) sequence have the verb seen with some target of same 'role'? list train targets per verb for those verbs
for v in sorted(set(df[~df[['verb','target_cat']].astype(str).apply(tuple,axis=1).isin(seen[('verb','target_cat')])].verb)):
    print('verb',v,'train targets:', tr[tr.verb==v].target_cat.value_counts().to_dict(), '| train tools:', tr[tr.verb==v].tool_cat.value_counts().to_dict())
# cross check novelty_tags
nt=pd.read_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/novelty_tags.csv')
m=df.merge(nt[['sequence_id','verb_seen','verb_target_seen','verb_tool_seen','target_cat_seen','target_mesh_seen','tool_mesh_seen']],on='sequence_id')
for c,k in [('verb_seen',('verb',)),('verb_target_seen',('verb','target_cat')),('verb_tool_seen',('verb','tool_cat')),('target_cat_seen',('target_cat',)),('target_mesh_seen',('target_mesh',)),('tool_mesh_seen',('tool_mesh',))]:
    mine = m[list(k)].astype(str).apply(tuple,axis=1).isin(seen[k])
    print(c,'agreement with novelty_tags:', (mine==m[c].astype(bool)).mean())
res.to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/verify/c1_novelty_counts.csv')
