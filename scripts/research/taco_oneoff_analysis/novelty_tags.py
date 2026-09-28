import pandas as pd, numpy as np
idx = pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
print(idx.split.value_counts())
tr = idx[idx.split=='train']
def S(cols): return set(map(tuple, tr[cols].astype(str).values))
seen = {
 'verb_seen': (['verb'],), 'tool_cat_seen': (['tool_cat'],), 'target_cat_seen': (['target_cat'],),
 'tool_mesh_seen': (['tool_mesh'],), 'target_mesh_seen': (['target_mesh'],),
 'verb_tool_seen': (['verb','tool_cat'],), 'verb_target_seen': (['verb','target_cat'],),
 'tool_target_cat_seen': (['tool_cat','target_cat'],), 'mesh_pair_seen': (['tool_mesh','target_mesh'],),
 'triplet_seen': (['triplet'],),
}
test = idx[idx.split!='train'].copy()
for k,(cols,) in seen.items():
    s = S(cols)
    test[k] = [tuple(map(str,r)) in s for r in test[cols].values]
for k,col in [('n_train_seq_same_triplet','triplet'),('n_train_seq_same_verb','verb'),('n_train_seq_same_tool_mesh','tool_mesh')]:
    c = tr[col].value_counts()
    test[k] = test[col].map(c).fillna(0).astype(int)
out = test[['sequence_id','split','triplet','verb','tool_cat','target_cat','tool_mesh','target_mesh','n_frames','n_windows']+list(seen)+['n_train_seq_same_triplet','n_train_seq_same_verb','n_train_seq_same_tool_mesh']]
out.to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/novelty_tags.csv', index=False)
summ = out.groupby('split')[list(seen)].apply(lambda d: 1-d.mean()).round(3)
summ['n_seq'] = out.groupby('split').size()
print('\nFraction NOVEL (seen==False) per split:\n', summ.T.to_string())
for sp in ['test_3','test_4']:
    d = out[out.split==sp]
    print(f'\n== {sp} crosstab verb_seen x tool_mesh_seen x target_mesh_seen ==')
    ct = d.groupby(['verb_seen','tool_mesh_seen','target_mesh_seen']).size().rename('n').reset_index()
    print(ct.to_string(index=False))
    print(f'{sp}: verbs never in train:', d[~d.verb_seen].verb.value_counts().to_dict(), ' n=', int((~d.verb_seen).sum()))
    print(f'{sp}: novel-triplet breakdown: verb_tool novel', int((~d.verb_tool_seen).sum()), 'verb_target novel', int((~d.verb_target_seen).sum()), 'tool_target_cat novel', int((~d.tool_target_cat_seen).sum()))
    print(f'{sp}: novel triplets by verb:', d[~d.triplet_seen].verb.value_counts().to_dict())
    print(f'{sp}: novel triplets by tool_cat:', d[~d.triplet_seen].tool_cat.value_counts().to_dict())
    print(f'{sp}: n_train_seq_same_verb quantiles:', d.n_train_seq_same_verb.describe()[['min','25%','50%','75%','max']].to_dict())
print('\ntrain verb counts:', tr.verb.value_counts().to_dict())
print('all verbs:', sorted(idx.verb.unique()))
for sp in ['test_1','test_2']:
    d = out[out.split==sp]; print(sp, 'verb novel', int((~d.verb_seen).sum()), 'tool_mesh novel', int((~d.tool_mesh_seen).sum()), 'triplet novel', int((~d.triplet_seen).sum()))
