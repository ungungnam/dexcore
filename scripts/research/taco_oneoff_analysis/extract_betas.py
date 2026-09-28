import pickle, numpy as np, pandas as pd, os
si = pd.read_csv('/result/uhnam/dexcore/bimart_taco/sequence_index.csv')
rows=[]; missing=[]
for r in si.itertuples():
    d = f"/backups/uhnam/TACO/Hand_Poses/{r.triplet}/{r.seq}"
    rec = dict(sequence_id=r.sequence_id, triplet=r.triplet, seq=r.seq, split=r.split)
    for h in ['left','right']:
        p = f"{d}/{h}_hand_shape.pkl"
        if not os.path.exists(p):
            missing.append((r.sequence_id,h)); continue
        x = pickle.load(open(p,'rb'))['hand_shape']
        if hasattr(x,'detach'): x = np.asarray(x.detach().cpu(), dtype=np.float64)
        else: x = np.asarray(x, dtype=np.float64)
        for i in range(10): rec[f'{h[0].upper()}{i}'] = x[i]
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/gen/betas_per_sequence.csv', index=False)
print('n seq', len(df), 'missing', missing)
print(df.iloc[:, 4:].describe().T)
