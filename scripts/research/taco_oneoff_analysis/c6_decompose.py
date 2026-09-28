"""Decompose WRONG predicted touched points (pred & ~rec_window) per hand-half:
   seq   : touched (>=5% frames) elsewhere in this same sequence (temporal/phase error)
   sverb : else in known set of same mesh & same verb train sequences (same-strategy variant)
   overb : else in known set of same mesh & other verb train sequences (other-action strategy)
   out   : else outside all known regions (nowhere anyone grasps)
Known sets are per object-half: tool half uses same tool_mesh train seqs, target half uses same target_mesh."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/result/uhnam/dexcore/bimart_taco'); OUT = ROOT/'contact_probe/verify'
TH=0.01; SEQ_FRAC=0.05; WIN_FRAC=0.20
off = pd.read_csv(ROOT/'train_store/offsets.csv')
tr = off[off.split=='train'].reset_index(drop=True)
seq_touch = np.load(OUT/'train_seq_touchsets.npy')
c = np.load(ROOT/'train_store/contact.npy', mmap_mode='r')
# whole-sequence touch sets for every sequence (all splits)
seqset = {}
for _, r in off.iterrows():
    m = np.asarray(c[r.start:r.start+r.n_frames_store]); seqset[r.sequence_id] = (m<TH).mean(0) >= SEQ_FRAC
np.save(OUT/'all_seq_touchsets.npy', np.stack([seqset[s] for s in off.sequence_id])); off[['sequence_id']].to_csv(OUT/'all_seq_touchsets_index.csv', index=False)
HALVES = {('L','tool'):slice(0,512), ('L','target'):slice(512,1024), ('R','tool'):slice(1024,1536), ('R','target'):slice(1536,2048)}
tm = tr.tool_mesh.values.astype(int); gm = tr.target_mesh.values.astype(int); vb = tr.verb.values; sid = tr.sequence_id.values
rows=[]
for split in ['train','test_1','test_2','test_3','test_4']:
    idx = pd.read_csv(ROOT/f'contact_probe/{split}_contact_maps_index.csv')
    z = np.load(ROOT/f'contact_probe/{split}_contact_maps.npz'); pred, gt = z['pred'], z['gt']
    for k, r in idx.iterrows():
        pt = (pred[k]<TH).mean(0)>=WIN_FRAC; rt = (gt[k]<TH).mean(0)>=WIN_FRAC
        S = seqset[r.sequence_id]
        for (h, obj), sl in HALVES.items():
            mesh = int(r.tool_mesh) if obj=='tool' else int(r.target_mesh)
            selm = ((tm==mesh) if obj=='tool' else (gm==mesh)) & (sid != r.sequence_id)
            if not selm.any(): continue
            Ksv = seq_touch[selm & (vb==r.verb)][:, sl].any(0)
            Kov = seq_touch[selm & (vb!=r.verb)][:, sl].any(0)
            K = Ksv|Kov
            p, g, s = pt[sl], rt[sl], S[sl]
            wrong = p & ~g
            row = dict(split=split, window=int(r.window), hand=h, obj=obj, sequence_id=r.sequence_id, verb=r.verb, mesh=mesh,
                       n_mesh_seqs=int(selm.sum()), n_sameverb_seqs=int((selm&(vb==r.verb)).sum()),
                       n_pred=int(p.sum()), n_rec=int(g.sum()), n_wrong=int(wrong.sum()), n_hit=int((p&g).sum()),
                       base_K=float(K.mean()), base_Ksv=float(Ksv.mean()), base_Kov_only=float((Kov&~Ksv).mean()), base_S=float(s.mean()),
                       rec_in_K=float((g&K).sum()/g.sum()) if g.sum() else np.nan,
                       rec_in_Ksv=float((g&Ksv).sum()/g.sum()) if g.sum() else np.nan,
                       pred_in_K=float((p&K).sum()/p.sum()) if p.sum() else np.nan)
            if wrong.sum():
                a = wrong & s; b = wrong & ~s & Ksv; d = wrong & ~s & ~Ksv & Kov; e = wrong & ~s & ~K
                n = wrong.sum()
                row.update(w_seq=a.sum()/n, w_sameverb=b.sum()/n, w_otherverb=d.sum()/n, w_out=e.sum()/n,
                           w_in_K_any=(wrong&K).sum()/n)
                # chance: among non-recorded points of this half, fraction in each category
                nr = ~g; m = nr.sum()
                row.update(ch_seq=(nr&s).sum()/m, ch_sameverb=(nr&~s&Ksv).sum()/m, ch_otherverb=(nr&~s&~Ksv&Kov).sum()/m, ch_out=(nr&~s&~K).sum()/m, ch_in_K_any=(nr&K).sum()/m)
            rows.append(row)
    print(split, 'done', flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT/'c6_wrong_point_decomposition.csv', index=False); print(df.shape)
