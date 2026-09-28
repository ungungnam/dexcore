"""C6: do predicted contact points land on known grasp regions (plausible-but-wrong) or nowhere anyone grasps?"""
import numpy as np, pandas as pd, json, sys
from pathlib import Path
ROOT = Path('/result/uhnam/dexcore/bimart_taco')
OUT = ROOT/'contact_probe/verify'; OUT.mkdir(parents=True, exist_ok=True)
TH = 0.01; SEQ_FRAC = 0.05; WIN_FRAC = 0.20
off = pd.read_csv(ROOT/'train_store/offsets.csv')
tr = off[off.split=='train'].reset_index(drop=True)
c = np.load(ROOT/'train_store/contact.npy', mmap_mode='r')
# 1. per train sequence: touched in >=5% of frames  -> [953, 2048] bool
seq_touch = np.zeros((len(tr), 2048), bool)
for i, r in tr.iterrows():
    m = np.asarray(c[r.start:r.start+r.n_frames_store])
    seq_touch[i] = (m < TH).mean(0) >= SEQ_FRAC
print('train seqs', len(tr), 'frames', tr.n_frames_store.sum(), 'mean touched pts/seq', seq_touch.sum(1).mean())
np.save(OUT/'train_seq_touchsets.npy', seq_touch)
tr['tool_mesh']=tr.tool_mesh.astype(int); tr['target_mesh']=tr.target_mesh.astype(int)
HALF = {'L': slice(0,1024), 'R': slice(1024,2048)}
TOOL = {'L': slice(0,512), 'R': slice(1024,1536)}; TARG = {'L': slice(512,1024), 'R': slice(1536,2048)}

def known_set(tool_mesh, target_mesh, exclude_seq=None, mode='pair'):
    """Return (known[2048] bool, n_seqs) . mode 'pair': union over same (tool,target) train seqs.
       mode 'half': tool halves from same tool_mesh seqs, target halves from same target_mesh seqs."""
    mask_ex = np.ones(len(tr), bool) if exclude_seq is None else (tr.sequence_id.values != exclude_seq)
    if mode == 'pair':
        sel = (tr.tool_mesh.values==tool_mesh)&(tr.target_mesh.values==target_mesh)&mask_ex
        return seq_touch[sel].any(0), int(sel.sum())
    known = np.zeros(2048, bool)
    st = (tr.tool_mesh.values==tool_mesh)&mask_ex; sg = (tr.target_mesh.values==target_mesh)&mask_ex
    for h in 'LR':
        known[TOOL[h]] = seq_touch[st][:, TOOL[h]].any(0) if st.any() else False
        known[TARG[h]] = seq_touch[sg][:, TARG[h]].any(0) if sg.any() else False
    return known, (int(st.sum()), int(sg.sum()))

rows = []
for split in ['train','test_1','test_2','test_3','test_4']:
    idx = pd.read_csv(ROOT/f'contact_probe/{split}_contact_maps_index.csv')
    z = np.load(ROOT/f'contact_probe/{split}_contact_maps.npz')
    pred, gt = z['pred'], z['gt']
    assert len(idx)==len(pred)
    for k, r in idx.iterrows():
        pt = (pred[k] < TH).mean(0) >= WIN_FRAC   # [2048]
        rt = (gt[k] < TH).mean(0) >= WIN_FRAC
        ex = r.sequence_id if split=='train' else None
        kp, np_ = known_set(int(r.tool_mesh), int(r.target_mesh), ex, 'pair')
        kh, nh = known_set(int(r.tool_mesh), int(r.target_mesh), ex, 'half')
        for h in 'LR':
            sl = HALF[h]; p, g = pt[sl], rt[sl]
            row = dict(split=split, window=int(r.window), hand=h, sequence_id=r.sequence_id, verb=r.verb,
                       tool_mesh=int(r.tool_mesh), target_mesh=int(r.target_mesh),
                       n_pred=int(p.sum()), n_rec=int(g.sum()), n_pair_seqs=np_, n_tool_seqs=nh[0], n_targ_seqs=nh[1],
                       pred_rec_overlap=int((p&g).sum()))
            for name, K in [('pair', kp[sl]), ('half', kh[sl])]:
                row[f'{name}_known_base'] = float(K.mean())
                row[f'{name}_known_tool_base'] = float(K[:512].mean()); row[f'{name}_known_targ_base'] = float(K[512:].mean())
                if p.sum():
                    row[f'{name}_pred_in_known'] = float((p&K).sum()/p.sum())
                    row[f'{name}_pred_known_not_rec'] = float((p&K&~g).sum()/p.sum())
                    row[f'{name}_pred_outside'] = float((p&~K).sum()/p.sum())
                    row[f'{name}_pred_in_rec'] = float((p&g).sum()/p.sum())
                if g.sum():
                    row[f'{name}_rec_in_known'] = float((g&K).sum()/g.sum())
            rows.append(row)
    print(split, len(idx), 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT/'c6_known_grasp_window_hand.csv', index=False)
print(df.shape)
