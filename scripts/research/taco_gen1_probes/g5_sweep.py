#!/usr/bin/env python
"""G5 verification: same 128 windows/split as ckpt_sweep_taco.py; motion model at best/ep100/ep200,
EMA and RAW weights; t=0 floor decomposed per hand into translation vs residual; plus nearest-training-
window distances for test_1 windows and the t=0 prediction (does the prediction move toward training data?)."""
import pathlib as _pl, sys as _sys
_REPO = "/home/uhnam/workspace/dexcore"
for _p in (_REPO, _REPO + "/third_party/BimArt"):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import json, logging, time
from pathlib import Path
import numpy as np, torch, yaml, pandas as pd
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset
from scripts.eval_bimart_taco import load_model, sample
from src.analysis.bimart.datasets import TacoMotionDataset
from utils import data_util

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("g5")
OUT = Path("/result/uhnam/dexcore/bimart_taco/contact_probe/gen/verify"); OUT.mkdir(parents=True, exist_ok=True)
KP_DIM = 600; K = 100
def kp(a, mean, std): v = (a * std + mean)[..., :KP_DIM]; return v.reshape(*v.shape[:-1], 200, 3)   # metres, L then R
def err(a, b, mean, std): return torch.linalg.norm(kp(a, mean, std) - kp(b, mean, std), dim=-1).mean(dim=(-1, -2)) * 1000
def decompose(pr, x, mean, std):
    """per window, per hand: total mm, per-frame translation mm (norm of the keypoint-mean displacement), residual mm."""
    d = kp(pr, mean, std) - kp(x, mean, std)                        # (B,T,200,3)
    out = {}
    for h, sl in (("L", slice(0, K)), ("R", slice(K, 2 * K))):
        dh = d[..., sl, :]                                            # (B,T,100,3)
        tr = dh.mean(dim=2, keepdim=True)                             # (B,T,1,3)
        out[f"tot_{h}"] = torch.linalg.norm(dh, dim=-1).mean(dim=(1, 2)) * 1000
        out[f"trans_{h}"] = torch.linalg.norm(tr[:, :, 0], dim=-1).mean(dim=1) * 1000
        out[f"resid_{h}"] = torch.linalg.norm(dh - tr, dim=-1).mean(dim=(1, 2)) * 1000
        wt = dh.mean(dim=(1, 2), keepdim=True)                        # whole-window constant offset
        out[f"wtrans_{h}"] = torch.linalg.norm(wt[:, 0, 0], dim=-1) * 1000
    return out

def main():
    dev = "cuda"; E = "/result/uhnam/dexcore/bimart_taco/experiments/"
    mcfg = yaml.safe_load(open(_REPO + "/configs/bimart_taco/train_motion_config.yaml"))
    T = 50; mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], dev)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="sample")
    ab = sched.alphas_cumprod; log.info("alpha_bar[0]=%.5f (noise std %.4f)  alpha_bar[49]=%.5f", ab[0], (1 - ab[0]).sqrt(), ab[-1])
    mean, std = mstat["action"]["mean"], mstat["action"]["std"]
    # ---- training bank for NN (stride 4 over all stride-1 train windows), keypoints only, metres ----
    tr_ds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split="train", pred_horizon=64, base_frame=8)
    store = tr_ds.store["action"]
    bank_idx = np.arange(0, len(tr_ds), 4)
    bank = np.empty((len(bank_idx), 64 * KP_DIM), dtype=np.float32); bank_seq = np.empty(len(bank_idx), dtype=object); bank_start = np.empty(len(bank_idx), dtype=int)
    for j, i in enumerate(bank_idx):
        sl, row = tr_ds._rows(int(i)); bank[j] = np.asarray(store[sl])[:, :KP_DIM].reshape(-1); bank_seq[j] = row["sequence_id"]; bank_start[j] = tr_ds.windows[int(i)]["start"]
    bank_t = torch.from_numpy(bank).to(dev); bank_sq = (bank_t ** 2).sum(1)
    log.info("bank: %d train windows (stride 4 of %d)", len(bank_idx), len(tr_ds))
    def nn(q, exclude_seq=None):
        """q: (B, 64*600) metres. returns (mm distance to nearest bank window, its seq, its start)."""
        d2 = (q ** 2).sum(1, keepdim=True) + bank_sq[None] - 2 * q @ bank_t.T                    # (B,N)
        if exclude_seq is not None:
            mask = torch.from_numpy(np.array([[s == e for s in bank_seq] for e in exclude_seq])).to(dev)
            d2 = d2.masked_fill(mask, float("inf"))
        j = d2.argmin(1)
        best = bank_t[j]
        mm = torch.linalg.norm((q - best).reshape(len(q), 64, 200, 3), dim=-1).mean(dim=(1, 2)) * 1000
        return mm.cpu().numpy(), bank_seq[j.cpu().numpy()], bank_start[j.cpu().numpy()]
    rows, per = [], []
    for split in ("train", "test_1"):
        mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split, pred_horizon=64, base_frame=8)
        pick = np.random.default_rng(0).choice(len(mds), min(128, len(mds)), replace=False)
        wseq = [mds.index.iloc[mds.windows[int(i)]["file_idx"]]["sequence_id"] for i in pick]; wstart = [mds.windows[int(i)]["start"] for i in pick]
        for tag, fname in (("best", "model_best.pth"), ("ep100", "model_epoch100.pth"), ("ep200", "model_epoch200.pth")):
            for use_ema in (True, False):
                mm_, mname, mep, _ = load_model("motion", E + "motion_model_taco_20260923_151845", mcfg, dev, prefer=fname, use_ema=use_ema)
                acc = {k: [] for k in ("sampled", "floor_t0", "floor_t0_clean", "condmean_t49")}; dec = []; nnrows = []
                for s in range(0, len(pick), 32):
                    idx = pick[s:s + 32]
                    mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
                    mn = data_util.preprocess_batch({k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, dev)
                    x = mn["action"]; cond = mn["obs"]
                    xs = sample(mm_, x.shape, sched, dev, cond, "motion", T, seed=0); acc["sampled"].append(err(xs, x, mean, std).cpu())
                    g = torch.Generator(device=dev).manual_seed(999); noise = torch.randn(x.shape, device=dev, generator=g)
                    def fwd(xt, t):
                        ts = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
                        with torch.no_grad(): return mm_(xt, ts, obj_feat=cond["object"], contact_cond=cond["contact_points"], contact_on_prob=1.0, global_states=cond["global_states"])
                    ts0 = torch.zeros(x.shape[0], device=dev, dtype=torch.long)
                    pr0 = fwd(sched.add_noise(x, noise, ts0), 0); acc["floor_t0"].append(err(pr0, x, mean, std).cpu())
                    prc = fwd(x, 0); acc["floor_t0_clean"].append(err(prc, x, mean, std).cpu())
                    pr49 = fwd(sched.add_noise(x, noise, ts0 + (T - 1)), T - 1); acc["condmean_t49"].append(err(pr49, x, mean, std).cpu())
                    d0 = decompose(pr0, x, mean, std); ds_ = decompose(xs, x, mean, std)
                    # NN: input window, t=0 prediction, sampled output -> nearest training window
                    seqs = wseq[s:s + 32]
                    excl = seqs if split == "train" else None
                    qx = kp(x, mean, std).reshape(len(idx), -1); q0 = kp(pr0, mean, std).reshape(len(idx), -1); qs = kp(xs, mean, std).reshape(len(idx), -1)
                    nx, nxs, nxst = nn(qx, excl); n0, n0s, n0st = nn(q0, excl); ns, nss, nsst = nn(qs, excl)
                    # distance of prediction to the INPUT's nearest training window (did it move toward it?)
                    if split == "train":
                        selfd = np.zeros(len(idx))
                    for b in range(len(idx)):
                        per.append({"split": split, "ckpt": tag, "ema": use_ema, "seq": seqs[b], "start": wstart[s + b],
                                    **{k: float(v[b]) for k, v in d0.items()}, **{"s_" + k: float(v[b]) for k, v in ds_.items()},
                                    "floor_t0": float(err(pr0, x, mean, std)[b]), "floor_t0_clean": float(err(prc, x, mean, std)[b]), "sampled": float(err(xs, x, mean, std)[b]),
                                    "nn_input_mm": float(nx[b]), "nn_pred0_mm": float(n0[b]), "nn_sampled_mm": float(ns[b]),
                                    "nn_input_seq": nxs[b], "nn_pred0_seq": n0s[b], "nn_pred0_same_as_input_nn": bool(nxs[b] == n0s[b])})
                r = {"split": split, "ckpt": tag, "motion_epoch": mep, "ema": use_ema, "weights": mname, **{k: float(torch.cat(v).mean()) for k, v in acc.items()}}
                pdf = pd.DataFrame([p for p in per if p["split"] == split and p["ckpt"] == tag and p["ema"] == use_ema])
                for c in ("tot_L", "tot_R", "trans_L", "trans_R", "resid_L", "resid_R", "wtrans_L", "wtrans_R", "nn_input_mm", "nn_pred0_mm", "nn_sampled_mm"):
                    r[c] = float(pdf[c].mean())
                r["frac_pred0_closer_to_train_than_input"] = float((pdf["nn_pred0_mm"] < pdf["nn_input_mm"]).mean())
                rows.append(r); log.info("%s %s ema=%s: sampled %.1f floor %.1f (clean %.1f) cm49 %.1f | t0 L tot/trans/resid %.1f/%.1f/%.1f R %.1f/%.1f/%.1f | NN in %.1f pred0 %.1f",
                                         split, tag, use_ema, r["sampled"], r["floor_t0"], r["floor_t0_clean"], r["condmean_t49"], r["tot_L"], r["trans_L"], r["resid_L"], r["tot_R"], r["trans_R"], r["resid_R"], r["nn_input_mm"], r["nn_pred0_mm"])
                del mm_; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(OUT / "g5_ckpt_sweep_ema_vs_raw.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "g5_per_window.csv", index=False)
    print(df.round(2).to_string(index=False)); print("->", OUT)

if __name__ == "__main__": main()
