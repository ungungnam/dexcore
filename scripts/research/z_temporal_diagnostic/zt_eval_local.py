#!/usr/bin/env python
"""Experiment A metrics on the TEST pairs (z_t, z_{t+h}), h = 1, 4, 8, for every method:
    persistence (z_hat = z_t), mean (train-mean z), shrink (per-dimension a_d z_t,d + b_d: plain mean reversion),
    ar1 (ridge on z_t alone with a full 64 x 64 map: linear autonomous dynamics of the latent, no G, no trajectory),
    linear (ridge on [z_t | G | tau_t | tau_{t+h}], lambda chosen on validation),
    probe (the nonlinear probe, primary), probe_notau (ablation without the object trajectory).
    CUDA_VISIBLE_DEVICES=5 python zt_eval_local.py --dataset taco        python zt_eval_local.py --merge
Latent metrics (standardised z, TRAIN statistics): RMSE_z, explained variance R^2 = 1 - SSE / SST (SST around the test mean), mean
per-dimension R^2, Gain_h = 1 - RMSE / RMSE_persistence (and the MSE version), cosine(z_hat, z), cosine of the predicted vs the true
CHANGE (z_hat - z_t vs z_{t+h} - z_t).  Decoded-contact diagnostic with the FROZEN Stage-1 decoder D_z at the geometry of frame t + h:
E_C against the GT map and the distance to D_z(z_{t+h}^GT).  All intervals: take-cluster bootstrap (1000 reps).
Writes <ds>/local_metrics.csv, <ds>/persistence.csv, <ds>/local_arrays.npz (per-t RMSE curves, per-dimension R^2, per-pair errors).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch

import zt_common as Z
from zt_data import Cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zt_eval_local")
METHODS = ("persistence", "mean", "shrink", "ar1", "linear", "probe", "probe_notau")
POSITION_BINS = (("t = 0 (first frame of the window)", 0, 0), ("t = 1 .. 7", 1, 7), ("t >= 8", 8, 63))
DECODE_METHODS = ("gt_z", "probe", "persistence", "mean", "linear")


def features(c, r, t, h, z_only=False):
    return c.zs[r, t] if z_only else np.concatenate([c.zs[r, t], c.S[r], c.tau[r, t], c.tau[r, t + h]], 1)


def ridge_predict(c, h, dev, z_only=False):
    """Ridge regression of the standardised change z_{t+h} - z_t on the standardised inputs; lambda by validation MSE. Returns test z_hat.
    z_only: the inputs are z_t alone — a linear AR(1) model with a FULL 64 x 64 map (direction-dependent decay and cross-dimension
    coupling), i.e. the linear autonomous dynamics of the latent without G or the trajectory (not merely shrinkage toward the mean)."""
    f64 = lambda a: torch.from_numpy(a).to(dev, torch.float64)
    P = {s: c.pairs(s, h) for s in ("train", "val", "test")}
    X = {s: f64(features(c, *P[s], h, z_only)) for s in P}; Y = {s: f64(c.zs[P[s][0], P[s][1] + h] - c.zs[P[s][0], P[s][1]]) for s in P}
    mu, sd = X["train"].mean(0), X["train"].std(0).clamp(min=1e-6)
    X = {s: torch.cat([(x - mu) / sd, torch.ones(len(x), 1, dtype=torch.float64, device=dev)], 1) for s, x in X.items()}
    A = X["train"].T @ X["train"]; B = X["train"].T @ Y["train"]; n = len(X["train"]); best = (np.inf, None, None)
    for lam in Z.RIDGE_LAMBDAS:
        reg = lam * n * torch.eye(A.shape[0], dtype=torch.float64, device=dev); reg[-1, -1] = 0.0
        W = torch.linalg.solve(A + reg, B); v = float(((X["val"] @ W - Y["val"]) ** 2).mean())
        if v < best[0]:
            best = (v, lam, W)
    z_hat = c.zs[P["test"][0], P["test"][1]] + (X["test"] @ best[2]).cpu().numpy().astype(np.float32)
    return z_hat, dict(lambda_=best[1], val_mse=best[0])


def shrink_predict(c, h):
    """Per-dimension shrinkage z_hat_d = a_d z_t,d + b_d fitted by least squares on the TRAIN pairs: plain mean reversion, no coupling."""
    r, t = c.pairs("train", h); x, y = c.zs[r, t].astype(np.float64), c.zs[r, t + h].astype(np.float64)
    xm, ym = x.mean(0), y.mean(0); a = ((x - xm) * (y - ym)).sum(0) / ((x - xm) ** 2).sum(0); b = ym - a * xm
    rt, tt = c.pairs("test", h)
    return (c.zs[rt, tt] * a + b).astype(np.float32)


def gain_ci(sse_m, sse_p, take, sel):
    cs = Z.cluster_sums([sse_m[sel], sse_p[sel]], take[sel])
    return Z.boot_stat(cs, lambda s: 1 - np.sqrt(s[0] / s[1]))


def cosine(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1); ok = (na > 1e-8) & (nb > 1e-8)
    out = np.full(len(a), np.nan); out[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok]); return out


def latent_metrics(z_hat, z_t, z_tgt, zbar, take, pers_sse):
    sse = ((z_hat - z_tgt) ** 2).sum(1); sst = ((z_tgt - zbar) ** 2).sum(1); n = len(sse); dz = z_tgt.shape[1]
    cs = Z.cluster_sums([sse, sst, pers_sse, np.ones(n)], take)
    out = {}
    for k, fn in (("rmse", lambda s: np.sqrt(s[0] / (s[3] * dz))), ("r2", lambda s: 1 - s[0] / s[1]), ("gain_rmse", lambda s: 1 - np.sqrt(s[0] / s[2])), ("gain_mse", lambda s: 1 - s[0] / s[2])):
        out[k], out[k + "_lo"], out[k + "_hi"] = Z.boot_stat(cs, fn)
    sse_d = ((z_hat - z_tgt) ** 2).sum(0); sst_d = ((z_tgt - zbar) ** 2).sum(0)
    out["r2_dim_mean"] = float((1 - sse_d / sst_d).mean()); out["r2_dim_min"] = float((1 - sse_d / sst_d).min()); out["r2_dim_max"] = float((1 - sse_d / sst_d).max())
    for k, v in (("cos", cosine(z_hat, z_tgt)), ("cos_delta", cosine(z_hat - z_t, z_tgt - z_t))):
        m, lo, hi = Z.cluster_bootstrap(v, take, n_boot=Z.N_BOOT); out[k], out[k + "_lo"], out[k + "_hi"] = m, lo, hi
    return out, sse, 1 - sse_d / sst_d


@torch.no_grad()
def decode_all(data, net, seq, t_tgt, z_std_by_method, c, bs=1024):
    """Frozen Stage-1 D_z at the geometry of the target frame: per-pair E_C vs the GT map and distance to D_z(z_{t+h}^GT)."""
    dev = data.device; n_all = torch.from_numpy(seq).to(dev); t_all = torch.from_numpy(t_tgt).to(dev)
    mu, sd = torch.from_numpy(c.z_mu).to(dev), torch.from_numpy(c.z_sd).to(dev)
    out = {m: dict(E_C=[], d_gtz=[]) for m in z_std_by_method}
    for i in range(0, len(seq), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]; b = data.batch(n, t); dec = {}
        for m, zs in z_std_by_method.items():
            z_raw = torch.from_numpy(zs[i:i + bs]).to(dev) * sd + mu
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                dec[m] = data.to_raw(net.decode_z(z_raw, b["geo"], b["S"]).float())
        for m in dec:
            out[m]["E_C"].append((dec[m] - b["C"]).norm(dim=-1).cpu().numpy()); out[m]["d_gtz"].append((dec[m] - dec["gt_z"]).norm(dim=-1).cpu().numpy())
    return {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=Z.DATASETS); ap.add_argument("--merge", action="store_true"); ap.add_argument("--no-decode", action="store_true"); a = ap.parse_args()
    if a.merge:
        for name, out in (("local_metrics.csv", "local_prediction_metrics.csv"), ("persistence.csv", "persistence_comparison.csv"), ("local_splits.csv", "local_prediction_splits.csv")):
            pd.concat([pd.read_csv(Z.ds_out(ds) / name) for ds in Z.DATASETS if (Z.ds_out(ds) / name).exists()], ignore_index=True).to_csv(Z.OUT / out, index=False)
        return
    ds = a.dataset; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = Cache(ds, with_contact=True)
    data = net = None
    if not a.no_decode:
        from cf_data import FrameData
        from cf_models import build as cf_build
        ck = torch.load(c.ckpt, map_location="cpu", weights_only=False); assert Z.md5(c.ckpt) == c.md5
        data = FrameData(ds, dev, stats=ck["stats"], need_masks=False)
        net = cf_build(ck["model"], data.d_geo(), data.d_static(), res_scale=ck.get("res_scale", 1.0)).to(dev); net.load_state_dict(ck["ema_state"]); net.eval()
    rows, prow, srow, arrays = [], [], [], {}
    for h in Z.HORIZONS:
        r, t = c.pairs("test", h); take = c.take[r]; z_t, z_tgt = c.zs[r, t], c.zs[r, t + h]; zbar = z_tgt.mean(0)
        preds = {"persistence": z_t, "mean": np.zeros_like(z_t)}                                   # standardised train mean = 0
        preds["linear"], lin_info = ridge_predict(c, h, dev)
        preds["ar1"], ar_info = ridge_predict(c, h, dev, z_only=True)
        preds["shrink"] = shrink_predict(c, h)
        for v in ("probe", "probe_notau"):
            p = np.load(Z.preds_path(ds, v, h)); assert np.array_equal(p["test_r"], r) and np.array_equal(p["test_t"], t)
            preds[v] = p["test_z_hat"]
        pers_sse = ((z_t - z_tgt) ** 2).sum(1)
        dec = decode_all(data, net, c.seq[r], t + h, {"gt_z": z_tgt, **{m: preds[m] for m in DECODE_METHODS if m != "gt_z"}}, c) if data is not None else {}
        dC = np.linalg.norm(c.C[r, t] - c.C[r, t + h], axis=1)
        for m in METHODS:
            met, sse, r2d = latent_metrics(preds[m], z_t, z_tgt, zbar, take, pers_sse)
            row = dict(dataset=ds, h=h, method=m, n_pairs=len(r), n_takes=len(set(take)), **met)
            if m in dec:
                for k in ("E_C", "d_gtz"):
                    mu_, lo, hi = Z.cluster_bootstrap(dec[m][k], take, n_boot=Z.N_BOOT); row.update({f"dec_{k}": mu_, f"dec_{k}_lo": lo, f"dec_{k}_hi": hi})
            if m in ("linear", "ar1"):
                row.update(ridge_lambda=(lin_info if m == "linear" else ar_info)["lambda_"])
            rows.append(row)
            prow.append(dict(dataset=ds, h=h, method=m, rmse_persistence=float(np.sqrt(pers_sse.sum() / pers_sse.size / c.dz)), rmse_method=met["rmse"],
                             **{k: met[k] for k in ("gain_rmse", "gain_rmse_lo", "gain_rmse_hi", "gain_mse", "gain_mse_lo", "gain_mse_hi", "cos_delta", "cos_delta_lo", "cos_delta_hi")}))
            # per-t RMSE curve (does the local error depend on where in the window the pair sits?)
            arrays[f"h{h}|{m}|rmse_t"] = np.array([np.sqrt(sse[t == k].mean() / c.dz) for k in range(Z.T - h)]); arrays[f"h{h}|{m}|r2_dim"] = r2d
            arrays[f"h{h}|{m}|sse"] = sse.astype(np.float32)
            # ---- splits: where in the window the pair starts (the windows begin at the first firm contact), and on ARCTIC whether the subject was seen in training
            if m in ("probe", "linear", "ar1", "shrink", "probe_notau"):
                for lab, lo_, hi_ in POSITION_BINS:
                    sel = (t >= lo_) & (t <= hi_)
                    if sel.any():
                        g, glo, ghi = gain_ci(sse, pers_sse, take, sel)
                        srow.append(dict(dataset=ds, h=h, method=m, split_kind="window position", split=lab, n_pairs=int(sel.sum()), rmse_persistence=float(np.sqrt(pers_sse[sel].sum() / sel.sum() / c.dz)),
                                         rmse_method=float(np.sqrt(sse[sel].sum() / sel.sum() / c.dz)), gain_rmse=g, gain_rmse_lo=glo, gain_rmse_hi=ghi, share_of_persistence_sse=float(pers_sse[sel].sum() / pers_sse.sum())))
                if ds == "arctic":
                    subj = np.array([x.split("/")[-1] for x in take]); seen = set(x.split("/")[-1] for x in c.take[c.rows("train")])
                    for lab, sel in (("subject seen in training", np.isin(subj, list(seen))), ("subject not in training", ~np.isin(subj, list(seen)))):
                        if sel.any():
                            g, glo, ghi = gain_ci(sse, pers_sse, take, sel)
                            srow.append(dict(dataset=ds, h=h, method=m, split_kind="subject", split=lab + " (" + ", ".join(sorted(set(subj[sel]))) + ")", n_pairs=int(sel.sum()),
                                             rmse_persistence=float(np.sqrt(pers_sse[sel].sum() / sel.sum() / c.dz)), rmse_method=float(np.sqrt(sse[sel].sum() / sel.sum() / c.dz)), gain_rmse=g, gain_rmse_lo=glo, gain_rmse_hi=ghi,
                                             share_of_persistence_sse=float(pers_sse[sel].sum() / pers_sse.sum())))
        if dec:
            g = dec["gt_z"]["E_C"]; mu_, lo, hi = Z.cluster_bootstrap(g, take, n_boot=Z.N_BOOT)
            rows.append(dict(dataset=ds, h=h, method="gt_z (decoder floor)", n_pairs=len(r), n_takes=len(set(take)), dec_E_C=mu_, dec_E_C_lo=lo, dec_E_C_hi=hi, dec_d_gtz=0.0))
            mu_, lo, hi = Z.cluster_bootstrap(dC, take, n_boot=Z.N_BOOT)
            rows.append(dict(dataset=ds, h=h, method="dense persistence C_t (reference)", n_pairs=len(r), n_takes=len(set(take)), dec_E_C=mu_, dec_E_C_lo=lo, dec_E_C_hi=hi))
        arrays[f"h{h}|r"], arrays[f"h{h}|t"] = r, t
        log.info("%s h=%d: %s", ds, h, {m: (round(x["rmse"], 3), round(x["r2"], 3), round(x["gain_rmse"], 3)) for m, x in ((rw["method"], rw) for rw in rows if rw["h"] == h and "rmse" in rw)})
    pd.DataFrame(rows).to_csv(Z.ds_out(ds) / "local_metrics.csv", index=False); pd.DataFrame(prow).to_csv(Z.ds_out(ds) / "persistence.csv", index=False)
    pd.DataFrame(srow).to_csv(Z.ds_out(ds) / "local_splits.csv", index=False)
    np.savez_compressed(Z.ds_out(ds) / "local_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
