#!/usr/bin/env python
"""Experiment B: temporal geometry of z (no training).    python zt_geometry.py --dataset taco      python zt_geometry.py --merge
For every GT frame pair (t, t + h), h = 1, 4, 8:
    delta_C = ||C_t - C_{t+h}||            raw canonical contact (the E_C convention)
    delta_z = ||z_t - z_{t+h}||            z standardised with TRAIN statistics
    delta_T = 0.5 ||u_R2,t - u_R2,t+h|| + 0.5 ||u_q,t - u_q,t+h||     the Stage-1 teacher distance (exact R2 + wrench), secondary
each divided by its mean over 200 000 random TRAIN frame pairs (so 1.0 = the distance between two unrelated training frames).
Reported on the TEST transitions: Pearson / Spearman (take-cluster bootstrap), conditional means of delta_z in TRAIN deciles of delta_C,
the four quadrants with thresholds fixed from the TRAIN 30 % / 70 % quantiles of each quantity at that horizon, what the teacher distance
does inside the low-delta_C / high-delta_z quadrant, the lag-h autocorrelation of z, and (optional, h = 4 only) the directional
consistency: cosine of the latent changes of transitions whose dense maps before and after are nearest neighbours.
Writes <ds>/geometry.csv, <ds>/quadrants.csv, <ds>/geometry_bins.csv, <ds>/direction.csv, <ds>/geometry_arrays.npz.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import zt_common as Z
from zt_data import Cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zt_geometry")
N_BOOT_CORR = 300


def deltas(c, r, t, h):
    dC = np.linalg.norm(c.C[r, t] - c.C[r, t + h], axis=1)
    dz = np.linalg.norm(c.zs[r, t] - c.zs[r, t + h], axis=1)
    dT = 0.5 * np.linalg.norm(c.u_r2[r, t] - c.u_r2[r, t + h], axis=1) + 0.5 * np.linalg.norm(c.u_q[r, t] - c.u_q[r, t + h], axis=1)
    flips = np.abs(c.a[r, t].astype(np.int16) - c.a[r, t + h].astype(np.int16)).sum(1)
    normC = np.linalg.norm(c.C[r, t], axis=1)                                     # how much contact the starting frame has (raw L2 norm of the map)
    return dC, dz, dT, flips, normC


def articulation_step(c, r, t, h):
    """ARCTIC only: |change of the object's articulation angle| between the two frames (z-scored units; the first object-state channel at
    window offset 0). The encoder reads the per-frame articulated geometry G_t, so z can move while the canonical contact map does not."""
    D = c.d_tau // 9; art = lambda tt: c.tau[r, tt].reshape(len(r), 9, D)[:, 4, 0]
    return np.abs(art(t + h) - art(t))


def boot_corr(x, y, take, fn, n_boot=N_BOOT_CORR, seed=0):
    u, inv = np.unique(take, return_inverse=True); groups = [np.where(inv == g)[0] for g in range(len(u))]; rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        sel = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))]); reps.append(fn(x[sel], y[sel])[0])
    return float(fn(x, y)[0]), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def partial_spearman(rxy, rxz, ryz):
    """Spearman correlation of x and y with z partialled out (from the three pairwise rank correlations)."""
    return float((rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=Z.DATASETS); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge:
        for name, out in (("geometry.csv", "temporal_geometry_metrics.csv"), ("quadrants.csv", "quadrant_metrics.csv"), ("geometry_bins.csv", "temporal_geometry_bins.csv"), ("direction.csv", "directional_consistency.csv")):
            pd.concat([pd.read_csv(Z.ds_out(ds) / name) for ds in Z.DATASETS if (Z.ds_out(ds) / name).exists()], ignore_index=True).to_csv(Z.OUT / out, index=False)
        return
    ds = a.dataset; c = Cache(ds, with_contact=True); rng = np.random.default_rng(1)
    # ---- normalisers: random TRAIN frame pairs from different sequences
    tr = c.rows("train"); i1, i2 = rng.choice(tr, Z.N_RANDOM_PAIRS), rng.choice(tr, Z.N_RANDOM_PAIRS); keep = i1 != i2; i1, i2 = i1[keep], i2[keep]
    t1, t2 = rng.integers(0, Z.T, len(i1)), rng.integers(0, Z.T, len(i1))
    sC = float(np.linalg.norm(c.C[i1, t1] - c.C[i2, t2], axis=1).mean()); sz = float(np.linalg.norm(c.zs[i1, t1] - c.zs[i2, t2], axis=1).mean())
    sT = float((0.5 * np.linalg.norm(c.u_r2[i1, t1] - c.u_r2[i2, t2], axis=1) + 0.5 * np.linalg.norm(c.u_q[i1, t1] - c.u_q[i2, t2], axis=1)).mean())
    log.info("%s normalisers (mean random train-pair distance): C %.3f, z %.3f, teacher %.3f", ds, sC, sz, sT)
    grow, qrow, brow, arrays = [], [], [], {}
    for h in Z.HORIZONS:
        D = {}
        for split in ("train", "test"):
            r, t = c.pairs(split, h); dC, dz, dT, fl, nC = deltas(c, r, t, h); D[split] = dict(r=r, t=t, dC=dC / sC, dz=dz / sz, dT=dT / sT, flips=fl, normC=nC, take=c.take[r])
        trn, te = D["train"], D["test"]
        thr = {k: (float(np.quantile(trn[k], Z.QUANT_LOW)), float(np.quantile(trn[k], Z.QUANT_HIGH))) for k in ("dC", "dz", "dT")}          # fixed from TRAIN
        # ---- correlations (test)
        res = dict(dataset=ds, h=h, n_test=len(te["dC"]), n_takes=len(set(te["take"])), scale_C=sC, scale_z=sz, scale_T=sT,
                   mean_dC=float(te["dC"].mean()), mean_dz=float(te["dz"].mean()), mean_dT=float(te["dT"].mean()), median_dC=float(np.median(te["dC"])), median_dz=float(np.median(te["dz"])))
        for name, x, y in (("C_z", te["dC"], te["dz"]), ("T_z", te["dT"], te["dz"]), ("C_T", te["dC"], te["dT"])):
            for cn, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
                v, lo, hi = boot_corr(x, y, te["take"], fn); res.update({f"{cn}_{name}": v, f"{cn}_{name}_lo": lo, f"{cn}_{name}_hi": hi})
        res["partial_spearman_z_T_given_C"] = partial_spearman(res["spearman_T_z"], res["spearman_C_T"], res["spearman_C_z"])
        res["partial_spearman_z_C_given_T"] = partial_spearman(res["spearman_C_z"], res["spearman_C_T"], res["spearman_T_z"])
        res["pearson_log_C_z"] = float(pearsonr(np.log(te["dC"] + 1e-4), np.log(te["dz"] + 1e-4))[0])
        # lag-h autocorrelation of the standardised latent (per dimension, averaged) and of the dense map (per point with non-zero variance)
        zt_, zh_ = c.zs[te["r"], te["t"]], c.zs[te["r"], te["t"] + h]
        res["acf_z"] = float(np.mean([np.corrcoef(zt_[:, d], zh_[:, d])[0, 1] for d in range(c.dz)]))
        res["persistence_r2_z"] = float(1 - ((zt_ - zh_) ** 2).sum() / ((zh_ - zh_.mean(0)) ** 2).sum())
        Ct, Ch = c.C[te["r"], te["t"]], c.C[te["r"], te["t"] + h]
        res["persistence_r2_C"] = float(1 - ((Ct - Ch) ** 2).sum() / ((Ch - Ch.mean(0)) ** 2).sum())
        res.update({f"thr_{k}_low": v[0] for k, v in thr.items()}); res.update({f"thr_{k}_high": v[1] for k, v in thr.items()})
        grow.append(res)
        # ---- conditional means in TRAIN deciles of delta_C
        edges = np.quantile(trn["dC"], np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
        b = np.digitize(te["dC"], edges[1:-1])
        for k in range(10):
            s = b == k
            if s.any():
                brow.append(dict(dataset=ds, h=h, decile=k + 1, n=int(s.sum()), dC_mean=float(te["dC"][s].mean()), dz_mean=float(te["dz"][s].mean()), dz_q25=float(np.quantile(te["dz"][s], 0.25)),
                                 dz_q75=float(np.quantile(te["dz"][s], 0.75)), dT_mean=float(te["dT"][s].mean()), ratio_dz_over_dC=float(te["dz"][s].mean() / max(te["dC"][s].mean(), 1e-9))))
        # ---- quadrants (test fractions; thresholds from train)
        lowC, highC = te["dC"] <= thr["dC"][0], te["dC"] >= thr["dC"][1]; lowz, highz = te["dz"] <= thr["dz"][0], te["dz"] >= thr["dz"][1]
        quads = {"A_lowC_lowz": lowC & lowz, "B_lowC_highz": lowC & highz, "C_highC_lowz": highC & lowz, "D_highC_highz": highC & highz}
        row = dict(dataset=ds, h=h, n_test=len(lowC), frac_lowC=float(lowC.mean()), frac_highC=float(highC.mean()), frac_lowz=float(lowz.mean()), frac_highz=float(highz.mean()))
        for k, s in quads.items():
            v, lo, hi = Z.cluster_bootstrap(s.astype(float), te["take"], n_boot=Z.N_BOOT); row.update({k: v, k + "_lo": lo, k + "_hi": hi})
        row["indep_B_at_test_marginals"] = float(lowC.mean() * highz.mean()); row["indep_C_at_test_marginals"] = float(highC.mean() * lowz.mean())
        # composition of the critical quadrant B: weak-contact start, participation flip, large teacher (R2 + wrench) step, or none of these
        sB = quads["B_lowC_highz"]; weak = te["normC"] <= np.quantile(trn["normC"], 0.10); flip = te["flips"] > 0; hiT = te["dT"] >= thr["dT"][1]
        if sB.any():
            row.update(B_n=int(sB.sum()), B_weak_frac=float(weak[sB].mean()), B_flip_frac=float(flip[sB].mean()), B_highT_frac=float(hiT[sB].mean()), B_structural_frac=float((flip | hiT)[sB].mean()),
                       B_any_frac=float((weak | flip | hiT)[sB].mean()), B_none_frac=float((~(weak | flip | hiT))[sB].mean()), B_n_takes=int(len(set(te["take"][sB]))))
        row["P_highz_given_lowC"] = float(highz[lowC].mean()) if lowC.any() else np.nan; row["P_lowz_given_highC"] = float(lowz[highC].mean()) if highC.any() else np.nan
        row["P_lowz_given_lowC"] = float(lowz[lowC].mean()) if lowC.any() else np.nan; row["P_highz_given_highC"] = float(highz[highC].mean()) if highC.any() else np.nan
        # what the structure (teacher distance) does inside the quadrants: is a large latent step with a small dense step a structural step?
        rank_T = np.searchsorted(np.sort(trn["dT"]), te["dT"]) / len(trn["dT"])                    # train-quantile rank of the teacher distance
        for k, s in quads.items():
            row[f"{k}_teacher_rank"] = float(rank_T[s].mean()) if s.any() else np.nan; row[f"{k}_flip_frac"] = float((te["flips"][s] > 0).mean()) if s.any() else np.nan
            row[f"{k}_highT_frac"] = float((te["dT"][s] >= thr["dT"][1]).mean()) if s.any() else np.nan
            row[f"{k}_dz_mean"] = float(te["dz"][s].mean()) if s.any() else np.nan; row[f"{k}_dC_mean"] = float(te["dC"][s].mean()) if s.any() else np.nan
            row[f"{k}_normC_median"] = float(np.median(te["normC"][s])) if s.any() else np.nan; row[f"{k}_n"] = int(s.sum())
            # share of the quadrant's transitions that start from a weak-contact frame (below the TRAIN 10 % quantile of the contact norm)
            row[f"{k}_weak_contact_frac"] = float((te["normC"][s] <= np.quantile(trn["normC"], 0.10)).mean()) if s.any() else np.nan
        if ds == "arctic":
            da_tr = articulation_step(c, trn["r"], trn["t"], h); da = articulation_step(c, te["r"], te["t"], h); a70 = float(np.quantile(da_tr, Z.QUANT_HIGH))
            for k, s in quads.items():
                row[f"{k}_articulation_step_median"] = float(np.median(da[s])) if s.any() else np.nan; row[f"{k}_articulating_frac"] = float((da[s] >= a70).mean()) if s.any() else np.nan
            row["all_articulation_step_median"] = float(np.median(da)); row["all_articulating_frac"] = float((da >= a70).mean()); row["lowC_articulating_frac"] = float((da[lowC] >= a70).mean())
            row["spearman_dz_articulation_given_lowC"] = float(spearmanr(da[lowC], te["dz"][lowC])[0])
        row["all_normC_median"] = float(np.median(te["normC"])); row["weak_contact_threshold"] = float(np.quantile(trn["normC"], 0.10)); row["all_weak_contact_frac"] = float((te["normC"] <= np.quantile(trn["normC"], 0.10)).mean())
        qrow.append(row)
        arrays[f"h{h}|weak_thr"] = np.array([np.quantile(trn["normC"], 0.10)])
        for k in ("r", "t", "dC", "dz", "dT", "flips", "normC"):
            arrays[f"h{h}|{k}"] = te[k]
        arrays[f"h{h}|thr"] = np.array([thr["dC"][0], thr["dC"][1], thr["dz"][0], thr["dz"][1], thr["dT"][0], thr["dT"][1]])
        log.info("%s h=%d: spearman C-z %.3f, T-z %.3f, C-T %.3f; quadrants A %.3f B %.3f C %.3f D %.3f; acf_z %.3f", ds, h, res["spearman_C_z"], res["spearman_T_z"], res["spearman_C_T"],
                 row["A_lowC_lowz"], row["B_lowC_highz"], row["C_highC_lowz"], row["D_highC_highz"], res["acf_z"])
    # ---- optional: directional consistency (h = DIRECTION_H): do transitions with similar maps before and after move z in the same direction?
    import torch
    h = Z.DIRECTION_H; r, t = c.pairs("test", h); dC = np.linalg.norm(c.C[r, t] - c.C[r, t + h], axis=1) / sC
    med = float(np.median(np.linalg.norm(c.C[c.pairs("train", h)[0], c.pairs("train", h)[1]] - c.C[c.pairs("train", h)[0], c.pairs("train", h)[1] + h], axis=1) / sC))
    sel = np.where(dC >= med)[0]; r, t = r[sel], t[sel]                                                 # transitions whose contact actually moves (above the TRAIN median)
    Fm = np.concatenate([c.C[r, t], c.C[r, t + h]], 1); dCv = c.C[r, t + h] - c.C[r, t]; dzv = c.zs[r, t + h] - c.zs[r, t]; take, mesh = c.take[r], c.mesh[r]
    cos = lambda a_, b_: (a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1) + 1e-12)
    nn_i, nn_j, rd_j = [], [], []
    for mid in np.unique(mesh):
        idx = np.where(mesh == mid)[0]
        if len(idx) < 4 or len(set(take[idx])) < 2:
            continue
        X = torch.from_numpy(Fm[idx]); d = torch.cdist(X, X); same = torch.from_numpy(take[idx][:, None] == take[idx][None]); d[same] = float("inf")
        j = d.argmin(1).numpy(); ok = np.isfinite(d.min(1).values.numpy())
        nn_i.append(idx[ok]); nn_j.append(idx[j[ok]])
        # random partner: another take of the same mesh
        cand = [idx[take[idx] != take[i]] for i in idx[ok]]; rd_j.append(np.array([rng.choice(cd) for cd in cand]))
    nn_i, nn_j, rd_j = np.concatenate(nn_i), np.concatenate(nn_j), np.concatenate(rd_j)
    drow = dict(dataset=ds, h=h, n_transitions=len(nn_i), n_meshes=int(len(np.unique(mesh[nn_i]))), min_dC_over_train_median=med)
    for name, j in (("nn", nn_j), ("random", rd_j)):
        cz, cC = cos(dzv[nn_i], dzv[j]), cos(dCv[nn_i], dCv[j])
        for k, v in (("cos_dz", cz), ("cos_dC", cC)):
            m_, lo, hi = Z.cluster_bootstrap(v, take[nn_i], n_boot=Z.N_BOOT); drow.update({f"{k}_{name}": m_, f"{k}_{name}_lo": lo, f"{k}_{name}_hi": hi})
        if name == "nn":
            s = cC >= 0.5; drow.update(frac_nn_similar_dense_change=float(s.mean()), cos_dz_nn_given_similar_dC=float(cz[s].mean()) if s.any() else np.nan,
                                       cos_dz_nn_given_dissimilar_dC=float(cz[~s].mean()) if (~s).any() else np.nan, spearman_cos_dz_cos_dC=float(spearmanr(cz, cC)[0]))
    log.info("%s directional consistency (h=%d, %d transitions): cos dz nn %.3f vs random %.3f; cos dC nn %.3f vs random %.3f; given similar dC %.3f", ds, h, len(nn_i), drow["cos_dz_nn"], drow["cos_dz_random"],
             drow["cos_dC_nn"], drow["cos_dC_random"], drow["cos_dz_nn_given_similar_dC"])
    pd.DataFrame(grow).to_csv(Z.ds_out(ds) / "geometry.csv", index=False); pd.DataFrame(qrow).to_csv(Z.ds_out(ds) / "quadrants.csv", index=False)
    pd.DataFrame(brow).to_csv(Z.ds_out(ds) / "geometry_bins.csv", index=False); pd.DataFrame([drow]).to_csv(Z.ds_out(ds) / "direction.csv", index=False)
    np.savez_compressed(Z.ds_out(ds) / "geometry_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
