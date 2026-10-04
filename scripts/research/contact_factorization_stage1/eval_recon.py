#!/usr/bin/env python
"""Evaluation question 1 (reconstruction), question 6 (what r explains) and the capacity / collapse checks of Section 11.
    CUDA_VISIBLE_DEVICES=4 python eval_recon.py --dataset taco
On the unique TEST frames (take-cluster bootstrap, 1000 reps):
  reconstruction   E_zonly = ||C_bar - C||, E_full = ||C_hat - C|| (raw L2 per frame), standardised MSE, R^2 of C_bar / C_hat
                   against the standardised map, Gain_r = (E_zonly - E_full) / E_zonly with the paired CI;
                   baselines: train mean map, per-mesh train mean, PCA-64 / PCA-128 of the training frames (linear low-rank);
                   the previous study's R2 decoder (residual_prediction.csv) as the hand-designed reference
  explained_by_r   1 - sum ||C - C_hat||^2 / sum ||C - C_bar||^2 (energy ratio, bootstrap)
  localisation     |Delta_C| and the residual energies by GT contact bin (off < 0.05, soft 0.05-0.45, hard >= 0.45), the fraction
                   of the z residual energy removed per bin, and per-point mean |Delta_C| maps of the three most frequent test
                   meshes (figure 8)
  latents          per-dimension std, participation-ratio effective rank, fraction of dimensions above 1 % of the top variance,
                   |Delta_C| (raw) and its energy share, R^2_z / R^2_full
Writes <ds>/reconstruction.csv, <ds>/latent_statistics.csv, <ds>/residual_localisation.csv, <ds>/residual_maps.npz.
"""
from __future__ import annotations

import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch

import cf_common as S
from encode import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recon")
BINS = [("off", 0.0, 0.05), ("soft", 0.05, 0.45), ("hard", 0.45, 10.0)]


def unique_test(data, lat):
    """Indices (sequence row, frame) of the unique test frames inside the latent arrays (sequence order of idx['test'])."""
    n, t = data.split_frames("test"); n, t = n.cpu().numpy(), t.cpu().numpy()
    pos = {int(v): i for i, v in enumerate(lat["n"])}
    rows = np.array([pos[int(v)] for v in n])
    return rows, t


def r2_score(res_sq, C_std, mean):
    return 1 - res_sq.sum() / ((C_std - mean) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device(os.environ.get("CF_DEVICE", "cuda"))
    data = load_data(ds, dev, S.run_name("A3", a.seed))
    n_te, t_te = data.split_frames("test"); take = data.frames["test"]["take"]
    C = data.C[n_te, t_te]; Cn = data.Cn[n_te, t_te]
    n_tr, t_tr = data.split_frames("train"); Cn_tr = data.Cn[n_tr, t_tr]; C_tr = data.C[n_tr, t_tr]
    mean_tr = Cn_tr.mean(0); sst = float(((Cn - mean_tr) ** 2).sum())
    rows, lat_rows, loc_rows, maps = [], [], [], {}

    def add(name, kind, C_bar_raw, C_hat_raw, extra=None):
        ez = (C_bar_raw - C).norm(dim=-1).cpu().numpy(); ef = (C_hat_raw - C).norm(dim=-1).cpu().numpy()
        rz = data.to_std(C_bar_raw) - Cn; rf = data.to_std(C_hat_raw) - Cn
        row = dict(dataset=ds, model=name, kind=kind)
        for k, v in (("E_zonly", ez), ("E_full", ef), ("mse_zonly", (rz ** 2).mean(-1).cpu().numpy()), ("mse_full", (rf ** 2).mean(-1).cpu().numpy())):
            m, lo, hi = S.cluster_bootstrap(v, take, n_boot=S.N_BOOT); row[k] = m; row[k + "_lo"] = lo; row[k + "_hi"] = hi
        row["R2_zonly"] = 1 - float((rz ** 2).sum()) / sst; row["R2_full"] = 1 - float((rf ** 2).sum()) / sst
        g, glo, ghi = S.ratio_cluster_bootstrap(ez - ef, ez, take, n_boot=S.N_BOOT); row.update(gain_r=g, gain_r_lo=glo, gain_r_hi=ghi)
        d, dlo, dhi = S.paired_cluster_bootstrap(ez, ef, take, n_boot=S.N_BOOT); row.update(E_zonly_minus_full=d, E_zonly_minus_full_lo=dlo, E_zonly_minus_full_hi=dhi)
        e, elo, ehi = S.ratio_cluster_bootstrap((rz ** 2).sum(-1).cpu().numpy() - (rf ** 2).sum(-1).cpu().numpy(), (rz ** 2).sum(-1).cpu().numpy(), take, n_boot=S.N_BOOT)
        row.update(explained_by_r=e, explained_by_r_lo=elo, explained_by_r_hi=ehi)
        row.update(extra or {}); rows.append(row); return row

    # ---- baselines
    mean_raw = C_tr.mean(0)
    add("mean", "baseline: train mean map", mean_raw[None].expand_as(C), mean_raw[None].expand_as(C))
    mesh_tr = data.mesh[n_tr.cpu().numpy()]; mesh_te = data.mesh[n_te.cpu().numpy()]
    pm = torch.stack([C_tr[torch.from_numpy(mesh_tr == m).to(dev)].mean(0) if (mesh_tr == m).any() else mean_raw for m in mesh_te])
    add("mesh_mean", "baseline: per-mesh train mean map", pm, pm, dict(n_unseen_mesh=int(sum((mesh_tr == m).sum() == 0 for m in np.unique(mesh_te)))))
    Xc = Cn_tr - mean_tr
    U, Sv, Vh = torch.linalg.svd(Xc, full_matrices=False)
    for k in (64, 128):
        P = Vh[:k]; rec = (Cn - mean_tr) @ P.T @ P + mean_tr
        add(f"pca{k}", f"baseline: PCA-{k} of the training frames (linear, fitted on train)", data.to_raw(rec), data.to_raw(rec))
    prev = pd.read_csv(S.SAT.SV.OUT / "residual_prediction.csv")
    prev = prev[(prev.dataset == ds) & (prev.metric == "decoder_r2_test") & (prev.block == "C") & (prev.zstar == "R2")]
    prev_r2 = float(prev.value.iloc[0]) if len(prev) else np.nan
    rows.append(dict(dataset=ds, model="R2_decoder_prev", kind="reference: previous study's MLP decoder of the exact R2 (48-D) -> C (test R^2 on all frames)", R2_zonly=prev_r2, R2_full=prev_r2))
    # ---- models
    for model in S.ALL_MODELS:
        name = S.run_name(model, a.seed)
        if not S.latents_path(ds, name, "test").exists():
            log.warning("missing latents %s", name); continue
        lat = np.load(S.latents_path(ds, name, "test")); lr_ = unique_test(data, lat)
        C_bar = torch.from_numpy(lat["C_bar"][lr_].astype(np.float32)).to(dev); C_hat = torch.from_numpy(lat["C_hat"][lr_].astype(np.float32)).to(dev)
        tr = np.load(S.latents_path(ds, name, "train")); va = np.load(S.latents_path(ds, name, "val"))
        ck = torch.load(S.ckpt_path(ds, name), map_location="cpu", weights_only=False)
        extra = dict(train_E_zonly=float(tr["e_zonly"].mean()), train_E_full=float(tr["e_full"].mean()), val_E_zonly=float(va["e_zonly"].mean()), val_E_full=float(va["e_full"].mean()),
                     train_mse_full=float(tr["mse_full"].mean()), val_mse_full=float(va["mse_full"].mean()), best_step=int(ck["best_step"]), n_params=int(ck["n_params"]))
        row = add(model, S.ALL_MODELS[model]["label"], C_bar, C_hat, extra)
        # latent statistics (test frames)
        z = lat["z"][lr_]; ls = dict(dataset=ds, model=model, code="h" if model == "A0" else "z", dim=z.shape[1])
        for code, x in (("h" if model == "A0" else "z", z),) + ((("r", lat["r"][lr_]),) if "r" in lat else ()):
            x = torch.from_numpy(x.astype(np.float32)); xc = x - x.mean(0); ev = torch.linalg.eigvalsh(xc.T @ xc / len(xc)).flip(0)
            sd = x.std(0)
            lat_rows.append(dict(dataset=ds, model=model, code=code, dim=x.shape[1], std_mean=float(sd.mean()), std_min=float(sd.min()), std_max=float(sd.max()),
                                 eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()), frac_dims_above_1pct=float((ev > 0.01 * ev[0]).float().mean()),
                                 top1_var_share=float(ev[0] / ev.sum()), top8_var_share=float(ev[:8].sum() / ev.sum())))
        if "r" in lat:
            delta = C_hat - C_bar; rz = C - C_bar; rf = C - C_hat
            lat_rows[-1].update(delta_abs_mean=float(delta.abs().mean()), delta_energy_share=float((delta ** 2).sum() / ((C - mean_raw) ** 2).sum()),
                                R2_z_over_R2_full=row["R2_zonly"] / max(row["R2_full"], 1e-9))
            # localisation by GT contact bin
            tot_d, tot_z = float((delta ** 2).sum()), float((rz ** 2).sum())
            for bname, lo, hi in BINS:
                msk = (C >= lo) & (C < hi)
                loc_rows.append(dict(dataset=ds, model=model, bin=bname, frac_points=float(msk.float().mean()), delta_abs_mean=float(delta[msk].abs().mean()),
                                     resid_z_abs_mean=float(rz[msk].abs().mean()), resid_full_abs_mean=float(rf[msk].abs().mean()),
                                     delta_energy_share=float((delta[msk] ** 2).sum()) / tot_d, resid_z_energy_share=float((rz[msk] ** 2).sum()) / tot_z,
                                     frac_resid_z_energy_removed=1 - float((rf[msk] ** 2).sum()) / max(float((rz[msk] ** 2).sum()), 1e-9),
                                     corr_delta_resid_z=float(torch.corrcoef(torch.stack([delta[msk], rz[msk]]))[0, 1])))
            # sign agreement of Delta_C with the z residual (does r correct amplitude in the right direction?)
            loc_rows.append(dict(dataset=ds, model=model, bin="all", frac_points=1.0, delta_abs_mean=float(delta.abs().mean()), resid_z_abs_mean=float(rz.abs().mean()),
                                 resid_full_abs_mean=float(rf.abs().mean()), delta_energy_share=1.0, resid_z_energy_share=1.0,
                                 frac_resid_z_energy_removed=1 - float((rf ** 2).sum()) / tot_z, corr_delta_resid_z=float(torch.corrcoef(torch.stack([delta.flatten(), rz.flatten()]))[0, 1]),
                                 sign_agreement=float(((delta.sign() == rz.sign()) & (rz.abs() > 0.02)).float().sum() / (rz.abs() > 0.02).float().sum())))
            # per-point maps of the three most frequent test meshes
            vals, cnts = np.unique(mesh_te, return_counts=True)
            for m in vals[np.argsort(-cnts)[:3]]:
                sel = torch.from_numpy(mesh_te == m).to(dev)
                maps[f"{model}_{m}_delta_abs"] = delta[sel].abs().mean(0).cpu().numpy(); maps[f"{model}_{m}_resid_z_abs"] = rz[sel].abs().mean(0).cpu().numpy()
                maps[f"{model}_{m}_C_mean"] = C[sel].mean(0).cpu().numpy(); maps[f"{model}_{m}_n"] = int(sel.sum())
            # three representative examples (one per frequent mesh: the frame with the median z-residual energy of that mesh) for figure 2
            for j, m in enumerate(vals[np.argsort(-cnts)[:3]]):
                sel = np.where(mesh_te == m)[0]; en = (rz[sel] ** 2).sum(-1).cpu().numpy(); k = sel[np.argsort(en)[len(en) // 2]]
                x, _, _, valid = data.geometry(n_te[k:k + 1], t_te[k:k + 1])
                for key, arr in (("C", C[k]), ("C_bar", C_bar[k]), ("C_hat", C_hat[k]), ("delta", delta[k]), ("x", x[0]), ("valid", valid[0])):
                    maps[f"{model}_ex{j}_{key}"] = arr.cpu().numpy()
                maps[f"{model}_ex{j}_meta"] = np.array([str(m), int(n_te[k]), int(t_te[k])])
            maps[f"{model}_delta_abs_hist"] = np.histogram(delta.abs().flatten().cpu().numpy(), bins=np.linspace(0, 0.5, 51))[0]
            maps[f"{model}_resid_z_abs_hist"] = np.histogram(rz.abs().flatten().cpu().numpy(), bins=np.linspace(0, 0.5, 51))[0]
        log.info("%s %s: E_zonly %.4f E_full %.4f gain_r %.3f R2 %.3f/%.3f", ds, model, row["E_zonly"], row["E_full"], row["gain_r"], row["R2_zonly"], row["R2_full"])
    out = S.ds_out(ds); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "reconstruction.csv", index=False); pd.DataFrame(lat_rows).to_csv(out / "latent_statistics.csv", index=False)
    pd.DataFrame(loc_rows).to_csv(out / "residual_localisation.csv", index=False)
    geo = {}
    for m in np.unique(mesh_te):
        i = int(np.where(mesh_te == m)[0][0]); x, _, _, valid = data.geometry(n_te[i:i + 1], t_te[i:i + 1]); geo[f"x_{m}"] = x[0].cpu().numpy(); geo[f"valid_{m}"] = valid[0].cpu().numpy()
    np.savez_compressed(out / "residual_maps.npz", **maps, **geo)
    print(pd.DataFrame(rows)[["model", "E_zonly", "E_full", "gain_r", "R2_zonly", "R2_full", "explained_by_r"]].to_string())


if __name__ == "__main__":
    main()
