#!/usr/bin/env python
"""How low-dimensional is the residual that the z-only map leaves?  CUDA_VISIBLE_DEVICES=6 python diag_residual_pca.py --dataset taco --name A1_seed0
Encodes the unique train / test frames with the given single-latent model, takes the standardised residual C~ - C_bar~, fits a PCA on the
training residuals (global, and per mesh for the meshes with >= 2000 training frames) and reports the test energy fraction captured by
k = 8 / 16 / 32 / 64 / 128 components — the ceiling of a LINEAR 64-D realisation code. Also the fraction of the residual energy on
the GT contact patch (C >= 0.05). Writes <ds>/diag_residual_pca_<name>.json."""
from __future__ import annotations

import argparse
import os
import json

import numpy as np
import torch

import cf_common as S
from encode import load_data, load_model, encode_frames


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--name", default="A1_seed0")
    a = ap.parse_args(); dev = torch.device(os.environ.get("CF_DEVICE", "cuda")); ds = a.dataset
    data = load_data(ds, dev, a.name); net, ck = load_model(ds, a.name, data, dev)
    out = {}; R = {}
    for split in ("train", "test"):
        n, t = data.split_frames(split); enc = encode_frames(net, data, n, t, keep_maps=True)
        C_bar = data.to_std(torch.from_numpy(enc["C_bar"].astype(np.float32)).to(dev)); Cn = data.Cn[n, t]
        R[split] = (Cn - C_bar); out[f"{split}_E_zonly"] = float(enc["e_zonly"].mean()); out[f"{split}_mse_zonly"] = float(enc["mse_zonly"].mean())
        C = data.C[n, t]; patch = C >= 0.05
        out[f"{split}_residual_energy_on_patch"] = float((R[split][patch] ** 2).sum() / (R[split] ** 2).sum()); out[f"{split}_patch_fraction"] = float(patch.float().mean())
    mu = R["train"].mean(0); Xc = R["train"] - mu; _, Sv, Vh = torch.linalg.svd(Xc, full_matrices=False)
    tot = float(((R["test"] - mu) ** 2).sum())
    for k in (8, 16, 32, 64, 128, 256):
        P = Vh[:k]; rec = (R["test"] - mu) @ P.T @ P; out[f"global_pca{k}_test_energy_captured"] = 1 - float(((R["test"] - mu - rec) ** 2).sum()) / tot
    out["train_var_spectrum_top64_share"] = float((Sv[:64] ** 2).sum() / (Sv ** 2).sum())
    # per-mesh PCA (the residual patterns are object specific)
    n_tr, _ = data.split_frames("train"); n_te, _ = data.split_frames("test"); m_tr = data.mesh[n_tr.cpu().numpy()]; m_te = data.mesh[n_te.cpu().numpy()]
    cap = {k: 0.0 for k in (8, 16, 32, 64)}; tot_m = 0.0; n_mesh = 0
    for m in np.unique(m_te):
        itr = torch.from_numpy(m_tr == m).to(dev); ite = torch.from_numpy(m_te == m).to(dev)
        if int(itr.sum()) < 2000:
            continue
        n_mesh += 1; Xm = R["train"][itr]; mum = Xm.mean(0); _, _, Vm = torch.linalg.svd(Xm - mum, full_matrices=False); Y = R["test"][ite] - mum; tot_m += float((Y ** 2).sum())
        for k in cap:
            P = Vm[:k]; cap[k] += float((Y ** 2).sum()) - float(((Y - Y @ P.T @ P) ** 2).sum())
    for k in cap:
        out[f"per_mesh_pca{k}_test_energy_captured"] = cap[k] / max(tot_m, 1e-9)
    out["per_mesh_n_meshes"] = n_mesh; out["per_mesh_test_energy_share"] = tot_m / tot
    S.write_json(S.ds_out(ds) / f"diag_residual_pca_{a.name}.json", out); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
