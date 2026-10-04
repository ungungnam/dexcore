#!/usr/bin/env python
"""Evaluation question 4 — the latent swap test.
    CUDA_VISIBLE_DEVICES=4 python eval_swap.py --dataset taco
Controlled pairs (A, B) of TEST frames: same mesh, different takes, teacher distance in the lowest 20 % of the mesh's cross-take
pairs (similar R2 / wrench) and dense distance above the median (meaningfully different realisation); up to ~400 pairs per
dataset, capped per mesh. For the factorised models: AA = D(z_A, r_A, G_A), BB, AB = D(z_A, r_B, G_A), BA = D(z_B, r_A, G_B).
Per swap (AB and BA both counted, 'z donor' / 'r donor'):
  structure   single-frame exact-rule R2 + wrench of the decoded map extracted with the reference frame's GT part labels, as
              a teacher distance to that frame's own extraction: d_struct(AB, z donor) vs d_struct(AB, r donor); the same for
              the R2 part and the wrench part separately; label-free: total amount, hard-contact IoU
  dense       ||AB - C_zdonor||, ||AB - C_rdonor||, ||AB - AA|| (what swapping r changes), the reconstruction floor
  detail      cos(Delta_AB, C_rdonor - C_bar_rdonor) vs cos(Delta_AB, C_zdonor - C_bar_zdonor)  (Delta_AB = AB - C_bar_zdonor)
Fractions 'structure follows z' and 'detail follows r' with take-cluster bootstrap over the z donor's take.
Writes <ds>/swap_metrics.csv, <ds>/swap_pairs_<model>.csv, <ds>/swap_examples.npz (figure 6).
"""
from __future__ import annotations

import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cf_common as S
from encode import load_data, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("swap")
N_PAIRS, Q_TEACHER, Q_DENSE = 800, 0.20, 0.50


def select_pairs(data, dev, seed=0):
    n, t = data.split_frames("test"); take = pd.factorize(data.frames["test"]["take"])[0]; mesh = data.mesh[n.cpu().numpy()]
    C = data.C[n, t]; u_r2, u_q = data.u_r2[n, t], data.u_q[n, t]
    g = torch.Generator(device=dev); g.manual_seed(seed); meshes = np.unique(mesh); cap = max(20, N_PAIRS // len(meshes)); pairs = []
    for m in meshes:
        idx = np.where(mesh == m)[0]
        if len(idx) < 30:
            continue
        ii = torch.from_numpy(idx).to(dev); tk = torch.from_numpy(take[idx]).to(dev); cross = tk[:, None] != tk[None, :]
        if cross.sum() == 0:
            continue
        D_t = data.teacher_distance(u_r2[ii], u_q[ii], u_r2[ii], u_q[ii]); D_c = torch.cdist(C[ii], C[ii])
        thr_t = torch.quantile(D_t[cross], Q_TEACHER); thr_c = torch.quantile(D_c[cross], Q_DENSE)
        ok = cross & (D_t <= thr_t) & (D_c >= thr_c) & (torch.arange(len(idx), device=dev)[:, None] < torch.arange(len(idx), device=dev)[None, :])
        cand = torch.nonzero(ok)
        if len(cand) == 0:
            continue
        cand = cand[torch.randperm(len(cand), device=dev, generator=g)[:cap]]
        for i, j in cand.cpu().numpy():
            pairs.append(dict(mesh=str(m), iA=int(idx[i]), iB=int(idx[j]), d_teacher=float(D_t[i, j]), d_dense=float(D_c[i, j]), thr_t=float(thr_t), thr_c=float(thr_c)))
    return pd.DataFrame(pairs)


def hard_iou(X, Y, c):
    a, b = X >= c, Y >= c
    return (a & b).float().sum(-1) / (a | b).float().sum(-1).clamp(min=1)


@torch.no_grad()
def swap_metrics(data, net, P, dev, bs=64):
    n, t = data.split_frames("test"); rows = []; ex = {}
    for s in range(0, len(P), bs):
        p = P.iloc[s:s + bs]; iA = torch.from_numpy(p.iA.values).to(dev); iB = torch.from_numpy(p.iB.values).to(dev)
        bA, bB = data.batch(n[iA], t[iA]), data.batch(n[iB], t[iB])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            oA = net(bA["Cn"], bA["geo"], bA["S"]); oB = net(bB["Cn"], bB["geo"], bB["S"])
            zA, rA, zB, rB = oA["z"].float(), oA["r"].float(), oB["z"].float(), oB["r"].float()
            AB = oA["C_bar"].float() + net.decode_r(zA, rB, bA["geo"], bA["S"], oA["C_bar"]).float(); BA = oB["C_bar"].float() + net.decode_r(zB, rA, bB["geo"], bB["S"], oB["C_bar"]).float()
        maps = {k: data.to_raw(v).clamp(0, 1) for k, v in dict(AA=oA["C_hat"].float(), BB=oB["C_hat"].float(), AB=AB, BA=BA, barA=oA["C_bar"].float(), barB=oB["C_bar"].float()).items()}
        CA, CB = bA["C"], bB["C"]; labA, nhA = data.lab_nh(n[iA], t[iA]); labB, nhB = data.lab_nh(n[iB], t[iB])
        T = {}
        for key, Cm, lab, nh, nn_, tt in (("A", CA, labA, nhA, n[iA], t[iA]), ("B", CB, labB, nhB, n[iB], t[iB]), ("AA|A", maps["AA"], labA, nhA, n[iA], t[iA]), ("BB|B", maps["BB"], labB, nhB, n[iB], t[iB]),
                                          ("AB|A", maps["AB"], labA, nhA, n[iA], t[iA]), ("AB|B", maps["AB"], labB, nhB, n[iB], t[iB]), ("BA|B", maps["BA"], labB, nhB, n[iB], t[iB]), ("BA|A", maps["BA"], labA, nhA, n[iA], t[iA])):
            T[key] = data.teacher_of(data.frame_r2(Cm, nn_, tt, lab, nh))
        d = lambda x, y: (S.TEACHER["w_r2"] * (T[x][0] - T[y][0]).norm(dim=-1) + S.TEACHER["w_q"] * (T[x][1] - T[y][1]).norm(dim=-1))
        d_r2 = lambda x, y: (T[x][0] - T[y][0]).norm(dim=-1); d_q = lambda x, y: (T[x][1] - T[y][1]).norm(dim=-1)
        alphaA, alphaB = bA["alpha"], bB["alpha"]; c = data.cal["c_hard"]
        resA, resB = CA - maps["barA"], CB - maps["barB"]
        for swap, zd, rd, Cz, Cr, Czz, resz, resr, alpha, own in (("AB", "A", "B", CA, CB, "AA", resA, resB, alphaA, "AA|A"), ("BA", "B", "A", CB, CA, "BB", resB, resA, alphaB, "BB|B")):
            M = maps[swap]; delta = M - maps["bar" + zd]
            r = dict(swap=swap, struct_to_zdonor=d(f"{swap}|{zd}", zd), struct_to_rdonor=d(f"{swap}|{rd}", rd), struct_floor=d(own, zd),
                     r2_to_zdonor=d_r2(f"{swap}|{zd}", zd), r2_to_rdonor=d_r2(f"{swap}|{rd}", rd), q_to_zdonor=d_q(f"{swap}|{zd}", zd), q_to_rdonor=d_q(f"{swap}|{rd}", rd),
                     dense_to_zdonor=(M - Cz).norm(dim=-1), dense_to_rdonor=(M - Cr).norm(dim=-1), dense_floor=(maps[Czz] - Cz).norm(dim=-1), dense_change=(M - maps[Czz]).norm(dim=-1),
                     dense_AB=(Cz - Cr).norm(dim=-1), amount_swap=(alpha * M).sum(-1), amount_zdonor=(alpha * Cz).sum(-1), amount_rdonor=(alpha * Cr).sum(-1),
                     iou_zdonor=hard_iou(M, Cz, c), iou_rdonor=hard_iou(M, Cr, c), iou_floor=hard_iou(maps[Czz], Cz, c),
                     detail_cos_rdonor=F.cosine_similarity(delta, resr, dim=-1), detail_cos_zdonor=F.cosine_similarity(delta, resz, dim=-1),
                     detail_cos_own=F.cosine_similarity(maps[Czz] - maps["bar" + zd], resz, dim=-1))
            for k in range(len(p)):
                rows.append(dict(pair=int(p.index[k]), mesh=p.mesh.iloc[k], take_z=data.frames["test"]["take"][p.iA.iloc[k] if zd == "A" else p.iB.iloc[k]], d_teacher_sel=float(p.d_teacher.iloc[k]), d_dense_sel=float(p.d_dense.iloc[k]),
                                 **{kk: float(v[k]) for kk, v in r.items() if kk != "swap"}, swap=swap))
        if s == 0:
            for k in range(min(6, len(p))):
                x, _, _, valid = data.geometry(n[iA[k:k + 1]], t[iA[k:k + 1]])
                ex[f"ex{k}"] = dict(C_A=CA[k].cpu().numpy(), C_B=CB[k].cpu().numpy(), AA=maps["AA"][k].cpu().numpy(), BB=maps["BB"][k].cpu().numpy(), AB=maps["AB"][k].cpu().numpy(), BA=maps["BA"][k].cpu().numpy(),
                                    barA=maps["barA"][k].cpu().numpy(), barB=maps["barB"][k].cpu().numpy(), x=x[0].cpu().numpy(), valid=valid[0].cpu().numpy(), mesh=p.mesh.iloc[k])
    return pd.DataFrame(rows), ex


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device(os.environ.get("CF_DEVICE", "cuda"))
    data = load_data(ds, dev, S.run_name("A3", a.seed), need_masks=True)
    P = select_pairs(data, dev); log.info("%s: %d controlled pairs over %d meshes", ds, len(P), P.mesh.nunique())
    P.to_csv(S.ds_out(ds) / "swap_pairs_selected.csv", index=False)
    rows, flat = [], {}
    for model in S.FACTORISED:
        name = S.run_name(model, a.seed)
        if not S.ckpt_path(ds, name).exists():
            continue
        net, _ = load_model(ds, name, data, dev)
        df, ex = swap_metrics(data, net, P, dev); df.insert(0, "model", model); df.to_csv(S.ds_out(ds) / f"swap_pairs_{model}.csv", index=False)
        tk = df.take_z.values; row = dict(dataset=ds, model=model, n_pairs=len(P), n_swaps=len(df))
        for k in ("struct_to_zdonor", "struct_to_rdonor", "struct_floor", "r2_to_zdonor", "r2_to_rdonor", "q_to_zdonor", "q_to_rdonor", "dense_to_zdonor", "dense_to_rdonor", "dense_floor", "dense_change", "dense_AB",
                  "iou_zdonor", "iou_rdonor", "iou_floor", "detail_cos_rdonor", "detail_cos_zdonor", "detail_cos_own", "d_teacher_sel", "d_dense_sel"):
            row[k], row[k + "_lo"], row[k + "_hi"] = S.cluster_bootstrap(df[k].values, tk, n_boot=S.N_BOOT)
        for k, cond in (("frac_struct_follows_z", df.struct_to_zdonor < df.struct_to_rdonor), ("frac_r2_follows_z", df.r2_to_zdonor < df.r2_to_rdonor), ("frac_q_follows_z", df.q_to_zdonor < df.q_to_rdonor),
                        ("frac_detail_follows_r", df.detail_cos_rdonor > df.detail_cos_zdonor), ("frac_iou_follows_z", df.iou_zdonor > df.iou_rdonor)):
            row[k], row[k + "_lo"], row[k + "_hi"] = S.cluster_bootstrap(cond.values.astype(float), tk, n_boot=S.N_BOOT)
        row["detail_cos_diff"], row["detail_cos_diff_lo"], row["detail_cos_diff_hi"] = S.paired_cluster_bootstrap(df.detail_cos_rdonor.values, df.detail_cos_zdonor.values, tk, n_boot=S.N_BOOT)
        row["struct_diff"], row["struct_diff_lo"], row["struct_diff_hi"] = S.paired_cluster_bootstrap(df.struct_to_rdonor.values, df.struct_to_zdonor.values, tk, n_boot=S.N_BOOT)
        row["amount_corr_zdonor"] = float(np.corrcoef(df.amount_swap, df.amount_zdonor)[0, 1]); row["amount_corr_rdonor"] = float(np.corrcoef(df.amount_swap, df.amount_rdonor)[0, 1])
        rows.append(row); log.info("%s %s: struct->z %.3f vs ->r %.3f (floor %.3f) frac %.2f | detail cos r %.3f vs z %.3f frac %.2f | dense change %.3f of AB %.3f", ds, model, row["struct_to_zdonor"], row["struct_to_rdonor"], row["struct_floor"], row["frac_struct_follows_z"], row["detail_cos_rdonor"], row["detail_cos_zdonor"], row["frac_detail_follows_r"], row["dense_change"], row["dense_AB"])
        for ek, e in ex.items():
            for k, v in e.items():
                flat[f"{model}__{ek}__{k}"] = v
    pd.DataFrame(rows).to_csv(S.ds_out(ds) / "swap_metrics.csv", index=False)
    np.savez_compressed(S.ds_out(ds) / "swap_examples.npz", **flat)


if __name__ == "__main__":
    main()
