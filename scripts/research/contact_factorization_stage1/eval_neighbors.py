#!/usr/bin/env python
"""Evaluation question 3 — do z neighbourhoods preserve R2 / wrench similarity?
    CUDA_VISIBLE_DEVICES=4 python eval_neighbors.py --dataset taco
Unique TEST frames; the retrieval pool of an anchor = the frames of the same mesh from OTHER takes (no temporal near-duplicates).
Spaces: h (A0), z (A1), z / r (A2, A3), the dense map C~ (standardised Euclidean), the teacher itself (oracle) and a random
draw. k = 10. Per anchor: mean teacher distance of the neighbours (also relative to the random-pair mean of the pool),
mean dense distance (raw L2), recall@10 against the teacher neighbours, and the rate of "structurally similar but densely
different" neighbours (teacher distance below the pool's 25th percentile AND dense distance above the pool median).
Take-cluster bootstrap over anchors. Writes <ds>/neighborhood_metrics.csv and <ds>/neighbor_examples.npz (figure 5).
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
from eval_probes import frame_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("knn")
MAX_ANCHORS_PER_MESH = 400
MIN_POOL = 30


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device(os.environ.get("CF_DEVICE", "cuda")); K = S.KNN_K
    data = load_data(ds, dev, S.run_name("A3", a.seed))
    n, t = data.split_frames("test"); fr = data.frames["test"]; take = fr["take"]; mesh = data.mesh[n.cpu().numpy()]
    Cn = data.Cn[n, t]; C = data.C[n, t]; u_r2, u_q = data.u_r2[n, t], data.u_q[n, t]
    spaces = {"C": Cn}
    for model in S.ALL_MODELS:
        name = S.run_name(model, a.seed)
        if not S.latents_path(ds, name, "test").exists():
            continue
        lat = np.load(S.latents_path(ds, name, "test")); rows_, _ = frame_rows(data, lat, "test")
        spaces["A0:h" if model == "A0" else f"{model}:z"] = torch.from_numpy(lat["z"][rows_, t.cpu().numpy()].astype(np.float32)).to(dev)
        if "r" in lat:
            spaces[f"{model}:r"] = torch.from_numpy(lat["r"][rows_, t.cpu().numpy()].astype(np.float32)).to(dev)
    g = torch.Generator(device=dev); g.manual_seed(0)
    per = {k: dict(teacher=[], teacher_rel=[], dense=[], recall=[], ssdd=[], take=[]) for k in list(spaces) + ["teacher", "random"]}
    examples = {}
    take_codes = pd.factorize(take)[0]
    for m in np.unique(mesh):
        idx = np.where(mesh == m)[0]
        if len(idx) < MIN_POOL + 1:
            continue
        ii = torch.from_numpy(idx).to(dev); tk = torch.from_numpy(take_codes[idx]).to(dev)
        allowed = tk[:, None] != tk[None, :]                                           # other takes only
        if (allowed.sum(1) < MIN_POOL).all():
            continue
        D_t = data.teacher_distance(u_r2[ii], u_q[ii], u_r2[ii], u_q[ii])              # (n, n)
        D_c = torch.cdist(C[ii], C[ii])
        anchors = torch.arange(len(idx), device=dev)[allowed.sum(1) >= MIN_POOL]
        if len(anchors) > MAX_ANCHORS_PER_MESH:
            anchors = anchors[torch.randperm(len(anchors), device=dev, generator=g)[:MAX_ANCHORS_PER_MESH]]
        inf = torch.full_like(D_t, float("inf"))
        Dt_m = torch.where(allowed, D_t, inf); Dc_m = torch.where(allowed, D_c, inf)
        teacher_nn = Dt_m[anchors].topk(K, largest=False).indices                        # (A, K)
        # pool statistics per anchor
        q25 = torch.stack([torch.quantile(D_t[a_][allowed[a_]], 0.25) for a_ in anchors]); med_c = torch.stack([torch.median(D_c[a_][allowed[a_]]) for a_ in anchors])
        rand_mean = torch.stack([D_t[a_][allowed[a_]].mean() for a_ in anchors])

        def record(key, nn_idx):
            dt = D_t[anchors[:, None], nn_idx]; dc = D_c[anchors[:, None], nn_idx]
            per[key]["teacher"].append(dt.mean(1).cpu().numpy()); per[key]["teacher_rel"].append((dt.mean(1) / rand_mean).cpu().numpy())
            per[key]["dense"].append(dc.mean(1).cpu().numpy())
            rec = (nn_idx[:, :, None] == teacher_nn[:, None, :]).any(-1).float().mean(1); per[key]["recall"].append(rec.cpu().numpy())
            per[key]["ssdd"].append(((dt < q25[:, None]) & (dc > med_c[:, None])).float().mean(1).cpu().numpy())
            per[key]["take"].append(take[idx][anchors.cpu().numpy()])
            return dt, dc

        record("teacher", teacher_nn)
        rnd = torch.stack([torch.nonzero(allowed[a_]).squeeze(1)[torch.randperm(int(allowed[a_].sum()), device=dev, generator=g)[:K]] for a_ in anchors]); record("random", rnd)
        nn_of = {}
        for key, X in spaces.items():
            D = torch.where(allowed, torch.cdist(X[ii], X[ii]), inf); nn_of[key] = D[anchors].topk(K, largest=False).indices; record(key, nn_of[key])
        if "A3:z" in nn_of and len(examples) < 4:
            # an anchor whose z neighbours are structurally close but densely different
            dt_z = D_t[anchors[:, None], nn_of["A3:z"]]; dc_z = D_c[anchors[:, None], nn_of["A3:z"]]
            score = (dc_z.mean(1) / med_c) - dt_z.mean(1) / rand_mean
            a_ = anchors[score.argmax()]; e = dict(mesh=str(m), anchor_n=int(n[idx][a_]), anchor_t=int(t[idx][a_]), anchor_C=C[idx][a_].cpu().numpy())
            for key in ("A3:z", "A3:r", "C"):
                nb = nn_of[key][score.argmax(), :3]; e[f"{key}_C"] = C[ii][nb].cpu().numpy(); e[f"{key}_dt"] = D_t[a_, nb].cpu().numpy(); e[f"{key}_dc"] = D_c[a_, nb].cpu().numpy()
            nb = teacher_nn[score.argmax(), :3]; e["teacher_C"] = C[ii][nb].cpu().numpy(); e["teacher_dt"] = D_t[a_, nb].cpu().numpy(); e["teacher_dc"] = D_c[a_, nb].cpu().numpy()
            x, _, _, valid = data.geometry(n[idx][a_:a_ + 1], t[idx][a_:a_ + 1]); e["x"] = x[0].cpu().numpy(); e["valid"] = valid[0].cpu().numpy()
            examples[f"ex{len(examples)}"] = e
    rows = []
    for key, d in per.items():
        if not d["take"]:
            continue
        tk = np.concatenate(d["take"]); row = dict(dataset=ds, space=key, n_anchors=len(tk), k=K)
        for mname in ("teacher", "teacher_rel", "dense", "recall", "ssdd"):
            v = np.concatenate(d[mname]); row[mname], row[mname + "_lo"], row[mname + "_hi"] = S.cluster_bootstrap(v, tk, n_boot=S.N_BOOT)
        rows.append(row); log.info("%s %s: teacher %.3f (rel %.3f) dense %.3f recall@%d %.3f ssdd %.3f", ds, key, row["teacher"], row["teacher_rel"], row["dense"], K, row["recall"], row["ssdd"])
    pd.DataFrame(rows).to_csv(S.ds_out(ds) / "neighborhood_metrics.csv", index=False)
    flat = {}
    for ek, e in examples.items():
        for k, v in e.items():
            flat[f"{ek}__{k}"] = v
    np.savez_compressed(S.ds_out(ds) / "neighbor_examples.npz", **flat)


if __name__ == "__main__":
    main()
