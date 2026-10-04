#!/usr/bin/env python
"""Evaluation question 2 — where does the structural / functional information live?
    CUDA_VISIBLE_DEVICES=4 python eval_probes.py --dataset taco
Frozen codes of the unique frames (train / val / test): h (A0), z (A1), z / r / [z, r] (A2, A3); references: the dense map C~
(512), its PCA-64 projection (the best linear 64-D dense code) and the trivial train-mean predictor. Probes: linear and a
2-layer MLP (hidden 256), one network with four heads (participation BCE, z-scored amount MSE, masked z-scored centroid /
normal MSE over the active parts, z-scored wrench MSE), Adam 1e-3, batch 1024, early-stopped on the validation loss.
Test metrics (take-cluster bootstrap): participation macro F1 / macro AUPRC / Hamming; amount L1 normalised by the train-mean
total amount and R^2; centroid distance (units of l) and normal angle (degrees) over the GT-active parts; wrench relative L1,
cosine and explained variance. Retention = (probe - trivial) / (C probe - trivial) per quantity (primary metrics: AUPRC,
amount R^2, centroid, normal, wrench EV), averaged = structure retention.
Writes <ds>/probe_metrics.csv.
"""
from __future__ import annotations

import argparse
import copy
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

import cf_common as S
from encode import load_data
from r2_ops import angle_deg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probes")
D_OUT = 6 + 6 + 36 + S.N_DIR


def frame_rows(data, lat, split):
    n, t = data.split_frames(split); n, t = n.cpu().numpy(), t.cpu().numpy()
    pos = {int(v): i for i, v in enumerate(lat["n"])}
    return np.array([pos[int(v)] for v in n]), t


class Targets:
    def __init__(self, data, split):
        n, t = data.split_frames(split); st = data.tstats
        self.a, self.m, self.p, self.nrm, self.q = data.a[n, t], data.m[n, t], data.p[n, t], data.nrm[n, t], data.q[n, t]
        self.m_z = (self.m - st["m"][0]) / st["m"][1]
        self.g_z = torch.cat([((self.p - st["p"][0]) / st["p"][1]).flatten(-2), ((self.nrm - st["nrm"][0]) / st["nrm"][1]).flatten(-2)], -1)
        self.g_mask = self.a.repeat_interleave(3, -1).repeat(1, 2)
        self.q_z = (self.q - st["q"][0]) / st["q"][1]
        self.take = data.frames[split]["take"]


def probe_loss(out, T, idx):
    la = F.binary_cross_entropy_with_logits(out[:, :6], T.a[idx]); lm = F.mse_loss(out[:, 6:12], T.m_z[idx])
    msk = T.g_mask[idx]; lg = (((out[:, 12:48] - T.g_z[idx]) ** 2) * msk).sum() / msk.sum().clamp(min=1); lq = F.mse_loss(out[:, 48:], T.q_z[idx])
    return la + lm + lg + lq


def fit_probe(X_tr, T_tr, X_va, T_va, kind, cfg=None, seed=0):
    cfg = cfg or (dict(S.PROBE, max_epochs=2) if os.environ.get("CF_SMOKE") else S.PROBE)
    torch.manual_seed(seed); d = X_tr.shape[1]
    net = nn.Linear(d, D_OUT) if kind == "linear" else nn.Sequential(nn.Linear(d, cfg["hidden"]), nn.SiLU(), nn.Linear(cfg["hidden"], D_OUT))
    net = net.to(X_tr.device); opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, bad = float("inf"), None, 0; g = torch.Generator(device=X_tr.device); g.manual_seed(seed)
    for ep in range(cfg["max_epochs"]):
        net.train(); perm = torch.randperm(len(X_tr), device=X_tr.device, generator=g)
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]; loss = probe_loss(net(X_tr[idx]), T_tr, idx)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([float(probe_loss(net(X_va[i:i + 4096]), T_va, torch.arange(i, min(i + 4096, len(X_va)), device=X_va.device))) for i in range(0, len(X_va), 4096)]))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best, ep + 1


@torch.no_grad()
def metrics(pred, T, data, take):
    """pred (N, 124) probe output (logits / z-scored) -> dict of metrics with bootstrap CIs."""
    st = data.tstats; a_hat = torch.sigmoid(pred[:, :6]); m_hat = pred[:, 6:12] * st["m"][1] + st["m"][0]
    p_hat = (pred[:, 12:30].view(-1, 6, 3) * st["p"][1] + st["p"][0]); n_hat = pred[:, 30:48].view(-1, 6, 3) * st["nrm"][1] + st["nrm"][0]
    q_hat = pred[:, 48:] * st["q"][1] + st["q"][0]
    a, act = T.a.cpu().numpy(), T.a > 0
    r = {}
    r["part_f1"] = float(np.mean([f1_score(a[:, k], (a_hat[:, k] > 0.5).cpu().numpy(), zero_division=0) for k in range(6)]))
    r["part_auprc"] = float(np.mean([average_precision_score(a[:, k], a_hat[:, k].cpu().numpy()) if a[:, k].any() else np.nan for k in range(6)]))
    ham = ((a_hat > 0.5).float() != T.a).float().mean(-1).cpu().numpy()
    r["part_hamming"], r["part_hamming_lo"], r["part_hamming_hi"] = S.cluster_bootstrap(ham, take, n_boot=S.N_BOOT)
    m_bar = float(data.tstats["m"][0].sum()) if "m_bar" not in data.tstats else data.tstats["m_bar"]
    l1 = ((m_hat - T.m).abs().sum(-1) / m_bar).cpu().numpy()
    r["amount_l1"], r["amount_l1_lo"], r["amount_l1_hi"] = S.cluster_bootstrap(l1, take, n_boot=S.N_BOOT)
    sse = ((m_hat - T.m) ** 2).sum(0); sst = ((T.m - T.m.mean(0)) ** 2).sum(0); r["amount_r2"] = float((1 - sse / sst.clamp(min=1e-9)).mean())
    cd = (p_hat - T.p).norm(dim=-1); cd = (cd * T.a).sum(-1) / T.a.sum(-1).clamp(min=1); ok = (T.a.sum(-1) > 0).cpu().numpy()
    r["centroid"], r["centroid_lo"], r["centroid_hi"] = S.cluster_bootstrap(cd.cpu().numpy()[ok], take[ok], n_boot=S.N_BOOT)
    ang = angle_deg(n_hat, T.nrm); ang = (ang * T.a).sum(-1) / T.a.sum(-1).clamp(min=1)
    r["normal"], r["normal_lo"], r["normal_hi"] = S.cluster_bootstrap(ang.cpu().numpy()[ok], take[ok], n_boot=S.N_BOOT)
    qs = T.q.sum(-1); okq = (qs > 0).cpu().numpy()
    rel = ((q_hat - T.q).abs().sum(-1) / qs.clamp(min=1e-9)).cpu().numpy()
    r["wrench_rel_l1"], r["wrench_rel_l1_lo"], r["wrench_rel_l1_hi"] = S.cluster_bootstrap(rel[okq], take[okq], n_boot=S.N_BOOT)
    cos = F.cosine_similarity(q_hat, T.q, dim=-1).cpu().numpy()
    r["wrench_cos"], r["wrench_cos_lo"], r["wrench_cos_hi"] = S.cluster_bootstrap(cos[okq], take[okq], n_boot=S.N_BOOT)
    num = ((q_hat - T.q) ** 2).sum(-1).cpu().numpy(); den = ((T.q - T.q.mean(0)) ** 2).sum(-1).cpu().numpy()
    ev, lo, hi = S.ratio_cluster_bootstrap(num, den, take, n_boot=S.N_BOOT); r["wrench_ev"], r["wrench_ev_lo"], r["wrench_ev_hi"] = 1 - ev, 1 - hi, 1 - lo
    return r


def standardise(X_tr, *others):
    mu, sd = X_tr.mean(0), X_tr.std(0).clamp(min=1e-6)
    return [(X - mu) / sd for X in (X_tr,) + others]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device(os.environ.get("CF_DEVICE", "cuda"))
    data = load_data(ds, dev, S.run_name("A3", a.seed))
    T = {s: Targets(data, s) for s in ("train", "val", "test")}
    inputs = {}
    Cn = {s: data.Cn[data.split_frames(s)[0], data.split_frames(s)[1]] for s in T}
    inputs["C"] = ("dense", Cn)
    mean_tr = Cn["train"].mean(0); _, _, Vh = torch.linalg.svd(Cn["train"] - mean_tr, full_matrices=False); P = Vh[:64]
    inputs["C_pca64"] = ("dense", {s: (Cn[s] - mean_tr) @ P.T for s in T})
    for model in S.ALL_MODELS:
        name = S.run_name(model, a.seed)
        if not S.latents_path(ds, name, "test").exists():
            log.warning("missing %s", name); continue
        lat = {s: np.load(S.latents_path(ds, name, s)) for s in T}
        rows_ = {s: frame_rows(data, lat[s], s) for s in T}
        f = lambda s, k: torch.from_numpy(lat[s][k][rows_[s]].astype(np.float32)).to(dev)
        if model == "A0":
            inputs["A0:h"] = (model, {s: f(s, "z") for s in T})
        else:
            inputs[f"{model}:z"] = (model, {s: f(s, "z") for s in T})
            if "r" in lat["test"]:
                inputs[f"{model}:r"] = (model, {s: f(s, "r") for s in T}); inputs[f"{model}:zr"] = (model, {s: torch.cat([f(s, "z"), f(s, "r")], -1) for s in T})
    rows = []
    # trivial predictor: train means (participation rate, mean amount, active-mean centroid / normal, mean wrench) as z-scored outputs
    st = data.tstats
    triv = torch.cat([torch.logit(st["a"][0].clamp(1e-4, 1 - 1e-4)), torch.zeros(6 + 36 + S.N_DIR, device=dev)])[None].expand(len(T["test"].a), -1)
    rows.append(dict(dataset=ds, model="trivial", input="trivial", probe="constant", d_in=0, **metrics(triv, T["test"], data, T["test"].take)))
    for key, (model, X) in inputs.items():
        Xtr, Xva, Xte = standardise(X["train"], X["val"], X["test"])
        for kind in ("linear", "mlp"):
            net, vl, ep = fit_probe(Xtr, T["train"], Xva, T["val"], kind)
            with torch.no_grad():
                pred = torch.cat([net(Xte[i:i + 4096]) for i in range(0, len(Xte), 4096)])
            r = metrics(pred, T["test"], data, T["test"].take)
            rows.append(dict(dataset=ds, model=model, input=key, probe=kind, d_in=Xtr.shape[1], val_loss=vl, epochs=ep, **r))
            log.info("%s %s %s: F1 %.3f AUPRC %.3f amount R2 %.3f centroid %.3f normal %.1f wrench EV %.3f cos %.3f", ds, key, kind, r["part_f1"], r["part_auprc"], r["amount_r2"], r["centroid"], r["normal"], r["wrench_ev"], r["wrench_cos"])
    df = pd.DataFrame(rows)
    # retention relative to the dense-C probe of the same kind (trivial = 0, C probe = 1)
    prim = [("part_auprc", 1), ("amount_r2", 1), ("centroid", -1), ("normal", -1), ("wrench_ev", 1)]
    tv = df[df.model == "trivial"].iloc[0]
    for kind in ("linear", "mlp"):
        ref = df[(df.input == "C") & (df.probe == kind)].iloc[0]
        for i, row in df[df.probe == kind].iterrows():
            rets = []
            for k, sgn in prim:
                den = ref[k] - tv[k]; rets.append((row[k] - tv[k]) / den if abs(den) > 1e-9 else np.nan)
                df.loc[i, f"ret_{k}"] = rets[-1]
            df.loc[i, "structure_retention"] = float(np.nanmean(rets))
    df.to_csv(S.ds_out(ds) / "probe_metrics.csv", index=False)
    print(df[["input", "probe", "part_auprc", "amount_r2", "centroid", "normal", "wrench_ev", "structure_retention"]].to_string())


if __name__ == "__main__":
    main()
