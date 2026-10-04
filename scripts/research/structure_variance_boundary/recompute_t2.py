#!/usr/bin/env python
"""Recompute the T2 (future amount) metric of every structured checkpoint with a constant normaliser.
The original per-frame relative L1 divided by the future total amount, which is ~0 on frames without contact and made the
metric explode. New definition (written into the same metrics files, keys T2_rel_l1 / T2_rel_l1_persist / T2_rel_l1_mean):
    T2 = sum_k |m^_k - m_k(t+h)| / M_bar,   M_bar = mean over TRAIN frames of sum_k m_k (the typical total contact amount)
    CUDA_VISIBLE_DEVICES=<gpu> python recompute_t2.py --dataset taco
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

import sv_common as S
from sv_data import Data
import train_temporal as TT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recompute_t2")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=S.DATASETS); a = ap.parse_args()
    ds = a.dataset; dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = Data(ds, dev, need_hand100=True)
    tr = D.idx["train"]
    m_tr = D.tg["m"][tr].reshape(-1, 6).astype(np.float64)
    M_bar = float(m_tr.sum(1).mean()); m_bar = torch.tensor(m_tr.mean(0), dtype=torch.float32, device=dev)
    log.info("%s: train-mean total amount M_bar = %.4f (per part %s)", ds, M_bar, np.round(m_tr.mean(0), 4))
    n_te, t_te = D.pairs("test"); N = len(n_te); nh = len(S.HORIZONS); bs = 4096
    out = S.ds_out(ds) / "temporal"
    for ck in sorted((out / "ckpt").glob("*_structured_seed*.pt")):
        st = torch.load(ck, map_location=dev)
        rep = st["rep"]; TT.rep_g = rep
        net = TT.RepPredictor(S.REP_DIM[rep] * S.HIST, D.d_static(), D.d_traj(), sum(TT.HEADS["structured"].values()), nh).to(dev)
        net.load_state_dict(st["state"]); net.eval()
        mp = out / "metrics" / f"{ck.stem}.npz"
        z = dict(np.load(mp, allow_pickle=True))
        assert np.array_equal(z["example"], n_te.cpu().numpy()) and np.array_equal(z["t"], t_te.cpu().numpy()), ck.stem
        new = {k: np.full((N, nh), np.nan, np.float32) for k in ("T2_rel_l1", "T2_rel_l1_persist", "T2_rel_l1_mean")}
        for i in range(0, N, bs):
            n, t = n_te[i:i + bs], t_te[i:i + bs]
            outs = TT.split_out(net(D.window(rep, n, t), *D.cond(n, t)), "structured"); sl = slice(i, i + len(n))
            for j, h in enumerate(S.HORIZONS):
                tg, valid = D.targets(n, t, h, ["m"]); now, _ = D.targets(n, t, 0, ["m"]); m_t = tg["m"]
                m_hat = D.destandardise("m", outs["m"][:, j]).clamp(min=0)
                new["T2_rel_l1"][sl, j] = ((m_hat - m_t).abs().sum(1) / M_bar).cpu().numpy()
                new["T2_rel_l1_persist"][sl, j] = ((now["m"] - m_t).abs().sum(1) / M_bar).cpu().numpy()
                new["T2_rel_l1_mean"][sl, j] = ((m_bar[None] - m_t).abs().sum(1) / M_bar).cpu().numpy()
        z.update(new); z["T2_normaliser"] = np.array(M_bar, np.float32)
        np.savez_compressed(mp, **z)
        v = z["valid"] > 0
        log.info("  %s: T2 %s persist %s mean %s", ck.stem, np.round([np.nanmean(new["T2_rel_l1"][:, j][v[:, j]]) for j in range(nh)], 3),
                 np.round([np.nanmean(new["T2_rel_l1_persist"][:, j][v[:, j]]) for j in range(nh)], 3), np.round([np.nanmean(new["T2_rel_l1_mean"][:, j][v[:, j]]) for j in range(nh)], 3))
    log.info("done %s", ds)


if __name__ == "__main__":
    main()
