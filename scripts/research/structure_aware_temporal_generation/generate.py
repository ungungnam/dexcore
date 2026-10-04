#!/usr/bin/env python
"""Generate the test (or validation) trajectories of one trained model.
    CUDA_VISIBLE_DEVICES=4 python generate.py --dataset taco --name D1_seed0 --protocol A
Protocol A  s_0 = the GT frame 0 of every example: deterministic models one trajectory, diffusion models K = 10 trajectories
            from the same (s_0, G, tau) with the noise generator seeded by (sample index k, training seed) only — identical
            initial noise for S0 / S1 / S2 of the same seed (paired comparisons).
Protocol B  s_0^(k), k = 1..10, the frozen sampler's draws of <ds>/s0_samples.npz (the same for every model): ONE trajectory
            per draw for both families (the diffusion noise seeded as in A by k).
Frame 0 of every stored trajectory is the conditioning s_0 (never generated). Writes <ds>/preds/<name>_<protocol>[_val].npz:
pred (B, K, 64, 512) float16 raw contact values, example indices, the s_0 used, the noise seeds, raw-range statistics.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import sat_common as S
import sat_models as MD
from sat_data import SeqData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("generate")
NOISE_BASE = 7_000_000


def load_model(ds, name, data, dev):
    ck = torch.load(S.ckpt_path(ds, name), map_location=dev, weights_only=False)
    net = MD.build(ck["model"], data.d_traj(), data.d_static(), ck["cfg"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck


@torch.no_grad()
def run(net, ck, data, n_all, s0_raw, k_seed, ddim_steps, bs=128):
    """One trajectory per example for the given s_0 (B, 512) raw; diffusion noise seeded by k_seed."""
    out = []
    g = None
    if S.MODELS[ck["model"]]["family"] == "diff":
        g = torch.Generator(device=data.device); g.manual_seed(NOISE_BASE + 1000 * k_seed + ck["seed"])
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; s0 = s0_raw[i:i + bs]
        s0_in, tau, s = data.s0_input(s0), data.tau(n), data.fd.S[n]
        if g is None:
            r, _ = net(s0_in, tau, s)
        else:
            r = net.sample(s0_in, tau, s, n_steps=ddim_steps, generator=g)
        out.append(torch.cat([s0[:, None], data.to_raw(r, s0)], 1).cpu())
    return torch.cat(out)                                                       # (B, 64, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--name", required=True)
    ap.add_argument("--protocol", required=True, choices=S.PROTOCOLS); ap.add_argument("--part", default="test", choices=["test", "val"])
    ap.add_argument("--k", type=int, default=S.K_SAMPLES); ap.add_argument("--ddim-steps", type=int, default=100)
    a = ap.parse_args()
    ds = a.dataset; t0 = time.time()
    dev = torch.device("cuda")
    ck = torch.load(S.ckpt_path(ds, a.name), map_location="cpu", weights_only=False)
    data = SeqData(ds, dev, need_masks=False, stats=ck["stats"])
    net, ck = load_model(ds, a.name, data, dev)
    fam = S.MODELS[ck["model"]]["family"]
    te = data.idx[a.part]; n_all = torch.from_numpy(te).to(dev)
    preds, s0_used, seeds = [], [], []
    if a.protocol == "A":
        s0 = data.C[n_all, 0]
        K = a.k if fam == "diff" else 1
        for k in range(K):
            preds.append(run(net, ck, data, n_all, s0, k, a.ddim_steps)); s0_used.append(s0.cpu()); seeds.append(NOISE_BASE + 1000 * k + ck["seed"] if fam == "diff" else -1)
    else:
        z = np.load(S.s0_path(ds))
        assert np.array_equal(z[f"example_{a.part}"], te), "s0 samples do not match the split"
        s0_all = torch.from_numpy(z[f"s0_{a.part}"]).to(dev)                    # (B, K, 512) raw
        for k in range(a.k):
            s0 = s0_all[:, k]
            preds.append(run(net, ck, data, n_all, s0, k, a.ddim_steps)); s0_used.append(s0.cpu()); seeds.append(NOISE_BASE + 1000 * k + ck["seed"] if fam == "diff" else -1)
    P = torch.stack(preds, 1).numpy()                                           # (B, K, 64, 512)
    gt = data.C[n_all].cpu().numpy()
    err = np.linalg.norm(P[:, :, 1:] - gt[:, None, 1:], axis=-1).mean(-1)      # (B, K)
    frame0_ok = bool(np.abs(P[:, :, 0] - torch.stack(s0_used, 1).numpy()).max() == 0)
    out = S.preds_path(ds, a.name, a.protocol + ("" if a.part == "test" else "_val")); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pred=P.astype(np.float16), example=te, s0=torch.stack(s0_used, 1).numpy().astype(np.float32), noise_seeds=np.array(seeds),
                        protocol=a.protocol, part=a.part, model=ck["model"], seed=ck["seed"], ddim_steps=a.ddim_steps, raw_min=float(P.min()), raw_max=float(P.max()),
                        frac_below_0=float((P < 0).mean()), frac_above_1=float((P > 1).mean()), frame0_equals_s0=frame0_ok)
    log.info("%s %s %s %s: %d examples x %d samples, E_C(1..63) K1 %.3f mean %.3f best %.3f, raw range [%.3f, %.3f], frame0 == s0 %s (%.0f s) -> %s",
             ds, a.name, a.protocol, a.part, len(te), P.shape[1], err[:, 0].mean(), err.mean(), err.min(1).mean(), P.min(), P.max(), frame0_ok, time.time() - t0, out)


if __name__ == "__main__":
    main()
