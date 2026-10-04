#!/usr/bin/env python
"""The frozen initial sampler p(s_0 | G) (hier_contact_gen sampler_G, DDPM / DDIM 100 steps, eta = 0): K = 10 draws per test
(and validation) example with stored seeds, shared by every temporal model of protocol B.
    CUDA_VISIBLE_DEVICES=4 python sample_s0.py --dataset taco
Seeds: generator seed 100000 + k for sample k (the hierarchical evaluation's seeds for sampler_G, fold 0), batches of 256
examples in index order, so the test draws reproduce the hierarchical study's S0_G up to float16 storage (checked).
Writes <ds>/s0_samples.npz: s0 (B, K, 512) raw float32 per part, example indices, seeds, sampler checkpoint metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys

import numpy as np
import torch

import sat_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--k", type=int, default=S.K_SAMPLES)
    ap.add_argument("--ddim-steps", type=int, default=100)
    a = ap.parse_args()
    ds = a.dataset
    os.environ["DC_DATASET"] = ds
    hier = str(S.REPO / "scripts/research/hier_contact_gen")
    if sys.path[0] != hier:
        sys.path.insert(0, hier)
    import hc_common as H  # noqa: E402
    from data import FoldData  # noqa: E402  (hier_contact_gen/data.py)
    from train import build  # noqa: E402  (hier_contact_gen/train.py)
    dev = torch.device("cuda")
    ck_path = H.CKPT / "fixed0" / "sampler_G.pt"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    fd = FoldData("fixed", 0, dev, stats=ck["stats"])
    net = build("sampler_G", fd, ck.get("width"), ck.get("dropout", 0.0)).to(dev); net.load_state_dict(ck["state"]); net.eval()
    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()[:16]
    out = {}
    for part in ("test", "val"):
        te = fd.idx[part]; n_all = torch.from_numpy(te).to(dev); B = len(te)
        samples, seeds = [], []
        for k in range(a.k):
            seed = 100000 + k; seeds.append(seed)
            g = torch.Generator(device=dev); g.manual_seed(seed)
            outs = []
            for i in range(0, B, 256):
                n = n_all[i:i + 256]
                outs.append(net.sample(fd.S[n], None, n_steps=a.ddim_steps, generator=g).cpu())
            samples.append(torch.cat(outs))
        s0 = fd.to_raw(torch.stack(samples, 1))                                   # (B, K, 512) raw
        out[f"s0_{part}"] = s0.astype(np.float32); out[f"example_{part}"] = te; out[f"seeds_{part}"] = np.array(seeds)
        log.info("%s %s: %d examples x %d samples, raw range [%.3f, %.3f], mean mass %.3f (GT %.3f)", ds, part, B, a.k, s0.min(), s0.max(),
                 np.clip(s0, 0, None).sum(-1).mean(), fd.X_raw[te, 0].sum(-1).mean())
    # check against the hierarchical study's stored test draws (float16)
    pz = np.load(H.OUT / "eval" / "fixed0" / "preds.npz")
    ref = pz["S0_G"].astype(np.float32); same = np.array_equal(pz["example"], out["example_test"])
    diff = float(np.abs(ref - out["s0_test"]).max()) if same and ref.shape == out["s0_test"].shape else np.nan
    log.info("match with hier preds.npz S0_G: same examples %s, max |diff| %.5f", same, diff)
    S.s0_path(ds).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(S.s0_path(ds), **out, ddim_steps=a.ddim_steps, sampler_ckpt=str(ck_path), sampler_sha256_16=sha,
                        sampler_best_step=ck["best_step"], hier_match_max_abs=diff)
    S.write_json(S.ds_out(ds) / "s0_sampler.json", dict(dataset=ds, checkpoint=str(ck_path), sha256_16=sha, best_step=int(ck["best_step"]),
                                                       converged=bool(ck["converged"]), ddim_steps=a.ddim_steps, k=a.k,
                                                       seeds_test=out["seeds_test"].tolist(), hier_match_max_abs=diff,
                                                       note="frozen; the hierarchical study's sampler_G, same seeds / batching as its evaluation"))


if __name__ == "__main__":
    main()
