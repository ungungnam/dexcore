#!/usr/bin/env python
"""Diagnostic for sanity check 18: is the frame-to-frame change of the diffusion samples a property of the model or an artefact of
the sampler?   CUDA_VISIBLE_DEVICES=4 python diag_jitter.py --dataset taco --name S0_seed2
Measures, on the first 96 test sequences: the jitter (mean_t ||C_{t+1} - C_t||, raw units) of (a) the GT, (b) DDIM samples with
50 / 100 / 250 / 1000 steps (eta = 0), (c) the x0 estimate of the EMA model from the GT residual noised to step k in
{20, 50, 100, 200, 500} (one forward pass: how smooth is the denoiser's own prediction?), (d) a 3-frame moving average of the
100-step samples (post-hoc smoothing, reference only). Writes <ds>/diag_jitter_<name>.json."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import sat_common as S
from generate import load_model
from sat_data import SeqData


def jitter(C):
    return float((C[:, 1:] - C[:, :-1]).norm(dim=-1).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--name", required=True); ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args(); dev = torch.device("cuda")
    ck = torch.load(S.ckpt_path(a.dataset, a.name), map_location="cpu", weights_only=False)
    data = SeqData(a.dataset, dev, need_masks=False, stats=ck["stats"]); net, ck = load_model(a.dataset, a.name, data, dev)
    n = torch.from_numpy(data.idx["test"][:a.n]).to(dev); s0 = data.C[n, 0]; s0_in, tau, s = data.s0_input(s0), data.tau(n), data.fd.S[n]
    out = {"gt": jitter(data.C[n]), "persistence": 0.0, "E_C": {}}
    r_gt = data.residual(n)
    with torch.no_grad():
        for steps in (50, 100, 250, 1000):
            g = torch.Generator(device=dev); g.manual_seed(1)
            r = net.sample(s0_in, tau, s, n_steps=steps, generator=g); C = torch.cat([s0[:, None], data.to_raw(r, s0)], 1)
            out[f"ddim_{steps}"] = jitter(C); out["E_C"][f"ddim_{steps}"] = float((C[:, 1:] - data.C[n, 1:]).norm(dim=-1).mean())
            if steps == 100:
                Cs = C.clone(); Cs[:, 1:-1] = (C[:, :-2] + C[:, 1:-1] + C[:, 2:]) / 3
                out["ddim_100_moving_avg3"] = jitter(Cs); out["E_C"]["ddim_100_moving_avg3"] = float((Cs[:, 1:] - data.C[n, 1:]).norm(dim=-1).mean())
        for k in (20, 50, 100, 200, 500):
            kk = torch.full((len(n),), k, device=dev); noise = torch.randn_like(r_gt); ab = net.alpha_bar[kk][:, None, None]
            x_k = ab.sqrt() * r_gt + (1 - ab).sqrt() * noise
            v, _ = net.v_pred(x_k, kk, s0_in, tau, s); x0 = ab.sqrt() * x_k - (1 - ab).sqrt() * v
            C = torch.cat([s0[:, None], data.to_raw(x0, s0)], 1)
            out[f"x0_from_k{k}"] = jitter(C); out["E_C"][f"x0_from_k{k}"] = float((C[:, 1:] - data.C[n, 1:]).norm(dim=-1).mean())
            out[f"x0_from_k{k}_resid_jitter_z"] = float((x0[:, 1:] - x0[:, :-1]).norm(dim=-1).mean()); out["gt_resid_jitter_z"] = float((r_gt[:, 1:] - r_gt[:, :-1]).norm(dim=-1).mean())
    print(json.dumps(out, indent=1))
    S.write_json(S.ds_out(a.dataset) / f"diag_jitter_{a.name}.json", out)


if __name__ == "__main__":
    main()
