#!/usr/bin/env python
"""Does the t=0 network read its input, or act as a conditioning-keyed lookup?  Feed window i's conditioning
with window j's (clean) trajectory as x_t at t=0.  If the output tracks the conditioning's own trajectory the
network is a lookup keyed by the conditioning (take identity); if it tracks the input, it is a denoiser."""
import sys as _sys
_REPO = "/home/uhnam/workspace/dexcore"
for _p in (_REPO, _REPO + "/third_party/BimArt"):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import logging; from pathlib import Path
import numpy as np, torch, yaml, pandas as pd
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset
from scripts.eval_bimart_taco import load_model
from src.analysis.bimart.datasets import TacoMotionDataset
from utils import data_util
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("g5b")
OUT = Path("/result/uhnam/dexcore/bimart_taco/contact_probe/gen/verify")
KP_DIM = 600; K = 100
def kp(a, mean, std): v = (a * std + mean)[..., :KP_DIM]; return v.reshape(*v.shape[:-1], 200, 3)
def err(a, b, mean, std): return torch.linalg.norm(kp(a, mean, std) - kp(b, mean, std), dim=-1).mean(dim=(-1, -2)) * 1000
dev = "cuda"; E = "/result/uhnam/dexcore/bimart_taco/experiments/"
mcfg = yaml.safe_load(open(_REPO + "/configs/bimart_taco/train_motion_config.yaml"))
T = 50; mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], dev)
sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="sample")
mean, std = mstat["action"]["mean"], mstat["action"]["std"]
prior = kp(torch.zeros(1, 1, 1200, device=dev), mean, std)[0, 0]      # training-mean keypoints (normalised 0), (200,3) metres
rows, per = [], []
for split in ("train", "test_1"):
    mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split, pred_horizon=64, base_frame=8)
    pick = np.random.default_rng(0).choice(len(mds), min(128, len(mds)), replace=False)
    for tag, fname in (("best", "model_best.pth"), ("ep100", "model_epoch100.pth"), ("ep200", "model_epoch200.pth")):
        mm_, mname, mep, _ = load_model("motion", E + "motion_model_taco_20260923_151845", mcfg, dev, prefer=fname, use_ema=True)
        acc = {k: [] for k in ("t0_own", "t0_swapped_vs_cond", "t0_swapped_vs_input", "t0_swapped_cond_vs_input", "t10_own", "t10_swapped_vs_cond", "t10_swapped_vs_input", "cos_R", "cos_L", "wt_R_mm", "wt_L_mm", "prior_R_mm", "prior_L_mm")}
        for s in range(0, len(pick), 32):
            idx = pick[s:s + 32]
            mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
            mn = data_util.preprocess_batch({k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, dev)
            x = mn["action"]; cond = mn["obs"]; B = x.shape[0]
            perm = torch.roll(torch.arange(B, device=dev), 1); xp = x[perm]          # window j's trajectory, i's conditioning
            def fwd(xt, t):
                ts = torch.full((B,), t, device=dev, dtype=torch.long)
                with torch.no_grad(): return mm_(xt, ts, obj_feat=cond["object"], contact_cond=cond["contact_points"], contact_on_prob=1.0, global_states=cond["global_states"])
            for t, key in ((0, "t0"), (10, "t10")):
                g = torch.Generator(device=dev).manual_seed(999); noise = torch.randn(x.shape, device=dev, generator=g)
                ts = torch.full((B,), t, device=dev, dtype=torch.long)
                own = fwd(sched.add_noise(x, noise, ts), t); sw = fwd(sched.add_noise(xp, noise, ts), t)
                acc[f"{key}_own"].append(err(own, x, mean, std).cpu()); acc[f"{key}_swapped_vs_cond"].append(err(sw, x, mean, std).cpu()); acc[f"{key}_swapped_vs_input"].append(err(sw, xp, mean, std).cpu())
                if t == 0:
                    acc["t0_swapped_cond_vs_input"].append(err(x, xp, mean, std).cpu())
                    d = kp(own, mean, std) - kp(x, mean, std)                        # (B,T,200,3)
                    gt = kp(x, mean, std)
                    for h, sl in (("R", slice(K, 2 * K)), ("L", slice(0, K))):
                        wt = d[:, :, sl].mean(dim=(1, 2))                           # (B,3) whole-window offset of hand
                        toprior = (prior[sl].mean(0)[None] - gt[:, :, sl].mean(dim=(1, 2)))   # gt hand centre -> training-mean hand centre
                        cos = torch.nn.functional.cosine_similarity(wt, toprior, dim=-1)
                        acc[f"cos_{h}"].append(cos.cpu()); acc[f"wt_{h}_mm"].append((wt.norm(dim=-1) * 1000).cpu()); acc[f"prior_{h}_mm"].append((toprior.norm(dim=-1) * 1000).cpu())
        r = {"split": split, "ckpt": tag, "epoch": mep, **{k: float(torch.cat(v).mean()) for k, v in acc.items()}}
        r["frac_cos_R_pos"] = float((torch.cat(acc["cos_R"]) > 0).float().mean()); r["frac_cos_L_pos"] = float((torch.cat(acc["cos_L"]) > 0).float().mean())
        rows.append(r); log.info("%s", r); del mm_; torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv(OUT / "g5_t0_lookup_probe.csv", index=False); print(df.round(2).to_string(index=False))
