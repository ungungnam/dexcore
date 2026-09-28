#!/usr/bin/env python
"""(1) Training sequences scored under TEST-STYLE windowing (starts 8, 72, 136, ...): same windowing as test_1,
different take identity.  (2) Contact model t=0 own/swapped-input probe at best/ep100/ep200 (does it also
turn into a conditioning-keyed lookup?)."""
import sys as _sys
_REPO = "/home/uhnam/workspace/dexcore"
for _p in (_REPO, _REPO + "/third_party/BimArt"):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import logging; from pathlib import Path
import numpy as np, torch, yaml, pandas as pd
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset
from scripts.eval_bimart_taco import load_model, sample
from src.analysis.bimart.datasets import TacoMotionDataset, TacoContactDataset
from utils import data_util
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("g5c")
OUT = Path("/result/uhnam/dexcore/bimart_taco/contact_probe/gen/verify")
KP_DIM = 600
def kp(a, mean, std): v = (a * std + mean)[..., :KP_DIM]; return v.reshape(*v.shape[:-1], 200, 3)
def err(a, b, mean, std): return torch.linalg.norm(kp(a, mean, std) - kp(b, mean, std), dim=-1).mean(dim=(-1, -2)) * 1000
dev = "cuda"; E = "/result/uhnam/dexcore/bimart_taco/experiments/"
mcfg = yaml.safe_load(open(_REPO + "/configs/bimart_taco/train_motion_config.yaml")); ccfg = yaml.safe_load(open(_REPO + "/configs/bimart_taco/train_contact_config.yaml"))
T = 50; mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], dev); cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], dev)
sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="sample")
mean, std = mstat["action"]["mean"], mstat["action"]["std"]; c_std = cstat["action"]["std"]
rows = []
# ---- (1) train sequences, test-style windows ----
mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split="train", pred_horizon=64, base_frame=8)
teststyle = np.array([i for i, w in enumerate(mds.windows) if (w["start"] - 8) % 64 == 0])
log.info("train windows total %d; test-style (start = 8 mod 64) %d", len(mds), len(teststyle))
pick = np.random.default_rng(0).choice(teststyle, 128, replace=False)
for tag, fname in (("best", "model_best.pth"), ("ep200", "model_epoch200.pth")):
    mm_, mname, mep, _ = load_model("motion", E + "motion_model_taco_20260923_151845", mcfg, dev, prefer=fname, use_ema=True)
    acc = {"sampled": [], "floor_t0": []}
    for s in range(0, len(pick), 32):
        idx = pick[s:s + 32]; mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
        mn = data_util.preprocess_batch({k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, dev); x = mn["action"]; cond = mn["obs"]
        xs = sample(mm_, x.shape, sched, dev, cond, "motion", T, seed=0); acc["sampled"].append(err(xs, x, mean, std).cpu())
        g = torch.Generator(device=dev).manual_seed(999); noise = torch.randn(x.shape, device=dev, generator=g); ts = torch.zeros(x.shape[0], device=dev, dtype=torch.long)
        with torch.no_grad(): pr = mm_(sched.add_noise(x, noise, ts), ts, obj_feat=cond["object"], contact_cond=cond["contact_points"], contact_on_prob=1.0, global_states=cond["global_states"])
        acc["floor_t0"].append(err(pr, x, mean, std).cpu())
    r = {"probe": "train_seqs_teststyle_windows", "split": "train", "ckpt": tag, "epoch": mep, "n": len(pick), **{k: float(torch.cat(v).mean()) for k, v in acc.items()}}
    rows.append(r); log.info("%s", r); del mm_; torch.cuda.empty_cache()
# ---- (2) contact model t=0 own / swapped ----
for split in ("train", "test_1"):
    cds = TacoContactDataset(root=ccfg["base_dir"], split=split, pred_horizon=64, base_frame=8)
    pick = np.random.default_rng(0).choice(len(cds), min(128, len(cds)), replace=False)
    for tag, fname in (("best", "model_best.pth"), ("ep100", "model_epoch100.pth"), ("ep200", "model_epoch200.pth")):
        cm, cname, cep, _ = load_model("contact", E + "contact_model_taco_20260923_151838", ccfg, dev, prefer=fname)
        acc = {k: [] for k in ("t0_own", "t0_swapped_vs_cond", "t0_swapped_vs_input", "cond_vs_input")}
        for s in range(0, len(pick), 32):
            idx = pick[s:s + 32]; cb = next(iter(DataLoader(Subset(cds, idx), batch_size=len(idx))))
            cn = data_util.preprocess_contact_batch({k: v for k, v in cb.items() if k != "aux"}, dev, cstat, True); x = cn["action"]; cond = cn["obs"]; B = x.shape[0]
            xp = x[torch.roll(torch.arange(B, device=dev), 1)]
            g = torch.Generator(device=dev).manual_seed(999); noise = torch.randn(x.shape, device=dev, generator=g); ts = torch.zeros(B, device=dev, dtype=torch.long)
            def fwd(xt):
                with torch.no_grad(): return cm(xt, ts, obj_feat=cond["obj_feat"], global_cond={"curr_global_states": cond["curr_global_states"]})
            own = fwd(sched.add_noise(x, noise, ts)); sw = fwd(sched.add_noise(xp, noise, ts))
            e = lambda a, b: (((a - b) * c_std).abs().mean(dim=(-1, -2)) * 1000).cpu()
            acc["t0_own"].append(e(own, x)); acc["t0_swapped_vs_cond"].append(e(sw, x)); acc["t0_swapped_vs_input"].append(e(sw, xp)); acc["cond_vs_input"].append(e(x, xp))
        r = {"probe": "contact_t0_lookup", "split": split, "ckpt": tag, "epoch": cep, "n": len(pick), **{k: float(torch.cat(v).mean()) for k, v in acc.items()}}
        rows.append(r); log.info("%s", r); del cm; torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv(OUT / "g5_windowing_and_contact_probe.csv", index=False); print(df.round(2).to_string(index=False))
