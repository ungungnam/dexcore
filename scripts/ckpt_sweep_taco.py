#!/usr/bin/env python
"""Overfitting dynamics: the same windows scored under each saved checkpoint.

For the motion model (best = epoch 19, epoch 100, epoch 200) on the RECORDED contact map: the sampled
hand error, the t = 0 reconstruction floor (a nearly clean recorded trajectory, one forward pass) and
the t = T-1 conditional mean. For the contact model (best = epoch 75, 100, 200): the sampled map error.
Train and test_1 side by side, so the gap's growth with training is visible directly.
"""
import pathlib as _pl, sys as _sys
_HERE = str(_pl.Path(__file__).resolve().parent); _REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
for _p in (_REPO, str(_pl.Path(_REPO) / "third_party/BimArt")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml

KP_DIM = 600
def kp(a, mean, std): v = (a * std + mean)[..., :KP_DIM]; return v.reshape(*v.shape[:-1], 200, 3)
def err(a, b, mean, std): return (torch.linalg.norm(kp(a, mean, std) - kp(b, mean, std), dim=-1).mean(dim=(-1, -2)) * 1000)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, default=128); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="/result/uhnam/dexcore/bimart_taco/contact_probe/ckpt_sweep")
    p.add_argument("--variant", default="", choices=("", "_fps", "_scene"),
                   help="'_fps' selects the corrected-contact-label configs and store")
    p.add_argument("--store", default=None, help="store subdirectory; defaults to train_store{variant}")
    p.add_argument("--contact-run", default="contact_model_taco_20260923_151838")
    p.add_argument("--motion-run", default="motion_model_taco_20260923_151845")
    p.add_argument("--ckpts", nargs="+", default=["model_best.pth", "model_epoch100.pth", "model_epoch200.pth"])
    a = p.parse_args()
    STORE = a.store or ("train_store_fps" if a.variant == "_fps" else "train_store")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("sweep")
    import pandas as pd
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset
    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from utils import data_util
    dev = "cuda"
    ccfg = yaml.safe_load(open(f"configs/bimart_taco/train_contact_config{a.variant}.yaml")); mcfg = yaml.safe_load(open(f"configs/bimart_taco/train_motion_config{a.variant}.yaml"))
    E = str(ccfg["base_dir"]).rstrip("/") + "/experiments/"   # the _scene variant lives under its own root
    T = ccfg["num_timesteps"]; cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], dev); mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], dev)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="sample")
    mean, std = mstat["action"]["mean"], mstat["action"]["std"]; c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]
    rows = []
    for split in ("train", "test_1"):
        mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split, pred_horizon=64, base_frame=8, store_dir=STORE)
        cds = TacoContactDataset(root=ccfg["base_dir"], split=split, pred_horizon=64, base_frame=8, store_dir=STORE)
        pick = np.random.default_rng(0).choice(len(mds), min(a.windows, len(mds)), replace=False)
        for fname in a.ckpts:
            tag = fname.replace("model_", "").replace(".pth", "")
            mm, mname, mep, _ = load_model("motion", E + a.motion_run, mcfg, dev, prefer=fname)
            cm, cname, cep, _ = load_model("contact", E + a.contact_run, ccfg, dev, prefer=fname)
            acc = {"sampled": [], "floor_t0": [], "condmean_t49": [], "cmap": []}
            for s in range(0, len(pick), a.batch_size):
                idx = pick[s:s + a.batch_size]
                mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx)))); cb = next(iter(DataLoader(Subset(cds, idx), batch_size=len(idx))))
                mn = data_util.preprocess_batch({k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, dev)
                cn = data_util.preprocess_contact_batch({k: v for k, v in cb.items() if k != "aux"}, dev, cstat, True)
                x = mn["action"]; cond = mn["obs"]
                xs = sample(mm, x.shape, sched, dev, cond, "motion", T, seed=0); acc["sampled"].append(err(xs, x, mean, std).cpu())
                g = torch.Generator(device=dev).manual_seed(999); noise = torch.randn(x.shape, device=dev, generator=g)
                for t, key in ((0, "floor_t0"), (T - 1, "condmean_t49")):
                    ts = torch.full((x.shape[0],), t, device=dev, dtype=torch.long); xt = sched.add_noise(x, noise, ts)
                    with torch.no_grad(): pr = mm(xt, ts, obj_feat=cond["object"], contact_cond=cond["contact_points"], contact_on_prob=1.0, global_states=cond["global_states"])
                    acc[key].append(err(pr, x, mean, std).cpu())
                ch = sample(cm, cn["action"].shape, sched, dev, cn["obs"], "contact", T, seed=0)
                acc["cmap"].append((((ch - cn["action"]) * c_std).abs().mean(dim=(-1, -2)) * 1000).cpu())
            r = {"split": split, "ckpt": tag, "motion_epoch": mep, "contact_epoch": cep, **{k: float(torch.cat(v).mean()) for k, v in acc.items()}}
            rows.append(r); log.info("%s %s: sampled %.1f floor %.1f condmean %.1f cmap %.1f", split, tag, r["sampled"], r["floor_t0"], r["condmean_t49"], r["cmap"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "ckpt_sweep.csv", index=False)
    hist = json.load(open(E + a.motion_run + "/loss_history.json"))
    (out / "motion_loss_history.json").write_text(json.dumps(hist))
    print("\n" + df.round(2).to_string(index=False)); print(f"\nloss history entries: {len(hist)}; first: {hist[0]}")
    print(f"-> {out}")

if __name__ == "__main__": main()
