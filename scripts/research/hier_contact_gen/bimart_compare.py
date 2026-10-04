#!/usr/bin/env python
"""B4: BimArt's contact stage on the SAME test sequences (TACO: the gen3 scene-scale contact model;
ARCTIC: BimArt's pretrained contact model), scored with the same metrics.

  DC_DATASET=taco   python bimart_compare.py --gpu 7 --n-samples 10
  DC_DATASET=arctic python bimart_compare.py --gpu 6 --n-samples 10

BimArt predicts, for a 64-frame object-trajectory window, the hand-surface distance (metres) at
512 BPS basis points per object part and hand (TACO: [L tool, L target, R tool, R target]; ARCTIC:
[L top, L bottom, R top, R bottom]). Each basis point gathers one mesh vertex (obj_cano_bps_inds,
per frame), so the prediction lives on 512 (TACO) or 1024 (ARCTIC) of the object's vertices. To
score it on the canonical 512-D support, both the prediction and the GT distances at those same
vertices are splatted with the study's operator RESTRICTED to those columns (rows renormalised):
  X_r(d) = W[:, inds_t] / rowsum @ exp(-d / 0.02)
Rows written (per test sequence, K samples with the study's best-of-K rule):
  bimart              X_r(BimArt) vs the full-support GT of the tables (what B0-B3 are scored on)
  bimart_vs_gtr       X_r(BimArt) vs X_r(GT): BimArt on its own support
  gt_restricted       X_r(GT) vs full GT: the representation gap of the BPS support (K=0)
  static_gtr          X_r(GT)_0 held static vs full GT (K=0), and static_gtr_vs_gtr vs X_r(GT)
Outputs: eval/fixed0/bimart_per_example.csv, bimart_preds.npz, results/bimart_comparison.{csv,md}.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:] = [p for p in sys.path if p != ""]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for p in (str(REPO), str(REPO / "third_party/BimArt")):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=7); ap.add_argument("--n-samples", type=int, default=10); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N test sequences")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import torch  # noqa: E402

import hc_common as H  # noqa: E402
from hc_common import C  # noqa: E402
from rollout_eval import metrics, KS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bimart")
DEV = torch.device("cuda")
SOFT = lambda d: np.exp(-np.clip(d, 0, None) / 0.02).astype(np.float32)


def restricted_splat(W, inds, d):
    """W (512,N) float32, inds (T,M) vertex ids, d (T,M) distances -> (T,512)."""
    out = np.zeros((len(inds), W.shape[0]), np.float32)
    s = SOFT(d)
    for t in range(len(inds)):
        Wr = W[:, inds[t]]
        rs = Wr.sum(1, keepdims=True); Wr = np.where(rs > 0, Wr / np.maximum(rs, 1e-12), 0.0)
        out[t] = Wr @ s[t]
    return out


# ----------------------------------------------------------------------------- predictions
def taco_predict(meta_te, n_samples, batch):
    """(B, K, 64, 512) distances at the BPS points of the sequence's (hand, role); (B, 64, 512) vertex ids;
    (B, 64, 512) GT distances at those vertices; W per (cat, mesh)."""
    import yaml
    from torch.utils.data import DataLoader, Subset
    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.bimart.datasets import TacoContactDataset
    from utils import data_util
    from diffusers import DDPMScheduler
    os.chdir(REPO)
    ccfg = yaml.safe_load(open(REPO / "configs/bimart_taco/train_contact_config_scene.yaml"))
    run = "/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/experiments/contact_model_taco_scene_20260927_174803"
    cm, cname, *_ = load_model("contact", run, ccfg, "cuda")
    T = ccfg["num_timesteps"]
    cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], "cuda")
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="sample")
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]
    ds = TacoContactDataset(root=ccfg["base_dir"], split="all", pred_horizon=64, base_frame=8, store_dir="train_store")
    fidx = {s: i for i, s in enumerate(ds.index.sequence_id)}
    ds.windows = [{"file_idx": fidx[s], "start": int(t0)} for s, t0 in zip(meta_te.sequence_id, meta_te.t0)]
    log.info("TACO: contact model %s, %d windows at the study's onsets, %d samples", cname, len(ds.windows), n_samples)
    files = pd.read_csv(C.SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    wi = pd.read_csv(C.STUDY / "window_index.csv").drop_duplicates("sequence_id").set_index("sequence_id")
    B = len(ds.windows)
    pred = np.zeros((B, n_samples, 64, 512), np.float32); inds_all = np.zeros((B, 64, 512), np.int64); gt_d = np.zeros((B, 64, 512), np.float32)
    for s0 in range(0, B, batch):
        idx = np.arange(s0, min(s0 + batch, B))
        b = next(iter(DataLoader(Subset(ds, idx), batch_size=len(idx))))
        nb = data_util.preprocess_contact_batch({k: v for k, v in b.items() if k != "aux"}, "cuda", cstat, True)
        for k in range(n_samples):
            c_hat = sample(cm, nb["action"].shape, sched, "cuda", nb["obs"], "contact", T, 1000 * k + s0)
            full = (c_hat * c_std + c_mean).float().cpu().numpy()                      # (b, 64, 2048)
            for j, i in enumerate(idx):
                r = meta_te.iloc[i]
                off = (0 if r.hand == "L" else 1024) + (0 if r.role == "tool" else 512)
                pred[i, k] = full[j, :, off:off + 512]
        log.info("  TACO windows %d-%d done", idx[0], idx[-1])
    for i, r in enumerate(meta_te.itertuples()):
        with np.load(C.SEQ / "sequences" / files[r.sequence_id]) as z:
            d = z[f"contact_{'left' if r.hand == 'L' else 'right'}"][r.t0:r.t0 + 64]
            inds = z["obj_cano_bps_inds"][r.t0:r.t0 + 64].astype(np.int64)
        n_tool = int(wi.loc[r.sequence_id].n_tool)
        ii = inds[:, :512] if r.role == "tool" else inds[:, 512:] - n_tool
        dd = d[:, :n_tool] if r.role == "tool" else d[:, n_tool:]
        inds_all[i] = ii; gt_d[i] = np.take_along_axis(dd, ii, 1)
    # operators
    from src.analysis.canonical import backends as BR
    sys.path.insert(0, str(REPO / "scripts/research/canonical_contact/dyn"))
    from build_full_cache import weight_matrix
    be = BR.load("normalized", str(C.BACKEND_DIR))
    W = {}
    for cat, mesh in set(zip(meta_te.category, meta_te.mesh_id)):
        W[(cat, mesh)] = weight_matrix(be, cat, mesh).astype(np.float32)
    return pred, inds_all, gt_d, W


def arctic_predict(meta_te, n_samples, batch):
    bim = REPO / "third_party/BimArt"
    os.chdir(bim)
    from utils import data_util, yaml_util
    from dataset import contact_data
    from contact_prior.contact_prior_util import load_contact_module, load_noise_scheduler
    ccfg = yaml_util.load_yaml("config_files/contact_inference.yaml")
    cats = set(meta_te.category)

    class SubContact(contact_data.ObjectContactData):
        def __get_file_paths__(self):
            super().__get_file_paths__()
            for s in ("train", "test"):
                self.data_files[s]["process_files"] = [p for p in self.data_files[s]["process_files"] if Path(p).parts[-3] in cats]
    ds = SubContact(split="test", base_dir=ccfg["base_dir"], end_frame=ccfg["end_frame"], pred_horizon=ccfg["pred_horizon"], return_aux_info=False)
    files = ds.data_files["test"]["process_files"]
    seq_of_file = {f_i: f"{Path(p).name.split('_processed_obj_features')[0]}/{Path(p).parent.name}" for f_i, p in enumerate(files)}
    fidx = {v: k for k, v in seq_of_file.items()}
    missing = [s for s in meta_te.sequence_id if s not in fidx]
    assert not missing, f"sequences not in BimArt's test files: {missing[:5]}"
    proto = {}
    for e in ds.indices:
        proto.setdefault(e["file_idx"], e)
    ds.indices = [dict(proto[fidx[s]], subsequence_idx=int(t0)) for s, t0 in zip(meta_te.sequence_id, meta_te.t0)]
    stat = data_util.load_stat_dict(ccfg["stat_dict_path"], "cuda")
    model, _, _ = load_contact_module(ccfg, "cuda"); model.eval()
    sched = load_noise_scheduler(ccfg)
    mean, std = stat["action"]["mean"], stat["action"]["std"]
    log.info("ARCTIC: pretrained contact model, %d windows at the study's onsets, %d samples", len(ds.indices), n_samples)
    B = len(ds.indices)
    pred = np.zeros((B, n_samples, 64, 1024), np.float32); inds_all = np.zeros((B, 64, 1024), np.int64); gt_d = np.zeros((B, 64, 1024), np.float32)
    for s0 in range(0, B, batch):
        idx = list(range(s0, min(s0 + batch, B)))
        items = [ds[i] for i in idx]
        b = {"action": torch.from_numpy(np.stack([np.asarray(it["action"]) for it in items])).float(),
             "obs": {k: torch.from_numpy(np.stack([np.asarray(it["obs"][k]) for it in items])).float() for k in items[0]["obs"]}, "aux": {}}
        nb = data_util.preprocess_contact_batch(b, "cuda", stat, ccfg["normalize_data"])
        for k in range(n_samples):
            torch.manual_seed(1000 * k + s0)
            x = torch.randn(nb["action"].shape, device="cuda").float()
            sched.set_timesteps(ccfg["num_timesteps"])
            with torch.no_grad():
                for t in sched.timesteps:
                    x0 = model(sample=x, timestep=t, obj_feat=nb["obs"]["obj_feat"], global_cond=nb["obs"])
                    x = sched.step(model_output=x0, timestep=t, sample=x).prev_sample
            full = data_util.unnormalize_item(x, mean, std).cpu().numpy()                 # (b, 64, 2048)
            for j, i in enumerate(idx):
                off = 0 if meta_te.iloc[i].hand == "L" else 1024
                pred[i, k] = full[j, :, off:off + 1024]
        log.info("  ARCTIC windows %d-%d done", idx[0], idx[-1])
    probe = Path("/result/uhnam/dexcore/arctic/20_bimart_contact_probe")
    si = pd.read_csv(probe / "sequence_index.csv").set_index("sequence_id")["file"]
    for i, r in enumerate(meta_te.itertuples()):
        with np.load(probe / "sequences" / si[r.sequence_id]) as z:
            d = z[f"contact_{'left' if r.hand == 'L' else 'right'}"][r.t0:r.t0 + 64]
            inds = z["obj_cano_bps_inds"][r.t0:r.t0 + 64].astype(np.int64)
        inds_all[i] = inds; gt_d[i] = np.take_along_axis(d, inds, 1)
    # operators: the cache's canonical points + the same gauss rule on the normalised mesh
    sys.path.insert(0, str(REPO / "scripts/research/canonical_contact/dyn"))
    from build_arctic_cache import weight_matrix
    md = np.load(probe / "assets/arctic_mesh_dict.npy", allow_pickle=True).item()
    W = {}
    for cat in cats:
        V = np.concatenate([md[f"{cat}_top"]["verts_original"], md[f"{cat}_bottom"]["verts_original"]]).astype(np.float64)
        c = V.mean(0); rad = np.linalg.norm(V - c, axis=1).max(); Vn = (V - c) / rad
        P = np.load(C.CACHE / "frames_full" / f"{cat}__obj__L.npz", allow_pickle=True)["canonical_points"].astype(np.float64)
        Wc, _ = weight_matrix(Vn, P)
        W[(cat, cat)] = Wc.astype(np.float32)
    return pred, inds_all, gt_d, W


def main():
    t_start = time.time()
    z, meta = H.load_sequences()
    man = json.load(open(H.OUT / "manifests" / "fixed0.json"))
    te = np.asarray(man["test"])
    if a.limit:
        te = te[:a.limit]
    meta_te = meta.iloc[te].reset_index(drop=True)
    gt_full = z["C"][te].astype(np.float32)
    cano = {str(c): p for c, p in zip(z["canonical_cats"], z["canonical_points"])}
    thr = C.zero_threshold()
    pred, inds, gt_d, W = (taco_predict if H.DATASET == "taco" else arctic_predict)(meta_te, a.n_samples, a.batch)
    os.chdir(HERE)
    B, K = pred.shape[:2]
    log.info("predictions done (%.0fs); restricted splat", time.time() - t_start)
    Xr_gt = np.zeros((B, 64, 512), np.float32); Xr_pred = np.zeros((B, K, 64, 512), np.float32)
    for i, r in enumerate(meta_te.itertuples()):
        Wc = W[(r.category, r.mesh_id)]
        Xr_gt[i] = restricted_splat(Wc, inds[i], gt_d[i])
        for k in range(K):
            Xr_pred[i, k] = restricted_splat(Wc, inds[i], pred[i, k])
    gap = np.linalg.norm(Xr_gt - gt_full, axis=2).mean(1)
    log.info("representation gap E_C(X_r(GT), GT full): mean %.3f (GT full static E_C for reference: %.3f)", gap.mean(),
             np.linalg.norm(np.repeat(gt_full[:, :1], 64, 1) - gt_full, axis=2).mean())
    rows, curves = [], {}

    def cat_metrics(pr, gt):
        m, cv = {}, {}
        for c in np.unique(meta_te.category):
            sel = np.where(meta_te.category.values == c)[0]
            mm, cc = metrics(pr[sel], gt[sel], thr, cano[c])
            for key, v in mm.items():
                m.setdefault(key, np.full(B, np.nan))[sel] = v
            for key, v in cc.items():
                cv.setdefault(key, np.full((B, v.shape[1]), np.nan))[sel] = v
        return m, cv

    def add(name, Kk, m, cv):
        df = pd.DataFrame(m); df["model"] = name; df["K"] = Kk; df["example"] = te; df["group"] = meta_te.group.values
        df["take_key"] = meta_te.take_key.values; df["first_onset"] = meta_te.first_onset.values; df["leaves_contact"] = meta_te.leaves_contact.values
        df["category"] = meta_te.category.values; df["split"] = "fixed"; df["fold"] = 0; df["rep_gap"] = gap
        rows.append(df); curves[(name, Kk)] = cv

    m, cv = cat_metrics(Xr_gt, gt_full); add("gt_restricted", 0, m, cv)
    m, cv = cat_metrics(np.repeat(Xr_gt[:, :1], 64, 1), gt_full); add("static_gtr", 0, m, cv)
    m, cv = cat_metrics(np.repeat(Xr_gt[:, :1], 64, 1), Xr_gt); add("static_gtr_vs_gtr", 0, m, cv)
    best = {}
    for name, ref in (("bimart", gt_full), ("bimart_vs_gtr", Xr_gt)):
        per_k = [cat_metrics(Xr_pred[:, k], ref) for k in range(K)]
        E = np.stack([p[0]["E_C"] for p in per_k], 1)
        for Kk in KS:
            ks = np.argmin(E[:, :Kk], 1)
            m = {key: np.array([per_k[ks[i]][0][key][i] for i in range(B)]) for key in per_k[0][0]}
            cv = {key: np.stack([per_k[ks[i]][1][key][i] for i in range(B)]) for key in per_k[0][1]}
            m["k_star"] = ks; add(name, Kk, m, cv)
            if name == "bimart" and Kk in (1, max(KS)):
                best[Kk] = np.stack([Xr_pred[i, ks[i]] for i in range(B)])
        m = {key: np.mean([p[0][key] for p in per_k], 0) for key in per_k[0][0]}
        cv = {key: np.mean([p[1][key] for p in per_k], 0) for key in per_k[0][1]}
        add(name, -1, m, cv)
        log.info("  %s: E_C K1 %.3f K5 %.3f K10 %.3f mean %.3f", name, E[:, 0].mean(), E[:, :5].min(1).mean(), E.min(1).mean(), E.mean())
    R = pd.concat(rows, ignore_index=True)
    out_dir = H.OUT / "eval" / "fixed0"; out_dir.mkdir(parents=True, exist_ok=True)
    R.to_csv(out_dir / "bimart_per_example.csv", index=False)
    np.savez_compressed(out_dir / "bimart_preds.npz", example=te, gt_restricted=Xr_gt.astype(np.float16), bimart_K1=best[1].astype(np.float16),
                        bimart_K10=best[max(KS)].astype(np.float16), bps_pred_metres_sample0=pred[:, 0].astype(np.float16), bps_inds=inds.astype(np.int32))
    np.savez_compressed(out_dir / "bimart_curves.npz", **{f"{n}__K{Kk}__{key}": v for (n, Kk), cv in curves.items() for key, v in cv.items()}, example=te)
    # ---- summary + paired comparison with the study's rows on the same sequences
    res = H.OUT / "results"; res.mkdir(exist_ok=True)
    S = []
    for (mname, Kk), g in R.groupby(["model", "K"]):
        row = dict(model=mname, K=Kk, n=len(g))
        for met in ("E_C", "E_dC", "dmag_pred", "dmag_true", "pattern_L2", "mass_abs", "s0_err", "E_C_last16"):
            pt, lo, hi = H.cluster_bootstrap(g[met].values, g.take_key.values, n_boot=500)
            row[met], row[f"{met}_lo"], row[f"{met}_hi"] = pt, lo, hi
            row[f"{met}_macro"] = float(g.groupby("group")[met].mean().mean())
        S.append(row)
    S = pd.DataFrame(S); S.to_csv(res / "bimart_comparison.csv", index=False)
    ours = pd.read_csv(out_dir / "per_example.csv")
    P = []
    for lab, (m1, k1), (m2, k2) in (("B4(K=1) - B1", ("bimart", 1), ("gtinit_vf", 0)), ("B4(best10) - B1", ("bimart", 10), ("gtinit_vf", 0)),
                                    ("B4(K=1) - B2(K=1)", ("bimart", 1), ("samplerG_vf", 1)), ("B4(best10) - B2(best10)", ("bimart", 10), ("samplerG_vf", 10)),
                                    ("B4(K=1) - B0", ("bimart", 1), ("static_gt", 0)), ("B4(best10) - B0", ("bimart", 10), ("static_gt", 0)),
                                    ("B4(K=1) - B3(K=1)", ("bimart", 1), ("samplerGT_vf", 1)), ("B4(best10) - B3(best10)", ("bimart", 10), ("samplerGT_vf", 10))):
        x = R[(R.model == m1) & (R.K == k1)].set_index("example"); y = ours[(ours.model == m2) & (ours.K == k2)].set_index("example")
        common = x.index.intersection(y.index)
        d = dict(comparison=lab, n=len(common))
        for met in ("E_C", "E_dC", "pattern_L2", "mass_abs", "s0_err"):
            diff = x.loc[common, met].values - y.loc[common, met].values
            pt, lo, hi = H.cluster_bootstrap(diff, x.loc[common, "take_key"].values, n_boot=500)
            d[met], d[f"{met}_lo"], d[f"{met}_hi"], d[f"{met}_frac_better"] = pt, lo, hi, float(np.nanmean(diff < 0))
        P.append(d)
    P = pd.DataFrame(P); P.to_csv(res / "bimart_paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["model", "K", "n", "E_C", "E_C_lo", "E_C_hi", "E_dC", "dmag_pred", "dmag_true", "pattern_L2", "mass_abs", "s0_err"]].round(3).to_string())
    print(P[["comparison", "n", "E_C", "E_C_lo", "E_C_hi", "E_C_frac_better", "s0_err"]].round(3).to_string())
    log.info("done in %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
