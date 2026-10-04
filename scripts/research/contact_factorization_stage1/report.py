#!/usr/bin/env python
"""Assemble report.md: the Stage-1 decision table first, then the 16 sections (narrative/<nn>_<name>.md) with Tables 1-8
and the figures inserted at the referenced places.    python report.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import cf_common as S

NARR = S.OUT / "narrative"
SECTIONS = ["research_question", "motivation", "hypothesis", "dataset_split", "architecture", "training_objective", "reconstruction", "information_location",
            "neighborhood", "swap", "temporal_persistence", "dataset_differences", "failure_cases", "sanity_checks", "limitations", "decision"]
TITLES = ["Research question", "Motivation from the previous experiments", "Hypothesis", "Dataset / split", "Architecture", "Training objective", "Reconstruction results",
          "Where the structural information lives", "Nearest-neighbour analysis", "Swap analysis", "Temporal persistence of r", "Dataset differences", "Failure cases",
          "Sanity checks", "Limitations", "Stage-1 decision"]
PROBE_INPUTS = ["A3:z", "A3:r", "A3:zr", "A2:z", "A2:r", "A1:z", "A0:h", "A3_z16:z", "A3_z16:r", "A3_z16:zr", "A2_z16:z", "A2_z16:r", "A1_z16:z", "C_pca64", "C", "trivial"]
KNN_SPACES = ["A3:z", "A3:r", "A2:z", "A2:r", "A1:z", "A0:h", "A3_z16:z", "A3_z16:r", "A2_z16:z", "A2_z16:r", "A1_z16:z", "C", "teacher", "random"]
TEMP_CODES = ["A3:r", "A3:z", "A2:r", "A2:z", "A1:z", "A0:h", "A3_z16:r", "A3_z16:z", "A2_z16:r", "A2_z16:z", "A1_z16:z", "A3:resid_z", "A3:resid_full", "dense_C", "teacher"]


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ci(row, k, d=3):
    lo, hi = row.get(k + "_lo", np.nan), row.get(k + "_hi", np.nan)
    return f(row[k], d) if np.isnan(lo) else f"{f(row[k], d)} [{f(lo, d)}, {f(hi, d)}]"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def load(name):
    p = S.OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def table1():
    m = load("model_table.csv")
    rows = [(r.model, r.description, int(r.latent_z) if r.latent_z else "–", int(r.latent_r) if r.latent_r else "–", int(r.latent_h) if r.latent_h else "–", int(r.total_latent), "yes" if r.relational_loss else "no",
             "yes" if r.residual_branch else "no", f"{r.n_params / 1e6:.1f} M" if not np.isnan(r.n_params) else "–") for r in m.itertuples()]
    return md_table(["model", "description", "|z|", "|r|", "|h|", "total latent", "relational loss", "r branch", "parameters"], rows)


def table2():
    rec = load("reconstruction_metrics.csv"); out = ""
    for ds in S.DATASETS:
        r = rec[rec.dataset == ds]
        if r.empty:
            continue
        rows = []
        for m in list(S.ALL_MODELS) + ["mean", "mesh_mean", "pca64", "pca128", "R2_decoder_prev"]:
            q = r[r.model == m]
            if q.empty:
                continue
            q = q.iloc[0]
            if m == "R2_decoder_prev":
                rows.append((m, "–", "–", "–", f(q.R2_full), f(q.R2_full), "–", "–", "–")); continue
            rows.append((m, ci(q, "E_zonly"), ci(q, "E_full"), ci(q, "gain_r", 2) if m in S.MODELS else "–", f(q.R2_zonly), f(q.R2_full), ci(q, "explained_by_r", 2) if m in S.FACTORISED else "–",
                         f"{f(q.get('train_E_full', np.nan))} / {f(q.get('val_E_full', np.nan))} / {f(q.E_full)}" if m in S.MODELS else "–", f"{f(q.get('train_E_zonly', np.nan))} / {f(q.get('val_E_zonly', np.nan))}" if m in S.MODELS else "–"))
        out += f"\n**{S.LABEL[ds]}** (unique test frames; raw L2 per frame; 95 % take-bootstrap CI)\n\n" + md_table(["model", "E_zonly", "E_full", "Gain_r", "R² (C̄)", "R² (Ĉ)", "explained_by_r", "E_full train / val / test", "E_zonly train / val"], rows)
    return out


def table3_4():
    pr = load("probe_metrics.csv"); t3 = t4 = ""
    for ds in S.DATASETS:
        p = pr[pr.dataset == ds]
        if p.empty:
            continue
        rows3, rows4 = [], []
        for inp in PROBE_INPUTS:
            for kind in ("linear", "mlp", "constant"):
                q = p[(p.input == inp) & (p.probe == kind)]
                if q.empty:
                    continue
                q = q.iloc[0]
                rows3.append((inp, kind, int(q.d_in), f(q.part_f1), f(q.part_auprc), ci(q, "part_hamming"), ci(q, "amount_l1"), f(q.amount_r2), ci(q, "centroid"), ci(q, "normal", 1)))
                rows4.append((inp, kind, ci(q, "wrench_rel_l1"), ci(q, "wrench_cos"), ci(q, "wrench_ev"), f(q.get("ret_part_auprc", np.nan), 2), f(q.get("ret_amount_r2", np.nan), 2), f(q.get("ret_centroid", np.nan), 2), f(q.get("ret_normal", np.nan), 2), f(q.get("ret_wrench_ev", np.nan), 2), f(q.get("structure_retention", np.nan), 2)))
        t3 += f"\n**{S.LABEL[ds]}**\n\n" + md_table(["input", "probe", "d_in", "part F1", "part AUPRC", "part Hamming", "amount L1 / m̄", "amount R²", "centroid (l)", "normal (°)"], rows3)
        t4 += f"\n**{S.LABEL[ds]}**\n\n" + md_table(["input", "probe", "wrench rel L1", "wrench cos", "wrench EV", "ret. AUPRC", "ret. amount", "ret. centroid", "ret. normal", "ret. wrench", "structure retention"], rows4)
    return t3, t4


def table5():
    kn = load("neighborhood_metrics.csv"); out = ""
    for ds in S.DATASETS:
        k = kn[kn.dataset == ds]
        if k.empty:
            continue
        rows = []
        for sp in KNN_SPACES:
            q = k[k.space == sp]
            if q.empty:
                continue
            q = q.iloc[0]; rows.append((sp, int(q.n_anchors), ci(q, "teacher"), ci(q, "teacher_rel", 2), ci(q, "dense"), ci(q, "recall", 2), ci(q, "ssdd", 3)))
        out += f"\n**{S.LABEL[ds]}** (k = 10; pool = same mesh, other takes)\n\n" + md_table(["space", "anchors", "teacher distance of the neighbours", "relative to random", "dense distance", "recall@10 vs teacher", "structure-similar & dense-different rate"], rows)
    return out


def table6():
    sw = load("swap_metrics.csv"); out = ""
    for ds in S.DATASETS:
        s = sw[sw.dataset == ds]
        if s.empty:
            continue
        rows = []
        for q in s.itertuples():
            q = q._asdict()
            rows.append((q["model"], int(q["n_pairs"]), ci(q, "struct_to_zdonor"), ci(q, "struct_to_rdonor"), f(q["struct_floor"]), ci(q, "frac_struct_follows_z", 2), f"{f(q['r2_to_zdonor'])} / {f(q['r2_to_rdonor'])}", f"{f(q['q_to_zdonor'])} / {f(q['q_to_rdonor'])}",
                         ci(q, "detail_cos_rdonor", 2), ci(q, "detail_cos_zdonor", 2), ci(q, "frac_detail_follows_r", 2), f"{f(q['dense_change'])} / {f(q['dense_AB'])}", f"{f(q['iou_zdonor'], 2)} / {f(q['iou_rdonor'], 2)} / {f(q['iou_floor'], 2)}"))
        out += f"\n**{S.LABEL[ds]}** (controlled pairs: teacher distance {f(s.iloc[0].d_teacher_sel, 2)}, dense distance {f(s.iloc[0].d_dense_sel, 2)} on average; each pair swapped both ways)\n\n" + md_table(
            ["model", "pairs", "struct. d → z donor", "struct. d → r donor", "floor (own recon.)", "frac. structure follows z", "R2 part → z / → r", "wrench part → z / → r", "detail cos → r donor", "detail cos → z donor", "frac. detail follows r", "dense change / pair distance", "hard IoU → z / → r / floor"], rows)
    return out


def table7():
    tp = load("temporal_persistence.csv"); out = ""
    g = lambda t, code, metric, h: (t[(t.code == code) & (t.metric == metric) & (t.h == h)].iloc[0] if ((t.code == code) & (t.metric == metric) & (t.h == h)).any() else None)
    for ds in S.DATASETS:
        t = tp[tp.dataset == ds]
        if t.empty:
            continue
        rows = []
        for code in TEMP_CODES:
            if not (t.code == code).any():
                continue
            acf = " / ".join(f(g(t, code, "autocorrelation", h).value, 2) for h in (1, 4, 8, 16, 32))
            disp = " / ".join(f(g(t, code, "displacement_ratio", h).value, 2) for h in (1, 4, 8, 16))
            d8 = g(t, code, "displacement_ratio", 8); cos = " / ".join(f(g(t, code, "cos_to_frame0", h).value, 2) for h in (8, 32, 63))
            r_self, r_z, gain = (g(t, code, m, 8) for m in ("r2_from_past_self", "r2_from_current_z3", "r2_gain_past_beyond_z3"))
            rows.append((code, acf, disp, f"[{f(d8.lo, 2)}, {f(d8.hi, 2)}]", cos, f(r_self.value, 2) if r_self is not None else "–", f(r_z.value, 2) if r_z is not None else "–", f(gain.value, 2) if gain is not None else "–"))
        out += f"\n**{S.LABEL[ds]}** (test trajectories)\n\n" + md_table(["code", "autocorrelation h = 1 / 4 / 8 / 16 / 32", "displacement ratio h = 1 / 4 / 8 / 16", "CI at h = 8", "cos to frame 0 at t = 8 / 32 / 63", "R² x_{t+8} from x_t", "from z_{t+8} (A3)", "gain of x_t beyond z_{t+8}"], rows)
    return out


def table8(model="A3"):
    d = load("decision_summary.csv")
    if d.empty:
        return "(decision table not available)\n"
    if "model" in d:
        d = d[d.model == model]
    qs = list(dict.fromkeys(d.question)); rows = []
    for q in qs:
        a = {ds: d[(d.dataset == ds) & (d.question == q)] for ds in S.DATASETS}
        rows.append((q, *(a[ds].iloc[0].answer if len(a[ds]) else "–" for ds in S.DATASETS), " <br> ".join(f"**{S.LABEL[ds]}**: {a[ds].iloc[0].evidence}" for ds in S.DATASETS if len(a[ds]))))
    return md_table(["Question", "TACO", "ARCTIC", "Evidence"], rows)


def extra_tables():
    lat = load("latent_statistics.csv"); ts = load("training_summary.csv"); out = {}
    rows = [(r.dataset, r.model, r.code, int(r.dim), f(r.std_mean), f(r.std_min), f(r.eff_rank, 1), f(r.frac_dims_above_1pct, 2), f(r.top1_var_share, 2), f(getattr(r, "delta_abs_mean", np.nan), 4), f(getattr(r, "delta_energy_share", np.nan), 3), f(getattr(r, "R2_z_over_R2_full", np.nan), 2)) for r in lat.itertuples()] if not lat.empty else []
    out["latent"] = md_table(["dataset", "model", "code", "dim", "std (mean)", "std (min)", "effective rank", "dims > 1 % top var", "top-1 var share", "mean |ΔC| (raw)", "ΔC energy share", "R²(C̄) / R²(Ĉ)"], rows)
    rows = [(r.dataset, r.model, f"{r.n_params / 1e6:.1f} M", int(r.steps), int(r.best_step), f(r.train_coarse_last, 4), f(r.train_full_last, 4), f(r.train_rel_last, 4), f(r.grad_z_last, 3), f(r.grad_r_last, 3), f(r.val_E_zonly_best), f(r.val_E_full_best), f(r.train_E_full_best), f(r.val_z_std, 2), f(r.val_r_std, 2), f"{r.hours:.1f} h") for r in ts.itertuples()] if not ts.empty else []
    out["training"] = md_table(["dataset", "model", "params", "steps", "best step", "train L_coarse", "train L_full", "train L_rel", "‖grad‖ z branch", "‖grad‖ r branch", "val E_zonly", "val E_full", "train E_full", "z std", "r std", "wall time"], rows)
    loc = load("residual_localisation.csv")
    rows = [(r.dataset, r.model, r.bin, f(r.frac_points, 2), f(r.delta_abs_mean, 4), f(r.resid_z_abs_mean, 4), f(r.resid_full_abs_mean, 4), f(r.delta_energy_share, 3), f(r.resid_z_energy_share, 3), f(r.frac_resid_z_energy_removed, 2), f(r.corr_delta_resid_z, 2), f(getattr(r, "sign_agreement", np.nan), 2)) for r in loc.itertuples()] if not loc.empty else []
    out["table_localisation"] = md_table(["dataset", "model", "GT contact bin", "fraction of points", "mean |ΔC|", "mean |C − C̄|", "mean |C − Ĉ|", "share of ΔC energy", "share of z-residual energy", "fraction of z-residual energy removed", "corr(ΔC, C − C̄)", "sign agreement"], rows)
    p = S.OUT / "sanity_summary.json"
    if p.exists():
        sj = json.loads(p.read_text()); rows = []
        for k in sorted(next(iter(sj["checks"].values())).keys()):
            rows.append((k, *(f"{sj['checks'][ds][k]['status']} — {sj['checks'][ds][k]['detail']}" if ds in sj["checks"] else "–" for ds in S.DATASETS)))
        out["sanity"] = md_table(["check", "TACO", "ARCTIC"], rows)
    else:
        out["sanity"] = "(sanity_summary.json missing)\n"
    return out


def main():
    cfg = json.loads((S.OUT / "experiment_config.json").read_text()); cases = cfg.get("cases", {})
    t3, t4 = table3_4(); ex = extra_tables()
    tables = {"table1": table1(), "table2": table2(), "table3": t3, "table4": t4, "table5": table5(), "table6": table6(), "table7": table7(), "table8": table8(), "table8_z16": table8("A3_z16"), "latent": ex["latent"], "training": ex["training"], "sanity": ex["sanity"], "table_localisation": ex["table_localisation"]}
    cv = cfg.get("cases_bottleneck_variant", {})
    head = ["# Contact factorization, Stage 1: can a dense contact map be split into a structure-oriented latent z and a complementary realisation code r?", "",
            f"Date 2026-10-02/03. Datasets TACO and ARCTIC (analysed separately). One training seed. Outputs under `{S.OUT}`.", "",
            "## Stage-1 decision", "", "Case per dataset (primary model A3, |z| = 64): " + ", ".join(f"**{S.LABEL[ds]}: Case {cases.get(ds, '?')}**" for ds in S.DATASETS) + "."
            + (" Bottleneck variant A3_z16 (|z| = 16): " + ", ".join(f"**{S.LABEL[ds]}: Case {cv.get(ds, '?')}**" for ds in S.DATASETS) + "." if cv else ""), "",
            "**Table 8 — final Stage-1 decision (primary model A3, |z| = 64)**", "", tables["table8"], ""]
    if cv:
        head += ["**Table 8b — the same questions for the bottleneck variant A3_z16 (|z| = 16, |r| = 64)**", "", tables["table8_z16"], ""]
    body = []
    for i, (sec, title) in enumerate(zip(SECTIONS, TITLES), 1):
        p = NARR / f"{i:02d}_{sec}.md"
        txt = p.read_text() if p.exists() else "(to be written)\n"
        txt = re.sub(r"\{\{(\w+)\}\}", lambda m: tables.get(m.group(1), f"(missing {m.group(1)})"), txt)
        body += [f"## {i}. {title}", "", txt, ""]
    (S.OUT / "report.md").write_text("\n".join(head + body))
    n_todo = sum(len(re.findall(r"TBD|\(to be written\)", (NARR / f'{i:02d}_{s}.md').read_text())) if (NARR / f'{i:02d}_{s}.md').exists() else 1 for i, s in enumerate(SECTIONS, 1))
    print("report.md written;", n_todo, "placeholder(s)")


if __name__ == "__main__":
    main()
