"""Frame-level data of the factorisation study (one dataset per process: the hierarchical loader is import-bound).

  contact      C (N, 64, 512) raw in [0, 1];  C~ = (C - mu) / s   the hierarchical study's standardisation (per-point mean,
               ONE scalar s, so raw L2 = s x standardised L2); the networks read and write C~.
  geometry     per canonical point: position x_v / l (mesh centroid at the origin; ARCTIC top part articulated per frame),
               outward normal n_v, the static descriptor's nearest-distance / coverage channels, cell area, validity (10 ch.);
               global: the 1032-D static descriptor G of the hierarchical study (surface descriptor + radius + extent + flags).
  teacher      u_R2 = [z(a) | z(m) | a_k z(p_k) | a_k z(n_k)] (48, each block / sqrt(dim)), u_q = z(q) / sqrt(76); z-scores
               with TRAIN statistics (centroid / normal over the active parts); both divided by the train median pairwise
               distance so that  d_teacher(i, j) = 0.5 ||u_R2,i - u_R2,j|| + 0.5 ||u_q,i - u_q,j||  has median ~1 per block.
  frames       unique (take, group, absolute frame) triples of a split (overlapping windows of the same take and object are
               counted once); training draws P(take, group) ∝ n_frames^0.5 and a frame uniformly within (balanced sampling).
  masks        GT part label / toward-hand normal field of every frame (evaluation of decoded maps only; never a model input).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import cf_common as S
from r2_ops import CanonGeometry, hard_r2, wrench_support


class FrameData:
    def __init__(self, ds, device, stats=None, need_masks=False, seed=0):
        self.ds, self.device = ds, device
        self.fd = S.SAT.HP.fold(ds, device, stats=stats["hier"] if stats else None)
        fd = self.fd
        self.idx, self.meta, self.N = fd.idx, fd.meta, fd.N
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.C = f(fd.X_raw)                                                    # (N, 64, 512) raw
        self.Cn = fd.C                                                          # standardised
        self.mu = f(fd.stats["mu"]); self.s = float(fd.stats["s"])
        self.S = fd.S                                                           # (N, 1032)
        z = np.load(S.SAT.HP.ROOTS[ds] / "sequences.npz")
        gi = z["g_index"]
        self.pstat = f(np.stack([z["G_nearest"][gi] / 0.3, z["G_coverage"][gi]], -1))   # (N, 512, 2)
        self.geo = CanonGeometry(ds, device, z["O"])
        self.take = self.meta.take_key.astype(str).values
        self.group_key = (self.meta.take_key.astype(str) + "|" + self.meta.group.astype(str)).values
        self.mesh = self.meta.mesh_id.astype(str).values
        # exact R2 / wrench of the feature cache (raw units)
        F = S.SAT.load_features(ds); R = S.SAT.exact_r2(F)
        self.a, self.m, self.p, self.nrm, self.q = f(R["a"]), f(R["m"]), f(R["p"]), f(R["nrm"]), f(R["q"])
        self.U = f(F["U"]); self.length = f(F["length"])
        # unique frames of every split
        self.frames = {k: self._frame_table(k) for k in ("train", "val", "test")}
        self.rng = torch.Generator(device=device); self.rng.manual_seed(seed)
        # teacher descriptors
        self.tstats = self._teacher_stats() if stats is None else {k: (f(v[0]), f(v[1])) if isinstance(v, (list, tuple)) else float(v) for k, v in stats["teacher"].items()}
        self.u_r2, self.u_q = self._teacher_vectors()
        if stats is None:
            self.tstats["med_r2"], self.tstats["med_q"] = self._teacher_medians()
            self.u_r2 = self.u_r2 / self.tstats["med_r2"]; self.u_q = self.u_q / self.tstats["med_q"]
        else:
            self.u_r2 = self.u_r2 / self.tstats["med_r2"]; self.u_q = self.u_q / self.tstats["med_q"]
        self.stats = dict(hier=fd.stats, teacher={k: (v[0].cpu().numpy(), v[1].cpu().numpy()) if isinstance(v, tuple) else v for k, v in self.tstats.items()})
        # evaluation-only privileged signal
        self.M = self.nh = self.row_of = None
        if need_masks:
            mk = S.SAT.load_masks(ds)
            self.M = torch.from_numpy(mk["M"]).to(device); self.nh = torch.from_numpy(mk["nh"]).to(device)
            self.row_of = torch.from_numpy(mk["row_of"]).long().to(device)
        self.cal = S.SAT.load_calibration(ds)

    # ------------------------------------------------------------------ frame tables
    def _frame_table(self, split):
        n = self.idx[split]; meta = self.meta.iloc[n]
        df = pd.DataFrame(dict(n=np.repeat(n, S.T), t=np.tile(np.arange(S.T), len(n)),
                               take=np.repeat(self.take[n], S.T), gkey=np.repeat(self.group_key[n], S.T),
                               abs_frame=np.repeat(meta.t0.values, S.T) + np.tile(np.arange(S.T), len(n))))
        n_all = len(df)
        dup = df.duplicated(subset=["gkey", "abs_frame"], keep="first")
        # duplicates (overlapping windows of one take / object) must carry the same contact map
        if dup.any():
            d = df[dup].head(200); first = df[~dup].set_index(["gkey", "abs_frame"])
            for r in d.itertuples():
                o = first.loc[(r.gkey, r.abs_frame)]
                o = o.iloc[0] if isinstance(o, pd.DataFrame) else o
                assert torch.allclose(self.C[int(o.n), int(o.t)], self.C[r.n, r.t], atol=2e-3), "overlapping windows disagree"
        df = df[~dup].reset_index(drop=True)
        cnt = df.groupby("gkey").n.transform("size").values.astype(np.float64)
        w = cnt ** (S.TRAIN["balance_power"] - 1.0)
        return dict(n=torch.from_numpy(df.n.values).to(self.device), t=torch.from_numpy(df.t.values).to(self.device),
                    take=df["take"].values, gkey=df["gkey"].values, abs_frame=df.abs_frame.values, n_all=n_all, n_unique=len(df),
                    weight=torch.from_numpy(w / w.sum()).float().to(self.device), n_groups=int(df.gkey.nunique()))

    def sample_train(self, batch):
        fr = self.frames["train"]
        i = torch.multinomial(fr["weight"], batch, replacement=True, generator=self.rng)
        return fr["n"][i], fr["t"][i]

    def split_frames(self, split):
        fr = self.frames[split]
        return fr["n"], fr["t"]

    # ------------------------------------------------------------------ teacher
    def _teacher_stats(self):
        n, t = self.split_frames("train")
        a, m, p, nr, q = self.a[n, t], self.m[n, t], self.p[n, t], self.nrm[n, t], self.q[n, t]
        st = {}
        st["a"] = (a.mean(0), a.std(0).clamp(min=1e-3))
        st["m"] = (m.mean(0), m.std(0).clamp(min=1e-6))
        act = a > 0
        for k, x in (("p", p), ("nrm", nr)):
            mu = torch.stack([x[act[:, j], j].mean(0) if act[:, j].any() else x.new_zeros(3) for j in range(S.N_PARTS)])
            sd = torch.stack([x[act[:, j], j].std(0) if act[:, j].any() else x.new_ones(3) for j in range(S.N_PARTS)]).clamp(min=1e-6)
            st[k] = (mu, sd)
        st["q"] = (q.mean(0), q.std(0).clamp(min=1e-6))
        return st

    def _teacher_vectors(self):
        st = self.tstats; a = self.a
        za = (a - st["a"][0]) / st["a"][1] / np.sqrt(6)
        zm = (self.m - st["m"][0]) / st["m"][1] / np.sqrt(6)
        zp = ((self.p - st["p"][0]) / st["p"][1] * a[..., None]).flatten(-2) / np.sqrt(18)
        zn = ((self.nrm - st["nrm"][0]) / st["nrm"][1] * a[..., None]).flatten(-2) / np.sqrt(18)
        u_r2 = torch.cat([za, zm, zp, zn], -1)                                  # (N, 64, 48)
        u_q = (self.q - st["q"][0]) / st["q"][1] / np.sqrt(S.N_DIR)              # (N, 64, 76)
        return u_r2, u_q

    def _teacher_medians(self, n_pairs=100000):
        n, t = self.split_frames("train"); g = torch.Generator(device=self.device); g.manual_seed(1)
        i = torch.randint(0, len(n), (n_pairs,), device=self.device, generator=g); j = torch.randint(0, len(n), (n_pairs,), device=self.device, generator=g)
        d_r2 = (self.u_r2[n[i], t[i]] - self.u_r2[n[j], t[j]]).norm(dim=-1); d_q = (self.u_q[n[i], t[i]] - self.u_q[n[j], t[j]]).norm(dim=-1)
        return float(d_r2.median()), float(d_q.median())

    def teacher_distance(self, u_r2_a, u_q_a, u_r2_b, u_q_b):
        """d_teacher between two sets (pairwise (A, B))."""
        return S.TEACHER["w_r2"] * torch.cdist(u_r2_a, u_r2_b) + S.TEACHER["w_q"] * torch.cdist(u_q_a, u_q_b)

    # ------------------------------------------------------------------ per-frame tensors
    def geometry(self, n, t):
        """x (B, 512, 3) positions / l, nrm (B, 512, 3) outward unit normals (articulated top part on ARCTIC), alpha, valid."""
        geo = self.geo; g = geo.geo_index[n]
        Xt, Xb, Nt, Nb = geo.X_top[g], geo.X_bot[g], geo.N_top[g], geo.N_bot[g]
        if geo.articulated:
            th = -geo.arti[n, t + S.PAD]
            c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
            rot = lambda v: torch.cat([c * v[..., :1] - s * v[..., 1:2], s * v[..., :1] + c * v[..., 1:2], v[..., 2:]], -1)
            X, Nn = rot(Xt) + Xb, rot(Nt) + Nb
        else:
            X, Nn = Xt + Xb, Nt + Nb
        x = (X - geo.centroid[g][:, None]) / geo.length[g][:, None, None]
        Nn = Nn / (Nn.norm(dim=-1, keepdim=True) + 1e-9)
        return x, Nn, geo.alpha[g], geo.valid[g]

    def batch(self, n, t):
        x, nrm, alpha, valid = self.geometry(n, t)
        geo = torch.cat([x, nrm, self.pstat[n], (alpha * S.N_PTS)[..., None], valid.float()[..., None]], -1)   # (B, 512, 10)
        return dict(n=n, t=t, C=self.C[n, t], Cn=self.Cn[n, t], geo=geo, x=x, alpha=alpha, valid=valid, S=self.S[n],
                    u_r2=self.u_r2[n, t], u_q=self.u_q[n, t], a=self.a[n, t], m=self.m[n, t], p=self.p[n, t], nrm=self.nrm[n, t], q=self.q[n, t])

    def to_raw(self, Cn):
        return Cn * self.s + self.mu

    def to_std(self, C):
        return (C - self.mu) / self.s

    def lab_nh(self, n, t):
        """GT part label (B, 512) long and toward-hand normal field (B, 512, 3) of the frames (evaluation only)."""
        rows = self.row_of[n]
        assert (rows >= 0).all(), "mask cache incomplete"
        M = self.M[rows, t]
        return M.argmax(-1), self.nh[rows, t].float() / 127.0

    @torch.no_grad()
    def frame_r2(self, C_raw, n, t, lab=None, nh=None):
        """Single-frame exact-rule R2 of raw maps (no causal majority: the frame's own hard-point flag) and the one-patch-
        per-part wrench profile. Returns a, m (B, 6), p, nrm (B, 6, 3), q (B, 76)."""
        if lab is None:
            lab, nh = self.lab_nh(n, t)
        x, _, alpha, _ = self.geometry(n, t)
        r = hard_r2(C_raw[:, None], lab[:, None], nh[:, None], x[:, None], alpha, self.cal)
        a, m, p, nr = r["a"][:, 0], r["m"][:, 0], r["p"][:, 0], r["n"][:, 0]
        q = wrench_support(p, nr, a, self.U)
        return dict(a=a, m=m, p=p, nrm=nr, q=q)

    def teacher_of(self, r2):
        """u_R2 / u_q (already divided by the train medians) of an R2 dict (e.g. of a decoded map)."""
        st = self.tstats; a = r2["a"]
        za = (a - st["a"][0]) / st["a"][1] / np.sqrt(6)
        zm = (r2["m"] - st["m"][0]) / st["m"][1] / np.sqrt(6)
        zp = ((r2["p"] - st["p"][0]) / st["p"][1] * a[..., None]).flatten(-2) / np.sqrt(18)
        zn = ((r2["nrm"] - st["nrm"][0]) / st["nrm"][1] * a[..., None]).flatten(-2) / np.sqrt(18)
        u_r2 = torch.cat([za, zm, zp, zn], -1) / st["med_r2"]
        u_q = (r2["q"] - st["q"][0]) / st["q"][1] / np.sqrt(S.N_DIR) / st["med_q"]
        return u_r2, u_q

    def d_geo(self):
        return 10

    def d_static(self):
        return self.S.shape[1]
