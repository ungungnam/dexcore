"""Tensors shared by training, generation and evaluation: the hierarchical study's FoldData (raw and standardised contact,
static descriptor G, standardised object states, the fixed split), the residual parameterisation r_t = (C_t - s_0) / sigma_r
(sigma_r = train std of the residual over frames 1..63), the per-frame trajectory context tau_local,t for t = 1..63, the GT
part weights / orientation signs of the mask cache, the canonical geometry, and the exact R2 / wrench targets of the feature
cache with their train statistics.
"""
from __future__ import annotations

import numpy as np
import torch

import sat_common as S
from r2_ops import CanonGeometry


class SeqData:
    def __init__(self, ds, device, need_masks=True, stats=None):
        self.ds, self.device = ds, device
        self.fd = S.HP.fold(ds, device, stats=stats["hier"] if stats else None)
        fd = self.fd
        self.idx, self.meta, self.N = fd.idx, fd.meta, fd.N
        self.C = torch.from_numpy(fd.X_raw).to(device)                                   # (N, 64, 512) raw
        tr = self.idx["train"]
        if stats is None:
            r = self.C[tr, 1:] - self.C[tr, :1]
            self.sigma_r = float(r.std())
        else:
            self.sigma_r = float(stats["sigma_r"])
        self.stats = dict(hier=fd.stats, sigma_r=self.sigma_r)
        O_raw = np.load(S.HP.ROOTS[ds] / "sequences.npz")["O"]
        self.geo = CanonGeometry(ds, device, O_raw)
        self.take = self.meta.take_key.astype(str).values
        # exact R2 / wrench targets of the feature cache (raw units)
        F = S.load_features(ds)
        R = S.exact_r2(F)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.a, self.m, self.p, self.nrm, self.q = f(R["a"]), f(R["m"]), f(R["p"]), f(R["nrm"]), f(R["q"])
        self.geom = torch.cat([self.p.flatten(-2), self.nrm.flatten(-2)], -1)            # (N, 64, 36)
        self.U = f(F["U"])
        self.C_pad_prev = f(F["C_pad"][:, S.PAD - 2:S.PAD].astype(np.float32))            # (N, 2, 512) the two frames before t = 0
        self.length = f(F["length"])
        # train statistics of the targets (as in the previous study: m per dim, geom over the active parts, q per dim)
        if stats is None:
            self.tstats = {}
            mt = self.m[tr].reshape(-1, 6); self.tstats["m"] = (mt.mean(0), mt.std(0).clamp(min=1e-6))
            g = self.geom[tr].reshape(-1, 36); act = self.a[tr].reshape(-1, 6).repeat_interleave(3, 1).repeat(1, 2) > 0
            mu = torch.stack([g[act[:, j], j].mean() if act[:, j].any() else g.new_zeros(()) for j in range(36)])
            sd = torch.stack([g[act[:, j], j].std() if act[:, j].any() else g.new_ones(()) for j in range(36)]).clamp(min=1e-6)
            self.tstats["geom"] = (mu, sd)
            qt = self.q[tr].reshape(-1, 76); self.tstats["q"] = (qt.mean(0), qt.std(0).clamp(min=1e-6))
            self.tstats["m_bar"] = float(self.m[tr].sum(-1).mean())                       # train-mean total amount (T2 normaliser)
            self.stats["tstats"] = {k: (v if isinstance(v, float) else (v[0].cpu().numpy(), v[1].cpu().numpy())) for k, v in self.tstats.items()}
        else:
            self.tstats = {k: (v if isinstance(v, float) else (f(v[0]), f(v[1]))) for k, v in stats["tstats"].items()}
            self.stats["tstats"] = stats["tstats"]
        # training-time privileged signal
        self.M = self.nh = self.row_of = None
        if need_masks:
            mk = S.load_masks(ds)
            self.M = torch.from_numpy(mk["M"]).to(device)                                 # (N_used, 64, 512, 6) uint8
            self.nh = torch.from_numpy(mk["nh"]).to(device)                               # (N_used, 64, 512, 3) int8 toward-hand normals
            self.row_of = torch.from_numpy(mk["row_of"]).long().to(device)
            assert (self.row_of[np.concatenate([self.idx[k] for k in ("train", "val", "test")])] >= 0).all(), "mask cache incomplete"
        self.frames_future = torch.arange(1, S.T, device=device)
        self.cal = S.load_calibration(ds) if S.calibration_path(ds).exists() else None

    # ------------------------------------------------------------------ network inputs
    def s0_input(self, s0_raw):
        """raw (B, 512) -> the hierarchical study's standardised contact (per-point mean, scalar scale)."""
        mu = torch.as_tensor(self.fd.stats["mu"], device=self.device); s = float(self.fd.stats["s"])
        return (s0_raw - mu) / s

    def tau(self, n, t=None):
        """(B, T', 9 * D) trajectory context of frames t (default 1..63)."""
        t = self.frames_future[None].expand(len(n), -1) if t is None else t
        idx = t[..., None] + self.fd.offsets[None, None]                                   # (B, T', 9), padded indices
        return self.fd.O[n[:, None, None], idx].reshape(len(n), t.shape[1], -1)

    def residual(self, n, s0_raw=None):
        """(B, 63, 512) standardised residual r_t = (C_t - s_0) / sigma_r for t = 1..63 (s_0 = the GT frame 0 by default)."""
        s0 = self.C[n, 0] if s0_raw is None else s0_raw
        return (self.C[n, 1:] - s0[:, None]) / self.sigma_r

    def to_raw(self, r, s0_raw):
        return s0_raw[:, None] + self.sigma_r * r

    # ------------------------------------------------------------------ privileged signal / geometry for frames t
    def masks(self, n, t):
        """M (B, T', 512, 6) float part weights, nh (B, T', 512, 3) toward-hand unit normals, lab (B, T', 512) long for frames t."""
        rows = self.row_of[n]
        M = self.M[rows[:, None], t].float() / 255.0
        nh = self.nh[rows[:, None], t].float() / 127.0
        return M, nh, M.argmax(-1)

    def geometry(self, n, t):
        return self.geo.frame(n, t)

    def targets(self, n, t):
        """Exact R2 targets at frames t: a (B, T', 6), m (B, T', 6), geom (B, T', 36), q (B, T', 76)."""
        return dict(a=self.a[n[:, None], t], m=self.m[n[:, None], t], geom=self.geom[n[:, None], t], q=self.q[n[:, None], t],
                    p=self.p[n[:, None], t], nrm=self.nrm[n[:, None], t])

    def standardise(self, k, x):
        mu, sd = self.tstats[k]
        return (x - mu) / sd

    def batch(self, n, s0_raw=None):
        """Everything a training step needs for example indices n (B,) and frames 1..63."""
        t = self.frames_future[None].expand(len(n), -1)
        s0 = self.C[n, 0] if s0_raw is None else s0_raw
        M, nh, lab = self.masks(n, t)
        x, alpha, valid = self.geometry(n, t)
        tg = self.targets(n, t)
        return dict(n=n, t=t, s0_raw=s0, s0_in=self.s0_input(s0), S=self.fd.S[n], tau=self.tau(n), r=self.residual(n, s0), C=self.C[n, 1:],
                    M=M, nh=nh, lab=lab, x=x, alpha=alpha, valid=valid, **tg)

    def d_static(self):
        return self.fd.S.shape[1]

    def d_traj(self):
        return len(self.fd.offsets) * self.fd.D


def assert_causal_conditioning(data):
    """The conditioning of frame t reads object states t-8 .. t+8 (known trajectory) and nothing of the hand / contact after
    s_0: the only contact input is s_0 (frame 0) and the static descriptor."""
    offs = (data.fd.offsets - S.PAD).tolist()
    assert offs == list(range(-8, 9, 2)), offs
