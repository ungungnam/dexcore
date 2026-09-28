"""Hand IK: recover a valid MANO configuration that tracks interpolated keypoints.

Interpolation produces KEYPOINTS. DexMachina trains on a HAND POSE (`configs_{side}`, 51 DoF), and
a keypoint cloud is not a hand -- nothing guarantees an interpolated cloud is reachable, inside the
joint limits, or even a consistent skeleton. IK is what turns the one into the other.

THE OBJECTIVE, four terms, each reported separately so a bad weight shows up as a term that is not
moving rather than as a mystery:

  keypoint    each of the 21 MANO keypoints is a MATERIAL point of the hand, fixed in its owner
              link's frame at an offset read off the source demo (see src/data/mano.py: the demo's
              keypoints are NOT the URDF link origins). IK asks the hand to put that same material
              point on the interpolated target.
  pose        stay near the warm start. Wrist and fingers weighted separately: across a transfer
              the wrist MUST travel, the fingers should only give way where the keypoints demand.
  limits      one-sided hinge penalty outside the URDF's own ranges. Soft, not clamped -- a clamp
              kills the gradient exactly when it is most needed.
  smoothness  penalises frame-to-frame change RELATIVE TO THE WARM START's own frame-to-frame
              change, not absolute change. Two reasons, and the first one is a correctness gate:
              the term is then exactly zero at q = q_init, so `All -> Identity -> Reconstruct`
              reproduces the source demo EXACTLY instead of approximately. A term penalising
              absolute change has a non-zero gradient at the truth and quietly drags a perfect
              reconstruction off it. Second, what we actually want to suppress is jitter IK
              INTRODUCED, not the motion the human performed.

SCALE. The keypoint term is divided by `keypoint_scale_m` squared, so it is dimensionless and reads
as "error in units of 1 mm". Without that normalisation it is measured in square metres: at the
ground-truth pose it is ~1e-18 while the smoothness term is ~4e-4, i.e. 14 orders of magnitude
smaller, and with comparable weights IK optimises smoothness and ignores the keypoints entirely.
Every weight below is therefore dimensionless and comparable to every other.

ALL FRAMES ARE SOLVED JOINTLY, and that is why. The brief asks for "previous frame as
initialization" plus temporal regularization; a joint solve with a smoothness term achieves the
same end more robustly than a sequential per-frame solve, because a sequential solve cannot revise
an early frame once a later one reveals it was a bad basin. The warm start plays the role of the
per-frame initialization: q is initialised at the interpolated demo pose, so IK begins from the
current best answer and can only improve it -- and a run that changes nothing is then a signal that
the objective is inert rather than a silent no-op.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.mano import ManoFK


@dataclass
class IKConfig:
    """Every solver knob. A reconstruction's name plus this config fully describes its IK."""
    iters: int = 600
    lr: float = 0.01
    keypoint_scale_m: float = 1e-3   # the keypoint term reads as "error in units of 1 mm"
    w_keypoint: float = 1.0
    w_pose_finger: float = 1e-3
    w_pose_wrist: float = 1e-4
    w_smooth: float = 1e-2
    w_limit: float = 1e3
    # "auto" -> CPU unless the user explicitly picked GPUs with CUDA_VISIBLE_DEVICES. This box is
    # shared and its GPUs are routinely at 100% for other people's jobs, where merely creating a
    # CUDA context blocks for minutes. IK here is small enough that CPU is the sane default; ask
    # for "cuda" (with CUDA_VISIBLE_DEVICES set to a free device) when you want it.
    device: str = "auto"
    verbose: bool = False
    # a frame whose final keypoint RMS exceeds this is REPORTED as not converged, never silently
    # accepted. 5 mm is well inside the ~13 mm ARCTIC-fit-vs-URDF joint discrepancy this hand has.
    converged_rms_m: float = 5e-3

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def validate(self) -> "IKConfig":
        if self.iters < 1:
            raise ValueError("iters must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.keypoint_scale_m <= 0:
            raise ValueError("keypoint_scale_m must be > 0")
        return self


@dataclass
class IKResult:
    """The solved pose plus everything needed to judge whether to trust it."""
    q: torch.Tensor                              # (F, ndof)
    per_frame_rms: np.ndarray                    # (F,) keypoint RMS error, metres
    history: List[dict] = field(default_factory=list)
    converged: Optional[np.ndarray] = None       # (F,) bool

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.per_frame_rms ** 2)))

    @property
    def num_unconverged(self) -> int:
        return int((~self.converged).sum()) if self.converged is not None else 0

    def summary(self) -> str:
        s = (f"IK: keypoint RMS {self.rms * 1000:.2f} mm "
             f"(worst frame {self.per_frame_rms.max() * 1000:.2f} mm)")
        if self.num_unconverged:
            bad = np.flatnonzero(~self.converged)
            s += (f"  -- {self.num_unconverged} frames NOT converged, "
                  f"first at row {bad[0]}")
        return s


def solve_keypoint_ik(fk: ManoFK, targets: np.ndarray, local: torch.Tensor,
                      q_init: torch.Tensor, cfg: Optional[IKConfig] = None,
                      tag: str = "") -> IKResult:
    """Fit the 51-DoF hand to interpolated keypoints over all frames jointly.

    fk        the hand's ManoFK
    targets   (F,21,3) interpolated world keypoints to track
    local     (F,21,3) each keypoint's offset in its OWNER LINK's frame
    q_init    (F,ndof) warm start; also the pose the `pose` term regularizes toward
    """
    cfg = (cfg or IKConfig()).validate()
    dev = fk.device
    tgt = torch.as_tensor(np.asarray(targets, dtype=np.float64), device=dev, dtype=fk.dtype)
    loc = local.to(device=dev, dtype=fk.dtype)
    q_ref = q_init.to(device=dev, dtype=fk.dtype)
    F, N = tgt.shape[0], tgt.shape[1]
    if loc.shape[:2] != (F, N) or q_ref.shape[0] != F:
        raise ValueError(f"shape mismatch: targets {tuple(tgt.shape)}, local {tuple(loc.shape)}, "
                         f"q_init {tuple(q_ref.shape)}")

    wrist_idx = torch.tensor(fk.wrist_dof_idx, device=dev, dtype=torch.long)
    finger_idx = torch.tensor(fk.finger_dof_idx, device=dev, dtype=torch.long)
    lo, hi = fk.limits[:, 0], fk.limits[:, 1]

    dref = (q_ref[1:] - q_ref[:-1]).detach() if F > 1 else None
    q = q_ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=cfg.lr)
    history: List[dict] = []

    for it in range(cfg.iters):
        opt.zero_grad()
        pred = fk.joints_from_dofs(q, loc)
        sq = ((pred - tgt) ** 2).sum(-1)                       # (F,21) squared metres
        L_kp = sq.mean() / (cfg.keypoint_scale_m ** 2)         # dimensionless: "in units of 1 mm"
        L_pf = ((q[:, finger_idx] - q_ref[:, finger_idx]) ** 2).mean()
        L_pw = ((q[:, wrist_idx] - q_ref[:, wrist_idx]) ** 2).mean()
        # deviation of the VELOCITY from the warm start's, so the term vanishes at q = q_ref
        L_sm = (((q[1:] - q[:-1]) - dref) ** 2).mean() if F > 1 else q.sum() * 0.0
        L_lim = (torch.relu(q - hi) ** 2 + torch.relu(lo - q) ** 2).mean()
        loss = (cfg.w_keypoint * L_kp + cfg.w_pose_finger * L_pf + cfg.w_pose_wrist * L_pw
                + cfg.w_smooth * L_sm + cfg.w_limit * L_lim)
        loss.backward()
        opt.step()

        if it % max(cfg.iters // 5, 1) == 0 or it == cfg.iters - 1:
            rec = dict(it=it, rms_mm=float(torch.sqrt(sq.mean())) * 1000,
                       pose_f=float(L_pf), pose_w=float(L_pw),
                       smooth=float(L_sm), limit=float(L_lim))
            history.append(rec)
            if cfg.verbose:
                print(f"    [{tag}] it {it:5d}  keypoint RMS {rec['rms_mm']:7.3f} mm | "
                      f"pose_f {rec['pose_f']:.5f} pose_w {rec['pose_w']:.5f} "
                      f"smooth {rec['smooth']:.2e} lim {rec['limit']:.2e}")

    with torch.no_grad():
        pred = fk.joints_from_dofs(q, loc)
        per_frame = torch.sqrt(((pred - tgt) ** 2).sum(-1).mean(-1)).cpu().numpy()
    return IKResult(q=q.detach(), per_frame_rms=per_frame, history=history,
                    converged=per_frame <= cfg.converged_rms_m)


def limit_violations(fk: ManoFK, q: torch.Tensor) -> np.ndarray:
    """(F, ndof) how far each DoF is outside its URDF limit, in radians/metres. 0 when inside.

    Reported rather than clamped: a pose that needed to leave its limits is a finding about the
    transfer or the budget, and clamping it away destroys the evidence.
    """
    lo, hi = fk.limits[:, 0], fk.limits[:, 1]
    with torch.no_grad():
        v = torch.relu(q - hi) + torch.relu(lo - q)
    return v.cpu().numpy()
