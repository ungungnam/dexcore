"""LinearKeypointReconstructor -- object-relative linear in-betweening, then MANO IK.

The deliberately simple baseline the whole research programme is measured against. If sparse
interaction states plus a dense object trajectory are already enough, THIS is what shows it, and
anything more elaborate has to beat it.

    transferred sparse keypoints
        |  interpolate in the TARGET OBJECT's frame            (src/reconstruction/interpolate.py)
    dense world keypoints
        |  51-DoF IK, joint over all frames                    (src/reconstruction/ik.py)
    dense MANO configuration
        |  FK  +  contact recomputation against the target
    a complete Demonstration DexMachina can train on

NO DIFFUSION, NO RL, NO LEARNED INTERPOLATION, NO PHYSICS. That is the point, not a limitation:
the first question is whether simple in-betweening suffices, and a complicated reconstructor cannot
answer it.

WHAT THE OUTPUT'S KEYPOINTS ARE. They are the FK of the solved configuration, NOT the interpolated
targets. The two differ by the IK residual, and writing the targets would produce a demo whose
`joints` and `configs` disagree -- a file that is internally inconsistent in a way nothing
downstream would catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np
import torch

from src import geometry as G
from src.data.demo import Demonstration
from src.data.hand import get_hand
from src.data.mano import MANO_HAND_LINKS
from src.data.object import ArticulatedObject, get_object
from src.data.trajectory import ObjectTrajectory
from src.paths import BASE_PART_ID, LID_PART_ID, SIDES

from src.data.contacts import recompute_contacts as _recompute_contacts
from src.reconstruction import interpolate as I
from src.reconstruction.base import (Reconstructor, ReconstructionDiagnostics,
                                     ReconstructionResult, register)
from src.reconstruction.ik import IKConfig, IKResult, limit_violations, solve_keypoint_ik


@dataclass
class LinearKeypointConfig:
    """Everything this reconstructor may vary."""
    frame: str = "root"                  # interpolation frame: "root" (object-relative) or "world"
    ik: IKConfig = field(default_factory=IKConfig)
    recompute_contacts: bool = True      # re-derive contact links against the TARGET geometry
    contact_threshold: Optional[float] = None   # metres; None -> the source demo's own threshold
    samples_per_link: int = 32   # per-link surface samples (penetration/diagnostics only)
    n_obj_sample: int = 3580     # object surface samples; matches ARCTIC's vertex density
    seed: int = 0

    def validate(self) -> "LinearKeypointConfig":
        if self.frame not in I.FRAMES:
            raise ValueError(f"frame must be one of {I.FRAMES}, got {self.frame!r}")
        self.ik.validate()
        return self


@register
class LinearKeypointReconstructor(Reconstructor):
    """Interpolate keypoints in the target object's frame, then solve for a valid hand pose."""

    name = "linear_keypoint"

    def __init__(self, config: Optional[LinearKeypointConfig] = None, **kwargs):
        ik_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs) if k.startswith("ik_")}
        cfg = config or LinearKeypointConfig(ik=IKConfig(**ik_kwargs), **kwargs)
        self.config = cfg.validate()
        super().__init__(frame=self.config.frame, ik_iters=self.config.ik.iters)

    def describe(self) -> str:
        return f"{self.name}(frame={self.config.frame}, ik_iters={self.config.ik.iters})"

    # ------------------------------------------------------------------ the stage
    def reconstruct(self, transferred_sparse_demo: Demonstration,
                    target_object_trajectory: ObjectTrajectory,
                    target_object: Optional[ArticulatedObject] = None,
                    **kwargs) -> ReconstructionResult:
        sparse = transferred_sparse_demo
        traj = target_object_trajectory
        cfg = self.config
        rows = self._check_inputs(sparse, traj)          # keyframe rows in trajectory numbering
        target_object = target_object or get_object(traj.obj_name)
        device = cfg.ik.resolve_device()

        T = len(traj)
        targets = np.arange(T, dtype=np.float64)
        key_at = rows.astype(np.float64)
        dense_T = traj.transform()                       # (T,4,4)
        key_T = dense_T[rows]                            # (K,4,4) target object pose AT keyframes

        diag = ReconstructionDiagnostics(method=self.describe())
        out_joints: Dict[str, np.ndarray] = {}
        out_configs: Dict[str, Dict[str, np.ndarray]] = {}

        for side in SIDES:
            hand = get_hand(side, device=device, samples_per_link=cfg.samples_per_link)
            fk = hand.fk

            # 1. interpolate the keypoints in the target object's frame
            kp_dense = I.interpolate_keypoints(
                sparse.joints[side], key_T, dense_T, key_at, targets, frame=cfg.frame)

            # 2. warm start: the sparse hand poses, linearly interpolated (see interpolate_configs
            #    on why Euler lerp is acceptable for an initial guess but not for output)
            if not sparse.configs.get(side):
                raise ValueError(f"the sparse demo carries no configs_{side}; IK needs a warm start "
                                 f"and the link-local keypoint offsets, both of which come from the "
                                 f"demonstrated hand pose")
            cfg_dense = I.interpolate_configs(sparse.configs[side], key_at, targets)
            q_init = fk.dofs_from_cfg(cfg_dense, frames=T)

            # 3. each keypoint's offset in its owner link's frame, measured at the KEYFRAMES (where
            #    a real hand pose exists) and interpolated. Constant in principle -- it is the
            #    ARCTIC-fit vs URDF discrepancy -- but measured rather than assumed.
            q_key = fk.dofs_from_cfg(sparse.configs[side], frames=sparse.num_frames)
            local_key = fk.joint_local_offsets(q_key, sparse.joints[side]).cpu().numpy()
            local_dense = torch.as_tensor(
                I.interpolate_scalar(local_key, key_at, targets), device=fk.device, dtype=fk.dtype)

            # 4. solve for a valid hand configuration
            ik: IKResult = solve_keypoint_ik(fk, kp_dense, local_dense, q_init, cfg.ik,
                                             tag=f"{side}")

            out_configs[side] = fk.cfg_from_dofs(ik.q, sparse.configs[side])
            out_joints[side] = hand.keypoints(ik.q, local_dense)   # FK, not the targets
            diag.per_frame_rms[side] = ik.per_frame_rms
            diag.converged[side] = ik.converged
            diag.ik_history[side] = ik.history
            diag.extras[f"{side}.interp_targets"] = kp_dense
            diag.extras[f"{side}.limit_violation_max"] = float(limit_violations(fk, ik.q).max())

        dense = self._assemble(sparse, traj, out_joints, out_configs, target_object, diag)
        diag.extras["contact_link_frames"] = {
            s: int(dense.contact_mask[s].sum()) for s in SIDES}
        return ReconstructionResult(demo=dense, diagnostics=diag)

    # ------------------------------------------------------------------ assembly
    def _assemble(self, sparse: Demonstration, traj: ObjectTrajectory,
                  joints: Dict[str, np.ndarray], configs: Dict[str, Dict[str, np.ndarray]],
                  target_object: ArticulatedObject,
                  diag: ReconstructionDiagnostics) -> Demonstration:
        """Build the dense output Demonstration. Contacts are re-derived, never carried."""
        T = len(traj)
        cfg = self.config
        threshold = cfg.contact_threshold
        if threshold is None:
            threshold = sparse.contact_threshold

        contact_pos, contact_part, contact_mask = {}, {}, {}
        obj_contacts, obj_valid = {}, {}
        for side in SIDES:
            if cfg.recompute_contacts:
                cp, pt, mk, oc, ov = self._contacts(
                    side, configs[side], joints[side], traj, target_object, threshold)
                obj_contacts[side], obj_valid[side] = oc, ov
            elif target_object.name == sparse.provenance.source_object and T == sparse.provenance.source_num_frames:
                # same object AND same length: the source's own contact annotation is valid here and
                # is closer to ARCTIC's ground truth than geometric re-derivation. Only reachable
                # for the identity case, which is exactly where it is meaningful.
                raise ValueError(
                    "recompute_contacts=False needs the dense source contacts, which the sparse "
                    "demo no longer carries. Pass the source demo via `source_contacts=`, or leave "
                    "recompute_contacts=True.")
            else:
                raise ValueError(
                    f"recompute_contacts=False is not valid for target {target_object.name!r} "
                    f"(source {sparse.provenance.source_object!r}): carrying the source's contact "
                    f"annotation onto different geometry would fabricate contacts that were never "
                    f"observed and that the target's surface may not even admit.")
            contact_pos[side], contact_part[side], contact_mask[side] = cp, pt, mk

        prov = replace(sparse.provenance,
                       target_object=target_object.name,
                       reconstructor=self.describe())
        name = f"{target_object.name}_use_{sparse.use_clip}"
        return Demonstration(
            name=name, obj_name=target_object.name, use_clip=sparse.use_clip,
            subject=sparse.subject, fps=sparse.fps,
            frame_start=sparse.frame_start, frame_end=sparse.frame_start + T,
            total_frames=sparse.total_frames,
            obj_pos=traj.obj_pos.copy(), obj_quat=traj.obj_quat.copy(),
            obj_arti=traj.obj_arti.copy(),
            joints=joints, contact_pos=contact_pos, contact_part=contact_part,
            contact_mask=contact_mask, configs=configs,
            obj_contacts=obj_contacts, obj_contact_valid=obj_valid,
            contact_threshold=float(threshold),
            path="", provenance=prov, extra_params={},
            frame_index=traj.frames(),
        )

    def _contacts(self, side: str, configs: Dict[str, np.ndarray], joints: np.ndarray,
                  traj: ObjectTrajectory, obj: ArticulatedObject, threshold: float):
        """The format's contact definition, shared with every other synthesis method.

        Lives in `src/data/contacts.py` rather than here: an end-to-end method has to produce the
        same fields, and a second implementation of "what touching means" would show up as a
        result rather than as a bug. See that module's docstring.
        """
        return _recompute_contacts(
            side, configs, joints, traj, obj, threshold,
            samples_per_link=self.config.samples_per_link,
            n_obj_sample=self.config.n_obj_sample, seed=self.config.seed)
