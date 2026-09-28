"""CorDexTransfer -- contact transfer by VISUAL CORRESPONDENCE between the two object surfaces.

    source contact point (part-local)
        |  render both objects from CANONICALLY ALIGNED viewpoints (view i of each is the same aspect)
        |  match source view i <-> target view i in DINOv2 descriptor space
        |  cluster the per-view evidence -> that contact's target point
    target contact point   ->  51-DoF IK onto it

WHAT SEPARATES IT FROM A PALM-FIRST TRANSFER. This one is CONTACT-DRIVEN: it decides where on the
target each demonstrated touch should land and asks IK to put the hand there. A palm-first operator
is POSE-DRIVEN: it places the palm and lets contact be wherever the fingers stop. Same sparse input,
different notion of what must be preserved -- which is exactly the Experiment 3 comparison.

Correspondence is computed ONCE on the canonically posed pair, because the cameras are fixed and
shared: one render for the whole clip, not one per frame.

IMPLEMENTATION. DexMachina's exp3 `func_warp` reference, driven by path -- nothing is copied, so
these numbers come from the same code that produced them elsewhere.

NOT the CorDex-Grasp repository under `third_party/`, despite the name. That is a learned grasp
SYNTHESIS model: it predicts a robot-hand (ShadowHand / Inspire) grasp from an object point cloud
and a text/image condition. It does not transfer a demonstration, does not consume a MANO
trajectory, and does not emit one -- so it cannot stand in for Phi without a retargeting stage that
does not exist here.

This file is deliberately self-contained: locating the reference tree, the calling convention it
needs, and the dexcore Transfer contract are all here, so the baseline reads top to bottom.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory
from src.paths import SIDES
from src.transfer.base import (FrameDiagnostic, Transfer, TransferDiagnostics, TransferResult,
                               register)

EXP3 = "dexmachina.eval.experiments.exp3_demo_gen_baselines"
PKG = "func_warp_reference"


# ------------------------------------------------------------------ locating the reference tree
def _source_root() -> Optional[Path]:
    """The DexMachina WORKING TREE carrying the exp3 experiments, or None.

    Not `third_party/dexmachina`: the exp3 experiments are unpublished additions and are absent
    from the upstream checkout, which is exactly why they cannot be a submodule.
    """
    env = os.environ.get("DEXCORE_DEXMACHINA_SRC")
    for c in ([Path(env)] if env else [Path.home() / "workspace" / "dexmachina"]):
        if (c / "dexmachina" / "eval" / "experiments" / "exp3_demo_gen_baselines").exists():
            return c
    return None


def _require_root() -> Path:
    root = _source_root()
    if root is None:
        raise FileNotFoundError(
            "the DexMachina exp3 reference tree was not found.\n"
            "`cordex` (func_warp) is a reference implementation living in an unpublished working "
            "tree, not in third_party/dexmachina (the upstream checkout has no eval/experiments/).\n"
            "Set $DEXCORE_DEXMACHINA_SRC to that tree.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@contextlib.contextmanager
def _in_source_root():
    """Run with cwd at the reference tree's root, then restore it.

    exp3 resolves its own assets through REPO-ROOT-RELATIVE paths
    (`dexmachina/assets/arctic/box/decomp/box_decomp.urdf`), because it was written to be launched
    as `python -m ...` from that root. Rewriting those paths would mean editing the reference
    implementation, which is the one thing a wrapper must not do -- so supply the working directory
    it documents instead. Restored on the way out, including on failure.
    """
    root = _require_root()
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def _source_object_is(name: Optional[str]):
    """Point the reference's SOURCE object at `name` for the duration of one call.

    `exp3.paths.SRC_OBJ_DIR` is a module constant -- "the trajectory every target is transferred
    FROM" -- fixed to the ARCTIC box, and `synth.py` binds it into its own namespace at import.
    It is NOT derived from the demo file: `src_proc` supplies only obj_trans/obj_quat/obj_arti,
    while the source GEOMETRY (its canonical frame, its joint range, its part maps) comes from that
    constant. So a laptop demo handed in unchanged would still be canonicalised and phase-normalised
    against the box.

    Rebinding it on the module is enough, and is the technique `_target_resolved_by_dexcore` already
    uses on the registry: dexcore's asset path stays the single authority for where an object lives.
    Restored on the way out, including on failure. `None` leaves the reference's constant alone.
    """
    if name is None:
        yield None
        return
    from src import paths
    _require_root()
    mod = importlib.import_module(f"{EXP3}.{PKG}.synth")
    saved = mod.SRC_OBJ_DIR
    mod.SRC_OBJ_DIR = str(paths.object_dir(name))
    try:
        yield mod.SRC_OBJ_DIR
    finally:
        mod.SRC_OBJ_DIR = saved


@contextlib.contextmanager
def _canon_frame_from_dexcore(target: ArticulatedObject):
    """Canonicalise the TARGET in DEXCORE's canonical frame for the duration of one call.

    The reference derives its own `canon_R` in `AnnotationState.seed_frame_from_metadata`, always by
    the same rule. dexcore instead DECLARES the rule per object category in
    `configs/canonical_rules.yaml` (a laptop's +Z is the base's own normal, not the axis most aligned
    with base->lid), and `configs/canonical_rules.yaml` is the single authority for what a canonical
    frame is -- the same principle `bimart._adopt_trajectory` already applies to tau_O. Left alone,
    the reference places the target in one frame while `src/data/target.py` places it in another, and
    `_finish` refuses the run: the hand would be posed against one placement and reconstructed
    against the other.

    Only the ROTATION is written. `canon_origin` is derived FROM the rotation (`_set_origin` ->
    `_base_bottom_center` projects the base vertices onto the canonical axes), so it is recomputed by
    the reference's own method rather than copied across -- that keeps the reference's units and
    origin_mode intact and leaves exactly one thing overridden.

    Where the two rules already agree this is a no-op BY CONSTRUCTION: the same matrix is written and
    the same origin recomputed. That covers every box target, i.e. every result produced so far.
    """
    AS = importlib.import_module(
        "dexmachina.eval.experiments.exp2_demo_transfer.annotation_tool.annotation_state")
    original = AS.AnnotationState.seed_frame_from_metadata
    R = np.asarray(target.canonical_rotation(), dtype=float)

    def seed_then_override(self):
        original(self)
        if getattr(self.meta, "object_id", None) == target.name:
            self.canon_R = R.copy()
            self._set_origin()

    AS.AnnotationState.seed_frame_from_metadata = seed_then_override
    try:
        yield R
    finally:
        AS.AnnotationState.seed_frame_from_metadata = original


@contextlib.contextmanager
def _target_resolved_by_dexcore(target_name: str):
    """Make the reference resolve the TARGET OBJECT through dexcore's asset path, not its own registry.

    WHY THIS EXISTS. The reference takes a target by NAME and looks the name up in its own
    `targets.json`. That file is a BUILD specification -- it records the raw PartNet-Mobility
    instance and the 0.35 calibration that `bake_target` applied to produce the asset. Used at RUN
    time it points at the pre-bake mesh, which is not the mesh that gets simulated: measured on
    pm100141 the two canonical frames sit 4.77 mm apart, larger than the grasp clearance itself,
    and the hand would be posed against one object while training used the other.

    dexcore already knows where the object is -- `paths.object_dir()` resolves it across the asset
    search path, and that is the single authority for "where is this object" in this repository. So
    the lookup is answered here for the duration of one call: `scale=1.0` and `dataset="arctic"`
    because what dexcore resolves IS the baked asset, used as-is.

    This is what keeps the operator self-contained: no dependency on the reference's registry, and
    nothing written into its tree. Restored on the way out, including on failure.
    """
    from src import paths
    obj_dir = str(paths.object_dir(target_name))
    spec = {"scale": 1.0, "obj_dir": obj_dir, "dataset": "arctic"}

    _require_root()
    reg = importlib.import_module(f"{EXP3}.registry")
    wrench = importlib.import_module(f"{EXP3}.wrench")
    saved = (reg.gen_spec_for, reg.install_for, wrench.installed_obj_dir)

    def gen_spec_for(name, *a, **kw):
        return dict(spec) if name == target_name else saved[0](name, *a, **kw)

    def install_for(name, strategy, *a, **kw):
        if name != target_name:
            return saved[1](name, strategy, *a, **kw)
        return target_name, saved[1](name, strategy, *a, **kw)[1]

    def installed_obj_dir(obj_name, *a, **kw):
        return obj_dir if obj_name == target_name else saved[2](obj_name, *a, **kw)

    reg.gen_spec_for, reg.install_for, wrench.installed_obj_dir = (
        gen_spec_for, install_for, installed_obj_dir)
    # The reference modules do `from ... import registry as R`, so they hold the MODULE object --
    # rebinding the attribute on the module is enough for their `R.gen_spec_for(...)` to see it.
    try:
        yield spec
    finally:
        reg.gen_spec_for, reg.install_for, wrench.installed_obj_dir = saved

def registered_targets() -> List[str]:
    """Targets this reference can build. It resolves through its OWN registry, not dexcore's
    asset path, so a target must be baked and registered there (`bake_target`) first."""
    if _source_root() is None:
        return []
    try:
        _require_root()
        return sorted(importlib.import_module(f"{EXP3}.registry").load())
    except Exception:
        return []


# ------------------------------------------------------------------------------ the operator
@register
class CorDexTransfer(Transfer):
    """Correspondence-driven contact transfer (exp3 `func_warp`), as a per-keyframe transfer.

    ONLY THE SELECTED KEYFRAMES ARE TRANSFERRED. The reference's interface happens to take a demo
    and return a demo, so it is handed the SPARSE demo: K keyframes in, K keyframes out. It never
    sees the dense clip, and it is never asked to produce one -- filling in the frames between is
    the reconstruction stage's job, and doing it here would make Phi and I indistinguishable.

    That is sound because the operator is per-frame by construction: each frame's contacts are
    answered on their own, with nothing smoothed across frames. Running it on K frames is the same
    computation as running it on those K frames out of T.
    """

    name = "cordex"

    def __init__(self, ik_iters: Optional[int] = None, corr: Optional[dict] = None,
                 source_object: Optional[str] = None, verbose: bool = False, **kwargs):
        super().__init__(source_object=source_object, **kwargs)
        self.ik_iters = ik_iters
        # WHICH DEMONSTRATION THE TRANSFER STARTS FROM. `synth.SRC_OBJ_DIR` is fixed to the ARCTIC
        # box, and it is the SOURCE GEOMETRY -- canonical frame, joint range, part maps -- not just
        # a file path, so a non-box source demo is silently canonicalised against the box without
        # this. None keeps the reference's constant, which is correct while the source IS the box.
        self.source_object = source_object
        # correspondence settings (view count, resolution, match thresholds). None keeps the
        # reference's own defaults, which is what its published numbers were produced with.
        self.corr = corr
        self.verbose = bool(verbose)

    def describe(self) -> str:
        bits = ([] if self.ik_iters is None else [f"ik_iters={self.ik_iters}"])
        if self.source_object is not None:
            bits.append(f"source={self.source_object}")
        return self.name if not bits else f"{self.name}(" + ", ".join(bits) + ")"

    def available(self) -> bool:
        return _source_root() is not None

    def transfer(self, sparse_demo: Demonstration, source_object: ArticulatedObject,
                 target_object: ArticulatedObject,
                 target_object_trajectory: ObjectTrajectory, **kwargs) -> TransferResult:
        _require_root()
        # No registry gate: the target is resolved through dexcore's asset path (see
        # `_target_resolved_by_dexcore`), so any object dexcore can load is a candidate. What the
        # reference itself cannot handle -- a missing warp checkpoint, an unrenderable mesh -- it
        # reports on its own, from the place that actually knows.
        from src import paths
        paths.object_dir(target_object.name)          # raises with the known objects if absent
        self._check_trajectory(sparse_demo, target_object_trajectory, target_object)

        td = tempfile.mkdtemp(prefix="dexcore_cordex_")
        # the stem MATTERS only as the demo FILENAME convention the reference expects. It does NOT
        # resolve the source geometry -- that comes from `synth.SRC_OBJ_DIR`, which is why
        # `_source_object_is` exists.
        src_proc = Path(td) / f"{sparse_demo.obj_name}_use_{sparse_demo.use_clip}.npy"
        sparse_demo.save(src_proc, write_sidecar=False)

        from src import paths
        mano_dir = str(paths.mano_urdf("right").parent.parent)   # absolute, before cwd moves
        kw = dict(verbose=self.verbose)
        if self.ik_iters is not None:
            kw["ik_iters"] = int(self.ik_iters)
        if self.corr is not None:
            kw["corr"] = dict(self.corr)

        synth = importlib.import_module(f"{EXP3}.func_warp_reference.synth").synth_func_warp
        with _target_resolved_by_dexcore(target_object.name), \
                _canon_frame_from_dexcore(target_object), \
                _source_object_is(self.source_object), _in_source_root():
            raw = synth(target_object.name, src_proc=str(src_proc.resolve()),
                        mano_dir=mano_dir, **kw)

        return _finish(self, raw, td, sparse_demo, target_object,
                       target_object_trajectory)


# ------------------------------------------------------------------------------ shared tail
def _finish(op: Transfer, raw, tmpdir: str, sparse: Demonstration,
            target_object: ArticulatedObject, dense_traj: ObjectTrajectory) -> TransferResult:
    """Turn the reference's processed-demo dict back into dexcore types, with diagnostics."""
    if not isinstance(raw, dict) or "world_coord" not in raw:
        raise RuntimeError(f"{op.name}: the reference returned {type(raw).__name__}, "
                           f"not a processed-format demo dict")
    out_path = Path(tmpdir) / f"{target_object.name}_use_{sparse.use_clip}.npy"
    np.save(out_path, raw, allow_pickle=True)
    got = Demonstration.load(path=str(out_path))
    if got.num_frames != sparse.num_frames:
        raise RuntimeError(
            f"{op.name}: returned {got.num_frames} frames for {sparse.num_frames} keyframes. "
            f"Phi must answer each selected frame; a length change means the operator resampled, "
            f"which breaks the alignment reconstruction relies on.")

    demo = replace(
        got,
        frame_start=sparse.frame_start, frame_end=sparse.frame_end,
        total_frames=sparse.total_frames, frame_index=sparse.frames(),
        subject=sparse.subject, fps=sparse.fps,
        provenance=replace(sparse.provenance, target_object=target_object.name,
                           transfer=op.describe()))
    # tau_O STAYS DENSE. The reference answers only the selected keyframes, but reconstruction has
    # to place the hand on every frame the object occupies -- handing it a 20-frame "trajectory"
    # would silently produce a 20-frame "dense" demo. The dense trajectory passed in is returned
    # unchanged; the transferred keyframes' own object poses are checked against it below, because
    # if the reference's canonical transfer disagreed with dexcore's, the hand would be posed
    # relative to one object placement and reconstructed against another.
    rows = [i for i, f in enumerate(dense_traj.frames()) if int(f) in set(int(x) for x in demo.frames())]
    dp = float(np.abs(demo.obj_pos - dense_traj.obj_pos[rows]).max())
    da = float(np.abs(demo.obj_arti - dense_traj.obj_arti[rows]).max())
    if dp > 1e-6 or da > 1e-6:
        raise RuntimeError(
            f"{op.name}: the reference placed the target object differently from dexcore's "
            f"canonical transfer (max |dpos| {dp:.2e} m, max |darti| {da:.2e} rad at the "
            f"keyframes). The hand would be posed against one placement and reconstructed against "
            f"another. Reconcile src/data/target.py with the reference before using this run.")
    traj = dense_traj

    # These operators aim AT contacts, so contact is what judges them. Two failure modes, and the
    # second is easy to miss: a frame with NO source contacts has no constraint at all -- the hand
    # is left wherever initialisation put it, tens of centimetres away on a distant target. Calling
    # that "ok" because no contact failed to transfer would be exactly backwards.
    diag = TransferDiagnostics(method=op.describe())
    for k in range(demo.num_frames):
        for side in SIDES:
            n_src = int(sparse.contact_mask[side][k].sum())
            n_out = int(demo.contact_mask[side][k].sum())
            disp = float(np.linalg.norm(
                demo.joints[side][k] - sparse.joints[side][k], axis=-1).max())
            reasons = []
            if n_src == 0:
                reasons.append("no source contacts: this operator is contact-driven, so nothing "
                               f"constrains the hand here (it moved {disp * 1000:.0f} mm anyway)")
            elif n_out == 0:
                reasons.append(f"{n_src} source contacts, none landed on the target")
            diag.frames.append(FrameDiagnostic(
                row=k, frame=int(demo.frames()[k]), side=side,
                ok=not reasons, reason="; ".join(reasons),
                metrics={"n_source_contacts": float(n_src), "n_target_contacts": float(n_out),
                         "max_displacement_mm": disp * 1000,
                         "contact_retention": float(n_out) / n_src if n_src else float("nan")}))
    return TransferResult(demo=demo, trajectory=traj, diagnostics=diag)
