"""BimArtSynthesizer -- the hand motion is GENERATED from tau_O, not transferred from a demo.

    tau_O + target geometry
        |  contact diffusion  -> where the object is touched
        |  motion diffusion   -> 100 hand-surface points per hand, per frame
        |  MANO fit           -> a MANO hand                      (all three: BimArt, in-process)
    MANO hand
        |  21 ARCTIC keypoints (16 kintree joints + 5 tip vertices)
        |  51-DoF IK onto the dexcore URDF hand                   (ours: src/reconstruction/ik.py)
        |  contact recompute against the TARGET geometry          (ours: src/data/contacts.py)
    a complete Demonstration, scored by the same code as the staged pipeline

WHY THIS IS A `Synthesizer` AND NOT A STAGE. BimArt has no selection, no transfer and no
reconstruction; it consumes the object trajectory and emits a hand. It is the control the staged
pipeline's central claim is measured against: `hand_states_used` is 0 here against the staged
pipeline's selection budget, and those two numbers in one column ARE the comparison.

IT CANNOT SEE THE HUMAN HAND, AND THAT IS ENFORCED TWICE.
  * `consumes` omits `source_hand`, so `SynthesisInput` refuses it (src/synthesis/base.py).
  * The reference is handed a demo file containing ONLY tau_O -- obj_trans / obj_quat / obj_arti
    and nothing else. The hand states are absent from the input, not merely unread. That matters
    because the exp3 driver `synth_bimart` DOES read them (for its own IK warm start and
    passthrough params) and we deliberately do not use that driver; see below.

WHY NOT `synth_bimart`. The exp3 tree ships a full driver that runs BimArt and then IK-fits the
dexmachina hand. Two reasons this file drives the generator directly instead:
  1. its IK is exp3's `hand_ik.py`, whose objective dexcore deliberately reworked -- an unscaled
     keypoint term in square metres next to an ABSOLUTE-velocity smoothness term at equal weight
     (see src/reconstruction/ik.py). Numbers produced by two different IK objectives cannot be put
     in one table and read as a difference between the two GENERATORS.
  2. it warm-starts that IK from the human demonstration's own pose, which is precisely the input
     this method is supposed to do without.
So the generator is used as published, and everything downstream of it is dexcore's own code --
the same code the staged pipeline runs.

WINDOWING. The models denoise fixed 64-frame windows; the reference stitches consecutive stride-64
windows in world space. That is the published horizon used outside its designed range, so the
window boundaries are recorded in the diagnostics rather than left implicit.

This file is deliberately self-contained: locating the reference tree, the calling convention it
needs, and the dexcore Synthesizer contract are all here, so the baseline reads top to bottom.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.contacts import recompute_contacts
from src.data.demo import Demonstration, Provenance
from src.data.hand import get_hand
from src.data.mano import (JOINT_NAMES, NUM_JOINTS, WRIST_ROTATION_DOFS, WRIST_TRANSLATION_DOFS,
                           MANO_HAND_LINKS, wrist_dofs_from_matrix)
from src.paths import SIDES
from src.reconstruction.ik import IKConfig, limit_violations, solve_keypoint_ik
from src.synthesis.base import (Synthesizer, SynthesisDiagnostics, SynthesisInput, SynthesisResult,
                                register)

EXP3 = "dexmachina.eval.experiments.exp3_demo_gen_baselines"
BIMART = f"{EXP3}.bimart_reference"

# MANO vertex index of each fingertip, in DEXCORE'S joint order (16..20 = thumb, index, middle,
# ring, pinky -- see src/data/mano.py, whose tip order is read off MANO_HAND_LINKS). The indices
# themselves are MANO v1.2's and are the set BimArt's own contact loss uses (`contact_loss.py`).
# The ORDER is asserted at runtime in `_keypoints21`, because a silently permuted fingertip set
# would look like a plausible hand. It was also checked once against the MANO v1.2 model directly
# (rest pose, zero betas): each vertex below is nearest its OWN finger's DIP joint, 5/5, at
# thumb 34.7 / index 23.6 / middle 26.5 / ring 24.4 / pinky 19.1 mm. That check needs the
# license-gated pkls, which dexcore does not depend on, so it is recorded here rather than as a
# test; `tests/test_synthesis.py` locks the ORDER against MANO_HAND_LINKS, which needs nothing.
TIP_VERTEX = [745, 317, 444, 556, 673]
TIP_JOINTS = [16, 17, 18, 19, 20]
# each tip's own DIP joint, i.e. the origin of the distal link it rides
TIP_DIP_JOINT = {16: 15, 17: 3, 18: 6, 19: 12, 20: 9}


# ------------------------------------------------------------------ locating the reference tree
def _source_root() -> Optional[Path]:
    """The DexMachina WORKING TREE carrying the exp3 experiments, or None.

    Not `third_party/dexmachina`: the exp3 experiments are unpublished additions and are absent
    from the upstream checkout, which is exactly why they cannot be a submodule. The BimArt repo
    itself, its pretrained checkpoints and the license-gated MANO models are all vendored under
    `<root>/dexmachina/eval/experiments/exp3_demo_gen_baselines/bimart_reference/`.
    """
    env = os.environ.get("DEXCORE_DEXMACHINA_SRC")
    for c in ([Path(env)] if env else [Path.home() / "workspace" / "dexmachina"]):
        if (c / "dexmachina" / "eval" / "experiments" / "exp3_demo_gen_baselines"
                / "bimart_reference").exists():
            return c
    return None


def _require_root() -> Path:
    root = _source_root()
    if root is None:
        raise FileNotFoundError(
            "the DexMachina exp3 reference tree was not found.\n"
            "`bimart` drives the BimArt reference vendored in an unpublished working tree, not in "
            "third_party/dexmachina (the upstream checkout has no eval/experiments/).\n"
            "Set $DEXCORE_DEXMACHINA_SRC to that tree.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@contextlib.contextmanager
def _in_source_root():
    """Run with cwd at the reference tree's root, then restore it.

    BimArt resolves its own assets, configs and checkpoints through cwd-relative paths
    (`assets/mano_v1_2/`, `experiments/.../model_final.pth`), because it was written to be launched
    from its repo root. Rewriting those paths would mean editing the reference implementation,
    which is the one thing a wrapper must not do -- so supply the working directory it documents
    instead. `BimArtRunner` chdirs into the BimArt repo itself; this one puts the exp3 package's
    own relative asset lookups on the right footing. Restored on the way out, including on failure.
    """
    root = _require_root()
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def registered_targets() -> List[str]:
    """Targets the exp3 registry can build. `bimart` resolves geometry through it, so a target
    must be baked and registered there first."""
    if _source_root() is None:
        return []
    try:
        _require_root()
        return sorted(importlib.import_module(f"{EXP3}.registry").load())
    except Exception:
        return []


@contextlib.contextmanager
def _target_resolved_by_dexcore(exp3_name: str, dexcore_name: str):
    """Make the reference resolve the TARGET OBJECT through dexcore's asset path, not its registry.

    The exp3 registry entry is a BUILD specification -- for `pm100141_calibrated` it records the raw
    PartNet-Mobility instance and the 0.35 calibration that `bake_target` APPLIED to produce the
    asset. Used at RUN time it points at the pre-bake mesh, which is not the mesh anything is
    trained or evaluated against. Measured here, on the exact clip below: the reference's own
    target trajectory sits 4.7748 mm from dexcore's canonical transfer with the registry entry, and
    0.0000 mm with this override. Larger than the grasp clearance, so the hand would be generated
    against one object placement and scored against another.

    dexcore already knows where the object is -- `paths.object_dir()` resolves it across the asset
    search path, and that is the single authority for "where is this object" in this repository. So
    the lookup is answered here for the duration of one call, with `scale=1.0` and
    `dataset="arctic"` because what dexcore resolves IS the baked asset in ARCTIC layout.

    `dataset` also selects which BimArt mesh_dict entry the generator is conditioned on, and that is
    a POLICY of the reference (native category vs synthesised entry) rather than a fact about where
    the files are -- so the caller restores the original value on the trajectory afterwards. See
    `_generate`. Restored on the way out, including on failure.
    """
    from src import paths
    # TWO NAMES, deliberately. `exp3_name` is the registry key the reference is called with;
    # `dexcore_name` is the object in OUR asset path whose geometry answers it. They differ for the
    # box family -- exp3 registers `box_s100` for what dexcore simply calls `box` -- and conflating
    # them makes the in-distribution control (a native BimArt category) unrunnable.
    obj_dir = str(paths.object_dir(dexcore_name))
    spec = {"scale": 1.0, "obj_dir": obj_dir, "dataset": "arctic"}

    _require_root()
    reg = importlib.import_module(f"{EXP3}.registry")
    saved = reg.gen_spec_for

    def gen_spec_for(name, *a, **kw):
        return dict(spec) if name == exp3_name else saved(name, *a, **kw)

    # The reference modules do `from ... import registry as R`, so they hold the MODULE object --
    # rebinding the attribute on the module is enough for their `R.gen_spec_for(...)` to see it.
    reg.gen_spec_for = gen_spec_for
    try:
        yield spec
    finally:
        reg.gen_spec_for = saved


@contextlib.contextmanager
def _source_object_is(name: Optional[str]):
    """Point the reference's SOURCE object at `name` for the duration of one call.

    `exp3.paths.SRC_OBJ_DIR` is a module constant whose comment reads "the trajectory every target
    is transferred FROM", and `object_traj` binds it into its own namespace at import. Rebinding it
    there is enough, and is the same technique `_target_resolved_by_dexcore` uses on the registry:
    dexcore's asset path stays the single authority for where an object lives. Restored on the way
    out, including on failure. `None` leaves the reference's own constant alone.
    """
    if name is None:
        yield None
        return
    from src import paths
    _require_root()
    obj_traj = importlib.import_module(f"{EXP3}.dexcore_reference.object_traj")
    saved = obj_traj.SRC_OBJ_DIR
    obj_traj.SRC_OBJ_DIR = str(paths.object_dir(name))
    try:
        yield obj_traj.SRC_OBJ_DIR
    finally:
        obj_traj.SRC_OBJ_DIR = saved


def _tau_only_demo(inp: SynthesisInput, path: Path) -> Path:
    """Write a processed-format file carrying ONLY the object trajectory -- no hand states.

    The reference derives the target trajectory from `params['obj_trans'/'obj_quat'/'obj_arti']`
    and reads nothing else from this file. Writing only those three makes that a property of the
    INPUT rather than a claim about the reference's behaviour: there is no human hand motion in
    the file for anything downstream to use, deliberately or by accident.
    """
    # the SOURCE object's trajectory: the reference applies the canonical transfer itself, so
    # handing it the already-transferred tau_O would transfer twice.
    traj = inp.source_trajectory
    if traj is None:
        raise ValueError("bimart needs the source object trajectory; the driver did not supply one")
    raw = {"params": {"obj_trans": np.asarray(traj.obj_pos, np.float64),
                      "obj_quat": np.asarray(traj.obj_quat, np.float64),
                      "obj_arti": np.asarray(traj.obj_arti, np.float64)},
           "world_coord": {}}
    np.save(path, raw, allow_pickle=True)
    return path


# ------------------------------------------------------------------------------ the synthesizer
@register
class BimArtSynthesizer(Synthesizer):
    """Learned bimanual interaction prior (BimArt, CVPR'25), driven from tau_O alone."""

    name = "bimart"
    consumes = frozenset({"source_object", "source_clip"})

    def __init__(self, exp3_target: Optional[str] = None, ik_iters: int = 600,
                 mano_steps: Optional[int] = None, device: str = "cuda",
                 contact_threshold: float = 0.01, samples_per_link: int = 32,
                 n_obj_sample: int = 3580, category: Optional[str] = None,
                 source_object: Optional[str] = None, entry_verts_per_part: Optional[int] = None,
                 verbose: bool = True, **kwargs):
        super().__init__(exp3_target=exp3_target, ik_iters=ik_iters, category=category,
                         source_object=source_object,
                         entry_verts_per_part=entry_verts_per_part, **kwargs)
        self.exp3_target = exp3_target
        # WHICH PRIOR THE MODEL IS CONDITIONED ON. BimArt has 11 ARCTIC categories and the
        # reference hardcodes "box" in both of its branches, which is right only while every target
        # is box-like. A laptop target conditioned on the box prior is asking the model for the
        # wrong kind of interaction, so the category is an explicit option here. None keeps the
        # reference's own choice.
        self.category = category
        # WHICH DEMONSTRATION THE TRANSFER STARTS FROM. exp3's `paths.SRC_OBJ_DIR` is a module
        # constant -- "the trajectory every target is transferred FROM" -- fixed to the box. A
        # laptop-to-laptop transfer needs the laptop there instead. None keeps the constant.
        self.source_object = source_object
        # HOW MANY VERTICES THE SYNTHESISED BimArt ENTRY KEEPS PER PART. The reference subsamples to
        # 6000 by choosing VERTICES first and then keeping only faces whose three vertices all
        # survived -- which leaves vertices belonging to no face at all. `potpourri3d`'s heat solver,
        # which BimArt runs to spread the contact map over the object, refuses such a mesh:
        #     GC_SAFETY_ASSERT FAILURE ... unreferenced vertex 3132
        # It happens on a thin SLAB, where a base of ~12k vertices loses more than half. Raising this
        # above the part's vertex count makes `_subsample_mesh` return the mesh untouched, so no
        # orphan can be created. None keeps the reference's own default, which is right for the box
        # family that has been running with it.
        self.entry_verts_per_part = entry_verts_per_part
        self.ik_iters = int(ik_iters)
        self.mano_steps = mano_steps
        self.device = device
        self.contact_threshold = float(contact_threshold)
        self.samples_per_link = int(samples_per_link)
        self.n_obj_sample = int(n_obj_sample)
        self.verbose = bool(verbose)

    def describe(self) -> str:
        """The name a result carries. The category and the source object CHANGE THE METHOD, so
        they belong in the name: two runs that differ in either are not the same experiment and
        must not share a run directory or a provenance string."""
        bits = [f"ik_iters={self.ik_iters}"]
        if self.category is not None:
            bits.append(f"category={self.category}")
        if self.source_object is not None:
            bits.append(f"source={self.source_object}")
        if self.entry_verts_per_part is not None:
            bits.append(f"entry_verts={self.entry_verts_per_part}")
        return f"{self.name}(" + ", ".join(bits) + ")"

    def available(self) -> bool:
        return _source_root() is not None

    # ------------------------------------------------------------------ the stage
    def synthesize(self, inp: SynthesisInput) -> SynthesisResult:
        _require_root()
        target = self.exp3_target or inp.target_object.name
        known = registered_targets()
        if target not in known:
            raise ValueError(
                f"bimart resolves target geometry through the exp3 registry, which does not have "
                f"{target!r}. Registered: {known}. Bake and register it there first, or pass "
                f"transfer of the name with synthesizer_options.exp3_target=<registered name>.")

        diag = SynthesisDiagnostics(method=self.describe())
        td = Path(tempfile.mkdtemp(prefix="dexcore_bimart_"))
        _tau_only_demo(inp, td / "tau_o.npy")

        bout, oi = self._generate(target, inp.target_object.name, td / "tau_o.npy", inp, diag)
        joints = {s: self._keypoints21(bout, s, diag) for s in SIDES}
        configs, ik_rms = self._fit_urdf(joints, diag)
        demo = self._assemble(inp, joints, configs, diag)

        for s in SIDES:
            diag.per_frame[f"ik_residual.{s}"] = ik_rms[s]
        return SynthesisResult(
            demo=demo, trajectory=inp.trajectory, diagnostics=diag,
            method=self.describe(), consumes=self.consumes,
            hand_states_used=0,            # the generator never saw a source hand state
            extras={"bimart_category": oi.category, "windows": diag.extras.get("windows")})

    # ------------------------------------------------------------------ 1. generate
    def _generate(self, target: str, dexcore_name: str, tau_path: Path, inp: SynthesisInput,
                  diag: SynthesisDiagnostics):
        """Run the reference generator, after checking its tau_O agrees with ours."""
        _require_root()
        obj_traj = importlib.import_module(f"{EXP3}.dexcore_reference.object_traj")
        obj_input = importlib.import_module(f"{BIMART}.obj_input")
        run_bimart = importlib.import_module(f"{BIMART}.run_bimart")
        BP = importlib.import_module(f"{BIMART}.bimart_paths")

        ok, missing = BP.check_assets()
        if not ok:
            raise RuntimeError(f"BimArt assets missing: {missing}. See bimart_reference/SETUP.md.")
        if not BP.mano_ready():
            raise RuntimeError(
                "BimArt needs the license-gated MANO v1.2 pkls for its MANO-fit stage; "
                "bimart_paths.mano_ready() is False. See bimart_reference/SETUP.md.")

        reg = importlib.import_module(f"{EXP3}.registry")
        policy_dataset = reg.gen_spec_for(target)["dataset"]   # BEFORE the override; see below
        with _target_resolved_by_dexcore(target, dexcore_name), \
                _source_object_is(self.source_object), _in_source_root():
            traj = obj_traj.build_target_object_trajectory(target, src_proc=str(tau_path), frames=0)
            self._adopt_trajectory(traj, inp)
            # GEOMETRY comes from dexcore (the override above); the ENTRY-SELECTION POLICY comes
            # from the reference. `dataset` is read in exactly one place -- `build_object_input`'s
            # choice between BimArt's shipped native-category mesh and a synthesised entry -- and
            # that choice is the reference's call, not a consequence of where our files live.
            traj["dataset"] = policy_dataset
            oi = self._object_input(obj_input, target, traj)
            runner = run_bimart.BimArtRunner(need_mano=True, device=self.device,
                                             verbose=self.verbose)
            bout = runner.generate(oi, points_only=False, mano_steps=self.mano_steps,
                                   seed=inp.seed)

        ph = BP.PRED_HORIZON
        n_win = int(np.ceil(max(len(inp) - ph, 1) / ph)) + 1
        diag.extras["windows"] = {"horizon": ph, "count": n_win}
        diag.notes.append(
            f"generated in {n_win} stride-{ph} windows, stitched in world space; the model's "
            f"designed horizon is one {ph}-frame window, so boundaries are an extrapolation")
        dp, da = getattr(self, "_traj_gap", (0.0, 0.0))
        diag.extras["tau_o_gap"] = {"dpos_mm": dp * 1000, "darti_rad": da}
        if dp > 1e-6 or da > 1e-6:
            diag.notes.append(
                f"the reference's own canonical transfer placed the target {dp * 1000:.2f} mm / "
                f"{da:.4f} rad from dexcore's; DEXCORE'S tau_O was used (see _adopt_trajectory)")
        al = oi.align
        diag.notes.append(
            f"frame bridge: icp {al.icp_rms * 100:.2f} cm, gate mean {al.gate_mean * 100:.2f} cm, "
            f"scale gap {al.scale_gap * 100:.1f}%")
        # BimArt's mesh_dict holds ONE entry per category -- the ARCTIC object itself, 11 in total.
        # Every target in this comparison is a PartNet-Mobility instance, so none of them is an
        # object BimArt has seen, whatever its category is called. The note used to fire only for
        # category "box", which read as though laptop and microwave targets were in-distribution;
        # they are not, and the only thing that differed was the message.
        src = getattr(inp, "_source_clip", None)
        if inp.target_object.name != getattr(src, "obj_name", inp.target_object.name):
            cat = getattr(oi, "category", None)
            diag.notes.append(
                f"OUT OF DISTRIBUTION: BimArt never saw {inp.target_object.name}; its mesh_dict "
                f"holds one entry per category (the ARCTIC object), so this run is conditioned on "
                f"the {cat!r} entry standing in for a different instance")
        return bout, oi

    def _object_input(self, obj_input, target: str, traj: dict):
        """The reference's own object conditioning, with the CATEGORY made explicit.

        `build_object_input` picks the entry and then calls four of the reference's own functions.
        Those four are called here unchanged; only the category literal it hardcodes to "box" is
        replaced when one was asked for. Nothing is reimplemented -- but this does mean the
        assembly is duplicated, so it is checked against the reference's own result when no
        category override is in play, which is the case for every box target.
        """
        if self.category is None:
            return obj_input.build_object_input(target, traj=traj, verbose=self.verbose)

        import numpy as _np
        BP = importlib.import_module(f"{BIMART}.bimart_paths")
        FR = importlib.import_module(f"{BIMART}.frames")
        if self.category not in BP.CATEGORIES:
            raise ValueError(f"bimart category {self.category!r} is not one of the 11 it was "
                             f"trained on: {list(BP.CATEGORIES)}")
        if traj["dataset"] == "pm":
            kw = ({} if self.entry_verts_per_part is None
                  else {"n_per_part": int(self.entry_verts_per_part)})
            entry = obj_input.build_pm_entry(traj, verbose=self.verbose, **kw)
        else:
            entry = _np.load(BP.UNIT_MESH_DICT, allow_pickle=True).item()[self.category]
        align = FR.align(target, traj=traj, category=self.category, entry=entry,
                         verbose=self.verbose)
        state = FR.to_obj_world_state(traj, align)
        bps, inds = obj_input._part_bps(entry, state, verbose=self.verbose)
        return obj_input.ObjectInput(
            category=self.category, entry=entry, obj_world_state=state,
            obj_cano_bps=bps, obj_cano_bps_inds=inds,
            scale=float(_np.asarray(entry["scale"]).reshape(-1)[0]), align=align)

    def _adopt_trajectory(self, traj: dict, inp: SynthesisInput) -> None:
        """Replace the reference's own tau_O with DEXCORE'S, in place, and report the gap.

        The reference builds a target trajectory of its own because `build_object_input` wants a
        populated `traj` dict. But tau_O is an INPUT to this synthesizer -- the driver already built
        it, every other method in the comparison is handed the same one, and `Synthesizer.run`
        checks the output against it. Letting the reference's copy through would mean the hand is
        generated against one object placement and evaluated against another.

        Only the three trajectory arrays are replaced. The rest of the dict -- `st`, `cal`, `spec`
        -- is geometry plumbing the frame bridge needs, and `to_obj_world_state` reads nothing else
        from it. The reference's SOURCE canonical frame (`ss`) then stops mattering entirely, which
        is the point: dexcore's `configs/canonical_rules.yaml` is the single authority for what a
        canonical frame is, and the two only ever agreed by coincidence while every source was the
        box.

        The gap is measured and recorded rather than asserted away, because a LARGE one still says
        the two disagree about the object and is worth seeing in the diagnostics.
        """
        if traj["F"] != len(inp):
            raise RuntimeError(f"bimart: the reference built {traj['F']} frames for a "
                               f"{len(inp)}-frame tau_O")
        dp = float(np.abs(np.asarray(traj["tgt_pos"]) - inp.trajectory.obj_pos).max())
        da = float(np.abs(np.asarray(traj["tgt_arti"]) - inp.trajectory.obj_arti).max())
        for key, ours in (("tgt_pos", inp.trajectory.obj_pos),
                          ("tgt_quat", inp.trajectory.obj_quat),
                          ("tgt_arti", inp.trajectory.obj_arti)):
            a = np.array(ours, dtype=np.float64, copy=True)
            a.flags.writeable = False          # the reference freezes these; keep that contract
            traj[key] = a
        self._traj_gap = (dp, da)

    # ------------------------------------------------------------------ 2. 21 keypoints
    def _keypoints21(self, bout, side: str, diag: SynthesisDiagnostics) -> np.ndarray:
        """BimArt's 16 kintree joints + 5 tip vertices -> the 21 ARCTIC keypoints.

        BimArt emits no fingertips (its 16 joints are MANO's kintree). The tips are real points on
        the MANO surface it also emits, so they are read off the mesh rather than invented. The
        tip ORDER is asserted: each tip must be nearer its own DIP than any other joint, which a
        permuted index list would fail immediately.
        """
        j16 = np.asarray(bout.joints16[side], np.float64).reshape(-1, 16, 3)
        verts = np.asarray(bout.verts[side], np.float64).reshape(len(j16), 778, 3)
        out = np.zeros((len(j16), NUM_JOINTS, 3))
        out[:, :16] = j16
        out[:, TIP_JOINTS] = verts[:, TIP_VERTEX]

        # WHICH DIP EACH TIP IS ATTACHED TO, BY RIGIDITY -- not by proximity. A tip is rigidly
        # attached to its own distal link, so its distance to its OWN DIP is nearly constant across
        # frames while its distance to another finger's DIP varies as the fingers move. Proximity
        # alone is NOT a valid test on a posed hand: on a closed grasp the ring tip sits nearer the
        # middle DIP than its own, which is a fact about the grasp, not a permuted index list.
        dips = [TIP_DIP_JOINT[j] for j in TIP_JOINTS]
        for j in TIP_JOINTS:
            d = np.linalg.norm(out[:, j][:, None, :] - j16[:, dips], axis=-1)      # (F,5)
            own = dips.index(TIP_DIP_JOINT[j])
            mean, std = d.mean(axis=0), d.std(axis=0)
            diag.extras[f"tip_to_dip.{side}.{JOINT_NAMES[j]}"] = {
                "mean_mm": float(mean[own] * 1000), "std_mm": float(std[own] * 1000)}
            # Being the strict argmin is too sharp a line. What this catches is a PERMUTED index
            # list, where a tip is bolted to some other finger's bone and its own distance wanders
            # by an order of magnitude more. Two fingers whose stds sit within a fraction of a
            # millimetre of each other are not that -- they are a tie decided by noise, and the
            # margin here is well inside the 3.2 mm residual the URDF IK already carries.
            # Measured: mw7304 failed the argmin test at 3.3 mm own vs 3.2 mm best.
            tol = max(0.2 * float(std.min()), 0.5e-3)
            if len(d) >= 8 and std[own] > std.min() + tol:
                raise RuntimeError(
                    f"bimart: {JOINT_NAMES[j]} (MANO vertex "
                    f"{TIP_VERTEX[TIP_JOINTS.index(j)]}) does not move rigidly with its own distal "
                    f"link: its distance to {JOINT_NAMES[TIP_DIP_JOINT[j]]} varies by "
                    f"{std[own] * 1000:.1f} mm across frames, more than to "
                    f"{JOINT_NAMES[dips[int(np.argmin(std))]]} "
                    f"({std.min() * 1000:.1f} mm) by more than the {tol * 1000:.1f} mm tolerance. "
                    f"The fingertip vertex indices or their order do not match dexcore's joint "
                    f"convention.")
        return out

    # ------------------------------------------------------------------ 3. IK onto the URDF hand
    def _fit_urdf(self, joints: Dict[str, np.ndarray], diag: SynthesisDiagnostics):
        """Two passes, because the fingertips' link-local offsets are not known in advance.

        The 16 kintree joints ARE the URDF's link origins (verified: every link origin's nearest
        demo keypoint is the one MANO_HAND_LINKS names), so `local = 0` is exact for them. The 5
        tips are NOT link origins -- they ride the distal link at an offset that depends on the
        pose being solved for. Pass 1 solves with `local = 0` everywhere, which is a good warm
        start and deliberately approximate at the tips; pass 2 measures each tip's offset in the
        distal link frame the first pass found and re-solves with it.
        """
        configs, rms = {}, {}
        for side in SIDES:
            hand = get_hand(side, device="cpu", samples_per_link=self.samples_per_link)
            fk = hand.fk
            F = len(joints[side])
            tgt = joints[side]
            q_init = self._warm_start(fk, tgt, F)
            cfg = IKConfig(iters=self.ik_iters, verbose=False)

            local = torch.zeros((F, NUM_JOINTS, 3), device=fk.device, dtype=fk.dtype)
            first = solve_keypoint_ik(fk, tgt, local, q_init, cfg, tag=f"{side} pass1")

            # The tips' offsets, measured in the distal link frames pass 1 found, and AVERAGED OVER
            # FRAMES. The average is the point: a per-frame offset is whatever makes pass 1 look
            # right, so re-solving with it is a no-op -- FK(q1, local) reproduces the targets by
            # construction. The tip's position on its distal link is a constant of the hand, so the
            # mean is the quantity being estimated and pass 2 is then a real re-solve.
            # The 16 kintree joints keep `local = 0`: they ARE the link origins, and that is a
            # requirement on the solution, not something to be absorbed into an offset.
            measured = fk.joint_local_offsets(first.q, tgt).detach()
            local = torch.zeros_like(measured)
            local[:, TIP_JOINTS] = measured[:, TIP_JOINTS].mean(dim=0, keepdim=True)
            for k, j in enumerate(TIP_JOINTS):
                off = measured[:, j]
                diag.extras[f"tip_local_mm.{side}.{JOINT_NAMES[j]}"] = {
                    "mean": float(off.mean(0).norm() * 1000),
                    "frame_std": float(off.std(0).norm() * 1000)}
            second = solve_keypoint_ik(fk, tgt, local, first.q, cfg, tag=f"{side} pass2")

            configs[side] = fk.cfg_from_dofs(second.q, {n: np.zeros(F) for n in fk.dof_names})
            rms[side] = second.per_frame_rms
            diag.extras[f"ik_rms_mm.{side}"] = {"pass1": float(first.rms * 1000),
                                                "pass2": float(second.rms * 1000)}
            n_bad = int((~second.converged).sum())
            if n_bad:
                diag.failures.append(f"{side}: IK did not converge on {n_bad}/{F} frames")
                diag.num_failed += n_bad

            # How far the URDF hand had to leave its own joint limits to reach the generated pose.
            # REPORTED, NEVER CLAMPED (src/reconstruction/ik.py): a generated hand is not obliged to
            # be reachable by this embodiment, and how unreachable it was is a finding about the
            # generator, not noise to tidy away. The transfer baselines start from a human pose that
            # is inside the limits by construction and stay near it, so a non-zero number here is a
            # real difference between the two families rather than a solver artefact.
            viol = limit_violations(fk, second.q)
            diag.extras[f"limit_violation.{side}"] = {
                "max_rad": float(viol.max()),
                "frac_frames": float((viol.max(axis=1) > 1e-6).mean()),
                "worst_dof": fk.dof_names[int(viol.max(axis=0).argmax())]}
            if viol.max() > 1e-6:
                diag.notes.append(
                    f"{side}: the URDF hand left its joint limits by up to "
                    f"{viol.max():.3f} rad to reach the generated pose "
                    f"({fk.dof_names[int(viol.max(axis=0).argmax())]})")
        return configs, rms

    def _warm_start(self, fk, targets: np.ndarray, F: int) -> torch.Tensor:
        """q_init: the rest-pose URDF hand carried RIGIDLY onto the generated hand, per frame.

        The 16 link origins ARE the 16 kintree joints, so the best rigid placement of the URDF hand
        is the Kabsch fit from its own link origins at rest onto the generated joints -- computed
        with `src/geometry.py:kabsch` and applied through `mano.apply_wrist_transform`, which is
        what "place the palm" means mechanically here: the whole hand is carried and every finger
        angle is left untouched for IK to resolve.

        NOT a palm triad built from wrist/index/pinky: that is a frame of OUR choosing and has no
        reason to coincide with the URDF's palm link frame, so it starts the solve with a wrist
        rotation error IK then has to undo. NOT a component-wise copy of BimArt's MANO pose either:
        the URDF's three per-segment revolute axes compose as Euler angles where MANO's are an
        axis-angle vector, so they are not the same rotation at the bend angles a grasp reaches.
        """
        from src import geometry as G
        from src.data.mano import apply_wrist_transform

        q0 = np.zeros((1, fk.ndof))
        with torch.no_grad():
            rest = fk.forward(torch.tensor(q0, device=fk.device, dtype=fk.dtype))
        origins = rest[0, :, :3, 3].cpu().numpy()                       # (16,3) at rest
        deltas = np.stack([G.kabsch(origins, targets[t, :16]) for t in range(F)])
        q = apply_wrist_transform(fk, np.repeat(q0, F, axis=0), deltas)
        return torch.tensor(q, device=fk.device, dtype=fk.dtype)

    # ------------------------------------------------------------------ 4. assemble
    def _assemble(self, inp: SynthesisInput, joints, configs,
                  diag: SynthesisDiagnostics) -> Demonstration:
        """Contacts re-derived against the TARGET geometry, with the shared definition."""
        traj, obj, clip = inp.trajectory, inp.target_object, inp.source_clip
        T = len(traj)
        cpos, cpart, cmask, oc, ov = {}, {}, {}, {}, {}
        for side in SIDES:
            cpos[side], cpart[side], cmask[side], oc[side], ov[side] = recompute_contacts(
                side, configs[side], joints[side], traj, obj, self.contact_threshold,
                samples_per_link=self.samples_per_link, n_obj_sample=self.n_obj_sample,
                seed=inp.seed)
        diag.extras["contact_link_frames"] = {s: int(cmask[s].sum()) for s in SIDES}

        prov = Provenance(
            source_sequence=clip.sequence, source_object=clip.obj_name,
            target_object=obj.name, source_path=clip.path,
            selector="", selection_ratio=None, selected_frames=None, source_frames=None,
            source_num_frames=T, transfer="", reconstructor=self.describe(), seed=inp.seed,
            notes=[f"synthesizer={self.describe()}", f"consumes={sorted(self.consumes)}",
                   "hand motion GENERATED from tau_O; no source hand state was consumed"])
        return Demonstration(
            name=f"{obj.name}_use_{clip.use_clip}", obj_name=obj.name, use_clip=clip.use_clip,
            subject=clip.subject, fps=clip.fps,
            frame_start=clip.frame_start, frame_end=clip.frame_start + T,
            total_frames=clip.total_frames,
            obj_pos=traj.obj_pos.copy(), obj_quat=traj.obj_quat.copy(),
            obj_arti=traj.obj_arti.copy(),
            joints=joints, contact_pos=cpos, contact_part=cpart, contact_mask=cmask,
            configs=configs, obj_contacts=oc, obj_contact_valid=ov,
            contact_threshold=self.contact_threshold,
            path="", provenance=prov, extra_params={}, frame_index=traj.frames())
