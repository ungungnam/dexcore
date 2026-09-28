"""The synthesizer contract: the level at which a staged pipeline and an end-to-end model meet.

The important test here is `TestEndToEndContract`. It defines a synthesizer that consumes ONLY
tau_O -- no selection, no transfer, no reconstruction, and no access to the human hand states --
and drives it through the same `pipeline.run` the staged method uses. That is the path BimArt will
take, and it is checked here without BimArt, so a failure means the interface is wrong rather than
that a third-party checkout is missing.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import paths
from src.data.demo import Demonstration
from src.paths import SIDES
from src.synthesis.base import (SourceClip, SynthesisDiagnostics, SynthesisInput, SynthesisResult,
                                Synthesizer, register)


def assets_available() -> bool:
    try:
        return paths.demo_path("box", "01", "s01").exists() and paths.mano_urdf("right").exists()
    except Exception:
        return False


needs_assets = unittest.skipUnless(assets_available(), "DexMachina assets not available")


# ------------------------------------------------------------------ a tau_O-only synthesizer
def _demo_from_trajectory(traj, clip: SourceClip, name_obj: str) -> Demonstration:
    """The minimum a method must build: a dense demo carrying tau_O and SOME hand state."""
    T = len(traj)
    zeros3 = {s: np.zeros((T, 21, 3)) for s in SIDES}
    return Demonstration(
        name=f"{name_obj}_use_{clip.use_clip}", obj_name=name_obj, use_clip=clip.use_clip,
        subject=clip.subject, fps=clip.fps,
        frame_start=clip.frame_start, frame_end=clip.frame_start + T,
        total_frames=clip.total_frames,
        obj_pos=traj.obj_pos.copy(), obj_quat=traj.obj_quat.copy(), obj_arti=traj.obj_arti.copy(),
        joints=zeros3,
        contact_pos={s: np.zeros((T, 16, 3)) for s in SIDES},
        contact_part={s: np.zeros((T, 16), dtype=np.int8) for s in SIDES},
        contact_mask={s: np.zeros((T, 16), dtype=bool) for s in SIDES},
        configs={s: {} for s in SIDES}, frame_index=traj.frames())


@register
class _TauOnlySynthesizer(Synthesizer):
    """Stands in for an end-to-end generative model: tau_O in, a dense demonstration out."""

    name = "_test_tau_only"
    consumes = frozenset({"source_clip"})       # names and frame numbers only; NO hand states

    def __init__(self, break_contract: str = "", **kw):
        super().__init__(**kw)
        self.break_contract = break_contract

    def synthesize(self, inp: SynthesisInput) -> SynthesisResult:
        demo = _demo_from_trajectory(inp.trajectory, inp.source_clip, inp.target_object.name)
        if self.break_contract == "length":
            demo = demo.subset(list(range(len(inp) - 1)))
        elif self.break_contract == "object":
            demo.obj_name = "not_the_target"
        elif self.break_contract == "trajectory":
            demo.obj_pos = demo.obj_pos + 0.01
        return SynthesisResult(
            demo=demo, trajectory=inp.trajectory,
            diagnostics=SynthesisDiagnostics(method=self.name),
            method=self.name, consumes=self.consumes, hand_states_used=0)


# --------------------------------------------------------------------------- input withholding
class TestInputWithholding(unittest.TestCase):
    def make(self, allowed):
        return SynthesisInput(trajectory=None, target_object=None, _source_object="OBJ",
                              _source_clip="CLIP", _source_demo="DEMO",
                              allowed=frozenset(allowed))

    def test_declared_inputs_are_available(self):
        inp = self.make({"source_clip", "source_object", "source_hand"})
        self.assertEqual(inp.source_clip, "CLIP")
        self.assertEqual(inp.source_object, "OBJ")
        self.assertEqual(inp.source_demo, "DEMO")

    def test_undeclared_hand_states_are_refused(self):
        """The claim 'this method never saw the human hand motion' must be enforced, not documented."""
        inp = self.make({"source_clip"})
        with self.assertRaises(AttributeError) as cm:
            inp.source_demo
        self.assertIn("source_hand", str(cm.exception))

    def test_undeclared_source_object_is_refused(self):
        with self.assertRaises(AttributeError):
            self.make({"source_clip"}).source_object

    def test_unknown_declaration_is_rejected_at_class_definition(self):
        with self.assertRaises(TypeError):
            class _Bad(Synthesizer):
                name = "_bad"
                consumes = frozenset({"the_answer"})

                def synthesize(self, inp):
                    ...


# ------------------------------------------------------------------------------- the registry
class TestRegistry(unittest.TestCase):
    def test_staged_is_registered_and_declares_hand_states(self):
        from src.synthesis.base import available, build
        self.assertIn("staged", available())
        self.assertIn("source_hand", build("staged").consumes)


# ------------------------------------------------------------------------ the common contract
@needs_assets
class TestEndToEndContract(unittest.TestCase):
    """A method with no stages must travel the same driver and be checked the same way."""

    def config(self, **over):
        from src.config import PipelineConfig
        return PipelineConfig(obj_name="box", use_clip="01", frame_end=60,
                              synthesizer="_test_tau_only", **over)

    def test_a_tau_only_method_runs_through_the_driver(self):
        from src import pipeline
        out = pipeline.run(self.config(), verbose=False)
        self.assertEqual(len(out.demo), len(out.trajectory))
        self.assertEqual(out.hand_states_used, 0)
        self.assertEqual(out.budget, 0.0)
        self.assertTrue(np.array_equal(out.demo.obj_pos, out.trajectory.obj_pos))

    def test_staged_intermediates_are_refused_with_a_reason(self):
        from src import pipeline
        out = pipeline.run(self.config(), verbose=False)
        with self.assertRaises(AttributeError) as cm:
            out.sparse_demo
        self.assertIn("no such stage", str(cm.exception))

    def test_output_length_must_match_tau_O(self):
        from src import pipeline
        with self.assertRaises(RuntimeError) as cm:
            pipeline.run(self.config(synthesizer_options={"break_contract": "length"}),
                         verbose=False)
        self.assertIn("must answer EVERY frame", str(cm.exception))

    def test_output_must_be_named_for_the_target_object(self):
        from src import pipeline
        with self.assertRaises(RuntimeError) as cm:
            pipeline.run(self.config(synthesizer_options={"break_contract": "object"}),
                         verbose=False)
        self.assertIn("wrong mesh", str(cm.exception))

    def test_synthesis_must_not_move_the_object_trajectory(self):
        from src import pipeline
        with self.assertRaises(RuntimeError) as cm:
            pipeline.run(self.config(synthesizer_options={"break_contract": "trajectory"}),
                         verbose=False)
        self.assertIn("object trajectory is an INPUT", str(cm.exception))

    def test_a_stageless_config_refuses_stage_settings(self):
        """A selector named next to a method that has none would be silently ignored."""
        with self.assertRaises(ValueError) as cm:
            self.config(selector="reconstruction_aware").validate()
        self.assertIn("has no stages", str(cm.exception))


# ------------------------------------------------------------------------------ the budget axis
@needs_assets
class TestBudgetIsReported(unittest.TestCase):
    def test_staged_reports_its_selection_budget(self):
        from src import pipeline
        from src.config import PipelineConfig
        out = pipeline.run(PipelineConfig(
            obj_name="box", use_clip="01", frame_end=90, selector_options={"ratio": 0.1},
            reconstructor_options={"ik_iters": 20}), verbose=False)
        self.assertEqual(out.hand_states_used, len(out.sparse_demo))
        self.assertAlmostEqual(out.budget, len(out.sparse_demo) / len(out.trajectory))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------- the bimart adapter
class TestBimArtConstants(unittest.TestCase):
    """Checks that need no reference tree and no GPU -- the conventions the adapter asserts."""

    def test_tip_joints_match_the_link_table(self):
        """TIP_JOINTS / TIP_DIP_JOINT must be derived from MANO_HAND_LINKS, not written twice."""
        from src.data.mano import MANO_HAND_LINKS
        from src.synthesis.bimart import TIP_DIP_JOINT, TIP_JOINTS, TIP_VERTEX
        spans = {idxs[1]: idxs[0] for name, idxs in MANO_HAND_LINKS if name.endswith("3")}
        self.assertEqual(sorted(TIP_JOINTS), sorted(spans))
        self.assertEqual(TIP_DIP_JOINT, spans)
        self.assertEqual(len(TIP_VERTEX), len(TIP_JOINTS))

    def test_tip_vertex_order_follows_joint_names(self):
        """The 5 MANO tip vertices are listed in dexcore's joint order, thumb first."""
        from src.data.mano import JOINT_NAMES
        from src.synthesis.bimart import TIP_JOINTS
        self.assertEqual([JOINT_NAMES[j] for j in TIP_JOINTS],
                         ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"])

    def test_it_declares_no_access_to_the_human_hand(self):
        from src.synthesis.base import build
        b = build("bimart")
        self.assertNotIn("source_hand", b.consumes)

    def test_the_file_handed_to_the_reference_carries_no_hand_state(self):
        """The claim is enforced by the INPUT, not by trusting the reference to look away."""
        import tempfile
        from src.data.trajectory import ObjectTrajectory
        from src.synthesis.base import SynthesisInput
        from src.synthesis.bimart import _tau_only_demo
        T = 5
        traj = ObjectTrajectory(obj_name="box", obj_pos=np.zeros((T, 3)),
                                obj_quat=np.tile([1.0, 0, 0, 0], (T, 1)), obj_arti=np.zeros(T))
        inp = SynthesisInput(trajectory=traj, target_object=None, source_trajectory=traj)
        with tempfile.TemporaryDirectory() as td:
            p = _tau_only_demo(inp, Path(td) / "tau.npy")
            raw = np.load(p, allow_pickle=True).item()
        self.assertEqual(sorted(raw["params"]), ["obj_arti", "obj_quat", "obj_trans"])
        self.assertEqual(raw["world_coord"], {})
        self.assertNotIn("configs_left", raw)
        self.assertNotIn("configs_right", raw)


# ------------------------------------------------------------------ the DexMachina export path
@needs_assets
class TestDexMachinaExport(unittest.TestCase):
    """Installing a generated demo where training will actually read it.

    Both properties checked here were broken and produced a file nothing reads: the destination
    doubled the subject (`.../processed/s01/s01/...`), and without an explicit root it resolved to
    whichever asset root won the search -- possibly the third_party checkout this repository
    refuses to write into.
    """

    def demo(self):
        from src.data.demo import Demonstration
        return Demonstration.load(obj_name="box", use_clip="01", frame_end=40)

    def test_installs_directly_under_the_named_root(self):
        import tempfile
        from eval.downstream.dexmachina import export
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = export(self.demo(), use_clip="pm189dx", subject="s01", dest_root=root)
            self.assertEqual(r.installed_path.parent, root,
                             "the subject must not be appended twice")
            self.assertEqual(r.installed_path.name, "box_use_pm189dx.npy")
            self.assertTrue(r.installed_path.exists())

    def test_default_root_is_repo_local_never_a_checkout(self):
        from src import paths
        from eval.downstream.dexmachina import export
        r = export(self.demo(), use_clip="_probe", dry_run=True)
        self.assertTrue(str(r.installed_path).startswith(str(paths.REPO_ROOT)),
                        f"default install went outside the repo: {r.installed_path}")

    def test_clip_string_carries_the_demo_own_window(self):
        """A clip string with the wrong range trains on a different segment, silently."""
        from src.data.demo import Demonstration
        from eval.downstream.dexmachina import export
        d = Demonstration.load(obj_name="box", use_clip="01", frame_start=0, frame_end=240)
        r = export(d, use_clip="pm189dx", dry_run=True)
        self.assertEqual(r.frame_start, 0)
        self.assertEqual(r.frame_end, 240)
        self.assertTrue(r.clip_string.endswith("-0-240-s01-upm189dx"), r.clip_string)


# ------------------------------------------------------- cordex: whose canonical frame is it
@needs_assets
class TestCanonicalFrameOverride(unittest.TestCase):
    """`_canon_frame_from_dexcore` makes the reference canonicalise the TARGET in dexcore's frame.

    Without it the reference derives its own frame, dexcore derives another for any category that
    declares a non-default rule, and cordex's `_finish` refuses the run because the hand would be
    posed against one placement and reconstructed against a different one.
    """

    def _states(self, name):
        from dexmachina.eval.experiments.exp2_demo_transfer.annotation_tool import (
            asset_loader as AL, annotation_state as AS)
        from src.data.object import get_object
        from src.transfer.cordex import _canon_frame_from_dexcore
        obj = get_object(name)
        d = str(paths.object_dir(name))
        plain = AS.AnnotationState(AL.load_object("arctic", d))
        with _canon_frame_from_dexcore(obj):
            overridden = AS.AnnotationState(AL.load_object("arctic", d))
        return obj, plain, overridden

    def test_noop_where_the_rules_agree(self):
        """Every reference produced so far is a default-rule box target. Bit-equality, not closeness:
        anything else would silently stop those results from being reproducible."""
        for name in ("pm102379_calibrated", "pm100243_calibrated", "pm100141_calibrated"):
            with self.subTest(object=name):
                try:
                    obj, plain, overridden = self._states(name)
                except FileNotFoundError:
                    self.skipTest(f"{name} is not installed in this asset path")
                self.assertEqual(obj.canonical_rule, "lid_hint")
                self.assertTrue(np.array_equal(overridden.canon_R, plain.canon_R))
                self.assertTrue(np.array_equal(overridden.canon_origin, plain.canon_origin))

    def test_applies_dexcores_rotation_on_a_declared_target(self):
        try:
            obj, plain, overridden = self._states("lap10239_calibrated")
        except FileNotFoundError:
            self.skipTest("lap10239_calibrated is not installed in this asset path")
        self.assertEqual(obj.canonical_rule, "base_normal")
        self.assertTrue(np.allclose(overridden.canon_R, obj.canonical_rotation(), atol=1e-12))
        # the reference really did disagree -- otherwise the test proves nothing
        self.assertFalse(np.allclose(plain.canon_R, obj.canonical_rotation(), atol=1e-6))

    def test_the_patch_is_restored_after_the_block(self):
        """A leaked override would silently re-frame every LATER transfer in the same process."""
        from dexmachina.eval.experiments.exp2_demo_transfer.annotation_tool import (
            asset_loader as AL, annotation_state as AS)
        from src.data.object import get_object
        from src.transfer.cordex import _canon_frame_from_dexcore
        before = AS.AnnotationState.seed_frame_from_metadata
        try:
            with _canon_frame_from_dexcore(get_object("lap10239_calibrated")):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        except FileNotFoundError:
            self.skipTest("lap10239_calibrated is not installed in this asset path")
        self.assertIs(AS.AnnotationState.seed_frame_from_metadata, before)
