"""dexcore tests. Plain `unittest`, so no test-runner dependency is needed:

    python -m unittest discover -s tests -v
    python tests/test_dexcore.py

The integration tests need the DexMachina asset tree (see src/paths.py). They SKIP rather than fail
when it is absent, so the pure-logic tests still run anywhere; a skip is reported, not hidden.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import geometry as G
from src import paths


def assets_available() -> bool:
    try:
        return paths.demo_path("box", "01", "s01").exists() and paths.mano_urdf("right").exists()
    except Exception:
        return False


def other_objects(n: int):
    """`n` objects that are NOT the box, from whatever asset roots are configured.

    Named rather than hard-coded because the asset search path is configurable: the upstream tree
    ships box/ketchup/laptop/..., while locally baked targets (box_s110, pm100141_calibrated) only
    exist on a machine that has built them. A test that names one of those passes or fails on where
    the assets happen to be, which is not what it is trying to measure.
    """
    from src import paths
    names = [o for o in paths.list_objects() if o != "box"]
    if len(names) < n:
        raise unittest.SkipTest(f"need {n} non-box objects, found {names}")
    return names[:n]


HAVE_ASSETS = assets_available()
needs_assets = unittest.skipUnless(HAVE_ASSETS, "DexMachina assets not available")


# --------------------------------------------------------------------------- coordinate transforms
class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def random_quat(self, n=None):
        return G.normalize(self.rng.normal(size=(4,) if n is None else (n, 4)))

    def test_quat_matrix_roundtrip(self):
        for _ in range(100):
            q = self.random_quat()
            R = G.wxyz_to_R(q)
            self.assertTrue(np.allclose(G.wxyz_to_R(G.R_to_wxyz(R)), R, atol=1e-10))

    def test_rotation_is_orthonormal(self):
        R = G.wxyz_to_R(self.random_quat(50))
        self.assertTrue(np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-10))
        self.assertTrue(np.allclose(np.linalg.det(R), 1.0, atol=1e-10))

    def test_transform_inverse_roundtrip(self):
        T = G.affine(G.wxyz_to_R(self.random_quat()), self.rng.normal(size=3))
        p = self.rng.normal(size=(11, 3))
        self.assertTrue(np.allclose(G.transform_points(G.transform_points(p, T), G.invert(T)),
                                    p, atol=1e-12))

    def test_unroll_quat_removes_sign_flips(self):
        q = np.tile(np.array([1.0, 0, 0, 0]), (10, 1))
        q[3:7] *= -1                                     # a double-cover flip
        u = G.unroll_quat(q)
        self.assertTrue(np.all(np.sum(u[1:] * u[:-1], axis=1) > 0))
        # flipping is exact: the ROTATION is unchanged
        self.assertTrue(np.allclose(G.wxyz_to_R(u), G.wxyz_to_R(q), atol=1e-12))

    def test_slerp_endpoints(self):
        q0, q1 = self.random_quat(), self.random_quat()
        self.assertTrue(np.allclose(G.wxyz_to_R(G.slerp(q0, q1, 0.0)), G.wxyz_to_R(q0), atol=1e-10))
        self.assertTrue(np.allclose(G.wxyz_to_R(G.slerp(q0, q1, 1.0)), G.wxyz_to_R(q1), atol=1e-10))


# ------------------------------------------------------------------------------ interpolation
class TestInterpolation(unittest.TestCase):
    def setUp(self):
        from src.reconstruction import interpolate as I
        self.I = I

    def test_endpoints_are_preserved_exactly(self):
        kf = np.array([0, 10, 20])
        t = np.arange(21, dtype=float)
        vals = np.array([[[1.0, 2, 3]], [[4.0, 5, 6]], [[7.0, 8, 9]]])
        eye = np.tile(np.eye(4), (21, 1, 1))
        out = self.I.interpolate_keypoints(vals, eye[kf], eye, kf, t, frame="world")
        for i, k in enumerate(kf):
            self.assertTrue(np.array_equal(out[k], vals[i]),
                            f"keyframe {k} not reproduced exactly")

    def test_refuses_to_extrapolate(self):
        with self.assertRaises(ValueError):
            self.I.bracket(np.array([0, 10]), np.array([11.0]))
        with self.assertRaises(ValueError):
            self.I.bracket(np.array([2, 10]), np.array([1.0]))

    def test_alpha_is_linear(self):
        lo, hi, a = self.I.bracket(np.array([0, 10]), np.array([0.0, 2.5, 5.0, 10.0]))
        self.assertTrue(np.allclose(a, [0.0, 0.25, 0.5, 0.0]))   # 10.0 lands exactly on a keyframe

    def test_object_relative_beats_world_for_a_static_grasp(self):
        """A keypoint fixed in the object frame must be reconstructed exactly in that frame."""
        kf = np.array([0, 20])
        t = np.arange(21, dtype=float)
        T = np.stack([G.affine(G.rodrigues([0, 0, 1], th), [th * 0.1, 0, 0])
                      for th in np.linspace(0, 1.5, 21)])
        rel = np.array([[[0.05, 0.02, 0.01]]])
        kp_key = G.transform_points(np.repeat(rel, len(kf), axis=0), T[kf])
        exact = G.transform_points(np.repeat(rel, 21, axis=0), T)
        root = self.I.interpolate_keypoints(kp_key, T[kf], T, kf, t, frame="root")
        world = self.I.interpolate_keypoints(kp_key, T[kf], T, kf, t, frame="world")
        self.assertLess(np.abs(root - exact).max(), 1e-12)
        self.assertGreater(np.abs(world - exact).max(), 1e-3)


# --------------------------------------------------------------------------------- selection
@needs_assets
class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data.demo import Demonstration
        import src.pipeline  # noqa: F401  populates the registries
        cls.demo = Demonstration.load(obj_name="box", use_clip="01", frame_end=200)

    def test_uniform_budget_and_endpoints(self):
        from src.selection.base import build
        for ratio in (1.0, 0.25, 0.1, 0.05, 0.01):
            s = build("uniform", ratio=ratio).select(self.demo)
            n = len(s)
            self.assertGreaterEqual(n, 2)
            self.assertLessEqual(n, len(self.demo))
            self.assertAlmostEqual(n / len(self.demo), ratio, delta=0.02)
            self.assertEqual(s.frames()[0], 0)
            self.assertEqual(s.frames()[-1], len(self.demo) - 1)

    def test_frame_indices_are_sorted_and_unique(self):
        from src.selection.base import build
        for name in ("uniform", "reconstruction_aware"):
            f = build(name, ratio=0.1).select(self.demo).frames()
            self.assertTrue(np.all(np.diff(f) > 0), f"{name} produced unsorted/duplicate frames")

    def test_all_selector_keeps_everything(self):
        from src.selection.base import build
        s = build("all").select(self.demo)
        self.assertEqual(len(s), len(self.demo))
        self.assertFalse(s.is_sparse)

    def test_num_frames_budget(self):
        from src.selection.base import build
        s = build("uniform", num_frames=17).select(self.demo)
        self.assertEqual(len(s), 17)

    def test_ratio_and_num_frames_are_mutually_exclusive(self):
        from src.selection.base import SelectionConfig
        with self.assertRaises(ValueError):
            SelectionConfig(ratio=0.1, num_frames=10).validate()

    def test_reconstruction_aware_beats_uniform_at_equal_budget(self):
        from src.selection.base import build
        from src.selection.reconstruction_aware import keypoint_criterion
        u = build("uniform", ratio=0.1).select(self.demo).frames()
        r = build("reconstruction_aware", ratio=0.1).select(self.demo).frames()
        self.assertLess(keypoint_criterion(self.demo, r).mean(),
                        keypoint_criterion(self.demo, u).mean())


# ------------------------------------------------------------------------ demo representation
@needs_assets
class TestDemonstration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data.demo import Demonstration
        cls.demo = Demonstration.load(obj_name="box", use_clip="01", frame_end=120)

    def test_object_relative_world_roundtrip(self):
        kp = self.demo.joints["right"]
        back = self.demo.object_to_world(self.demo.world_to_object(kp))
        self.assertLess(np.abs(back - kp).max(), 1e-12)

    def test_save_load_roundtrip_is_lossless(self):
        from src.data.demo import Demonstration
        with tempfile.TemporaryDirectory() as td:
            p = self.demo.save(Path(td) / "rt_use_01.npy")
            back = Demonstration.load(path=str(p))
            self.assertEqual(len(back), len(self.demo))
            for f in ("obj_pos", "obj_quat", "obj_arti"):
                self.assertTrue(np.array_equal(getattr(back, f), getattr(self.demo, f)), f)
            for side in ("left", "right"):
                self.assertTrue(np.array_equal(back.joints[side], self.demo.joints[side]))
                self.assertTrue(np.array_equal(back.contact_mask[side],
                                               self.demo.contact_mask[side]))
                for k in self.demo.configs[side]:
                    self.assertTrue(np.array_equal(back.configs[side][k],
                                                   self.demo.configs[side][k]), k)

    def test_contact_zero_row_means_no_contact(self):
        for side in ("left", "right"):
            pos, mask = self.demo.contact_pos[side], self.demo.contact_mask[side]
            self.assertTrue(np.all(np.linalg.norm(pos[~mask], axis=-1) == 0))
            self.assertTrue(np.all(self.demo.contact_part[side][~mask] == 0))

    def test_subset_records_metadata(self):
        rows = [0, 5, 11, 60, 119]
        s = self.demo.subset(rows, selector="test", ratio=len(rows) / len(self.demo))
        self.assertEqual(len(s), len(rows))
        self.assertEqual(list(s.frames()), rows)
        self.assertEqual(s.provenance.selected_frames, rows)
        self.assertEqual(s.provenance.source_num_frames, len(self.demo))
        self.assertEqual(s.provenance.selector, "test")
        self.assertTrue(s.is_sparse)

    def test_subset_bounds_are_checked(self):
        with self.assertRaises(IndexError):
            self.demo.subset([0, 10_000])


# ---------------------------------------------------------------------------------- transfer
@needs_assets
class TestIdentityTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data.demo import Demonstration
        from src.data.object import get_object
        from src.data.target import build_target_trajectory
        import src.pipeline  # noqa: F401
        cls.demo = Demonstration.load(obj_name="box", use_clip="01", frame_end=120)
        cls.box = get_object("box")
        cls.tau = build_target_trajectory(cls.demo, cls.box)

    def sparse(self):
        from src.selection.base import build
        return build("uniform", ratio=0.1).select(self.demo)

    def test_identity_changes_nothing(self):
        from src.transfer.base import build
        s = self.sparse()
        r = build("identity").transfer(s, self.box, self.box, self.tau)
        self.assertTrue(r.diagnostics.all_ok)
        self.assertEqual(len(r.demo), len(s))
        for side in ("left", "right"):
            self.assertTrue(np.array_equal(r.demo.joints[side], s.joints[side]))
        self.assertTrue(np.array_equal(r.trajectory.obj_pos, self.demo.obj_pos))

    def test_identity_refuses_a_different_object(self):
        """A scaled object is a DIFFERENT object; doing nothing to it is not a transfer."""
        from src.data.object import get_object
        from src.data.trajectory import ObjectTrajectory
        from src.transfer.base import build
        name = other_objects(1)[0]
        other = get_object(name)
        tau = ObjectTrajectory(obj_name=name, obj_pos=self.demo.obj_pos,
                               obj_quat=self.demo.obj_quat, obj_arti=self.demo.obj_arti)
        with self.assertRaises(ValueError):
            build("identity").transfer(self.sparse(), self.box, other, tau)

    def test_identity_allows_an_explicit_noop_baseline(self):
        from src.data.object import get_object
        from src.data.trajectory import ObjectTrajectory
        from src.transfer.base import build
        name = other_objects(1)[0]
        other = get_object(name)
        tau = ObjectTrajectory(obj_name=name, obj_pos=self.demo.obj_pos,
                               obj_quat=self.demo.obj_quat, obj_arti=self.demo.obj_arti)
        r = build("identity", allow_different_object=True).transfer(
            self.sparse(), self.box, other, tau)
        self.assertEqual(r.demo.provenance.target_object, name)

    def test_cross_object_trajectory_preserves_articulation_phase(self):
        """The canonical transfer must remap the joint by PHASE, not copy the angle.

        pm100141's joint range is different from the box's, so a correct transfer produces
        different angles at the same instants while keeping the timing identical -- correlation 1,
        range mapped onto the target's own limits. Copying the source angle would score correlation
        1 too, which is why the range is checked as well.
        """
        from src.data.object import get_object
        from src.data.target import build_target_trajectory
        for name in other_objects(2):
            tgt = get_object(name)
            traj = build_target_trajectory(self.demo, tgt, self.box)
            self.assertEqual(len(traj), len(self.demo))
            self.assertEqual(traj.obj_name, name)
            corr = np.corrcoef(self.demo.obj_arti, traj.obj_arti)[0, 1]
            self.assertGreater(corr, 0.999, f"{name}: articulation timing changed")
            lo, hi = sorted((tgt.q_closed, tgt.q_open))
            self.assertGreaterEqual(traj.obj_arti.min(), lo - 1e-6, f"{name} below its joint range")
            self.assertLessEqual(traj.obj_arti.max(), hi + 1e-6, f"{name} above its joint range")

    def test_same_object_trajectory_is_bit_exact(self):
        from src.data.target import build_target_trajectory
        traj = build_target_trajectory(self.demo, self.box, self.box)
        self.assertTrue(np.array_equal(traj.obj_pos, self.demo.obj_pos))
        self.assertTrue(np.array_equal(traj.obj_arti, self.demo.obj_arti))

    def test_all_transfer_operators_are_registered(self):
        """Adding an operator must not require touching the pipeline; the registry is the contract."""
        from src.transfer.base import available
        for name in ("identity", "palm_first_finger", "cordex", "shapegen"):
            self.assertIn(name, available())


# ------------------------------------------------------------------------------- integration
@needs_assets
class TestPipelineIntegration(unittest.TestCase):
    """End-to-end on a real ARCTIC sequence. Short window, few IK iters -- a gate, not a benchmark."""

    def run_pipeline(self, **over):
        from src.config import PipelineConfig
        from src import pipeline
        cfg = PipelineConfig(obj_name="box", use_clip="01", frame_end=90,
                             reconstructor_options={"ik_iters": 120}, **over)
        return pipeline.run(cfg, verbose=False)

    def test_output_length_matches_the_dense_trajectory(self):
        out = self.run_pipeline(selector_options={"ratio": 0.1})
        self.assertEqual(len(out.demo), len(out.source_demo))
        self.assertEqual(len(out.demo), len(out.trajectory))
        self.assertLess(len(out.sparse_demo), len(out.source_demo))

    def test_metadata_is_preserved_through_every_stage(self):
        out = self.run_pipeline(selector_options={"ratio": 0.1}, seed=7)
        p = out.demo.provenance
        self.assertEqual(p.source_sequence, "box_use_01")
        self.assertEqual(p.source_object, "box")
        self.assertEqual(p.target_object, "box")
        self.assertIn("uniform", p.selector)
        self.assertEqual(p.transfer, "identity")
        self.assertIn("linear_keypoint", p.reconstructor)
        self.assertEqual(p.seed, 7)
        self.assertEqual(len(p.selected_frames), len(out.sparse_demo))
        self.assertEqual(p.source_num_frames, len(out.source_demo))
        self.assertTrue(p.config)

    def test_object_trajectory_is_untouched_by_the_hand_stages(self):
        out = self.run_pipeline(selector_options={"ratio": 0.1})
        self.assertTrue(np.array_equal(out.demo.obj_pos, out.source_demo.obj_pos))
        self.assertTrue(np.array_equal(out.demo.obj_arti, out.source_demo.obj_arti))

    def test_all_identity_reconstruct_reproduces_the_source(self):
        """The correctness gate: the dense case must travel the pipeline and come back unchanged."""
        out = self.run_pipeline(selector="all", selector_options={})
        for side in ("left", "right"):
            err = np.abs(out.demo.joints[side] - out.source_demo.joints[side]).max()
            self.assertLess(err, 1e-3, f"{side} keypoints moved by {err*1000:.3f} mm")

    def test_generated_demo_is_readable_as_a_dexmachina_reference(self):
        from src.data.demo import Demonstration
        out = self.run_pipeline(selector_options={"ratio": 0.1})
        with tempfile.TemporaryDirectory() as td:
            p = out.demo.save(Path(td) / f"{out.demo.obj_name}_use_gen.npy")
            raw = np.load(p, allow_pickle=True).item()
            self.assertIn("params", raw)
            self.assertIn("world_coord", raw)
            for k in ("obj_trans", "obj_quat", "obj_arti"):
                self.assertIn(k, raw["params"])
            for side in ("left", "right"):
                self.assertIn(f"joints.{side}", raw["world_coord"])
                self.assertIn(f"contact_links_{side}", raw["world_coord"])
                self.assertEqual(len(raw[f"configs_{side}"]), 51)
            self.assertEqual(Demonstration.load(path=str(p)).num_frames, len(out.demo))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ------------------------------------------------------------------ canonical-frame rules
@needs_assets
class TestCanonicalRules(unittest.TestCase):
    """The +Z axis rule is DECLARED per object, never inferred from a flatness threshold.

    The property that matters is that adding an object to `configs/canonical_rules.yaml` cannot
    change an object that is not listed: the canonical frame is what every cross-object stage is
    expressed in, so a rule change reaching the wrong object silently rotates its whole trajectory.
    """

    def test_unlisted_objects_keep_the_default(self):
        from src.data.object import DEFAULT_CANONICAL_RULE, canonical_rule_for
        self.assertEqual(canonical_rule_for("box"), DEFAULT_CANONICAL_RULE)
        self.assertEqual(canonical_rule_for("_no_such_object_"), DEFAULT_CANONICAL_RULE)

    def test_the_rule_changes_the_frame_it_is_meant_to_change(self):
        """On a SLAB the two rules disagree, and base_normal is the one that finds the thickness."""
        from src.data.object import ArticulatedObject
        try:
            hint = ArticulatedObject.load("laptop", canonical_rule="lid_hint")
            norm = ArticulatedObject.load("laptop", canonical_rule="base_normal")
        except FileNotFoundError:
            raise unittest.SkipTest("laptop asset not available")
        e_norm = norm.canonical_extent()
        # the base's own thinnest direction must land on +Z, i.e. Z is the smallest canonical extent
        self.assertEqual(int(np.argmin(e_norm)), 2,
                         f"base_normal did not put the thin axis on +Z: {e_norm}")
        self.assertAlmostEqual(float(np.linalg.norm(e_norm)),
                               float(np.linalg.norm(hint.canonical_extent())), places=3,
                               msg="the two rules must describe the same object, only rotated")

    def test_an_unknown_rule_is_refused(self):
        from src.data.object import ArticulatedObject
        with self.assertRaises(ValueError):
            ArticulatedObject.load("box", canonical_rule="whatever")

    def test_boxy_objects_are_unchanged_by_the_rule_registry(self):
        """Regression: every object that was working before must still get lid_hint's frame."""
        from src.data.object import ArticulatedObject, canonical_rule_for
        for name in ("box",):
            self.assertEqual(canonical_rule_for(name), "lid_hint")
            a = ArticulatedObject.load(name)
            b = ArticulatedObject.load(name, canonical_rule="lid_hint")
            self.assertTrue(np.allclose(a.canonical_rotation(), b.canonical_rotation()))
