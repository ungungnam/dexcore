"""Loader tests against the REAL datasets. Every case skips when the data is not mounted.

These exist because the expensive mistakes in a loader are not logic errors, they are convention
errors -- a factor of 1000 in a unit, a quaternion read in the wrong order, an action label parsed
off by one field. None of those can be caught with synthetic data, and all of them are cheap to
catch with one sequence.
"""
import unittest

import numpy as np

from src.analysis import features as F
from src.analysis.loaders import arctic, taco

TACO_ROOT = taco.default_root()
ARCTIC_ROOT = arctic.default_root()
has_taco = (TACO_ROOT / "Hand_Poses").is_dir()
has_arctic = (ARCTIC_ROOT / "raw_seqs").is_dir()


@unittest.skipUnless(has_taco, f"TACO not mounted at {TACO_ROOT}")
class TestTaco(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = taco.index()
        cls.traj = taco.load(cls.refs[0])

    def test_index_finds_the_release(self):
        self.assertGreater(len(self.refs), 2000)
        self.assertEqual(len({r.triplet for r in self.refs}), 151)

    def test_triplet_with_a_multiword_verb_parses(self):
        a = taco.parse_triplet("(pour in some, kettle, cup)")
        self.assertEqual((a.verb, a.tool, a.target), ("pour in some", "kettle", "cup"))

    def test_every_triplet_directory_parses(self):
        for r in self.refs:
            self.assertTrue(r.action.verb, r.triplet)

    def test_a_sequence_has_a_tool_a_target_and_two_hands(self):
        self.assertIsNotNone(self.traj.tool)
        self.assertIsNotNone(self.traj.target)
        self.assertEqual(set(self.traj.hands), {"left", "right"})

    def test_objects_are_rigid(self):
        self.assertFalse(any(o.is_articulated for o in self.traj.objects))

    def test_all_tracks_share_one_timeline(self):
        lens = {len(o) for o in self.traj.objects} | {len(h) for h in self.traj.hands.values()}
        self.assertEqual(len(lens), 1)

    def test_positions_are_metres_in_a_tabletop_workspace(self):
        # a mis-scaled loader (millimetres, or a centimetre mesh scale applied to poses) lands
        # orders of magnitude outside this box, which is the point of the check
        p = self.traj.tool.pos
        self.assertLess(np.abs(p).max(), 3.0)
        self.assertGreater(np.ptp(p, axis=0).max(), 0.01)

    def test_the_hand_is_near_the_tool_it_holds(self):
        d = np.linalg.norm(self.traj.hands["right"].root_pos - self.traj.tool.pos, axis=1)
        self.assertLess(float(np.median(d)), 0.5)

    def test_hand_pose_is_45_finger_dimensions(self):
        self.assertEqual(self.traj.hands["right"].finger_pose.shape[1], 45)
        self.assertEqual(self.traj.hands["right"].shape.shape, (10,))

    def test_mesh_is_resolved_and_declared_in_centimetres(self):
        self.assertIsNotNone(self.traj.tool.mesh_path)
        self.assertTrue(self.traj.tool.mesh_path.endswith("_cm.obj"))
        self.assertAlmostEqual(self.traj.tool.mesh_scale, 0.01)

    def test_features_are_finite_where_they_must_be(self):
        row = F.sequence_features(self.traj)
        for k in ("duration_s", "tool_path_m", "tool_speed_mean", "rel_dist_mean_m"):
            self.assertTrue(np.isfinite(row[k]), k)


@unittest.skipUnless(has_arctic, f"ARCTIC not mounted at {ARCTIC_ROOT}")
class TestArctic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = arctic.index()
        cls.traj = arctic.load(cls.refs[0])

    def test_index_finds_every_sequence_including_retakes(self):
        self.assertEqual(len(self.refs), 301)
        self.assertTrue(any("retake" in r.name for r in self.refs))

    def test_verbs_are_the_two_arctic_has(self):
        self.assertEqual({r.action.verb for r in self.refs}, {"grab", "use"})

    def test_the_object_is_articulated(self):
        self.assertTrue(self.traj.objects[0].is_articulated)

    def test_articulation_is_radians_not_degrees(self):
        arti = np.concatenate([arctic.load(r).objects[0].arti for r in self.refs[:20]])
        self.assertLessEqual(float(arti.max()), np.pi + 1e-6)     # degrees would blow past this
        self.assertGreaterEqual(float(arti.min()), -1e-6)

    def test_positions_are_metres_not_millimetres(self):
        p = self.traj.objects[0].pos
        self.assertLess(np.abs(p).max(), 5.0)                     # millimetres would be ~1000x

    def test_the_hand_is_within_arms_reach_of_the_object(self):
        d = np.linalg.norm(self.traj.hands["right"].root_pos - self.traj.objects[0].pos, axis=1)
        self.assertLess(float(np.median(d)), 1.5)

    def test_no_target_object(self):
        self.assertIsNone(self.traj.target)
        self.assertIsNotNone(self.traj.tool)

    def test_target_features_are_nan(self):
        row = F.sequence_features(self.traj)
        self.assertTrue(np.isnan(row["rel_dist_mean_m"]))
        self.assertTrue(np.isfinite(row["tool_arti_range_deg"]))


@unittest.skipUnless(has_taco and has_arctic, "both datasets required")
class TestSchemaIsNeutral(unittest.TestCase):
    def test_both_datasets_produce_identical_feature_columns(self):
        a = F.sequence_features(taco.load(taco.index()[0]))
        b = F.sequence_features(arctic.load(arctic.index()[0]))
        self.assertEqual(set(a) - {"notes"}, set(b) - {"notes"})


if __name__ == "__main__":
    unittest.main()
