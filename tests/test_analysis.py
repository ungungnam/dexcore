"""Tests for src/analysis: the schema's invariants, the features' arithmetic, the grouping.

Everything here is SYNTHETIC. The loaders are exercised against the real datasets by
`test_analysis_datasets.py`, which skips when the data is not mounted; these tests must pass on a
checkout with no data at all.
"""
import unittest

import numpy as np

from src.analysis import distribution as D
from src.analysis import features as F
from src.analysis.schema import ActionLabel, HandTrack, HOITrajectory, ObjectTrack


def _quat(n, axis=(0, 0, 1), rate=0.0):
    """n unit quaternions rotating about `axis` by `rate` radians per frame."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    ang = np.arange(n) * rate
    return np.stack([np.cos(ang / 2)] + [s * np.sin(ang / 2) for s in a], axis=1)


def _track(pos, role="tool", quat=None, arti=None, name="obj"):
    n = len(pos)
    return ObjectTrack(name=name, role=role, pos=pos,
                       quat=_quat(n) if quat is None else quat, arti=arti)


def _hand(n, side="right", start=(0.0, 0.0, 0.0), step=(0.01, 0.0, 0.0), pose=None):
    pos = np.array(start) + np.arange(n)[:, None] * np.array(step)
    return HandTrack(side=side, root_pos=pos, root_quat=_quat(n),
                     finger_pose=np.zeros((n, 45)) if pose is None else pose)


def _traj(objects, hands=(), fps=30.0, verb="stir"):
    return HOITrajectory(dataset="synthetic", sequence_id="t/0",
                         action=ActionLabel(verb=verb, tool="spoon", target="bowl"),
                         objects=list(objects), hands={h.side: h for h in hands}, fps=fps)


class TestSchema(unittest.TestCase):
    def test_quaternions_are_sign_unrolled_on_construction(self):
        q = np.array([[1.0, 0, 0, 0], [-0.9999, -0.01, 0, 0]])
        t = _track(np.zeros((2, 3)), quat=q)
        self.assertGreater(float(t.quat[1] @ t.quat[0]), 0.0)

    def test_ragged_track_is_rejected(self):
        with self.assertRaises(ValueError):
            ObjectTrack(name="o", role="tool", pos=np.zeros((5, 3)), quat=_quat(4))

    def test_tracks_of_different_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            _traj([_track(np.zeros((5, 3)))], [_hand(4)])

    def test_roles_resolve(self):
        tr = _traj([_track(np.zeros((3, 3)), role="target", name="bowl"),
                    _track(np.ones((3, 3)), role="tool", name="spoon")])
        self.assertEqual(tr.tool.name, "spoon")
        self.assertEqual(tr.target.name, "bowl")

    def test_object_role_answers_tool_for_single_object_datasets(self):
        tr = _traj([_track(np.zeros((3, 3)), role="object", name="box")])
        self.assertEqual(tr.tool.name, "box")
        self.assertIsNone(tr.target)

    def test_duration_uses_fps(self):
        self.assertAlmostEqual(_traj([_track(np.zeros((60, 3)))], fps=30.0).duration_s, 2.0)

    def test_ragged_finger_pose_is_rejected_not_reshaped(self):
        # (2T,45) against a (T,3) root reshapes to (T,90) without complaint and welds frame pairs
        # together; len(track) reads T either way, so nothing downstream could ever notice
        with self.assertRaises(ValueError):
            HandTrack(side="right", root_pos=np.zeros((45, 3)), root_quat=_quat(45),
                      finger_pose=np.zeros((90, 45)))

    def test_bad_side_is_rejected(self):
        with self.assertRaises(ValueError):
            HandTrack(side="third", root_pos=np.zeros((2, 3)), root_quat=_quat(2),
                      finger_pose=np.zeros((2, 45)))


class TestPrimitives(unittest.TestCase):
    def test_speed_is_metres_per_second(self):
        pos = np.arange(10)[:, None] * np.array([0.1, 0.0, 0.0])   # 0.1 m per frame
        np.testing.assert_allclose(F.speeds(pos, fps=30.0), 3.0)   # 3 m/s

    def test_angular_speed_is_radians_per_second(self):
        q = _quat(10, rate=0.1)                                    # 0.1 rad per frame
        np.testing.assert_allclose(F.angular_speeds(q, fps=30.0), 3.0, atol=1e-8)

    def test_path_length_counts_the_whole_path_not_the_displacement(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], float)
        self.assertAlmostEqual(F.path_length(pos), 2.0)

    def test_short_track_gives_empty_speeds(self):
        self.assertEqual(len(F.speeds(np.zeros((1, 3)), 30.0)), 0)

    def test_tilt_is_zero_for_a_yaw_only_rotation(self):
        t = _track(np.zeros((10, 3)), quat=_quat(10, axis=(0, 0, 1), rate=0.3))
        np.testing.assert_allclose(F.tilt_angles(t), 0.0, atol=1e-8)

    def test_tilt_follows_a_pitch(self):
        t = _track(np.zeros((10, 3)), quat=_quat(10, axis=(1, 0, 0), rate=0.1))
        np.testing.assert_allclose(F.tilt_angles(t), np.arange(10) * 0.1, atol=1e-8)

    def test_relative_position_is_expressed_in_the_frame_object(self):
        n = 8
        # frame yaws by 90 degrees; the moving object sits 1 m along world +x throughout.
        frame = _track(np.zeros((n, 3)), role="target",
                       quat=_quat(n, axis=(0, 0, 1), rate=np.pi / 2))
        moving = _track(np.tile([1.0, 0.0, 0.0], (n, 1)))
        rel = F.relative_positions(moving, frame)
        np.testing.assert_allclose(rel[0], [1, 0, 0], atol=1e-8)     # no rotation yet
        np.testing.assert_allclose(rel[1], [0, -1, 0], atol=1e-8)    # frame turned +90 deg


class TestRhythm(unittest.TestCase):
    def test_a_sine_recovers_its_frequency(self):
        fps, hz, n = 30.0, 1.5, 300
        t = np.arange(n) / fps
        sig = np.stack([0.05 * np.sin(2 * np.pi * hz * t), np.zeros(n), np.zeros(n)], axis=1)
        out = F.rhythm(sig, fps)
        self.assertAlmostEqual(out["freq_hz"], hz, delta=0.2)
        self.assertGreater(out["autocorr_peak"], 0.5)

    def test_oscillation_survives_a_transport_drift(self):
        fps, hz, n = 30.0, 1.2, 300
        t = np.arange(n) / fps
        drift = np.stack([0.3 * t, np.zeros(n), np.zeros(n)], axis=1)     # 0.3 m/s carry
        sig = drift + np.stack([0.04 * np.sin(2 * np.pi * hz * t)] + [np.zeros(n)] * 2, axis=1)
        self.assertAlmostEqual(F.rhythm(sig, fps)["freq_hz"], hz, delta=0.2)

    def test_monotone_motion_has_no_period(self):
        n = 200
        sig = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
        self.assertTrue(np.isnan(F.rhythm(sig, 30.0)["freq_hz"]))

    def test_too_short_is_nan_not_an_error(self):
        self.assertTrue(np.isnan(F.rhythm(np.zeros((10, 3)), 30.0)["freq_hz"]))

    def test_frequency_does_not_depend_on_sequence_length(self):
        # the defect this pins: capping the lag search at T//2 raised the lowest reachable
        # frequency for a short sequence, so the SAME motion reported a higher rate when clipped
        fps, hz = 30.0, 1.2
        t = np.arange(400) / fps
        sig = np.stack([0.05 * np.sin(2 * np.pi * hz * t)] + [np.zeros(400)] * 2, axis=1)
        full = F.rhythm(sig, fps)["freq_hz"]
        for n in (140, 200, 300):
            self.assertAlmostEqual(F.rhythm(sig[:n], fps)["freq_hz"], full, delta=0.05,
                                   msg=f"frequency moved when truncated to {n} frames")

    def test_a_sequence_too_short_for_the_whole_band_is_nan(self):
        fps, hz = 30.0, 1.2
        t = np.arange(90) / fps                    # 90 frames: under the 106 the 0.8 Hz band needs
        sig = np.stack([0.05 * np.sin(2 * np.pi * hz * t)] + [np.zeros(90)] * 2, axis=1)
        self.assertTrue(np.isnan(F.rhythm(sig, fps)["freq_hz"]))

    def test_an_oscillation_above_the_band_is_not_reported_at_the_band_edge(self):
        fps, hz, n = 30.0, 6.0, 400                # 6 Hz, well above the 3 Hz ceiling
        t = np.arange(n) / fps
        sig = np.stack([0.05 * np.sin(2 * np.pi * hz * t)] + [np.zeros(n)] * 2, axis=1)
        self.assertTrue(np.isnan(F.rhythm(sig, fps)["freq_hz"]))


class TestSequenceFeatures(unittest.TestCase):
    def setUp(self):
        n = 120
        pos = np.stack([np.linspace(0, 1.0, n), np.zeros(n), np.full(n, 0.5)], axis=1)
        self.tool = _track(pos, role="tool", name="spoon")
        self.target = _track(np.tile([0.5, 0.0, 0.4], (n, 1)), role="target", name="bowl")
        self.traj = _traj([self.tool, self.target],
                          [_hand(n, "right", start=(0.0, 0.05, 0.5), step=(1 / (n - 1), 0, 0)),
                           _hand(n, "left", start=(-1.0, 0.0, 0.5), step=(0, 0, 0))])
        self.row = F.sequence_features(self.traj)

    def test_labels_are_carried_through(self):
        self.assertEqual(self.row["verb"], "stir")
        self.assertEqual(self.row["n_frames"], 120)
        self.assertAlmostEqual(self.row["duration_s"], 4.0)

    def test_tool_path_matches_the_synthetic_motion(self):
        self.assertAlmostEqual(self.row["tool_path_m"], 1.0, places=6)
        self.assertAlmostEqual(self.row["tool_disp_m"], 1.0, places=6)

    def test_static_target_has_zero_path(self):
        self.assertAlmostEqual(self.row["target_path_m"], 0.0)

    def test_acting_hand_is_the_one_near_the_tool(self):
        self.assertEqual(self.row["acting_hand"], "right")
        self.assertLess(self.row["rh_tool_dist_mean_m"], self.row["lh_tool_dist_mean_m"])

    def test_rigid_object_has_no_articulation(self):
        self.assertTrue(np.isnan(self.row["tool_arti_range_deg"]))

    def test_articulated_object_reports_its_range(self):
        n = 60
        arti = np.linspace(0, np.pi / 2, n)
        tr = _traj([_track(np.zeros((n, 3)), role="object", arti=arti)])
        self.assertAlmostEqual(F.sequence_features(tr)["tool_arti_range_deg"], 90.0, places=4)

    def test_a_missing_target_yields_nan_not_zero(self):
        tr = _traj([self.tool])
        row = F.sequence_features(tr)
        for k in ("target_path_m", "rel_dist_mean_m", "near_frac"):
            self.assertTrue(np.isnan(row[k]), k)

    def test_frame_count_is_a_label_not_a_second_copy_of_duration(self):
        # duration_s = n_frames / fps with fps constant per dataset: ranking both would put one
        # physical fact in two rows of the result
        cols = F.feature_columns(self.row.keys())
        self.assertIn("duration_s", cols)
        self.assertNotIn("n_frames", cols)

    def test_notes_is_always_present(self):
        self.assertIn("notes", self.row)

    def test_every_row_has_the_same_columns(self):
        a = set(F.sequence_features(_traj([self.tool, self.target])))
        b = set(F.sequence_features(_traj([self.tool])))
        self.assertEqual(a, b)

    def test_finger_activity_is_zero_for_a_still_hand(self):
        self.assertAlmostEqual(self.row["rh_finger_std_deg"], 0.0)

    def test_finger_activity_reads_in_degrees(self):
        n = 60
        pose = np.zeros((n, 45))
        pose[:, 0] = np.radians(10.0) * np.sign(np.sin(np.arange(n)))   # one joint, +-10 deg
        tr = _traj([self.tool.__class__(name="o", role="tool", pos=np.zeros((n, 3)),
                                        quat=_quat(n))], [_hand(n, "right", pose=pose)])
        # 1 of 15 joints moves by ~10 deg about a mean of ~0 -> mean per-joint deviation ~10/15
        self.assertAlmostEqual(F.sequence_features(tr)["rh_finger_std_deg"], 10.0 / 15, delta=0.2)


class TestDistribution(unittest.TestCase):
    def _frame(self, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for verb, mean in (("stir", 0.1), ("hit", 1.0)):
            for i in range(12):
                n = 90
                pos = np.stack([np.linspace(0, mean, n), np.zeros(n), np.zeros(n)], axis=1) \
                    + rng.normal(0, 1e-4, (n, 3))
                rows.append(F.sequence_features(
                    _traj([_track(pos, role="tool")], [_hand(n)], verb=verb)))
        return D.feature_table(rows)

    def test_table_puts_labels_first(self):
        df = self._frame()
        self.assertEqual(list(df.columns)[:3], ["dataset", "sequence_id", "verb"])

    def test_counts_are_per_group(self):
        counts = D.group_counts(self._frame(), "verb")
        self.assertEqual(sorted(counts["n_sequences"]), [12, 12])

    def test_summary_has_one_row_per_group_and_feature(self):
        df = self._frame()
        s = D.group_summary(df, "verb")
        self.assertEqual(set(s["verb"]), {"stir", "hit"})
        self.assertTrue((s["n"] <= 12).all())
        # a feature every sequence has is summarised over all 12
        complete = s[s["feature"] == "tool_path_m"]
        self.assertTrue((complete["n"] == 12).all())

    def test_a_separating_feature_outranks_a_constant_one(self):
        disc = D.discriminability(self._frame(), "verb")
        # the two groups differ only in how far the tool travels, so every feature that sees that
        # distance separates them perfectly and ties at the top; the point is that it IS at the top
        row = disc[disc["feature"] == "tool_path_m"]
        self.assertEqual(len(row), 1)
        self.assertGreater(float(row.iloc[0]["epsilon_sq"]), 0.5)
        self.assertLess(int(row.index[0]), 5)
        self.assertNotIn("n_frames", set(disc["feature"]))      # constant -> no ranks -> dropped

    def test_confound_comparison_flags_a_tool_driven_feature(self):
        # verb and tool are perfectly confounded here, so the two groupings must score alike
        df = self._frame()
        df["tool_name"] = df["verb"].map({"stir": "spoon", "hit": "hammer"})
        cmp = D.compare_groupings(df, ("verb", "tool_name"))
        row = cmp[cmp["feature"] == "tool_path_m"].iloc[0]
        self.assertAlmostEqual(row["epsilon_sq_verb"], row["epsilon_sq_tool_name"], places=6)
        self.assertAlmostEqual(row["ratio"], 1.0, places=6)

    def test_confound_comparison_separates_an_action_driven_feature(self):
        # one tool per verb no longer: each verb is done with both tools, so a feature that
        # follows the verb must score higher by verb than by tool
        df = self._frame()
        df["tool_name"] = ["spoon", "hammer"] * (len(df) // 2)
        cmp = D.compare_groupings(df, ("verb", "tool_name"))
        row = cmp[cmp["feature"] == "tool_path_m"].iloc[0]
        self.assertGreater(row["ratio"], 2.0)

    def test_a_single_level_grouping_returns_empty_not_a_key_error(self):
        # reached by an ordinary `--verbs stir`: one verb, several tools. The sweep is already
        # paid for by this point, so a crash here throws away the whole run
        df = self._frame()
        df["verb"] = "stir"
        df["tool_name"] = ["spoon", "hammer"] * (len(df) // 2)
        self.assertTrue(D.compare_groupings(df, ("verb", "tool_name")).empty)

    def test_groups_too_small_to_test_return_empty_not_a_key_error(self):
        df = self._frame()
        df["verb"] = [f"v{i}" for i in range(len(df))]        # every group has one member
        self.assertTrue(D.compare_groupings(df, ("verb", "tool_name")).empty)

    def test_direction_is_reported(self):
        disc = D.discriminability(self._frame(), "verb")
        top = disc[disc["feature"] == "tool_path_m"].iloc[0]
        self.assertEqual(top["highest_verb"], "hit")
        self.assertEqual(top["lowest_verb"], "stir")


if __name__ == "__main__":
    unittest.main()
