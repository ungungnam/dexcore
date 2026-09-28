"""Tests for src/analysis/action_structure. All synthetic -- no dataset needed.

The cases here are the ones that would have caught real mistakes: a relative pose composed the
wrong way round, an increment taken in the world frame instead of the body frame, a chunking
protocol that quietly overlaps, and -- the one that actually bit -- a directional statistic that
cancels itself out on reversing motion.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.analysis.action_structure import compact as CP
from src.analysis.action_structure import representations as REP
from src.analysis.action_structure import screw as SC
from src.analysis.action_structure import stats as ST
from src.analysis.action_structure import structure as SR
from src.analysis.action_structure.chunks import CHUNK_LEN, ChunkSet, _episode_chunks


def _se3(R, p):
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def _chunkset(tool_p, tool_R, targ_p, targ_R):
    return ChunkSet(tool_p, tool_R, targ_p, targ_R, None)


def _still(n, t):
    return np.zeros((n, t, 3)), np.tile(np.eye(3), (n, t, 1, 1))


class TestRotationCodecs(unittest.TestCase):
    def test_rot6d_is_the_first_two_columns(self):
        R = Rotation.random(5, random_state=0).as_matrix()
        six = REP.rot6d(R)
        np.testing.assert_allclose(six[:, :3], R[:, :, 0])
        np.testing.assert_allclose(six[:, 3:], R[:, :, 1])

    def test_so3_log_inverts_the_exponential(self):
        v = np.random.default_rng(0).normal(0, 0.5, (7, 3))
        np.testing.assert_allclose(REP.so3_log(Rotation.from_rotvec(v).as_matrix()), v, atol=1e-9)

    def test_so3_log_handles_identity(self):
        np.testing.assert_allclose(REP.so3_log(np.eye(3)[None]), 0.0, atol=1e-12)


class TestRelativeMotion(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n, self.t = 3, 64
        self.tR = Rotation.random(self.n * self.t, random_state=1).as_matrix().reshape(self.n, self.t, 3, 3)
        self.oR = Rotation.random(self.n * self.t, random_state=2).as_matrix().reshape(self.n, self.t, 3, 3)
        self.tp = rng.normal(0, 0.3, (self.n, self.t, 3))
        self.op = rng.normal(0, 0.3, (self.n, self.t, 3))
        self.cs = _chunkset(self.tp, self.tR, self.op, self.oR)

    def test_relative_pose_equals_the_matrix_expression(self):
        p_Y, R_Y = REP.relative_pose(self.cs)
        Y = np.linalg.inv(_se3(self.oR, self.op).reshape(-1, 4, 4)) @ _se3(self.tR, self.tp).reshape(-1, 4, 4)
        np.testing.assert_allclose(Y[:, :3, :3].reshape(R_Y.shape), R_Y, atol=1e-10)
        np.testing.assert_allclose(Y[:, :3, 3].reshape(p_Y.shape), p_Y, atol=1e-10)

    def test_increments_are_taken_in_the_previous_frame(self):
        p_Y, R_Y = REP.relative_pose(self.cs)
        log_dR, dp = REP.increments(p_Y, R_Y)
        self.assertEqual(log_dR.shape, (self.n, self.t - 1, 3))
        expect_dR = np.einsum("ntji,ntjk->ntik", R_Y[:, :-1], R_Y[:, 1:])
        np.testing.assert_allclose(REP.so3_log(expect_dR), log_dR, atol=1e-12)
        expect_dp = np.einsum("ntji,ntj->nti", R_Y[:, :-1], p_Y[:, 1:] - p_Y[:, :-1])
        np.testing.assert_allclose(expect_dp, dp, atol=1e-12)

    def test_representation_shapes_match_the_protocol(self):
        self.assertEqual(REP.raw_world(self.cs).shape, (self.n, 64, 18))
        self.assertEqual(REP.world_origin_aligned(self.cs).shape, (self.n, 64, 18))
        self.assertEqual(REP.virtual_base(self.cs).shape, (self.n, 64, 18))
        self.assertEqual(REP.differential(self.cs).shape, (self.n, 9 + 63 * 6))

    def test_only_the_differential_is_invariant_to_a_world_transform(self):
        Rw = Rotation.random(1, random_state=7).as_matrix()[0]
        tw = np.array([1.0, -2.0, 3.0])
        moved = _chunkset(self.tp @ Rw.T + tw, np.einsum("ij,ntjk->ntik", Rw, self.tR),
                          self.op @ Rw.T + tw, np.einsum("ij,ntjk->ntik", Rw, self.oR))
        self.assertTrue(np.allclose(REP.differential(self.cs), REP.differential(moved), atol=1e-9))
        for fn in (REP.raw_world, REP.world_origin_aligned, REP.virtual_base):
            self.assertFalse(np.allclose(fn(self.cs), fn(moved), atol=1e-6), fn.__name__)

    def test_world_origin_removes_only_the_offset(self):
        shifted = _chunkset(self.tp + 5.0, self.tR, self.op + 5.0, self.oR)
        np.testing.assert_allclose(REP.world_origin_aligned(self.cs),
                                   REP.world_origin_aligned(shifted), atol=1e-9)


class TestChunking(unittest.TestCase):
    class _Traj:
        def __init__(self, tool, targ):
            self.tool, self.target = tool, targ

    class _Track:
        def __init__(self, n, name):
            self.pos = np.arange(n * 3, dtype=float).reshape(n, 3)
            self.quat = np.tile([1.0, 0, 0, 0], (n, 1))
            self.name = name

        def __len__(self):
            return len(self.pos)

    class _Ref:
        sequence_id, triplet = "trip/seq", "(v, t, o)"

        class action:
            verb, tool, target = "v", "t", "o"

    def _chunks(self, n):
        traj = self._Traj(self._Track(n, "T"), self._Track(n, "O"))
        return _episode_chunks(traj, self._Ref())

    def test_chunk_count_is_the_floor(self):
        for n, k in ((64, 1), (127, 1), (128, 2), (200, 3), (497, 7)):
            self.assertEqual(len(self._chunks(n)["rows"]), k, f"{n} frames")

    def test_short_episode_gives_nothing(self):
        self.assertIsNone(self._chunks(63))

    def test_chunks_do_not_overlap_and_start_at_multiples_of_64(self):
        c = self._chunks(200)
        pos = c["tool_p"]
        for k in range(len(pos)):
            np.testing.assert_allclose(pos[k, 0], np.arange(3) + 3 * CHUNK_LEN * k)

    def test_the_remainder_is_discarded(self):
        c = self._chunks(200)                       # 3 chunks = 192 frames, 8 dropped
        self.assertEqual(c["tool_p"].shape, (3, 64, 3))

    def test_metadata_is_inherited_and_indexed(self):
        rows = self._chunks(200)["rows"]
        self.assertEqual([r["chunk_index"] for r in rows], [0, 1, 2])
        self.assertTrue(all(r["n_chunks"] == 3 for r in rows))
        self.assertTrue(all(r["episode_id"] == "trip/seq" for r in rows))


class TestScrew(unittest.TestCase):
    """The virtual-joint statistics, and the sign bug that made them necessary."""

    N, T = 2, 64

    def _rot_chunk(self, R):
        p, _ = _still(self.N, self.T)
        _, eye = _still(self.N, self.T)
        return _chunkset(p, np.tile(R, (self.N, 1, 1, 1)), p, eye)

    def test_back_and_forth_about_one_axis_is_maximally_concentrated(self):
        ang = 0.3 * np.sin(np.linspace(0, 6 * np.pi, self.T))
        R = Rotation.from_rotvec(np.outer(ang, [0, 0, 1.0])).as_matrix()
        f = SC.chunk_screw_features(self._rot_chunk(R))
        self.assertGreater(f["screw_axis_concentration"][0], 0.99)

    def test_the_old_statistic_misses_exactly_that_case(self):
        # the regression this module exists for: `rel_axis_stability` averages SIGNED rotation
        # vectors, so a reversal cancels and a perfectly fixed axis scores near zero
        ang = 0.3 * np.sin(np.linspace(0, 6 * np.pi, self.T))
        R = Rotation.from_rotvec(np.outer(ang, [0, 0, 1.0])).as_matrix()
        cs = self._rot_chunk(R)
        p_Y, R_Y = REP.relative_pose(cs)
        w, _ = REP.increments(p_Y, R_Y)
        self.assertLess(float(ST._resultant(w)[0]), 0.2)          # old measure: near zero
        self.assertGreater(SC.chunk_screw_features(cs)["screw_axis_concentration"][0], 0.99)

    def test_random_tumbling_is_not_concentrated(self):
        R = Rotation.random(self.T, random_state=0).as_matrix()
        f = SC.chunk_screw_features(self._rot_chunk(R))
        self.assertLess(f["screw_axis_concentration"][0], 0.7)

    def test_a_pure_slide_concentrates_the_translation_direction(self):
        slide = np.zeros((self.N, self.T, 3))
        slide[:, :, 0] = np.linspace(0, 0.3, self.T)
        p, eye = _still(self.N, self.T)
        f = SC.chunk_screw_features(_chunkset(slide, eye, p, eye))
        self.assertGreater(f["screw_dir_concentration"][0], 0.99)

    def test_a_helix_recovers_its_pitch(self):
        pitch, ang = 0.05, np.linspace(0, 4 * np.pi, self.T)
        p = np.zeros((self.N, self.T, 3))
        p[:, :, 2] = pitch * ang
        R = np.tile(Rotation.from_rotvec(np.outer(ang, [0, 0, 1.0])).as_matrix(), (self.N, 1, 1, 1))
        z, eye = _still(self.N, self.T)
        f = SC.chunk_screw_features(_chunkset(p, R, z, eye))
        self.assertAlmostEqual(f["screw_pitch_m_per_rad"][0], pitch, places=3)
        self.assertGreater(f["screw_axial_fraction"][0], 0.99)

    def test_a_pure_rotation_has_zero_pitch(self):
        ang = np.linspace(0, 2 * np.pi, self.T)
        R = Rotation.from_rotvec(np.outer(ang, [0, 0, 1.0])).as_matrix()
        f = SC.chunk_screw_features(self._rot_chunk(R))
        self.assertAlmostEqual(f["screw_pitch_m_per_rad"][0], 0.0, places=6)

    def test_reversal_fraction_separates_monotone_from_oscillating(self):
        ang_mono = np.linspace(0, 2 * np.pi, self.T)
        ang_osc = 0.3 * np.sin(np.linspace(0, 8 * np.pi, self.T))
        mono = SC.chunk_screw_features(
            self._rot_chunk(Rotation.from_rotvec(np.outer(ang_mono, [0, 0, 1.0])).as_matrix()))
        osc = SC.chunk_screw_features(
            self._rot_chunk(Rotation.from_rotvec(np.outer(ang_osc, [0, 0, 1.0])).as_matrix()))
        self.assertAlmostEqual(mono["screw_reversal_fraction"][0], 0.0, places=6)
        self.assertGreater(osc["screw_reversal_fraction"][0], 0.05)

    def test_every_feature_is_one_value_per_chunk(self):
        cs = self._rot_chunk(Rotation.random(self.T, random_state=3).as_matrix())
        for k, v in SC.chunk_screw_features(cs).items():
            self.assertEqual(np.shape(v), (self.N,), k)
        self.assertEqual(set(SC.chunk_screw_features(cs)), set(SC.SCREW_COLUMNS))


class TestCompactRepresentation(unittest.TestCase):
    def test_columns_are_the_documented_twenty(self):
        self.assertEqual(len(CP.JOINT_MOTION_COLUMNS), 20)
        self.assertEqual(len(set(CP.JOINT_MOTION_COLUMNS)), 20)

    def test_it_uses_only_relative_and_joint_quantities(self):
        # the world-frame tool and target motion is what carried object identity, and dropping it
        # is most of why this representation scores where it does
        for c in CP.JOINT_MOTION_COLUMNS:
            self.assertTrue(c.startswith(("rel_", "screw_")), c)

    def test_the_superseded_axis_statistic_is_excluded(self):
        self.assertNotIn("rel_axis_stability", CP.JOINT_MOTION_COLUMNS)
        self.assertIn("screw_axis_concentration", CP.JOINT_MOTION_COLUMNS)

    def test_net_displacement_quantities_are_excluded(self):
        for c in ("rel_trans_disp_m", "rel_rot_disp_rad", "screw_net_disp_m",
                  "screw_net_angle_rad"):
            self.assertNotIn(c, CP.JOINT_MOTION_COLUMNS)

    def test_columns_intersect_with_what_a_table_has(self):
        have = list(CP.JOINT_MOTION_COLUMNS[:5]) + ["something_else"]
        self.assertEqual(CP.columns(have), list(CP.JOINT_MOTION_COLUMNS[:5]))

    def test_build_fills_nan_with_the_column_median(self):
        import pandas as pd
        cols = list(CP.JOINT_MOTION_COLUMNS[:3])
        df = pd.DataFrame({cols[0]: [1.0, np.nan, 3.0], cols[1]: [0.0, 1.0, 2.0],
                           cols[2]: [5.0, 5.0, 5.0]})
        X = CP.build(df, cols)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertAlmostEqual(X[1, 0], 2.0)              # median of [1, 3]

    def test_redundant_list_names_only_means_of_kept_sums(self):
        # each dropped "..._inc_mean" has a "..._path_..." partner that is kept
        for c in CP.REDUNDANT:
            if c.endswith("_inc_mean"):
                stem = c.rsplit("_", 3)[0]
                self.assertTrue(any(k.startswith(stem) and "_path_" in k
                                    for k in ("tool_trans_path_m", "tool_rot_path_rad",
                                              "target_trans_path_m", "target_rot_path_rad",
                                              "rel_trans_path_m", "rel_rot_path_rad")), c)


class TestDistanceMetric(unittest.TestCase):
    def test_cosine_matches_sklearn(self):
        from sklearn.metrics import pairwise_distances as skl
        X = np.random.default_rng(0).normal(size=(8, 5))
        np.testing.assert_allclose(SR.pairwise_distances(X, "cosine"),
                                   skl(X, metric="cosine"), atol=1e-6)

    def test_euclidean_is_still_the_default_and_correct(self):
        from sklearn.metrics import pairwise_distances as skl
        X = np.random.default_rng(1).normal(size=(8, 5))
        np.testing.assert_allclose(SR.pairwise_distances(X), skl(X), atol=1e-4)

    def test_cosine_ignores_vector_magnitude(self):
        # the property the representation relies on: scaling a chunk's feature vector must not
        # move it, because the scale is mostly "how much motion was in this window"
        X = np.random.default_rng(2).normal(size=(6, 4))
        S = X * np.array([1.0, 5.0, 0.2, 3.0, 1.0, 1.0])[:, None]
        np.testing.assert_allclose(SR.pairwise_distances(X, "cosine"),
                                   SR.pairwise_distances(S, "cosine"), atol=1e-6)


class TestNormalisation(unittest.TestCase):
    def test_channel_zscore_pools_over_time(self):
        rng = np.random.default_rng(0)
        X = rng.normal(3.0, 2.0, (50, 64, 18))
        Z = REP.zscore(X, channels=18).reshape(50, 64, 18)
        np.testing.assert_allclose(Z.mean(axis=(0, 1)), 0.0, atol=1e-9)
        np.testing.assert_allclose(Z.std(axis=(0, 1)), 1.0, atol=1e-9)

    def test_flat_zscore_normalises_each_column(self):
        X = np.random.default_rng(0).normal(3.0, 2.0, (50, 12))
        Z = REP.zscore(X, channels=0)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)

    def test_a_constant_channel_does_not_divide_by_zero(self):
        X = np.ones((10, 64, 18))
        self.assertTrue(np.all(np.isfinite(REP.zscore(X, channels=18))))


if __name__ == "__main__":
    unittest.main()
