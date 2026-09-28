"""Synthetic checks for the canonical-mapping baselines: rotated/scaled copies of one shape must
land on each other, the maps must invert exactly, and the sanity wrapper must scramble the vector
and nothing else. No data needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis.canonical import backends as B
from src.analysis.canonical import canonical_to_object, object_to_canonical


def _shape(rng, n=1500):
    """An asymmetric blob (so PCA axes are well defined) with a long handle."""
    body = rng.normal(size=(n, 3)) * np.array([0.05, 0.03, 0.02])
    handle = np.stack([rng.uniform(0.05, 0.25, n // 3), rng.normal(size=n // 3) * 0.005,
                       rng.normal(size=n // 3) * 0.005], 1)
    return np.concatenate([body, handle])


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def _meshes():
    rng = np.random.default_rng(1)
    base = _shape(rng)
    out = {}
    for i in range(4):
        V = (base + rng.normal(size=base.shape) * 1e-3) * rng.uniform(0.7, 1.4) @ _rot(rng).T
        out[f"{i:03d}"] = (V + rng.uniform(-1, 1, 3), np.zeros((0, 3), dtype=np.int64))
    return out


class TestCanonical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes = _meshes()
        cls.aligned = B.AlignedBackend(n_sample=800, k=128)
        cls.aligned.fit("thing", cls.meshes)

    def test_aligned_overlaps_and_inverts(self):
        bk = self.aligned
        q = bk.quality("thing")
        tpl = bk.template("thing")
        for i, (V, _) in self.meshes.items():
            C = object_to_canonical("thing", i, V, bk)
            self.assertLess(np.abs(canonical_to_object("thing", i, C, bk) - V).max(), 1e-9)
            self.assertAlmostEqual(np.linalg.det(bk.transform("thing", i).R), 1.0, places=9)
            if i != tpl:
                self.assertLess(q[i]["chamfer_icp"], 0.02)
                self.assertLess(q[i]["chamfer_icp"], q[i]["chamfer_centred"])
        self.assertTrue(np.allclose(bk.transform("thing", tpl).t, 0))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.aligned.save(Path(d) / "aligned")
            bk2 = B.AlignedBackend.load(Path(d) / "aligned")
        V = self.meshes["001"][0]
        self.assertTrue(np.allclose(bk2.to_canonical("thing", "001", V),
                                    self.aligned.to_canonical("thing", "001", V)))
        self.assertEqual(bk2.canonical_points("thing").shape, (128, 3))
        self.assertEqual(bk2.template("thing"), self.aligned.template("thing"))

    def test_splat_and_random_perm(self):
        al = self.aligned
        rp = B.RandomPermBackend.from_aligned(al)
        V = self.meshes["002"][0]
        vals = (V[:, 0] > V[:, 0].mean()).astype(float)
        vec, cov = al.splat("thing", "002", vals, return_coverage=True)
        self.assertEqual(vec.shape, (128,)); self.assertEqual(cov.shape, (128,))
        self.assertGreater(cov.mean(), 0.9)
        g = al.splat("thing", "002", vals, kernel="gauss")
        self.assertGreaterEqual(g.min(), 0); self.assertLessEqual(g.max(), 1)
        pv = rp.splat("thing", "002", vals)
        p = rp.permutation("thing", "002")
        self.assertTrue(np.array_equal(pv, vec[p]))
        self.assertFalse(np.array_equal(pv, vec))
        # two meshes get DIFFERENT shuffles, otherwise correspondence would survive
        self.assertFalse(np.array_equal(p, rp.permutation("thing", "003")))
        self.assertTrue(np.array_equal(rp.to_canonical("thing", "002", V),
                                       al.to_canonical("thing", "002", V)))
        with tempfile.TemporaryDirectory() as d:
            rp.save(Path(d) / "random_perm")
            rp2 = B.load("random_perm", d)
        self.assertTrue(np.array_equal(rp2.permutation("thing", "002"), p))

    def test_normalized_no_rotation(self):
        bk = B.NormalizedBackend(n_sample=800, k=64)
        bk.fit("thing", self.meshes)
        for i, (V, _) in self.meshes.items():
            sim = bk.transform("thing", i)
            self.assertTrue(np.allclose(sim.R, np.eye(3)))
            C = bk.to_canonical("thing", i, V)
            self.assertAlmostEqual(np.linalg.norm(C, axis=1).max(), 1.0, places=9)
            self.assertLess(np.abs(bk.from_canonical("thing", i, C) - V).max(), 1e-9)




# ------------------------------------------------------------------------------------------------
# DINO (semantic) backend: exercised with an ORACLE mapper, so no renderer, no GPU, no weights.
# The oracle knows the true correspondence because the synthetic meshes are row-matched copies of
# one base shape; the real engine is only replaced at the `mapper` seam.
# ------------------------------------------------------------------------------------------------
from src.analysis.canonical.backends.dino import DinoBackend, MapResult   # noqa: E402


class _Oracle:
    """Maps an instance point to the template point with the same base index. Every 10th query is
    reported invalid and every 17th ambiguous, to exercise the fallback path."""

    def __init__(self):
        self.tpl = {}

    def prepare_template(self, category, verts, faces):
        self.tpl[category] = np.asarray(verts)

    flip = None          # a (3,3) proper flip to report as chosen (None: aligned frame)

    def map_points(self, category, verts, faces, pts):
        from scipy.spatial import cKDTree
        _d, j = cKDTree(verts).query(pts)           # pts are instance vertices -> base index
        n = len(pts)
        i = np.arange(n)
        valid = (i % 10) != 0
        top1 = np.where((i % 17) == 0, 0.3, 1.0)
        dst = np.where(valid[:, None], self.tpl[category][j], np.nan)
        return MapResult(dst, top1, np.where(valid, 0.8, np.nan), np.full(n, 5), np.full(n, 4),
                         valid.astype(int), valid, ["" if v else "oracle:invalid" for v in valid],
                         flip=self.flip)


def _stretched_meshes():
    """The 4 rigid copies plus one whose handle is half as long (same vertex order). Shortened,
    not lengthened: similarity ICP fits a short handle onto a long one cheaply, which would make
    the long-handled outlier the medoid; a short one is penalised in both directions."""
    rng = np.random.default_rng(1)
    base = _shape(rng)
    out = {}
    for i in range(4):
        V = (base + rng.normal(size=base.shape) * 1e-3) * rng.uniform(0.7, 1.4) @ _rot(rng).T
        out[f"{i:03d}"] = (V + rng.uniform(-1, 1, 3), np.zeros((0, 3), dtype=np.int64))
    S = base.copy()
    handle = S[:, 0] > 0.05
    S[handle, 0] = 0.05 + 0.5 * (S[handle, 0] - 0.05)
    out["004"] = (S @ _rot(rng).T + rng.uniform(-1, 1, 3), np.zeros((0, 3), dtype=np.int64))
    return out, base


class TestDino(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes, cls.base = _stretched_meshes()
        aligned = B.AlignedBackend(n_sample=800, k=128)
        aligned.fit("thing", cls.meshes)
        cls.aligned = aligned
        cls.dino = DinoBackend(aligned, n_sample=400, mapper=_Oracle())
        cls.dino.fit("thing", cls.meshes)

    def test_shares_template_and_points_with_aligned(self):
        d, a = self.dino, self.aligned
        self.assertEqual(d.template("thing"), a.template("thing"))
        self.assertTrue(np.array_equal(d.canonical_points("thing"), a.canonical_points("thing")))
        self.assertEqual(d.mesh_ids("thing"), a.mesh_ids("thing"))
        tpl = d.template("thing")
        m = d.mesh_map("thing", tpl)
        self.assertFalse(m.fallback.any())
        self.assertTrue(np.array_equal(m.src, m.dst))       # the template maps to itself

    def test_handle_tip_lands_on_template_tip(self):
        """The short handle's tip must land on the template's tip under dino, while aligned
        (surface overlap) leaves it part-way along the template's handle."""
        d, a = self.dino, self.aligned
        tpl = d.template("thing")
        self.assertNotEqual(tpl, "004")
        tip = int(np.argmax(self.base[:, 0]))
        V4 = self.meshes["004"][0]
        Vt = self.meshes[tpl][0]
        true_tip = a.to_canonical("thing", tpl, Vt[tip][None])[0]
        via_dino = d.to_canonical("thing", "004", V4[tip][None])[0]
        via_aligned = a.to_canonical("thing", "004", V4[tip][None])[0]
        e_dino = np.linalg.norm(via_dino - true_tip)
        e_aligned = np.linalg.norm(via_aligned - true_tip)
        self.assertLess(e_dino, 0.05)                 # within an FPS cell of the truth
        self.assertLess(e_dino, e_aligned)            # and strictly better than overlap

    def test_fallback_flags_and_quality(self):
        m = self.dino.mesh_map("thing", "004")
        i = np.arange(len(m.src))
        expect = ((i % 10) == 0) | ((i % 17) == 0)
        self.assertTrue(np.array_equal(m.fallback, expect))
        self.assertTrue(np.array_equal(m.valid, (i % 10) != 0))
        self.assertTrue(np.array_equal(m.dst[m.fallback], m.src[m.fallback]))
        self.assertTrue(np.isnan(m.dst_raw[~m.valid]).all())
        self.assertTrue(all(r.startswith("ambiguous") for r in
                            np.array(m.reason)[(i % 17 == 0) & (i % 10 != 0)]))
        self.assertTrue((m.displacement[m.fallback] == 0).all())
        q = self.dino.quality("thing")["004"]
        self.assertAlmostEqual(q["fallback_frac"], expect.mean())
        self.assertEqual(q["n_valid"], int(((i % 10) != 0).sum()))
        self.assertAlmostEqual(q["conf_mean"], 0.8)

    def test_fallback_follows_the_chosen_flip(self):
        """When the flip search picked F, fallback points sit at F * src (aligned under that sign),
        not at src."""
        from src.analysis.canonical.backends.dino import flip_name
        F = np.diag([-1.0, 1.0, -1.0])
        oracle = _Oracle(); oracle.flip = F
        d = DinoBackend(self.aligned, n_sample=400, mapper=oracle)
        d.fit("thing", self.meshes)
        m = d.mesh_map("thing", "004")
        self.assertEqual(flip_name(m.flip), "-+-")
        self.assertTrue(np.allclose(m.dst[m.fallback], m.src[m.fallback] @ F.T))
        self.assertTrue((m.displacement[m.fallback] > 0).any())
        tpl = d.template("thing")
        self.assertTrue(np.array_equal(d.mesh_map("thing", tpl).flip, np.eye(3)))
        with tempfile.TemporaryDirectory() as root:
            d.save(Path(root) / "dino")
            d2 = B.load("dino", root)
        self.assertTrue(np.array_equal(d2.mesh_map("thing", "004").flip, F))
        self.assertAlmostEqual(d2.quality("thing")["004"]["flip_margin"], 0.0)

    def test_inverse_is_approximate_and_splat_works(self):
        d = self.dino
        V = self.meshes["001"][0]
        C = object_to_canonical("thing", "001", V, d)
        back = canonical_to_object("thing", "001", C, d)
        err = np.linalg.norm(back - V, axis=1)
        self.assertLess(err.max(), 0.05)               # one FPS spacing in object units
        self.assertGreater(err.max(), 0.0)             # NOT exact: many-to-one map
        self.assertLess(d.roundtrip_error("thing", "001"), 0.05)
        vals = (V[:, 0] > V[:, 0].mean()).astype(float)
        vec, cov = d.splat("thing", "001", vals, kernel="gauss", return_coverage=True)
        self.assertEqual(vec.shape, (128,)); self.assertEqual(cov.shape, (128,))
        self.assertGreaterEqual(vec.min(), 0); self.assertLessEqual(vec.max(), 1)
        self.assertGreater(cov.mean(), 0.8)
        with self.assertRaises(ValueError):
            d.splat("thing", "001", vals[:-1])

    def test_save_load(self):
        d = self.dino
        with tempfile.TemporaryDirectory() as root:
            d.save(Path(root) / "dino")
            d2 = B.load("dino", root)
        V = self.meshes["004"][0]
        self.assertTrue(np.allclose(d2.to_canonical("thing", "004", V),
                                    d.to_canonical("thing", "004", V)))
        self.assertTrue(np.allclose(d2.from_canonical("thing", "004", d.canonical_points("thing")),
                                    d.from_canonical("thing", "004", d.canonical_points("thing"))))
        m, m2 = d.mesh_map("thing", "004"), d2.mesh_map("thing", "004")
        self.assertTrue(np.array_equal(m.fallback, m2.fallback))
        self.assertEqual(m.reason, m2.reason)
        self.assertEqual(d2.template("thing"), d.template("thing"))
        self.assertTrue(np.array_equal(d2.canonical_points("thing"), d.canonical_points("thing")))
        self.assertTrue(np.allclose(d2.aligned.to_canonical("thing", "004", V),
                                    d.aligned.to_canonical("thing", "004", V)))

if __name__ == "__main__":
    unittest.main()
