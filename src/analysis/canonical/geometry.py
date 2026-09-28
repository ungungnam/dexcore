"""Point-set geometry the canonical backends share: similarity transforms, surface sampling, PCA
frames, symmetric Chamfer distance and similarity (rigid + isotropic scale) ICP.

Everything is plain numpy + scipy's cKDTree. Meshes are treated as point sets throughout; the face
lists are carried but never used, which is deliberate: the baselines are about where vertices sit in
space, and an alignment that needs surface connectivity is already half-way to a correspondence
method, which these baselines are explicitly not.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


class Similarity(NamedTuple):
    """q = s * R @ (p - c) + t.

    `c` is the point the map rotates and scales about (the centroid for every backend here), so
    a fitted transform stays readable: `s` is the unit-sphere scale, `R` the principal-axis
    rotation (times the ICP correction), `t` the ICP translation in canonical units.
    """
    R: np.ndarray    # (3, 3), det +1
    t: np.ndarray    # (3,)
    s: float
    c: np.ndarray    # (3,)

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(p, dtype=np.float64) - self.c) @ self.R.T + self.t

    def invert(self, q: np.ndarray) -> np.ndarray:
        return ((np.asarray(q, dtype=np.float64) - self.t) / self.s) @ self.R + self.c

    @classmethod
    def identity(cls) -> "Similarity":
        return cls(np.eye(3), np.zeros(3), 1.0, np.zeros(3))


def compose(outer: Similarity, inner: Similarity) -> Similarity:
    """outer(inner(p)) as one Similarity about inner's centre."""
    R = outer.R @ inner.R
    s = outer.s * inner.s
    t = outer.s * outer.R @ (inner.t - outer.c) + outer.t
    return Similarity(R, t, s, inner.c)


#: The four proper rotations that keep a PCA frame's axes but flip their signs in pairs.
#: A single-axis flip has det -1 and would mirror the object; a mirrored spatula is not a
#: spatula anyone recorded, so those are never evaluated.
PROPER_FLIPS: Tuple[np.ndarray, ...] = tuple(np.diag(d).astype(np.float64) for d in
                                             ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))


def fps_indices(verts: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sample; deterministic (starts at the vertex farthest from the centroid).
    Same convention as `src.analysis.bimart.fps_contact.fps_indices`, repeated here so this
    package does not import BimArt's preprocessing."""
    verts = np.asarray(verts, dtype=np.float64)
    n = len(verts)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = int(np.argmax(np.linalg.norm(verts - verts.mean(0), axis=1)))
    d = np.linalg.norm(verts - verts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(verts - verts[sel[i]], axis=1))
    return sel


def centre_scale(verts: np.ndarray) -> Similarity:
    """Centroid to the origin, isotropic scale so the farthest vertex sits at radius 1."""
    verts = np.asarray(verts, dtype=np.float64)
    c = verts.mean(0)
    r = float(np.linalg.norm(verts - c, axis=1).max())
    return Similarity(np.eye(3), np.zeros(3), 1.0 / max(r, 1e-12), c)


def pca_axes(centred: np.ndarray) -> np.ndarray:
    """Rows are the principal axes, longest first, det +1. Each axis is signed so the third
    moment of the points along it is positive -- a deterministic convention, nothing more: for
    a symmetric shape the skewness is ~0 and the sign is meaningless, which is exactly why the
    aligned backend still searches the four proper flips against the template."""
    X = np.asarray(centred, dtype=np.float64)
    w, V = np.linalg.eigh(X.T @ X / len(X))
    axes = V[:, ::-1].T.copy()                      # rows, descending eigenvalue
    skew = ((X @ axes.T) ** 3).mean(0)
    axes[skew < 0] *= -1
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def chamfer(A: np.ndarray, B: np.ndarray, tree_a: Optional[cKDTree] = None,
            tree_b: Optional[cKDTree] = None) -> float:
    """Symmetric Chamfer: mean nearest-neighbour distance A->B and B->A, averaged. Not squared,
    so the value reads directly as "typical gap" in the units of the points."""
    tree_a = tree_a or cKDTree(A)
    tree_b = tree_b or cKDTree(B)
    return 0.5 * (float(tree_b.query(A)[0].mean()) + float(tree_a.query(B)[0].mean()))


def umeyama(X: np.ndarray, Y: np.ndarray) -> Similarity:
    """Least-squares similarity (rotation, isotropic scale, translation) taking X onto Y,
    row-matched. Kabsch with the scale term (Umeyama 1991)."""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_x = float((Xc ** 2).sum(1).mean())
    s = float(np.trace(np.diag(S) @ D) / max(var_x, 1e-12))
    return Similarity(R, my - s * R @ mx, s, np.zeros(3))


def icp_similarity(P: np.ndarray, T: np.ndarray, tree_t: Optional[cKDTree] = None,
                   iters: int = 30, tol: float = 1e-7,
                   scale_bounds: Tuple[float, float] = (0.5, 2.0)) -> Similarity:
    """Point-to-point ICP of P onto T with a similarity transform. Correspondences are
    P -> nearest T. Returns the Similarity (about the origin) mapping the INPUT P onto T. The
    scale is clamped so a degenerate correspondence set cannot collapse the shape."""
    P = np.asarray(P, dtype=np.float64); T = np.asarray(T, dtype=np.float64)
    tree_t = tree_t or cKDTree(T)
    cur = Similarity.identity()
    Q = P
    prev = np.inf
    for _ in range(iters):
        d, idx = tree_t.query(Q)
        err = float(d.mean())
        if prev - err < tol:
            break
        prev = err
        cur = umeyama(P, T[idx])
        s = float(np.clip(cur.s, *scale_bounds))
        cur = Similarity(cur.R, cur.t, s, cur.c)
        Q = cur.apply(P)
    return cur
