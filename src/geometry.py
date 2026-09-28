"""Rotations, rigid transforms and the quaternion conventions the demo files use.

CONVENTION, fixed once here: quaternions are **wxyz** (scalar first), matching
`params["obj_quat"]` in a processed demo. Every function in dexcore that touches a quaternion goes
through this module, so the convention is stated in one place rather than assumed in ten.
"""
from __future__ import annotations

import numpy as np


def normalize(v, eps: float = 1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def wxyz_to_R(q) -> np.ndarray:
    """(...,4) wxyz quaternion -> (...,3,3) rotation matrix."""
    q = normalize(np.asarray(q, dtype=np.float64))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R_to_wxyz(R) -> np.ndarray:
    """(...,3,3) -> (...,4) wxyz. Shepperd's method: picks the largest component to divide by."""
    R = np.asarray(R, dtype=np.float64)
    m = lambda i, j: R[..., i, j]
    tr = m(0, 0) + m(1, 1) + m(2, 2)
    q = np.zeros(R.shape[:-2] + (4,), dtype=np.float64)

    c0 = tr > 0
    c1 = (~c0) & (m(0, 0) >= m(1, 1)) & (m(0, 0) >= m(2, 2))
    c2 = (~c0) & (~c1) & (m(1, 1) >= m(2, 2))
    c3 = ~(c0 | c1 | c2)

    def fill(mask, s_expr, comps):
        if not np.any(mask):
            return
        s = s_expr(mask)
        for k, f in enumerate(comps):
            q[..., k][mask] = f(mask, s)

    fill(c0, lambda k: np.sqrt(tr[k] + 1.0) * 2.0,
         [lambda k, s: 0.25 * s,
          lambda k, s: (m(2, 1)[k] - m(1, 2)[k]) / s,
          lambda k, s: (m(0, 2)[k] - m(2, 0)[k]) / s,
          lambda k, s: (m(1, 0)[k] - m(0, 1)[k]) / s])
    fill(c1, lambda k: np.sqrt(1.0 + m(0, 0)[k] - m(1, 1)[k] - m(2, 2)[k]) * 2.0,
         [lambda k, s: (m(2, 1)[k] - m(1, 2)[k]) / s,
          lambda k, s: 0.25 * s,
          lambda k, s: (m(0, 1)[k] + m(1, 0)[k]) / s,
          lambda k, s: (m(0, 2)[k] + m(2, 0)[k]) / s])
    fill(c2, lambda k: np.sqrt(1.0 + m(1, 1)[k] - m(0, 0)[k] - m(2, 2)[k]) * 2.0,
         [lambda k, s: (m(0, 2)[k] - m(2, 0)[k]) / s,
          lambda k, s: (m(0, 1)[k] + m(1, 0)[k]) / s,
          lambda k, s: 0.25 * s,
          lambda k, s: (m(1, 2)[k] + m(2, 1)[k]) / s])
    fill(c3, lambda k: np.sqrt(1.0 + m(2, 2)[k] - m(0, 0)[k] - m(1, 1)[k]) * 2.0,
         [lambda k, s: (m(1, 0)[k] - m(0, 1)[k]) / s,
          lambda k, s: (m(0, 2)[k] + m(2, 0)[k]) / s,
          lambda k, s: (m(1, 2)[k] + m(2, 1)[k]) / s,
          lambda k, s: 0.25 * s])
    return normalize(q)


def unroll_quat(quat) -> np.ndarray:
    """Make a (F,4) quaternion sequence sign-continuous: flip q[t] when it opposes q[t-1].

    q and -q are the same rotation and a per-frame fit picks whichever the solver landed on. A
    finite difference or an interpolation across a flip reads as a 180-degree jump. Flipping is
    exact, not a smoothing step -- but it must happen before ANY temporal operation on obj_quat.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    for t in range(1, len(q)):
        if float(q[t] @ q[t - 1]) < 0.0:
            q[t] *= -1.0
    return q


def slerp(q0, q1, alpha) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions. `alpha` may be an array."""
    q0 = normalize(np.asarray(q0, float))
    q1 = normalize(np.asarray(q1, float))
    if float(q0 @ q1) < 0.0:                       # take the short way round
        q1 = -q1
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    a = np.atleast_1d(np.asarray(alpha, float))
    if dot > 1.0 - 1e-9:                           # nearly identical: lerp is numerically safer
        out = (1 - a)[:, None] * q0 + a[:, None] * q1
    else:
        th = np.arccos(dot)
        s = np.sin(th)
        out = (np.sin((1 - a) * th)[:, None] * q0 + np.sin(a * th)[:, None] * q1) / s
    out = normalize(out)
    return out if np.ndim(alpha) else out[0]


def affine(R, t) -> np.ndarray:
    """(3,3) + (3,) -> (4,4). Also batches: (...,3,3) + (...,3) -> (...,4,4)."""
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    T = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert(T) -> np.ndarray:
    """Inverse of a rigid (...,4,4) transform, using R^T rather than a general solve."""
    T = np.asarray(T, float)
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return affine(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def transform_points(pts, T) -> np.ndarray:
    """Apply (4,4) or (...,4,4) to (...,3) / (...,N,3) points."""
    pts = np.asarray(pts, float)
    T = np.asarray(T, float)
    if T.ndim == 2:
        return pts @ T[:3, :3].T + T[:3, 3]
    return np.einsum("...ij,...nj->...ni", T[..., :3, :3], pts) + T[..., None, :3, 3]


def rodrigues(axis, angle) -> np.ndarray:
    """(3,) axis + scalar angle -> (3,3)."""
    a = normalize(np.asarray(axis, float))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def quat_geodesic(q0, q1) -> np.ndarray:
    """Geodesic angle (rad) between wxyz quaternions, sign-insensitive. Batches over leading dims."""
    q0 = normalize(np.asarray(q0, float))
    q1 = normalize(np.asarray(q1, float))
    d = np.abs(np.sum(q0 * q1, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def orthonormal_frame(x_axis, up_hint) -> np.ndarray:
    """(3,3) rotation whose columns are an orthonormal (x, y, z) built from an axis and a hint.

    `x_axis` is taken exactly; `up_hint` only chooses where +Z points, and is re-orthogonalised
    against x. Used to build an object's canonical frame from its hinge line and its "up when
    closed" direction, where the hinge is a hard fact and up is a preference.
    """
    x = normalize(np.asarray(x_axis, float))
    z = np.asarray(up_hint, float)
    z = z - (z @ x) * x
    n = np.linalg.norm(z)
    if n < 1e-9:
        # the hint was parallel to the axis and says nothing; take any perpendicular
        alt = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z = alt - (alt @ x) * x
        n = np.linalg.norm(z)
    z = z / n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)          # columns


def kabsch(src, dst, weights=None) -> np.ndarray:
    """Least-squares rigid transform T with T @ src ~= dst. -> (4,4).

    src, dst: (N,3). `weights`: (N,) non-negative, optional.

    This is the classic orthogonal Procrustes solution with the reflection guard: if the SVD
    produces a determinant of -1 the fit has mirrored the point set, which is not a rotation, so
    the last singular direction is flipped. Omitting that guard yields a "transform" that fits
    beautifully and turns the hand inside out.
    """
    P = np.asarray(src, dtype=np.float64)
    Q = np.asarray(dst, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"kabsch needs matching (N,3) arrays, got {P.shape} and {Q.shape}")
    if len(P) < 3:
        raise ValueError(f"kabsch needs at least 3 points, got {len(P)}")

    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    if np.any(w < 0):
        raise ValueError("kabsch weights must be non-negative")
    s = w.sum()
    if s <= 0:
        raise ValueError("kabsch weights sum to zero")
    w = w / s

    cp = (w[:, None] * P).sum(0)
    cq = (w[:, None] * Q).sum(0)
    H = (w[:, None] * (P - cp)).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return affine(R, cq - R @ cp)


def obb_axes(points) -> np.ndarray:
    """(3,3) principal axes of a point set, as columns, longest first.

    A PCA box, not a minimal-volume one: it is used only to pick which perpendicular direction is
    "up" on a part, where the principal directions are what matter and the exact box is not.
    """
    P = np.asarray(points, dtype=np.float64)
    C = np.cov((P - P.mean(0)).T)
    vals, vecs = np.linalg.eigh(C)
    return vecs[:, np.argsort(vals)[::-1]]
