"""Contact-patch wrench model and directional wrench capacity.

Patch j: centroid r_j and normal n_j (pointing from the object INTO the hand) in the object frame,
finger / palm label k(j). The hand pushes the object along -n_j with Coulomb friction, linearised
into N_CONE edges of unit normal force:
    e_{j,m} = -n_j + mu (cos(theta_m) t1_j + sin(theta_m) t2_j),   theta_m = 2 pi m / N_CONE
Force  f_j = sum_m lambda_{j,m} e_{j,m},  lambda >= 0  (normal force = sum_m lambda_{j,m}),
wrench w_j = [ f_j ; (r_j - o) / l  x  f_j ]  with the object centroid o and characteristic length
l (max vertex radius of the rest mesh), so force and torque share one normalised unit.
Budgets: per group g (a finger, a patch, or the whole hand)  sum_{j in g} sum_m lambda_{j,m} <= B_g.

Directional capacity for a unit 6-D direction u:
    alpha*(u) = max alpha  s.t.  sum_j w_j = alpha u,  lambda >= 0,  budgets      (an LP, HiGHS).
alpha* = 0 when the grasp cannot produce any wrench along u; no patches -> 0 for every u.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

N_CONE = 8
N_RANDOM = 64


def cone_edges(normal, mu, n_edges=N_CONE):
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("zero patch normal")
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    th = 2 * np.pi * np.arange(n_edges) / n_edges
    return -n[None] + mu * (np.cos(th)[:, None] * t1[None] + np.sin(th)[:, None] * t2[None])


def wrench_basis(patches, origin, length, mu, n_edges=N_CONE):
    """(P * n_edges, 6) edge wrenches and the patch index of every row."""
    rows, owner = [], []
    for j, p in enumerate(patches):
        E = cone_edges(p["normal"], mu, n_edges)
        r = (np.asarray(p["centroid"], float) - origin) / length
        rows.append(np.concatenate([E, np.cross(r[None], E)], 1)); owner += [j] * n_edges
    return (np.concatenate(rows) if rows else np.zeros((0, 6))), np.asarray(owner, int)


def budget_constraints(patches, owner, scheme, hand_total=5.0):
    """A_ub (G, n), b_ub (G,) for the budget scheme."""
    n = len(owner)
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0)
    if scheme == "finger_equal":
        parts = np.array([p["part"] for p in patches]); groups = np.unique(parts)
        A = np.stack([(parts[owner] == g).astype(float) for g in groups]); b = np.ones(len(groups))
    elif scheme == "per_patch":
        A = np.stack([(owner == j).astype(float) for j in range(len(patches))]); b = np.ones(len(patches))
    elif scheme == "hand_total":
        A = np.ones((1, n)); b = np.array([hand_total])
    else:
        raise ValueError(scheme)
    return A, b


def directional_capacity(Wb, A_ub, b_ub, U):
    """alpha*(u) for every row of U (M, 6)."""
    n = len(Wb); out = np.zeros(len(U))
    if n == 0:
        return out
    A = np.concatenate([A_ub, np.zeros((len(A_ub), 1))], 1)
    c = np.zeros(n + 1); c[-1] = -1.0
    bounds = [(0, None)] * (n + 1)
    for i, u in enumerate(U):
        A_eq = np.concatenate([Wb.T, -np.asarray(u, float)[:, None]], 1)
        res = linprog(c, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(6), bounds=bounds, method="highs")
        out[i] = max(-res.fun, 0.0) if res.status == 0 else 0.0
    return out


def support_capacity(Wb, A_ub, b_ub, U):
    """h(u) = max u . (Wb^T lambda) s.t. lambda >= 0 and the group budgets: the support function of
    the achievable wrench polytope. With disjoint groups it is separable: each group spends its whole
    budget on its best edge, h(u) = sum_g B_g max(0, max_{rows of g} (Wb u)_row)."""
    if len(Wb) == 0:
        return np.zeros(len(U))
    S = Wb @ np.asarray(U, float).T                                   # (n, M)
    out = np.zeros(len(U))
    for g in range(len(A_ub)):
        rows = A_ub[g] > 0
        out += b_ub[g] * np.maximum(S[rows].max(0), 0.0)
    return out


def capacity(patches, origin, length, mu, U, scheme="finger_equal", n_edges=N_CONE, strict=True):
    """(h(u), alpha*(u)) over the rows of U: the support-function capacity (primary: the largest wrench
    component the hand can contribute along u; a single hand rarely acts alone) and the strict
    exact-direction capacity (the wrench must be exactly along u; zero outside the grasp's cone)."""
    Wb, owner = wrench_basis(patches, origin, length, mu, n_edges)
    A, b = budget_constraints(patches, owner, scheme)
    h = support_capacity(Wb, A, b, U)
    a = directional_capacity(Wb, A, b, U) if strict else np.full(len(U), np.nan)
    return h, a


def direction_set(n_random=N_RANDOM, seed=0):
    """Shared 6-D unit directions: n_random random (fixed seed) + 12 axis directions (+-force x/y/z,
    +-torque x/y/z). Returns (U (M,6), names)."""
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n_random, 6)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    axes = np.concatenate([np.eye(6), -np.eye(6)])
    names = [f"rand{i}" for i in range(n_random)] + [f"+f{a}" for a in "xyz"] + [f"+t{a}" for a in "xyz"] + [f"-f{a}" for a in "xyz"] + [f"-t{a}" for a in "xyz"]
    return np.concatenate([R, axes]), names


def retention(alpha_pre, alpha_post, eps=1e-6, tiny=1e-6):
    """mean_u min(alpha_pre / (alpha_post + eps), 1) over the directions where either grasp has capacity."""
    m = np.maximum(alpha_pre, alpha_post) > tiny
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.minimum(alpha_pre[m] / (alpha_post[m] + eps), 1.0)))


def profile_metrics(q_pre, q_post, prefix="", eps=1e-9):
    """Scalar summaries of two capability profiles over the shared direction set."""
    Qp, Qq = float(q_pre.mean()), float(q_post.mean())
    cov_p, cov_q = float((q_pre > 1e-6).mean()), float((q_post > 1e-6).mean())
    cos = float((q_pre * q_post).sum() / (np.linalg.norm(q_pre) * np.linalg.norm(q_post) + eps))
    l1_rel = float(np.abs(q_post - q_pre).sum() / ((q_pre + q_post).sum() + eps))
    d = dict(Q_pre=Qp, Q_post=Qq, delta_Q=Qq - Qp, ratio_Q=(Qq + eps) / (Qp + eps), rel_change_Q=(Qq - Qp) / (Qp + eps),
             Qmin_pre=float(q_pre.min()), Qmin_post=float(q_post.min()), Qmed_pre=float(np.median(q_pre)), Qmed_post=float(np.median(q_post)),
             coverage_pre=cov_p, coverage_post=cov_q, cosine=cos, l1_rel_distance=l1_rel,
             R_pre_to_post=retention(q_pre, q_post), R_post_to_pre=retention(q_post, q_pre))
    return {prefix + k: v for k, v in d.items()}


# ------------------------------------------------------------------------------ synthetic check
def synthetic_check(mu=0.5, verbose=True):
    """Two antipodal patches on a unit sphere (thumb at +x with normal into the object, index at -x):
    expected capacities (unit budget per finger, l = 1): push along -x from the thumb alone = 1;
    tangential +y: both press and shear in the same direction = 2 mu cos(pi/8) (8-edge polygon);
    pure torque about z from opposite shears = 2 mu cos(pi/8); a lone patch cannot pull (alpha = 0
    along the direction leaving its surface)."""
    c8 = np.cos(np.pi / 8)
    th = dict(part=1, centroid=np.array([1.0, 0, 0]), normal=np.array([1.0, 0, 0]))     # outward normal +x, hand pushes -x
    ix = dict(part=2, centroid=np.array([-1.0, 0, 0]), normal=np.array([-1.0, 0, 0]))
    o, l = np.zeros(3), 1.0
    s = np.sin(np.pi / 8); c = np.cos(np.pi / 8)
    U = np.array([[-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
                  [0, c, s, 0, 0, 0]], float)                                           # last: shear half-way between two cone edges
    h2, two = capacity([th, ix], o, l, mu, U); h1, one = capacity([th], o, l, mu, U)
    rows = [("strict: two patches, force -x (thumb pushes)", two[0], 1.0), ("strict: two patches, force +x (index pushes)", two[1], 1.0),
            ("strict: two patches, force +y (shear on an edge, both)", two[2], 2 * mu), ("strict: two patches, torque z (opposite shears)", two[3], 2 * mu),
            ("strict: two patches, torque x (no lever arm)", two[4], 0.0), ("strict: two patches, shear between edges", two[5], 2 * mu * c8),
            ("strict: one patch, force -x", one[0], 1.0), ("strict: one patch, force +x (pull)", one[1], 0.0),
            ("strict: one patch, force +y (needs -x balanced)", one[2], 0.0),
            ("support: one patch, force -x", h1[0], 1.0), ("support: one patch, force +x (pull)", h1[1], 0.0),
            ("support: one patch, force +y (shear, unbalanced -x allowed)", h1[2], mu), ("support: two patches, force +y", h2[2], 2 * mu),
            ("support: two patches, torque z", h2[3], 2 * mu), ("support: two patches, force -x (thumb only helps)", h2[0], 1.0)]
    ok = all(abs(a - b) < 1e-3 for _, a, b in rows)
    if verbose:
        for name, a, b in rows:
            print(f"  {name:62s} got {a:.4f}  expected {b:.4f}")
        print("synthetic grasp-map check:", "PASS" if ok else "FAIL")
    return ok, rows


if __name__ == "__main__":
    synthetic_check()
