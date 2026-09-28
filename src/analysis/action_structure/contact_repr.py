"""WHERE on an object the hand touched, encoded against the same basis the shape code uses.

CIRCULARITY, STATED UP FRONT. Contact was derived from the hand, so feeding it in to predict the
hand is leakage, not a result. It is used here in exactly two ways, and neither pretends otherwise:

    as an ORACLE   contact -> hand answers "if a method somehow knew the contact region, would that
                   be enough to place the fingers?" It is an upper bound on any pipeline that routes
                   through contact, and it is only worth measuring because the answer is not
                   obviously yes.
    as a TARGET    object -> contact is not circular at all: the objects are the input and contact
                   is the thing predicted. If this works, it is the missing bridge -- object
                   geometry to contact region to fingers.

THE ENCODING. Contact points are expressed in the OBJECT'S OWN frame, so the code says "the hand
gripped the handle" rather than "the hand was over there on the table", and then measured against
`bps.basis()` -- the same ruler the shape code uses, so a contact code and a shape code are
directly comparable and can be concatenated without either dominating.

An empty chunk -- no contact at all -- encodes as the clip radius everywhere, which is the honest
reading of "nothing near any basis point" rather than a zero that would look like contact at the
origin.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.analysis.action_structure import bps as BPS
from src.analysis.action_structure.chunks import CHUNK_LEN
from src.analysis.action_structure.contact import TARGET_PART_ID, TOOL_PART_ID

log = logging.getLogger(__name__)

DEFAULT_DIR = Path("/result/uhnam/dexcore/analysis/taco/contacts/full/episodes")


def episode_contact_local(npz, track, part_id: int):
    """(T, n_hits, 3) contact points for one object, in that object's own frame, both hands pooled.

    Returned as a list per frame because the number of hits varies; the caller reduces it.
    """
    from src import geometry as G

    R = G.wxyz_to_R(track.quat)
    out = []
    T = len(track)
    for t in range(T):
        pts = []
        for side in ("left", "right"):
            key = f"contacts_{side}"
            if key not in npz:
                continue
            c = npz[key][t]
            v = npz[f"valid_{side}"][t] & (c[:, 3] == part_id)
            if v.any():
                pts.append(c[v][:, :3])
        if pts:
            p = np.concatenate(pts)
            out.append((p - track.pos[t]) @ R[t])          # R^T (p - pos)
        else:
            out.append(np.zeros((0, 3)))
    return out


def encode_chunk(points_per_frame, basis: np.ndarray, radius: float) -> np.ndarray:
    """(n_basis,) distance from each basis point to the nearest contact point in the whole chunk."""
    from scipy.spatial import cKDTree

    pts = [p for p in points_per_frame if len(p)]
    if not pts:
        return np.full(len(basis), radius)
    allp = np.concatenate(pts)
    d, _ = cKDTree(allp).query(basis, k=1, workers=1)
    return np.minimum(d, radius)


def build(meta, root=None, contact_dir=None, n_basis: int = BPS.N_BASIS,
          radius: float = BPS.BALL_RADIUS_M, seed: int = BPS.BASIS_SEED) -> Dict[str, np.ndarray]:
    """(N, n_basis) contact-region codes per chunk, for the tool and for the target."""
    from src.analysis.loaders import taco

    B = BPS.basis(n_basis, radius, seed)
    contact_dir = Path(contact_dir or DEFAULT_DIR)
    root = Path(root or taco.default_root())
    refs = {r.sequence_id: r for r in taco.index(root)}

    n = len(meta)
    tool = np.full((n, n_basis), radius)
    targ = np.full((n, n_basis), radius)
    pos = {cid: i for i, cid in enumerate(meta["chunk_id"])}

    for ep, g in meta.groupby("episode_id", sort=False):
        f = contact_dir / f"{ep.replace('/', '__')}.npz"
        ref = refs.get(ep)
        if ref is None or not f.exists():
            log.warning("%s: no contact file", ep)
            continue
        z = np.load(f)
        traj = taco.load(ref, root=root, with_hands=False)
        local = {"tool": episode_contact_local(z, traj.tool, TOOL_PART_ID),
                 "target": episode_contact_local(z, traj.target, TARGET_PART_ID)}
        for k in range(int(g["n_chunks"].iloc[0])):
            i = pos.get(f"{ep}#{k}")
            if i is None:
                continue
            sl = slice(k * CHUNK_LEN, (k + 1) * CHUNK_LEN)
            tool[i] = encode_chunk(local["tool"][sl], B, radius)
            targ[i] = encode_chunk(local["target"][sl], B, radius)
    return {"contact_tool": tool, "contact_target": targ}


def scalar_summary(meta, contact_dir=None) -> "object":
    """A handful of scalars per chunk: how much contact, on which object, with how many links.

    Hand-derived like everything else here, so oracle-only.
    """
    import pandas as pd

    contact_dir = Path(contact_dir or DEFAULT_DIR)
    rows = []
    for _, r in meta.iterrows():
        ep, k = r["episode_id"], int(r["chunk_index"])
        f = contact_dir / f"{ep.replace('/', '__')}.npz"
        sl = slice(k * CHUNK_LEN, (k + 1) * CHUNK_LEN)
        rec = {"chunk_id": r["chunk_id"]}
        if not f.exists():
            rows.append(rec)
            continue
        z = np.load(f)
        for side in ("left", "right"):
            v = z[f"valid_{side}"][sl]
            c = z[f"contacts_{side}"][sl]
            l = z[f"links_{side}"][sl]
            pid = np.where(v, c[..., 3], 0)
            rec[f"{side}_frac"] = float(v.any(axis=1).mean())
            rec[f"{side}_pts"] = float(v.sum(axis=1).mean())
            rec[f"{side}_tool_frac"] = float(((pid == TOOL_PART_ID).sum(1) >
                                              (pid == TARGET_PART_ID).sum(1)).mean())
            rec[f"{side}_links"] = float((np.linalg.norm(l[..., :3], axis=-1) > 0).sum(1).mean())
        rows.append(rec)
    return pd.DataFrame(rows)
