"""Shared helpers for Figures B-E (canonical-space figures).

Template rendering: the category template mesh (the mesh `aligned` picked as medoid) is put in the
aligned canonical frame with the backend's own similarity; `dino` reuses the same template and the
same 512 canonical points, so one template rendering serves both backends. A (K,) canonical
vector is splatted onto template vertices by nearest canonical point (cKDTree).
No torch is imported here so the rendering scripts stay open3d-safe.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
CACHE = ROOT / "canonical_contact_cache"
FIG = ROOT / "figures"
FIGDATA = ROOT / "figures_data"
MESH_DICT = "/result/uhnam/dexcore/bimart_taco/assets/taco_mesh_dict.npy"
sys.path.insert(0, "/home/uhnam/workspace/dexcore")

# the six (category, role, hand) triples used by Figures C-E
SIX = [("spatula", "tool", "R"), ("spoon", "tool", "R"), ("bowl", "target", "L"),
       ("plate", "target", "L"), ("pan", "target", "L"), ("kettle", "tool", "R")]
# camera per category (same for every panel of a category); z is up in the PCA frame
# chosen from an 8-view sweep so the contact-bearing part (handle / interior) faces the camera
CAT_VIEW = {"spatula": [0.15, -0.25, 1.0], "spoon": [0.15, -0.25, 1.0], "knife": [0.15, -0.25, 1.0],
            "plate": [0.15, -0.25, 1.0], "bowl": [0.62, 0.95, 0.42], "pan": [0.62, -0.95, 0.42],
            "kettle": [-0.62, 0.95, 0.42]}
CAT_DIST = {"spatula": 2.4, "spoon": 2.4, "knife": 2.4, "plate": 2.6, "bowl": 2.9, "pan": 2.7,
            "kettle": 2.7}


def cat_camera(category: str):
    """(view direction, camera distance) -- the same for every panel of a category."""
    return np.asarray(CAT_VIEW.get(category, [0.62, -0.95, 0.42])), CAT_DIST.get(category, 2.9)
BACKENDS_CANON = ["normalized", "aligned", "dino"]


class Template:
    """Template mesh of a category in the aligned canonical frame + nearest-canonical-point map."""

    def __init__(self, category: str):
        from src.analysis.canonical import backends
        be = backends.load("aligned", ROOT / "canonical_backend")
        self.category = category
        self.template_id = be.template(category)
        meshes = np.load(MESH_DICT, allow_pickle=True).item()
        md = meshes[self.template_id]
        verts_obj = np.asarray(md["verts_original"], dtype=np.float64)
        self.faces = np.asarray(md["faces"], dtype=np.int32)
        self.verts = be.to_canonical(category, self.template_id, verts_obj)       # (N,3) canonical
        self.canonical_points = be.canonical_points(category)                     # (K,3)
        assert self.canonical_points.shape[0] == 512
        self.nearest = cKDTree(self.canonical_points).query(self.verts)[1]        # (N,) -> k

    def splat(self, x: np.ndarray) -> np.ndarray:
        """(K,) canonical vector -> (N,) per-vertex values on the template."""
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.canonical_points.shape[0],)
        return x[self.nearest]


def load_cache(backend: str, category: str, role: str, seq: bool = False):
    p = CACHE / backend / f"{category}__{role}{'__seq' if seq else ''}.npz"
    return np.load(p, allow_pickle=True)


def seq_row(z, sequence_id: str, hand: str) -> int:
    m = np.where((z["sequence_id"] == sequence_id) & (z["hand"] == hand))[0]
    assert len(m) == 1, (sequence_id, hand, len(m))
    return int(m[0])


def dino_fallback() -> dict:
    """(category, mesh_id str) -> fallback fraction from canonical_backend/dino/quality.csv."""
    import pandas as pd
    q = pd.read_csv(ROOT / "canonical_backend" / "dino" / "quality.csv", dtype={"mesh_id": str})
    return {(r.category, f"{int(r.mesh_id):03d}"): float(r.fallback_frac) for r in q.itertuples()}
