"""The SEMANTIC backend: every instance mesh is mapped onto its category's template through a
DINOv2 multi-view correspondence, on top of the `aligned` geometric map.

Why a second stage at all. `aligned` overlaps SURFACES: a spatula whose blade is 20% longer than
the template's lands its blade tip in empty space or on the template's blade middle. Nothing in a
similarity transform can say "this is the tip". DINOv2 dense descriptors, matched between renders
of the two objects seen from the SAME viewpoints, can: the descriptor of a rendered blade tip is
closer to the descriptor of the template's blade tip than to its blade middle, so the map moves
the point along the surface to where its counterpart is, not to where it happens to overlap.

What is fixed, deliberately, so the vectors of this backend and of `aligned` are comparable:

    template          the SAME mesh `aligned` chose (its medoid), so `canonical_points` is
                      IDENTICAL to aligned's 512 FPS points and a K-vector from either backend
                      indexes the same template location.
    working frame     the aligned canonical frame. Both the instance and the template are mapped
                      through `aligned.to_canonical` first (centred, unit max-radius, PCA + ICP
                      rotation) and the correspondence engine is called with `frame_R = I` for
                      both, so camera i of the instance and camera i of the template look at the
                      same aspect of the two objects. This is the "paired views" requirement the
                      engine's README calls load-bearing; without it, matches compare different
                      sides of the objects and come out bimodal.
    what is mapped    a 2000-point FPS subset of the instance's vertices; every point gets the
                      engine's top-1 clustered candidate on the template surface. All other
                      vertices (and any query point) follow their nearest FPS point.
    fallback          a point the engine cannot map (invalid, or the best cluster carries less
                      than `min_top1_conf` of the candidate mass, i.e. the match is ambiguous)
                      keeps its aligned coordinate (dst = src, or F * src when the flip search
                      below chose the flip F for the mesh) and is FLAGGED in `fallback`;
                      `quality()` reports the fallback fraction per mesh. The semantic backend
                      therefore degrades to `aligned`, never to a random point.

The map is NOT a similarity and NOT a bijection. `to_canonical` is aligned + nearest-FPS lookup
-> dst; `from_canonical` is the approximate inverse, nearest dst -> src -> aligned inverse. Two
instance points that share an FPS point map to one template point, and a template region no FPS
point was mapped onto is never returned by `from_canonical`. Read `roundtrip_error` as the FPS
spacing, not as a numerical error.

Flip search. `aligned` resolves the sign ambiguity of its PCA axes by Chamfer distance, which
for a spatula whose blade is as long as its handle can put the blade at the handle's end
(spatula 062 -> 184 in the study). Rendered under that frame the paired views compare blade with
handle and nothing matches. So the instance is rendered under each of the four PROPER flips of
the aligned frame (the same set `aligned` searches, passed as the engine's `frame_R` for the
instance cameras) and a flip REPLACES aligned's choice only when it yields at least
`flip_min_margin` (5% of the points) more valid matches than the aligned frame itself. The margin
matters: the two faces of a flat blade, or the two ends of a box, are indistinguishable in a
render, and without it a 7-point difference (spatula 026: 1840 vs 1833) would turn the object
over relative to `aligned` for no reason. With it, DINO overrides Chamfer only where the evidence
is clear (kettle 029: 620 vs 163; spatula 062: 363 vs 11). `flip_n_valid` keeps all four counts
so the ambiguity of a mesh can be read off. When a flip F is taken, the fallback of that mesh is
F * src -- the aligned coordinate with the sign DINO chose -- not src, otherwise a fallback point
of a blade-to-handle-flipped spatula would sit at the wrong end of the template. The stored source
points stay in the aligned frame, so `to_canonical` and the
displacement diagnostic are unchanged; a flipped instance simply shows a displacement of the
order of the object's length, which is the honest number. Axis ORDER swaps (a tall tumbler whose
long axis is its height, against a wide teacup template) are not searched: that is 24 frames per
mesh, and those categories are ambiguous for DINO as well.

Engine knobs. The correspondence engine's thresholds are metric (metres) because it was written for
objects in their own frames. Here everything lives in the unit-radius canonical frame, so they are
restated in canonical units: DBSCAN eps 0.06 (a few render pixels at 256 px, camera at 2.6 radii),
resnap / snap 0.05, depth tolerance 0.01. Two matcher settings were changed after measuring TACO's
thin tools (a knife occupies ~2500 of 65536 pixels, i.e. ~13 DINO patches at the native 14 px
patch): the render is bilinearly UPSAMPLED 2x before the ViT so the effective patch is 7 render
pixels, and the mutual-consistency tolerance / confidence floor were loosened to 16 px / 0.25.
On knife 045 -> 202 that took the valid fraction from 10% to 50%; on spatula and spoon from
~80% to ~92%. Everything is deterministic: FPS, Fibonacci cameras, the rasteriser, DINO in eval
mode, DBSCAN.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.canonical.backends._similarity import (CategoryFit, slug, splat_values)
from src.analysis.canonical.backends.aligned import AlignedBackend
from src.analysis.canonical.geometry import PROPER_FLIPS, fps_indices
from src.analysis.canonical.interface import (DEFAULT_RADIUS, N_CANONICAL, CanonicalBackend,
                                              MeshDict)

log = logging.getLogger(__name__)

#: Fields stored per FPS point, in the order they appear in the npz (prefix "dino_").
POINT_FIELDS = ("src", "dst", "dst_raw", "top1_conf", "match_conf", "cluster_size", "n_views",
                "n_cand", "valid", "fallback", "surf_dist")

DEFAULT_ENGINE = dict(n_views=24, image_size=256, camera_radius_scale=2.6, upsample=2,
                      tau=0.6, mutual_px_tol=16.0, min_match_confidence=0.25,
                      dbscan_eps=0.06, resnap_max_dist=0.05, surface_snap_max_dist=0.05,
                      visibility_depth_tol=0.01, top_k=3, flip_search=True, flip_min_margin=0.05)


# ==================================================================================================
# The correspondence call (the only part that touches the DexMachina engine)
# ==================================================================================================
@dataclass
class MapResult:
    """Top-1 correspondence of n source points onto the template surface (canonical frame)."""
    dst: np.ndarray            # (n,3) NaN where the engine returned no valid candidate
    top1_conf: np.ndarray      # (n,) normalised confidence of the best cluster (1 = unimodal)
    match_conf: np.ndarray     # (n,) mean DINO match confidence of the best cluster, NaN if invalid
    cluster_size: np.ndarray   # (n,) int
    n_views: np.ndarray        # (n,) int, distinct target views supporting the best cluster
    n_cand: np.ndarray         # (n,) int, number of clusters the engine returned (0..top_k)
    valid: np.ndarray          # (n,) bool, the engine's own validity
    reason: List[str]          # (n,) failure reason, "" when valid
    flip: np.ndarray = None    # (3,3) proper flip of the aligned frame the instance was rendered in
    flip_n_valid: np.ndarray = None   # (4,) valid count under each of PROPER_FLIPS


def _dexmachina_root() -> Path:
    """The DexMachina WORKING TREE with the exp3 experiments (same rule as src/transfer/shapegen)."""
    env = os.environ.get("DEXCORE_DEXMACHINA_SRC")
    for c in ([Path(env)] if env else [Path.home() / "workspace" / "dexmachina"]):
        if (c / "dexmachina" / "eval" / "experiments" / "exp3_demo_gen_baselines").exists():
            return c
    raise FileNotFoundError("DexMachina exp3 tree not found; set $DEXCORE_DEXMACHINA_SRC")


def _import_engine():
    try:
        from dexmachina.eval.experiments.exp3_demo_gen_baselines.func_warp_reference import \
            correspondence as C
    except ImportError:
        root = str(_dexmachina_root())
        if root not in sys.path:
            sys.path.append(root)                     # append: never shadow the repo's own packages
        from dexmachina.eval.experiments.exp3_demo_gen_baselines.func_warp_reference import \
            correspondence as C
    from dexmachina.eval.experiments.exp3_demo_gen_baselines.func_warp_reference.correspondence \
        import dino_matcher
    return C, dino_matcher


class EngineMapper:
    """Wraps `ContactCorrespondenceEngine` + `DinoMatcher` for canonical-frame meshes.

    `prepare_template(category, verts, faces)` renders and caches the template once;
    `map_points(category, verts, faces, pts)` maps `pts` (on the instance surface, canonical frame)
    onto the template. Construction is cheap; the engine and DINO weights load on first use.
    """

    def __init__(self, device: str = "cuda", **engine_kw):
        self.device = device
        self.kw = {**DEFAULT_ENGINE, **engine_kw}
        self._engine = None
        self._matcher = None
        self._C = None
        self._templates: Dict[str, Tuple[object, list]] = {}      # category -> (cache, views rgb)

    # ---- lazy construction ------------------------------------------------------------------
    def _ensure(self):
        if self._engine is not None:
            return
        C, dm = _import_engine()
        kw = self.kw
        cfg = C.CorrespondenceConfig(
            top_k=kw["top_k"], n_target_views=kw["n_views"], image_height=kw["image_size"],
            image_width=kw["image_size"], camera_radius_scale=kw["camera_radius_scale"],
            paired_views=True, feature_channels="rgb", restrict_to_part=True, device="cpu",
            min_match_confidence=kw["min_match_confidence"],
            surface_snap_max_dist=kw["surface_snap_max_dist"],
            visibility_depth_tol=kw["visibility_depth_tol"], resnap_max_dist=kw["resnap_max_dist"],
            dbscan_eps=kw["dbscan_eps"])

        class UpsampledDinoMatcher(dm.DinoMatcher):
            """DinoMatcher on a bilinearly upsampled render: effective patch = 14 / upsample px."""

            def __init__(self, upsample: int, **mkw):
                super().__init__(**mkw)
                self.upsample = int(upsample)

            def _features_for(self, rgb):
                import torch
                import torch.nn.functional as Fnn
                model = self._lazy_model()
                P = dm.PATCH
                H, W = rgb.shape[:2]
                x = torch.as_tensor(np.ascontiguousarray(rgb[..., :3]), dtype=torch.float32,
                                    device=self.device).permute(2, 0, 1)[None]
                Hu, Wu = (H * self.upsample) // P * P, (W * self.upsample) // P * P
                x = Fnn.interpolate(x, size=(Hu, Wu), mode="bilinear", align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
                x = (x - mean) / std
                with torch.no_grad():
                    tok = model.forward_features(x)["x_norm_patchtokens"][0]
                feat = tok.reshape(Hu // P, Wu // P, -1).permute(2, 0, 1)[None]
                feat = Fnn.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
                feat = Fnn.normalize(feat[0], dim=0).permute(1, 2, 0)
                return feat.cpu().numpy()

            def drop_cache_except(self, keep) -> None:
                """Free source-frame descriptors; keep the (shared) template views' ones."""
                keep_ids = {id(img) for img in keep}
                self._cache = {k: v for k, v in self._cache.items() if k in keep_ids}

        self._matcher = UpsampledDinoMatcher(kw["upsample"], device=self.device, tau=kw["tau"],
                                             mutual_px_tol=kw["mutual_px_tol"])
        self._engine = C.ContactCorrespondenceEngine(cfg, self._matcher)
        self._C = C

    def _cameras(self, verts: np.ndarray, faces: np.ndarray, frame_R: np.ndarray) -> list:
        """Cameras in the aligned canonical frame; `frame_R` = identity for the template, a
        proper flip for the instance during the flip search."""
        C, kw = self._C, self.kw
        parts = [C.PartMesh(0, verts, faces)]
        return C.default_render_cameras(parts, kw["n_views"], kw["image_size"], kw["image_size"],
                                        self._engine.cfg.camera_fov_deg,
                                        kw["camera_radius_scale"], frame_R=frame_R)

    # ---- API --------------------------------------------------------------------------------
    def prepare_template(self, category: str, verts: np.ndarray, faces: np.ndarray) -> None:
        self._ensure()
        t0 = time.time()
        cams = self._cameras(verts, faces, np.eye(3))
        cache = self._engine.prepare_target((verts, faces), target_render_cameras=cams)
        self._templates[category] = (cache, [v.rgb for v in cache.views])
        log.info("%-12s template rendered (%d views) in %.1fs", category, len(cams),
                 time.time() - t0)

    def map_points(self, category: str, verts: np.ndarray, faces: np.ndarray,
                   pts: np.ndarray) -> MapResult:
        """Instance -> template. With `flip_search` the instance is rendered under each proper
        flip of the aligned frame (index 0 = the aligned frame itself); a flip is taken only if
        it has at least `flip_min_margin` * n more valid matches than the aligned frame."""
        self._ensure()
        flips = PROPER_FLIPS if self.kw["flip_search"] else PROPER_FLIPS[:1]
        results, n_valid = [], np.zeros(len(PROPER_FLIPS), np.int64)
        for fi, F in enumerate(flips):
            results.append(self._map_once(category, verts, faces, pts, F))
            n_valid[fi] = int(results[-1].valid.sum())
        best = int(np.argmax(n_valid[:len(flips)]))            # ties -> lowest index (aligned)
        if n_valid[best] - n_valid[0] < self.kw["flip_min_margin"] * len(pts):
            best = 0
        res = results[best]
        res.flip_n_valid = n_valid
        return res

    def _map_once(self, category: str, verts: np.ndarray, faces: np.ndarray, pts: np.ndarray,
                  frame_R: np.ndarray) -> MapResult:
        cache, keep = self._templates[category]
        cams = self._cameras(verts, faces, frame_R)
        frames = self._engine.render_source_frames((verts, faces), cams)
        results = self._engine.transfer_contacts((verts, faces), pts, frames, cams, cache)
        self._matcher.drop_cache_except(keep)
        n = len(pts)
        out = MapResult(np.full((n, 3), np.nan), np.zeros(n), np.full(n, np.nan),
                        np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
                        np.zeros(n, bool), [""] * n, flip=np.asarray(frame_R, float))
        for r in results:
            i = r.source_index
            out.n_cand[i] = len(r.candidates)
            if not r.valid:
                out.reason[i] = str(r.failure_reason)
                continue
            b = r.candidates[0]
            out.dst[i] = b.target_point
            out.top1_conf[i] = b.confidence
            out.match_conf[i] = b.mean_match_confidence
            out.cluster_size[i] = b.cluster_size
            out.n_views[i] = b.n_target_views
            out.valid[i] = True
        return out


# ==================================================================================================
# Per-category state
# ==================================================================================================
@dataclass
class MeshMap:
    """The semantic map of one mesh: FPS subset -> template surface, plus diagnostics."""
    fps_idx: np.ndarray        # (n,) indices into the mesh's vertices
    src: np.ndarray            # (n,3) FPS points, canonical (aligned) frame
    dst: np.ndarray            # (n,3) on the template surface; == src @ flip where `fallback`
    dst_raw: np.ndarray        # (n,3) engine output, NaN where invalid
    top1_conf: np.ndarray      # (n,)
    match_conf: np.ndarray     # (n,) NaN where invalid
    cluster_size: np.ndarray   # (n,)
    n_views: np.ndarray        # (n,)
    n_cand: np.ndarray         # (n,)
    valid: np.ndarray          # (n,) bool  engine validity
    fallback: np.ndarray       # (n,) bool  dst = src (invalid OR ambiguous)
    surf_dist: np.ndarray      # (n,) distance of dst to the template surface (0 for fallback)
    reason: List[str]          # (n,)
    time_s: float = 0.0
    flip: np.ndarray = field(default_factory=lambda: np.eye(3))      # frame the instance was rendered in
    flip_n_valid: np.ndarray = field(default_factory=lambda: np.zeros(4, np.int64))
    _src_tree: Optional[cKDTree] = field(default=None, repr=False)
    _dst_tree: Optional[cKDTree] = field(default=None, repr=False)

    @property
    def displacement(self) -> np.ndarray:
        """|dst - src| per point: how far the semantic map moves a point away from where the
        geometric alignment put it (canonical units). Where `fallback`, this is 0 for a mesh in
        the aligned frame and the flip displacement for a mesh whose sign DINO corrected."""
        return np.linalg.norm(self.dst - self.src, axis=1)

    def src_tree(self) -> cKDTree:
        if self._src_tree is None:
            self._src_tree = cKDTree(self.src)
        return self._src_tree

    def dst_tree(self) -> cKDTree:
        if self._dst_tree is None:
            self._dst_tree = cKDTree(self.dst)
        return self._dst_tree


@dataclass
class DinoCategory:
    category: str
    aligned: CategoryFit                       # the geometric map this backend builds on
    maps: Dict[str, MeshMap]
    _mapped: Dict[str, Tuple[np.ndarray, cKDTree]] = field(default_factory=dict, repr=False)

    @property
    def mesh_ids(self) -> List[str]:
        return list(self.aligned.mesh_ids)

    @property
    def template(self) -> str:
        return self.aligned.template


# ==================================================================================================
# The backend
# ==================================================================================================
class DinoBackend(CanonicalBackend):
    """aligned + DINOv2 correspondence onto the aligned template. Semantic; not a bijection."""

    name = "dino"

    def __init__(self, aligned: Optional[AlignedBackend] = None, n_sample: int = 2000,
                 k: int = N_CANONICAL, min_top1_conf: float = 0.5, device: str = "cuda",
                 mapper: Optional[object] = None, **engine_kw):
        self.aligned = aligned if aligned is not None else AlignedBackend(n_sample=n_sample, k=k)
        self.n_sample = int(n_sample)
        self.k = int(k)
        self.min_top1_conf = float(min_top1_conf)
        self.device = device
        self.engine_kw = {**DEFAULT_ENGINE, **engine_kw}
        # `mapper` is injectable (tests use an oracle); the real one is built lazily
        self._mapper = mapper
        self._cats: Dict[str, DinoCategory] = {}

    @property
    def mapper(self):
        if self._mapper is None:
            self._mapper = EngineMapper(device=self.device, **self.engine_kw)
        return self._mapper

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, category: str, meshes: MeshDict) -> None:
        """Map every mesh of `category` onto the aligned template. Fits `aligned` first if it has
        not seen the category (in the study it is loaded from cache so the templates agree)."""
        if category not in self.aligned.categories():
            self.aligned.fit(category, meshes)
        faces = {str(i): np.asarray(f, dtype=np.int64) for i, (_v, f) in meshes.items()}
        fit = self.aligned._cat(category)
        tpl = fit.template
        CT = fit.sims[tpl].apply(fit.verts[tpl])
        self.mapper.prepare_template(category, CT, faces[tpl])
        tpl_surface = _SurfaceDistance(CT, faces[tpl])
        maps: Dict[str, MeshMap] = {}
        for mid in fit.mesh_ids:
            t0 = time.time()
            CI = fit.sims[mid].apply(fit.verts[mid])
            idx = fps_indices(CI, self.n_sample)
            src = CI[idx]
            if mid == tpl:
                maps[mid] = self._identity_map(idx, src)
            else:
                res = self.mapper.map_points(category, CI, faces[mid], src)
                maps[mid] = self._finish(idx, src, res, tpl_surface)
            maps[mid].time_s = time.time() - t0
            m = maps[mid]
            log.info("%-12s %s -> %s  valid %.3f  fallback %.3f  conf %.3f  disp %.3f  %.1fs",
                     category, mid, tpl, m.valid.mean(), m.fallback.mean(),
                     np.nanmean(m.match_conf) if m.valid.any() else np.nan,
                     m.displacement[~m.fallback].mean() if (~m.fallback).any() else 0.0, m.time_s)
        self._cats[category] = DinoCategory(category, fit, maps)

    def _identity_map(self, idx: np.ndarray, src: np.ndarray) -> MeshMap:
        n = len(src)
        return MeshMap(idx, src, src.copy(), src.copy(), np.ones(n), np.ones(n),
                       np.zeros(n, np.int64), np.zeros(n, np.int64), np.ones(n, np.int64),
                       np.ones(n, bool), np.zeros(n, bool), np.zeros(n), ["template:identity"] * n,
                       flip=np.eye(3), flip_n_valid=np.array([n, 0, 0, 0]))

    def _finish(self, idx: np.ndarray, src: np.ndarray, res: MapResult,
                tpl_surface: "_SurfaceDistance") -> MeshMap:
        ambiguous = res.valid & (res.top1_conf < self.min_top1_conf)
        fallback = ~res.valid | ambiguous
        reason = list(res.reason)
        for i in np.flatnonzero(ambiguous):
            reason[i] = f"ambiguous: top-1 confidence {res.top1_conf[i]:.2f} < {self.min_top1_conf}"
        flip = np.eye(3) if res.flip is None else np.asarray(res.flip, float)
        fnv = np.zeros(4, np.int64) if res.flip_n_valid is None else np.asarray(res.flip_n_valid)
        # fallback = the geometric map under the sign DINO chose (identity: plain aligned)
        dst = np.where(fallback[:, None], src @ flip.T, res.dst)
        surf = np.zeros(len(src))
        if (~fallback).any():
            surf[~fallback] = tpl_surface(dst[~fallback])
        return MeshMap(idx, src, dst, res.dst, res.top1_conf, res.match_conf, res.cluster_size,
                       res.n_views, res.n_cand, res.valid, fallback, surf, reason,
                       flip=flip, flip_n_valid=fnv)

    # ---- CanonicalBackend -------------------------------------------------------------------
    def categories(self) -> List[str]:
        return sorted(self._cats)

    def mesh_ids(self, category: str) -> List[str]:
        return self._cat(category).mesh_ids

    def template(self, category: str) -> str:
        return self._cat(category).template

    def mesh_map(self, category: str, mesh_id: str) -> MeshMap:
        return self._cat(category).maps[mesh_id]

    def to_canonical(self, category: str, mesh_id: str, points: np.ndarray) -> np.ndarray:
        """aligned.to_canonical, then each point takes the template point of its nearest FPS
        point (piecewise constant; fallback points return their aligned coordinate)."""
        m = self._cat(category).maps[mesh_id]
        C = self.aligned.to_canonical(category, mesh_id, points)
        _d, k = m.src_tree().query(np.atleast_2d(C))
        return m.dst[k]

    def from_canonical(self, category: str, mesh_id: str, coords: np.ndarray) -> np.ndarray:
        """APPROXIMATE inverse: nearest mapped template point -> its FPS source point -> aligned
        inverse. Not a bijection (see the module docstring)."""
        m = self._cat(category).maps[mesh_id]
        _d, k = m.dst_tree().query(np.atleast_2d(coords))
        return self.aligned.from_canonical(category, mesh_id, m.src[k])

    def canonical_points(self, category: str) -> np.ndarray:
        return self._cat(category).aligned.canonical_points          # identical to aligned's

    def mapped(self, category: str, mesh_id: str) -> Tuple[np.ndarray, cKDTree]:
        """All vertices of the mesh mapped onto the template (cached), with a KD-tree."""
        cat = self._cat(category)
        if mesh_id not in cat._mapped:
            V = self.to_canonical(category, mesh_id, cat.aligned.verts[mesh_id])
            cat._mapped[mesh_id] = (V, cKDTree(V))
        return cat._mapped[mesh_id]

    def splat(self, category: str, mesh_id: str, per_vertex_values: np.ndarray,
              kernel: str = "nearest", radius: float = DEFAULT_RADIUS,
              sigma: Optional[float] = None, return_coverage: bool = False):
        cat = self._cat(category)
        vals = np.asarray(per_vertex_values, dtype=np.float64)
        n = len(cat.aligned.verts[mesh_id])
        if vals.shape != (n,):
            raise ValueError(f"expected ({n},) values for mesh {mesh_id}, got {vals.shape}")
        _V, tree = self.mapped(category, mesh_id)
        out, covered = splat_values(tree, vals, cat.aligned.canonical_points, kernel, radius, sigma)
        return (out, covered) if return_coverage else out

    def coverage(self, category: str, mesh_id: str, radius: float = DEFAULT_RADIUS) -> float:
        _V, tree = self.mapped(category, mesh_id)
        return float((tree.query(self.canonical_points(category))[0] <= radius).mean())

    def roundtrip_error(self, category: str, mesh_id: str) -> float:
        """Max |from_canonical(to_canonical(v)) - v| over the vertices, in metres. This is the
        FPS spacing in object units, NOT a numerical error: the map is many-to-one."""
        V = self._cat(category).aligned.verts[mesh_id]
        return float(np.linalg.norm(self.from_canonical(
            category, mesh_id, self.to_canonical(category, mesh_id, V)) - V, axis=1).max())

    def quality(self, category: str) -> Dict[str, Dict[str, float]]:
        """mesh_id -> n_valid, fallback_frac, mean/median match confidence (valid points), mean
        displacement vs aligned (non-fallback points, canonical units), mean distance of the
        mapped points to the template surface, wall time."""
        out = {}
        for mid, m in self._cat(category).maps.items():
            ok, kept = m.valid, ~m.fallback
            out[mid] = {
                "n_points": int(len(m.src)), "n_valid": int(ok.sum()),
                "fallback_frac": float(m.fallback.mean()),
                "conf_mean": float(np.nanmean(m.match_conf[ok])) if ok.any() else float("nan"),
                "conf_median": float(np.nanmedian(m.match_conf[ok])) if ok.any() else float("nan"),
                "top1_conf_mean": float(m.top1_conf[ok].mean()) if ok.any() else float("nan"),
                "disp_mean": float(m.displacement[kept].mean()) if kept.any() else float("nan"),
                "disp_median": float(np.median(m.displacement[kept])) if kept.any() else float("nan"),
                "surf_dist_mean": float(m.surf_dist[kept].mean()) if kept.any() else float("nan"),
                "n_views_mean": float(m.n_views[ok].mean()) if ok.any() else float("nan"),
                "flip": flip_name(m.flip),
                "flip_margin": float((np.sort(m.flip_n_valid)[-1] - np.sort(m.flip_n_valid)[-2])
                                     / max(len(m.src), 1)),
                "time_s": float(m.time_s),
            }
        return out

    # ---- persistence ------------------------------------------------------------------------
    def _category_arrays(self, cat: DinoCategory) -> Dict[str, np.ndarray]:
        arrays = cat.aligned.to_arrays()             # the aligned fit travels with the file
        ids = cat.mesh_ids
        maps = [cat.maps[i] for i in ids]
        arrays["dino_fps_idx"] = np.stack([m.fps_idx for m in maps])
        for f in POINT_FIELDS:
            arrays["dino_" + f] = np.stack([getattr(m, f) for m in maps])
        arrays["dino_reason"] = np.array([m.reason for m in maps])
        arrays["dino_time_s"] = np.array([m.time_s for m in maps])
        arrays["dino_flip"] = np.stack([m.flip for m in maps])
        arrays["dino_flip_n_valid"] = np.stack([m.flip_n_valid for m in maps])
        return arrays

    def save_category(self, category: str, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{slug(category)}.npz"
        np.savez_compressed(path, **self._category_arrays(self._cat(category)))
        return path

    def _meta(self) -> dict:
        return {"backend": self.name, "n_sample": self.n_sample, "k": self.k,
                "min_top1_conf": self.min_top1_conf, "engine": self.engine_kw,
                "categories": {cat: {"template": c.template, "n_meshes": len(c.mesh_ids),
                                     "meshes": self.quality(cat)}
                               for cat, c in self._cats.items()}}

    def write_templates_json(self, directory: Union[str, Path]) -> Path:
        path = Path(directory) / "templates.json"
        path.write_text(json.dumps(self._meta(), indent=1))
        return path

    def save(self, directory: Union[str, Path]) -> Path:
        for cat in self._cats:
            self.save_category(cat, directory)
        self.write_templates_json(directory)
        return Path(directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DinoBackend":
        directory = Path(directory)
        meta = {}
        if (directory / "templates.json").exists():
            meta = json.loads((directory / "templates.json").read_text())
        engine_kw = meta.get("engine", {})
        self = cls(aligned=AlignedBackend(n_sample=meta.get("n_sample", 2000),
                                          k=meta.get("k", N_CANONICAL)),
                   n_sample=meta.get("n_sample", 2000), k=meta.get("k", N_CANONICAL),
                   min_top1_conf=meta.get("min_top1_conf", 0.5), **engine_kw)
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                fit = CategoryFit.from_arrays(z)
                maps = {}
                for n, mid in enumerate(fit.mesh_ids):
                    kw = {f: z["dino_" + f][n] for f in POINT_FIELDS}
                    maps[mid] = MeshMap(z["dino_fps_idx"][n], reason=[str(s) for s in z["dino_reason"][n]],
                                        time_s=float(z["dino_time_s"][n]), flip=z["dino_flip"][n],
                                        flip_n_valid=z["dino_flip_n_valid"][n], **kw)
            self.aligned._cats[fit.category] = fit
            self._cats[fit.category] = DinoCategory(fit.category, fit, maps)
        return self

    # ---- helpers ----------------------------------------------------------------------------
    def _cat(self, category: str) -> DinoCategory:
        try:
            return self._cats[category]
        except KeyError:
            raise KeyError(f"{self.name}: category {category!r} not fitted; "
                           f"have {self.categories()}") from None


def flip_name(F: np.ndarray) -> str:
    """'+++', '+--', '-+-' or '--+': the sign of each aligned axis the instance was rendered with."""
    return "".join("+" if d > 0 else "-" for d in np.diag(F))


class _SurfaceDistance:
    """Distance of points to the template: to its triangles when it has faces (the TACO meshes),
    to its vertices when it is a bare point set (the interface allows empty face lists; the
    geometric backends never read them)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.verts = np.asarray(verts, float)
        faces = np.asarray(faces, np.int64).reshape(-1, 3)
        self.mesh = None
        if len(faces):
            import trimesh
            self.mesh = trimesh.Trimesh(self.verts, faces, process=False)
        self._tree = None if self.mesh is not None else cKDTree(self.verts)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        if self.mesh is None:
            return self._tree.query(pts)[0]
        import trimesh
        _c, d, _f = trimesh.proximity.closest_point(self.mesh, pts)
        return np.asarray(d, float)
