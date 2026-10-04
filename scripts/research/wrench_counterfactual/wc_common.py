"""Wrench counterfactual analysis of post-grasp contact reconfiguration: shared definitions.

Question. When a persistent contact transition occurs inside a maintained grasp, could the
PRE-transition grasp have supported essentially the same manipulation wrench as the POST-transition
grasp?  Analysis only: GT hands, GT objects, the existing event labels; no model, no physics rollout.

Everything is inherited:
  events / frames    reports/hand_contact_predictive_info/<ds>/cache/{events_all,frames_all}.csv (the
                     temporal-event definitions on every split; identical to the earlier test-set labels)
  sequences, split   the hierarchical study (hp_common.load_raw): C_t (512-D), take keys, t0 per sequence
  contact distances  TACO gen3 `contact_{left,right}` (hand skin -> object vertex, tool verts then target
                     verts); ARCTIC BimArt `contact_dict[side]['dist' | 'index']` (global vertex order,
                     nearest hand vertex per object vertex) -- both reproduce exactly from their meshes and
                     hands (verified in plan.md)
  meshes             TACO `taco_mesh_dict.npy` (verts_original, faces; the object's own frame, metres);
                     ARCTIC object templates (`mesh.obj` / 1000 + parts.json; part 0 = top, 1 = bottom),
                     articulated by rotvec (0, 0, -arti) on the top part (BimArt's object_verts_to_world)
  hands              TACO recomputed from raw (loader + MANO, reproduces the cached sampled points);
                     ARCTIC `{left,right}_hand_verts` (world) -> rigid frame with obj_world_state
  finger labels      MANO skinning-weight argmax (16 joints -> palm / thumb / index / middle / ring / little)

Frames. An event [s, e] is a run of spike TRANSITIONS; transition t changes C_t -> C_{t+1}. So the
pre-event map is C_s and the post-event map is C_{e+1}: pre frame = s, post frame = e + 1 (sequence
frames; take frame = t0 + frame). Both must be firm-contact frames for the primary class.

Object frame. The contacted object's own frame (TACO: tool or target mesh frame; ARCTIC: the rigid
canonical frame R^T (x - t), in which the bottom part is fixed and the top part articulates).
Contact patches are attached to mesh vertex ids, so their transport through the object trajectory is
the identity for rigid objects and the articulation for ARCTIC's top part.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path("/home/uhnam/workspace/dexcore")
for _p in (REPO / "scripts/research/hand_contact_predictive_info", REPO / "scripts/research/temporal_contact_events",
           REPO / "scripts/research/hier_contact_gen", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hp_common as HP  # noqa: E402

OUT = Path("/result/uhnam/dexcore/reports/wrench_counterfactual")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T = 64
CONTACT_THR = {"primary": 0.010, "tight": 0.005, "loose": 0.015}     # metres; 1 cm = the hard-contact rule
STORE_THR = 0.02                                                     # vertices stored per frame (<= 2 cm)
PARTS = ["palm", "thumb", "index", "middle", "ring", "little"]
JOINT_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])   # MANO joint order: wrist, index x3, middle x3, little x3, ring x3, thumb x3
MANO_ROOT = Path("/home/uhnam/workspace/TACO-Instructions/dataset_utils/manopth/mano/models")
BIMART = REPO / "third_party/BimArt"
ARCTIC_PROC = BIMART / "data/arctic_processed_data"
ARCTIC_TEMPL = BIMART / "data/arctic_raw/meta/object_vtemplates"
TACO_GEN3 = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
TACO_WINDOW_INDEX = Path("/result/uhnam/dexcore/taco/40_representation_study/window_index.csv")
K_FUTURE = (8, 16)
MU_SET = (0.3, 0.5, 0.8, 1.0)
MU_PRIMARY = 0.5
N_CONE = 8
MERGE_R = {"primary": 0.010, "none": 0.0, "wide": 0.020}           # patch merge radius (metres) within a finger
BUDGETS = {"finger_equal": "one unit of normal force per finger (palm too), shared by the finger's patches",
           "per_patch": "one unit per PATCH (the independent-actuator assumption, for contrast)",
           "hand_total": "five units for the whole hand, no per-finger limit"}
BUDGET_PRIMARY = "finger_equal"
UP_WORLD = np.array([0.0, 0.0, 1.0])                                 # +z up in both datasets (object heights 0.55-0.76 m TACO, 0.97-1.43 m ARCTIC)
CLASSES = ["persistent_spatial", "persistent_mixed", "onset", "release", "transient", "amount"]
CLASS_LABEL = {"persistent_spatial": "persistent spatial", "persistent_mixed": "persistent mixed (firm throughout)", "onset": "onset / regrasp",
               "release": "release", "transient": "transient (diagnostic)", "amount": "amount-dominant (diagnostic)"}


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=HP._json_default))


# ------------------------------------------------------------------------------ MANO finger labels
def mano_labels():
    """(778,) part index per MANO vertex (0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 little), from the
    skinning-weight argmax; identical for the left and right templates (checked)."""
    p = OUT / "cache" / "mano_vertex_parts.npz"
    if p.exists():
        return np.load(p)["part"]
    for name, val in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("str", str), ("unicode", str)):
        if not hasattr(np, name):
            setattr(np, name, val)
    if str(MANO_ROOT.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(MANO_ROOT.parent.parent.parent))
    m = pickle.load(open(MANO_ROOT / "MANO_RIGHT.pkl", "rb"), encoding="latin1")
    joint = np.asarray(m["weights"]).argmax(1)
    part = JOINT_PART[joint].astype(np.int8)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, part=part, joint=joint.astype(np.int8), v_template=np.asarray(m["v_template"]), faces=np.asarray(m["f"]))
    return part


def mano_faces():
    mano_labels()
    return np.load(OUT / "cache" / "mano_vertex_parts.npz")["faces"]


# ------------------------------------------------------------------------------ event selection
def classify(E):
    cat = E.event_category.values; pers = E.persistent_flag.values
    cls = np.where(np.isin(cat, ["onset", "onset+release"]), "onset",
          np.where(cat == "release", "release",
          np.where(cat == "spatial", np.where(pers, "persistent_spatial", "transient"),
          np.where(cat == "mixed", np.where(pers, "persistent_mixed", "transient"),
          np.where(cat == "amount", "amount", "other")))))
    return cls


def select_events(ds, split="test", classes=None):
    """Events of one split with pre / post frames and validity flags. Written to
    <ds>/events_selected.csv (test) or <ds>/events_selected_train.csv (train: persistent spatial only,
    used for data-independent thresholds)."""
    E = HP.load_events(ds); F = HP.load_frames(ds)
    D = HP.load_raw(ds); meta = D["meta"]; C = D["C"]
    te = E[E.split_set == split].copy().reset_index(drop=True)
    if classes is not None:
        te = te[np.isin(classify(te), classes)].reset_index(drop=True)
    firm = F.pivot(index="example", columns="t", values="firm_t")
    # frame 63 has no transition row: its firm state is firm_t1 of transition 62
    f63 = F[F.t == 62].set_index("example").firm_t1 if "firm_t1" in F.columns else None
    rows = []
    for r in te.itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        pre, post = s, e + 1
        firm_row = firm.loc[r.example]
        def is_firm(t):
            if t <= 62:
                return bool(firm_row[t])
            return bool(r.firm_after)                                 # frame 63: the event's stored post state
        firm_pre, firm_post = is_firm(pre), is_firm(post)
        firm_through = all(is_firm(t) for t in range(pre, post + 1))
        m = meta.iloc[r.example]
        dC = float(np.linalg.norm(C[r.example, post].astype(np.float32) - C[r.example, pre].astype(np.float32)))
        n_take = int(m.n_frames_take); t0 = int(m.t0)
        rows.append(dict(dataset=ds, event_id=r.event_id, example=int(r.example), sequence_id=r.sequence_id, take_key=r.take_key, group=r.group,
                         category=m.category, role=m.role, hand=m.hand, mesh_id=str(m.mesh_id), subject=str(m.subject), t0=t0, n_frames_take=n_take,
                         cls=classify(te.iloc[[r.Index]])[0], event_category=r.event_category, start=s, end=e, pre_frame=pre, post_frame=post,
                         firm_pre=firm_pre, firm_post=firm_post, firm_through=firm_through, duration=int(r.duration), peak_d=float(r.peak_d),
                         persistence_h4=float(r.persistence_h4) if pd.notna(r.persistence_h4) else np.nan, persistence_event=float(r.persistence_event),
                         amount_ratio=float(r.amount_ratio) if pd.notna(r.amount_ratio) else np.nan, dC=dC,
                         mass_pre=float(C[r.example, pre].astype(np.float32).sum()), mass_post=float(C[r.example, post].astype(np.float32).sum()),
                         valid_K8=t0 + s + 8 < n_take, valid_K16=t0 + s + 16 < n_take))
    S = pd.DataFrame(rows)
    S["primary"] = (S.cls == "persistent_spatial") & S.firm_pre & S.firm_post & S.firm_through
    S["secondary"] = (S.cls == "persistent_mixed") & S.firm_through
    ds_out(ds).mkdir(parents=True, exist_ok=True)
    S.to_csv(ds_out(ds) / ("events_selected.csv" if split == "test" else f"events_selected_{split}.csv"), index=False)
    return S


def load_selected(ds):
    return pd.read_csv(ds_out(ds) / "events_selected.csv", dtype={"sequence_id": str, "take_key": str, "event_id": str, "mesh_id": str, "subject": str})


# ------------------------------------------------------------------------------ meshes
class ObjectMesh:
    """One contacted object: vertices in its own frame (rest pose), faces, outward vertex normals,
    adjacency, characteristic length; ARCTIC objects carry the part labels and articulate."""
    def __init__(self, verts, faces, parts=None, name=""):
        self.name = name
        m = trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces, int), process=False)
        trimesh.repair.fix_normals(m)                                  # consistent winding, outward by volume sign
        self.verts = np.asarray(m.vertices); self.faces = np.asarray(m.faces)
        nrm = np.array(m.vertex_normals, dtype=float)
        bad = ~np.isfinite(nrm).all(1) | (np.linalg.norm(nrm, axis=1) < 1e-6)   # unreferenced / degenerate vertices
        if bad.any():                                                  # fall back to the radial direction from the centroid
            c = self.verts.mean(0); r = self.verts[bad] - c
            nrm[bad] = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        self.normals = nrm; self.n_bad_normals = int(bad.sum())
        self.parts = None if parts is None else np.asarray(parts).reshape(-1)
        self.centroid = self.verts.mean(0)
        self.length = float(np.linalg.norm(self.verts - self.centroid, axis=1).max())   # characteristic length l (max radius)
        self.extent = (self.verts.max(0) - self.verts.min(0)).astype(float)
        e = m.edges_unique
        n = len(self.verts)
        self.adj = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        self.volume = float(m.volume); self.watertight = bool(m.is_watertight)
        self.frac_outward = float(((self.normals * (self.verts - self.centroid)).sum(1) > 0).mean())

    def verts_at(self, arti=None):
        """Vertices in the object's rigid frame; ARCTIC: top part rotated by rotvec (0, 0, -arti)."""
        if arti is None or self.parts is None:
            return self.verts
        v = self.verts.copy(); top = self.parts == 0
        v[top] = v[top] @ Rotation.from_rotvec((0, 0, -float(arti))).as_matrix().T
        return v

    def normals_at(self, arti=None):
        if arti is None or self.parts is None:
            return self.normals
        n = self.normals.copy(); top = self.parts == 0
        n[top] = n[top] @ Rotation.from_rotvec((0, 0, -float(arti))).as_matrix().T
        return n


_TACO_MD = None


def taco_mesh(mesh_id):
    global _TACO_MD
    if _TACO_MD is None:
        _TACO_MD = np.load(TACO_GEN3 / "assets/taco_mesh_dict.npy", allow_pickle=True).item()
    d = _TACO_MD[mesh_id]
    return ObjectMesh(d["verts_original"], d["faces"], name=f"taco_{mesh_id}")


def arctic_mesh(category):
    m = trimesh.load(ARCTIC_TEMPL / category / "mesh.obj", process=False)
    parts = np.array(json.load(open(ARCTIC_TEMPL / category / "parts.json")))
    return ObjectMesh(np.asarray(m.vertices) / 1000.0, np.asarray(m.faces), parts=parts, name=f"arctic_{category}")


def arctic_world_verts(mesh, state):
    """BimArt's object_verts_to_world: state = [arti, rotvec(3), trans(3)] (metres)."""
    v = mesh.verts_at(state[0])
    Rg = Rotation.from_rotvec(state[1:4]).as_matrix()
    return v @ Rg.T + state[4:7]


def arctic_rigid(x_world, state):
    Rg = Rotation.from_rotvec(state[1:4]).as_matrix()
    return (x_world - state[4:7]) @ Rg


# ------------------------------------------------------------------------------ kinematics helpers
def world_rates(R, p):
    """R (T,3,3), p (T,3) world pose -> linear velocity (T,3) [m/frame] and angular velocity (T,3)
    [rad/frame, world frame] by central differences (one-sided at the ends)."""
    v = np.zeros_like(p); v[1:-1] = 0.5 * (p[2:] - p[:-2]); v[0] = p[1] - p[0]; v[-1] = p[-1] - p[-2]
    rel = np.einsum("tij,tkj->tik", R[1:], R[:-1])                   # R_{t+1} R_t^T: world-frame rotation increment
    w = Rotation.from_matrix(rel).as_rotvec()
    om = np.zeros_like(p); om[1:-1] = 0.5 * (w[:-1] + w[1:]); om[0] = w[0]; om[-1] = w[-1]
    return v, om


def unit(x, eps=1e-9):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > eps else np.zeros_like(x)


# ------------------------------------------------------------------------------ contact patches
def contact_geometry(dist, index_or_hand, mesh_verts, mesh_normals, hand_verts, labels, thr=STORE_THR):
    """Contact vertices of one frame: ids with hand distance < thr, their positions / mesh normals (object
    frame), the nearest hand vertex (index array or by KD-tree on hand_verts, both in the object frame),
    its finger label, and the normal orientation flag (True when the hand sits on the mesh's outward side).
    Returns a dict of arrays (may be empty)."""
    ids = np.where(dist < thr)[0]
    if len(ids) == 0:
        return dict(ids=ids, dist=np.zeros(0), pos=np.zeros((0, 3)), nrm=np.zeros((0, 3)), label=np.zeros(0, np.int8), hand_nn=np.zeros(0, np.int64), outward=np.zeros(0, bool))
    if index_or_hand is None:
        d2, nn = cKDTree(hand_verts).query(mesh_verts[ids])
    else:
        nn = np.asarray(index_or_hand)[ids]; d2 = dist[ids]
    pos = mesh_verts[ids]; nrm = mesh_normals[ids]
    outward = ((hand_verts[nn] - pos) * nrm).sum(1) > 0
    return dict(ids=ids, dist=dist[ids].astype(np.float32), pos=pos.astype(np.float32), nrm=nrm.astype(np.float32), label=labels[nn].astype(np.int8),
                hand_nn=nn.astype(np.int64), outward=outward, nn_dist=np.asarray(d2, np.float32))


def build_patches(g, adj, thr, merge_r, min_verts=1):
    """Semantic contact patches from a contact_geometry dict at contact threshold thr: per finger, the
    mesh-connected components of the contact vertices, merged when their centroids are closer than
    merge_r. Normals point from the object INTO the hand (flipped where the hand touches the inner side
    of a hollow surface). Returns a list of dicts (part, ids, centroid, normal, n, area_proxy)."""
    sel = g["dist"] < thr
    if sel.sum() == 0:
        return []
    ids, pos, nrm, lab, outw = g["ids"][sel], g["pos"][sel], g["nrm"][sel], g["label"][sel], g["outward"][sel]
    nrm = np.where(outw[:, None], nrm, -nrm)                          # toward the hand
    patches = []
    for part in range(len(PARTS)):
        m = lab == part
        if m.sum() < min_verts:
            continue
        pid, ppos, pn = ids[m], pos[m], nrm[m]
        sub = adj[pid][:, pid]
        n_comp, comp = connected_components(sub, directed=False)
        cents = np.stack([ppos[comp == c].mean(0) for c in range(n_comp)])
        # merge components closer than merge_r (union-find)
        parent = list(range(n_comp))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        if merge_r > 0 and n_comp > 1:
            for a in range(n_comp):
                for b in range(a + 1, n_comp):
                    if np.linalg.norm(cents[a] - cents[b]) < merge_r:
                        parent[find(a)] = find(b)
        groups = {}
        for c in range(n_comp):
            groups.setdefault(find(c), []).append(c)
        for cs in groups.values():
            mm = np.isin(comp, cs)
            if mm.sum() < min_verts:
                continue
            nmean = pn[mm].mean(0); cen = ppos[mm].mean(0)
            coh = float(np.linalg.norm(nmean))
            if coh < 0.2:                                              # normals cancel (thin plate / edge): take the vertex nearest the centroid
                nmean = pn[mm][np.argmin(np.linalg.norm(ppos[mm] - cen, axis=1))]
            patches.append(dict(part=part, ids=pid[mm], centroid=cen, normal=unit(nmean), n=int(mm.sum()), coherence=coh))
    return patches


# ------------------------------------------------------------------------------ statistics
cluster_bootstrap = HP.cluster_bootstrap


def cluster_bootstrap_median(values, clusters, n_boot=1000, seed=0):
    import tce_common as K
    return K.cluster_bootstrap(values, clusters, n_boot=n_boot, seed=seed, stat="median")


def cluster_bootstrap_frac(flags, clusters, n_boot=1000, seed=0):
    return HP.cluster_bootstrap(np.asarray(flags, float), clusters, n_boot=n_boot, seed=seed)
