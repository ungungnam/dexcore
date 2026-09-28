"""(b)(i) Are scan frames already consistent within a category?  Per category and per mesh:
 - Chamfer of the mesh to the ALIGNED template under (1) centre+scale only (scan frame, 'identity rotation'),
   (2) aligned's stored map (PCA + proper flip + ICP);  and the mean all-pairs Chamfer between meshes of the
   category under the normalized map vs the aligned map (same FPS-2000 samples, same Chamfer).
 - The rotation aligned applies relative to the template's own PCA rotation: angle(R_i R_T^T). If scan frames
   were consistent AND aligned were right, this would be ~0; ~180 deg = a flip relative to the scan frame.
 - ICP scale factor relative to the unit-radius scale.
Writes verify/frames_per_mesh.csv, verify/frames_per_category.csv."""
import numpy as np, pandas as pd, sys
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/uhnam/workspace/dexcore")
from src.analysis.canonical import backends
from src.analysis.canonical.geometry import centre_scale, chamfer, fps_indices, PROPER_FLIPS, pca_axes
R = "/result/uhnam/dexcore/canonical_contact/"
al = backends.load("aligned", R + "canonical_backend"); no = backends.load("normalized", R + "canonical_backend")
dn = backends.load("dino", R + "canonical_backend")

def rot_angle(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))

per_mesh, per_cat = [], []
for cat in al.categories():
    fit = al._cat(cat); nfit = no._cat(cat); dfit = dn._cat(cat)
    ids = fit.mesh_ids; T = fit.template
    if len(ids) < 2:
        continue
    samp = {i: fit.verts[i][fps_indices(fit.verts[i], 2000)] for i in ids}
    cs = {i: centre_scale(fit.verts[i]) for i in ids}
    S_norm = {i: cs[i].apply(samp[i]) for i in ids}                 # scan frame, centred, unit radius
    S_al = {i: fit.sims[i].apply(samp[i]) for i in ids}             # aligned canonical frame
    # scan frame but expressed in the template's PCA frame (rotate everything by R_T): same relative geometry
    trees_norm = {i: cKDTree(S_norm[i]) for i in ids}; trees_al = {i: cKDTree(S_al[i]) for i in ids}
    M = len(ids)
    Cn = np.zeros((M, M)); Ca = np.zeros((M, M))
    for a, i in enumerate(ids):
        for b, j in enumerate(ids):
            if b <= a: continue
            Cn[a, b] = Cn[b, a] = chamfer(S_norm[i], S_norm[j], trees_norm[i], trees_norm[j])
            Ca[a, b] = Ca[b, a] = chamfer(S_al[i], S_al[j], trees_al[i], trees_al[j])
    tj = ids.index(T)
    RT = fit.sims[T].R
    flip_id = {tuple(np.diag(F).astype(int)): k for k, F in enumerate(PROPER_FLIPS)}
    for a, i in enumerate(ids):
        Ri = fit.sims[i].R
        rel = Ri @ RT.T                          # rotation aligned applies to mesh i beyond the template's PCA rotation
        r_i = 1.0 / cs[i].s
        # best proper flip of the SCAN frame (no PCA) onto template scan frame, to see if a flip alone explains things
        ch_flips = [chamfer(S_norm[i] @ F, S_norm[T], tree_b=trees_norm[T]) for F in PROPER_FLIPS]
        # PCA-frame flips (what aligned searched), without ICP
        Pi = pca_axes(cs[i].apply(fit.verts[i])); PT = pca_axes(cs[T].apply(fit.verts[T]))
        Si_p = cs[i].apply(samp[i]) @ Pi.T; ST_p = cs[T].apply(samp[T]) @ PT.T
        tree_T_p = cKDTree(ST_p)
        ch_pca_flips = [chamfer(Si_p @ F, ST_p, tree_b=tree_T_p) for F in PROPER_FLIPS]
        best = int(np.argmin(ch_pca_flips)); srt = np.sort(ch_pca_flips)
        # rotation between scan-frame-based PCA of i and of T (both in scan frame): angle of Pi^T PT.
        # If scan frames are consistent and shapes similar, PCA axes are similar up to sign.
        d_flip = np.diag(dfit.__dict__.get("dino_flip", np.eye(3)[None])[a]) if hasattr(dfit, "dino_flip") else None
        per_mesh.append(dict(category=cat, mesh_id=i, is_template=int(i == T), n_meshes=M, max_radius_m=r_i,
                             icp_scale=fit.sims[i].s * r_i,
                             chamfer_identity_to_T=Cn[a, tj], chamfer_aligned_to_T=Ca[a, tj],
                             chamfer_icp_stored=fit.quality[i]["chamfer_icp"], chamfer_pca_stored=fit.quality[i]["chamfer_pca"],
                             chamfer_identity_meanothers=Cn[a].sum() / (M - 1), chamfer_aligned_meanothers=Ca[a].sum() / (M - 1),
                             rot_angle_vs_template_deg=rot_angle(rel), rot_axis_dominant=int(np.argmax(np.abs(
                                 np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])))),
                             best_scanflip=int(np.argmin(ch_flips)), scanflip_margin=(np.sort(ch_flips)[1] - np.sort(ch_flips)[0]),
                             pcaflip_best=best, pcaflip_margin=srt[1] - srt[0], pcaflip_rel_margin=(srt[1] - srt[0]) / srt[0],
                             pca_flip_chamfers=";".join(f"{c:.4f}" for c in ch_pca_flips)))
    iu = np.triu_indices(M, 1)
    per_cat.append(dict(category=cat, n_meshes=M, template=T,
                        allpairs_chamfer_normalized=Cn[iu].mean(), allpairs_chamfer_aligned=Ca[iu].mean(),
                        frac_pairs_normalized_better=float((Cn[iu] < Ca[iu]).mean()),
                        toT_chamfer_identity=np.delete(Cn[:, tj], tj).mean(), toT_chamfer_aligned=np.delete(Ca[:, tj], tj).mean(),
                        n_rot_gt_90=int(sum(1 for r in per_mesh if r["category"] == cat and r["rot_angle_vs_template_deg"] > 90)),
                        n_rot_gt_150=int(sum(1 for r in per_mesh if r["category"] == cat and r["rot_angle_vs_template_deg"] > 150)),
                        mean_rot_angle=np.mean([r["rot_angle_vs_template_deg"] for r in per_mesh if r["category"] == cat and not r["is_template"]]),
                        icp_scale_mean=np.mean([r["icp_scale"] for r in per_mesh if r["category"] == cat]),
                        icp_scale_min=np.min([r["icp_scale"] for r in per_mesh if r["category"] == cat]),
                        icp_scale_max=np.max([r["icp_scale"] for r in per_mesh if r["category"] == cat])))
    print(f"{cat:12s} M={M:2d} allpairs Chamfer: normalized {Cn[iu].mean():.4f}  aligned {Ca[iu].mean():.4f} "
          f"(norm better on {100*(Cn[iu] < Ca[iu]).mean():.0f}% of pairs)  to-T identity {per_cat[-1]['toT_chamfer_identity']:.4f} "
          f"aligned {per_cat[-1]['toT_chamfer_aligned']:.4f}  rot>90: {per_cat[-1]['n_rot_gt_90']}/{M-1}  meanrot {per_cat[-1]['mean_rot_angle']:.0f}")
    sys.stdout.flush()
pd.DataFrame(per_mesh).to_csv(R + "verify/frames_per_mesh.csv", index=False)
pd.DataFrame(per_cat).to_csv(R + "verify/frames_per_category.csv", index=False)
print("done")
