"""(b)(ii) scale / contrast artefact check on the cached X_soft vectors, and a contact-landmark consistency probe.
Per (backend, category, role, hand), samples with touch_frac >= 0.2:
 - mean ||X_soft||_2, mean coverage fraction, mean X_soft sum (total splatted contact), n nonzero
 - contact centroid per MESH (mean X_soft over the mesh's samples, weighted centroid over canonical_points);
   spread = mean distance of a mesh's centroid to the leave-one-out mean of the other meshes' centroids
   (canonical units; template has unit max radius). Lower = hand-contact landmarks land in the same place.
 - also the per-mesh 'side' along canonical axis 0 (sign of centroid x) to spot flips.
Writes verify/cache_scale.csv, verify/contact_centroids.csv, verify/contact_centroid_spread.csv."""
import numpy as np, pandas as pd, glob, os, sys
R = "/result/uhnam/dexcore/canonical_contact/"
rows, cents = [], []
for be in ["normalized", "aligned", "dino", "random_perm"]:
    for f in sorted(glob.glob(R + f"canonical_contact_cache/{be}/*__*.npz")):
        if f.endswith("__seq.npz"): continue
        z = np.load(f, allow_pickle=False)
        cat, role = str(z["category"]), str(z["role"])
        m = z["touch_frac"] >= 0.2
        if m.sum() == 0: continue
        X = z["X_soft"][m].astype(np.float64); cov = z["coverage"][m]; P = z["canonical_points"]
        hand = z["hand"][m]; mesh = z["mesh_id"][m]
        for h in ["L", "R"]:
            mh = hand == h
            if mh.sum() == 0: continue
            Xh = X[mh]
            rows.append(dict(backend=be, category=cat, role=role, hand=h, n=int(mh.sum()), n_meshes=len(set(mesh[mh])),
                             norm_mean=np.linalg.norm(Xh, axis=1).mean(), sum_mean=Xh.sum(1).mean(),
                             max_mean=Xh.max(1).mean(), nnz_mean=(Xh > 1e-3).sum(1).mean(),
                             coverage_mean=cov[mh].mean(), radius=float(z["radius"])))
            for mid in sorted(set(mesh[mh])):
                mm = mh & (mesh == mid)
                xm = X[mm].mean(0)
                if xm.sum() <= 0: continue
                c = (xm[:, None] * P).sum(0) / xm.sum()
                cents.append(dict(backend=be, category=cat, role=role, hand=h, mesh_id=mid, n=int(mm.sum()),
                                  cx=c[0], cy=c[1], cz=c[2], norm_mean=np.linalg.norm(X[mm], axis=1).mean(),
                                  coverage_mean=cov[mm].mean()))
sc = pd.DataFrame(rows); sc.to_csv(R + "verify/cache_scale.csv", index=False)
ce = pd.DataFrame(cents); ce.to_csv(R + "verify/contact_centroids.csv", index=False)
spread = []
for (be, cat, role, h), g in ce.groupby(["backend", "category", "role", "hand"]):
    if len(g) < 3: continue
    C = g[["cx", "cy", "cz"]].values; w = g.n.values.astype(float)
    d = []
    for k in range(len(g)):
        others = np.delete(C, k, 0); wo = np.delete(w, k)
        d.append(np.linalg.norm(C[k] - (others * wo[:, None]).sum(0) / wo.sum()))
    d = np.array(d)
    spread.append(dict(backend=be, category=cat, role=role, hand=h, n_meshes=len(g), spread_mean=d.mean(),
                       spread_wmean=(d * w).sum() / w.sum(), spread_max=d.max(),
                       worst_mesh=g.mesh_id.values[int(np.argmax(d))],
                       frac_x_minority=float(min((C[:, 0] > 0).mean(), (C[:, 0] < 0).mean()))))
sp = pd.DataFrame(spread); sp.to_csv(R + "verify/contact_centroid_spread.csv", index=False)
print(sc.pivot_table(index=["category", "role", "hand"], columns="backend", values=["norm_mean", "coverage_mean"]).round(3).to_string())
print(sp.pivot_table(index=["category", "role", "hand"], columns="backend", values="spread_wmean").round(3).to_string())
print("done")
