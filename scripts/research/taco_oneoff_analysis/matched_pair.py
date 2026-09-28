"""The critic's decisive test: same verb, same TOOL MESH, only the target changes (pan/plate in train or test_1 vs bowl in test_3).
Hand-resolved and map-half-resolved. H1 (action strategy) needs the TOOL-hand grip error to rise on the bowl side even though
(verb, tool_mesh) was seen thousands of times; H2 needs the excess to sit on the target half / target-holding hand.
Restricted to windows where the DENSE truth says the right hand actually holds the tool (>50% of frames), so the
BPS-truncation artefact cannot masquerade as error.
"""
import numpy as np, pandas as pd
from scipy import stats
P="/result/uhnam/dexcore/bimart_taco/contact_probe/"
tr=pd.read_csv(P+"truncation_per_window.csv").drop(columns=["contact_mm_R","contact_mm_L","motion_mm_R"])
def half_err(split):
    z=np.load(P+f"{split}_contact_maps.npz"); pr=z["pred"].astype(np.float32); gt=z["gt"].astype(np.float32)
    e=np.abs(pr-gt)*1000
    di=pd.read_csv(P+f"{split}_contact_maps_index.csv")
    di["err_L_tool"]=e[...,0:512].mean((1,2)); di["err_L_targ"]=e[...,512:1024].mean((1,2))
    di["err_R_tool"]=e[...,1024:1536].mean((1,2)); di["err_R_targ"]=e[...,1536:2048].mean((1,2))
    di["split"]=split; return di[["split","window","err_L_tool","err_L_targ","err_R_tool","err_R_targ"]]
H=pd.concat([half_err(s) for s in ("train","test_1","test_3")])
pr=pd.concat([pd.read_csv(P+f"{s}.csv") for s in ("train","test_1","test_3")])
pw=pr.groupby(["split","window","hand"])[["contact_mm","motion_mm","contact_trans_mm","contact_resid_mm","gt_touch_frac"]].mean().unstack("hand")
pw.columns=[f"{a}_{b}" for a,b in pw.columns]; pw=pw.reset_index()
D=tr.merge(H,on=["split","window"]).merge(pw,on=["split","window"])
held=D[D.dense_R_tool>0.5].copy()
print("windows with tool actually held (dense R-tool >0.5):", held.groupby("split").size().to_dict())
# (verb, tool_mesh) pairs present in test_3-bowl AND in train/test_1 with a seen target
t3=held[(held.split=="test_3")&(held.target_cat=="bowl")]
ref=held[(held.split.isin(["train","test_1"]))&(held.target_cat!="bowl")]
pairs=set(zip(t3.verb,t3.tool_mesh))&set(zip(ref.verb,ref.tool_mesh))
print("matched (verb, tool_mesh) pairs:", len(pairs))
cols=["contact_mm_R","contact_trans_mm_R","contact_resid_mm_R","contact_mm_L","contact_trans_mm_L","contact_resid_mm_L","err_R_tool","err_R_targ","err_L_tool","err_L_targ","motion_mm_R","motion_mm_L","gath_R_tool","pred_R_tool","dense_R_tool"]
rows=[]
for v,m in sorted(pairs):
    a=t3[(t3.verb==v)&(t3.tool_mesh==m)]; b=ref[(ref.verb==v)&(ref.tool_mesh==m)]
    r={"verb":v,"tool_mesh":m,"n_bowl":len(a),"n_seq_bowl":a.sequence_id.nunique(),"n_ref":len(b),"n_seq_ref":b.sequence_id.nunique(),"ref_targets":",".join(sorted(b.target_cat.unique()))}
    for c in cols: r[c+"_bowl"]=a[c].mean(); r[c+"_ref"]=b[c].mean()
    rows.append(r)
M=pd.DataFrame(rows); M.to_csv(P+"matched_pairs.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print("\n=== per matched pair (bowl = test_3, ref = train/test_1 with pan/plate/etc.) ===")
show=["verb","tool_mesh","n_seq_bowl","n_seq_ref","ref_targets","contact_mm_R_bowl","contact_mm_R_ref","contact_mm_L_bowl","contact_mm_L_ref","err_R_tool_bowl","err_R_tool_ref","err_L_targ_bowl","err_L_targ_ref","gath_R_tool_bowl","gath_R_tool_ref","pred_R_tool_bowl","pred_R_tool_ref"]
print(M[show].round(1).to_string(index=False))
# pooled, weighting pairs equally, with sequence-clustered bootstrap on the bowl side
def pooled(df,col): return df[col].mean()
print("\n=== pooled over matched pairs (pair-weighted means) ===")
out={}
for c in cols: out[c]={"bowl(test_3)":M[c+"_bowl"].mean(),"ref(seen target)":M[c+"_ref"].mean()}
T=pd.DataFrame(out).T; T["excess"]=T["bowl(test_3)"]-T["ref(seen target)"]; T["ratio"]=T["bowl(test_3)"]/T["ref(seen target)"]
print(T.round(2).to_string())
# sequence-clustered bootstrap for the two key contrasts
rng=np.random.default_rng(0)
def boot(df,col,n=2000):
    seqs=df.sequence_id.unique(); g=df.groupby("sequence_id")[col].mean()
    bs=[g.loc[rng.choice(seqs,len(seqs))].mean() for _ in range(n)]; return np.percentile(bs,[2.5,97.5])
A=t3[[ (v,m) in pairs for v,m in zip(t3.verb,t3.tool_mesh)]]; B=ref[[ (v,m) in pairs for v,m in zip(ref.verb,ref.tool_mesh)]]
print("\n=== sequence-clustered 95% CIs on the matched windows ===")
for c in ("contact_mm_R","contact_mm_L","err_R_tool","err_L_targ","contact_resid_mm_R","contact_trans_mm_R"):
    print(f"  {c:20s} bowl {A[c].mean():6.1f} [{boot(A,c)[0]:.1f},{boot(A,c)[1]:.1f}]   ref {B[c].mean():6.1f} [{boot(B,c)[0]:.1f},{boot(B,c)[1]:.1f}]   (bowl n_seq={A.sequence_id.nunique()}, ref n_seq={B.sequence_id.nunique()})")
