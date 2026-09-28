import numpy as np, pandas as pd, pickle, time, sys
meta=pd.read_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/train_seq_meta.csv')
out={}; t0=time.time()
NB=16
def resample(x,nb=NB):
    T=len(x); edges=np.linspace(0,T,nb+1).astype(int)
    return np.stack([x[edges[i]:max(edges[i+1],edges[i]+1)].mean(0) for i in range(nb)])
for i,r in meta.iterrows():
    z=np.load('/result/uhnam/dexcore/bimart_taco/sequences/'+r.file)
    L=z['contact_left']; R=z['contact_right']; ind=z['obj_cano_bps_inds']
    n_tool=int(ind[:,:512].max())+1; n_tool_ub=int(ind[:,512:].min())
    soft=np.stack([np.exp(-L/0.02).mean(0),np.exp(-R/0.02).mean(0)]).astype(np.float32)  # (2,V)
    hard=np.stack([(L<0.01).mean(0),(R<0.01).mean(0)]).astype(np.float32)
    # temporal profile: per frame fraction of verts in contact, [L-tool, L-targ, R-tool, R-targ]
    prof=np.stack([(L[:,:n_tool]<0.01).mean(1),(L[:,n_tool:]<0.01).mean(1),(R[:,:n_tool]<0.01).mean(1),(R[:,n_tool:]<0.01).mean(1)],1)
    # also min distance profile (metres) per channel
    mind=np.stack([L[:,:n_tool].min(1),L[:,n_tool:].min(1),R[:,:n_tool].min(1),R[:,n_tool:].min(1)],1)
    out[r.sequence_id]=dict(soft=soft,hard=hard,n_tool=n_tool,n_tool_ub=n_tool_ub,V=L.shape[1],T=len(L),
                            prof16=resample(prof).astype(np.float32),mind16=resample(mind).astype(np.float32),
                            prof_mean=prof.mean(0).astype(np.float32))
    if i%50==0: print(i,len(meta),'%.0fs'%(time.time()-t0),flush=True)
pickle.dump(out,open('/result/uhnam/dexcore/bimart_taco/contact_probe/train_dense_summaries.pkl','wb'))
print('DONE',time.time()-t0)
