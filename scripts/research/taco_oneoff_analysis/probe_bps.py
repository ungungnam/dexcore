import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
off = pd.read_csv('/result/uhnam/dexcore/bimart_taco/train_store/offsets.csv'); off['date']=off.seq.str.split('_').str[0]
C = np.load('/result/uhnam/dexcore/bimart_taco/train_store/contact.npy', mmap_mode='r')
B = np.load('/result/uhnam/dexcore/bimart_taco/train_store/bps.npy', mmap_mode='r')
G = np.load('/result/uhnam/dexcore/bimart_taco/train_store/global.npy', mmap_mode='r')
basis=np.load('/home/uhnam/workspace/dexcore/third_party/BimArt/assets/bps_normalized_part_based.npy'); print("basis",basis.shape, "norm range",np.linalg.norm(basis,axis=-1).min().round(3),np.linalg.norm(basis,axis=-1).max().round(3))
basis=basis.reshape(-1,3)[:512]
TH=0.01
# layout check on one sequence: which block is static
r=off.iloc[0]; b=np.asarray(B[r.start:r.start+r.n_frames_store]).reshape(-1,1024,3)
print("block0 temporal std:",b[:,:512].std(0).mean().round(5)," block1 temporal std:",b[:,512:].std(0).mean().round(5), " (static block = target)")
rows=[]
sample=pd.concat([off[off.split==s].sample(n=min(110,(off.split==s).sum()),random_state=4) for s in ['train','test_1','test_2','test_3','test_4']])
for _,r in sample.iterrows():
    x=np.asarray(C[r.start:r.start+r.n_frames_store],dtype=np.float32); mR=x[:,1024:1536].min(1)
    b=np.asarray(B[r.start:r.start+r.n_frames_store]).reshape(-1,1024,3)
    tool=b[:,:512]+basis[None]   # selected tool vertex coords in normalised canonical frame
    tgt=b[0,512:]+basis
    g=np.asarray(G[r.start:r.start+r.n_frames_store])
    tnorm=np.linalg.norm(tool,axis=-1)
    # coverage: extent of selected tool vertices (max pairwise proxy: max-min per axis, norm), in normalised units and metres (x target scale)
    ext=np.linalg.norm(tool.max(1)-tool.min(1),axis=-1).mean()
    rows.append(dict(split=r.split,date=r.date,verb=r.verb,tool_cat=r.tool_cat,target_cat=r.target_cat,bowl=r.target_cat=='bowl',
        R_tool=(mR<TH).mean(),tgt_scale=float(g[:,6].mean()),tool_scale=float(g[:,7].mean()),
        tool_maxnorm=float(tnorm.max(1).mean()),tool_meannorm=float(tnorm.mean()),tool_ext_norm=float(ext),
        tgt_ext_norm=float(np.linalg.norm(tgt.max(0)-tgt.min(0))),
        tool_dist_from_basis=float(np.linalg.norm(b[:,:512],axis=-1).mean())))
df=pd.DataFrame(rows); df.to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/bps_coverage_probe.csv',index=False)
cols=['R_tool','tgt_scale','tool_scale','tool_maxnorm','tool_meannorm','tool_ext_norm','tgt_ext_norm','tool_dist_from_basis']
print("\n=== by split"); print(df.groupby('split')[cols].mean().round(3))
print("\n=== by target_cat"); print(df.groupby('target_cat')[cols+['bowl']].agg(['mean']).round(3).droplevel(1,axis=1).join(df.groupby('target_cat').size().rename('n')))
print("\n=== spatula/spoon tools by target_cat x split"); d=df[df.tool_cat.isin(['spatula','spoon','knife'])]
print(d.groupby(['tool_cat','target_cat','split'])[['R_tool','tgt_scale','tool_scale','tool_maxnorm','tool_ext_norm']].agg(['mean','count']).round(3))
print("\n=== bowl-target by split x date: R_tool, tgt_scale"); print(df[df.bowl].groupby(['split','date'])[['R_tool','tgt_scale','tool_maxnorm']].agg(['mean','count']).round(3))
print("\ncorr(R_tool, tgt_scale) all:",df[['R_tool','tgt_scale','tool_scale','tool_maxnorm','tool_dist_from_basis']].corr().round(3))
