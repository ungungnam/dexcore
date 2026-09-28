import pandas as pd, numpy as np
df=pd.read_csv('/result/uhnam/dexcore/bimart_taco/train_store/offsets.csv')
tr=df[df.split=='train'].reset_index(drop=True)
c=np.load('/result/uhnam/dexcore/bimart_taco/train_store/contact.npy',mmap_mode='r')
soft=np.zeros((len(tr),2048),np.float32); meand=np.zeros((len(tr),2048),np.float32)
hard=np.zeros((len(tr),2048),np.float32); frac_contact=np.zeros((len(tr),2),np.float32)
for i,r in tr.iterrows():
    x=np.asarray(c[r.start:r.start+r.n_frames_store],dtype=np.float32)
    soft[i]=np.exp(-x/0.02).mean(0); meand[i]=x.mean(0); hard[i]=(x<0.01).mean(0)
    # fraction of frames where each hand touches anything
    frac_contact[i,0]=(x[:,:1024].min(1)<0.01).mean(); frac_contact[i,1]=(x[:,1024:].min(1)<0.01).mean()
np.savez('/result/uhnam/dexcore/bimart_taco/contact_probe/train_seq_summaries.npz',soft=soft,meand=meand,hard=hard,frac_contact=frac_contact,sequence_id=tr.sequence_id.values)
tr.assign(frac_L=frac_contact[:,0],frac_R=frac_contact[:,1]).to_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/train_seq_meta.csv',index=False)
print('done',soft.shape, 'mean soft',soft.mean(), 'mean frac_contact L/R',frac_contact.mean(0))
