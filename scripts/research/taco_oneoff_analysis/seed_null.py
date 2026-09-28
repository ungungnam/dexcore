"""Seed-to-seed floor of contact_mm: sample stage 2 twice on the RECORDED map with two seeds and measure how far the
hand moves. This is what 'swapping the contact map' costs when the map does not change at all -- the zero point that
contact_mm on each split must be read against."""
import sys, pathlib
for p in ("/home/uhnam/workspace/dexcore","/home/uhnam/workspace/dexcore/third_party/BimArt"):
    if p not in sys.path: sys.path.insert(0,p)
import numpy as np, torch, yaml, pandas as pd
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset
from scripts.eval_bimart_taco import load_model, sample
from src.analysis.bimart.datasets import TacoMotionDataset
from utils import data_util
HAND_KP=100; KP_DIM=600
def kp(a,mean,std): v=(a*std+mean)[...,:KP_DIM]; return v.reshape(*v.shape[:-1],200,3)
dev="cuda"; mcfg=yaml.safe_load(open("configs/bimart_taco/train_motion_config.yaml"))
mm,*_=load_model("motion","/result/uhnam/dexcore/bimart_taco/experiments/motion_model_taco_20260923_151845",mcfg,dev)
T=mcfg["train"]["num_diffusion_iters"]; st=data_util.load_stat_dict(mcfg["data"]["stat_dict_path"],dev)
sched=DDPMScheduler(num_train_timesteps=T,beta_schedule="squaredcos_cap_v2",clip_sample=False,prediction_type="sample")
mean,std=st["action"]["mean"],st["action"]["std"]
rows=[]
for split in ("train","test_1","test_2","test_3","test_4"):
    ds=TacoMotionDataset(root=mcfg["data"]["base_dir"],split=split,pred_horizon=64,base_frame=8)
    pick=np.random.default_rng(0).choice(len(ds),min(160,len(ds)),replace=False)
    for s in range(0,len(pick),32):
        idx=pick[s:s+32]; b=next(iter(DataLoader(Subset(ds,idx),batch_size=len(idx))))
        nb=data_util.preprocess_batch({k:v for k,v in b.items() if k not in ("aux","viz")},st,dev)
        rec=kp(nb["action"],mean,std)
        x0=kp(sample(mm,nb["action"].shape,sched,dev,nb["obs"],"motion",T,seed=0),mean,std)
        x1=kp(sample(mm,nb["action"].shape,sched,dev,nb["obs"],"motion",T,seed=1),mean,std)
        for h,sl in (("L",slice(0,100)),("R",slice(100,200))):
            d=torch.linalg.norm(x1[:,:,sl]-x0[:,:,sl],dim=-1).mean((1,2))*1000
            e0=torch.linalg.norm(x0[:,:,sl]-rec[:,:,sl],dim=-1).mean((1,2))*1000
            e1=torch.linalg.norm(x1[:,:,sl]-rec[:,:,sl],dim=-1).mean((1,2))*1000
            tr=torch.linalg.norm((x1[:,:,sl]-x0[:,:,sl]).mean(-2),dim=-1).mean(1)*1000
            for i in range(len(idx)): rows.append({"split":split,"window":int(idx[i]),"hand":h,"seed_delta_mm":d[i].item(),"seed_delta_trans_mm":tr[i].item(),"err_seed0":e0[i].item(),"err_seed1":e1[i].item()})
    print(split,"done",flush=True)
D=pd.DataFrame(rows); D.to_csv("/result/uhnam/dexcore/bimart_taco/contact_probe/seed_null.csv",index=False)
print(D.groupby("split")[["seed_delta_mm","seed_delta_trans_mm","err_seed0","err_seed1"]].mean().round(1).to_string())
