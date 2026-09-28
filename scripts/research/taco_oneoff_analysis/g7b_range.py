import sys, pathlib
REPO='/home/uhnam/workspace/dexcore'; sys.path[:0]=[REPO, REPO+'/third_party/BimArt']
import numpy as np, pandas as pd, yaml, logging
logging.basicConfig(level=logging.WARNING)
from src.analysis.bimart.datasets import TacoMotionDataset
mcfg=yaml.safe_load(open(REPO+'/configs/bimart_taco/train_motion_config.yaml'))
P='/result/uhnam/dexcore/bimart_taco/contact_probe/'
w=pd.read_csv(P+'window_level.csv')
rows=[]
for split in ['train','test_1','test_2','test_3','test_4']:
    ds=TacoMotionDataset(root=mcfg['data']['base_dir'],split=split,pred_horizon=mcfg['data']['pred_horizon'],base_frame=mcfg['data']['base_frame'])
    wins=sorted(w[w.split==split].window.unique())
    print(split,len(ds),len(wins),max(wins))
    for i in wins:
        it=ds[i]; a=it['action']; c=it['obs']['contact_points']
        K=a[:,:600].reshape(64,200,3)
        for h,(sl,csl) in enumerate(((slice(0,100),slice(0,1024)),(slice(100,200),slice(1024,2048)))):
            k=K[:,sl]; cen=k.mean(1)  # [64,3]
            path=np.linalg.norm(np.diff(cen,axis=0),axis=-1).sum()*1000
            ext=np.linalg.norm(cen.max(0)-cen.min(0))*1000
            kp_speed=np.linalg.norm(np.diff(k,axis=0),axis=-1).mean()*1000  # per-frame per-kp
            # finger articulation: variation of keypoints relative to centroid
            rel=k-cen[:,None]; artic=np.linalg.norm(rel-rel[:1],axis=-1).mean()*1000
            cm=c[:,csl]  # metres? check scale
            touch=(cm<0.01).mean(); mind=cm.min(-1).mean()*1000
            rows.append(dict(split=split,window=i,hand='LR'[h],cen_path_mm=path,cen_extent_mm=ext,kp_speed_mm=kp_speed,artic_mm=artic,touch_rec=touch,min_dist_rec_mm=mind,
                             cen_x=cen[:,0].mean()*1000,cen_y=cen[:,1].mean()*1000,cen_z=cen[:,2].mean()*1000))
df=pd.DataFrame(rows)
df.to_csv(P+'gen/verify/g7b_recorded_hand_range.csv',index=False)
print(df.groupby(['split','hand'])[['cen_path_mm','cen_extent_mm','kp_speed_mm','artic_mm','touch_rec','min_dist_rec_mm']].mean().round(2))
