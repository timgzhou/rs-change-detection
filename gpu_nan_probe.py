import os,sys; sys.path.insert(0,os.path.join(os.getcwd(),"olmo_shims"))
import torch, torch.nn as nn
from pathlib import Path
from lp_urbansarfloods import ConcatPrePostPerPixelHead, TiledFeatureDataset
from torch.utils.data import DataLoader
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device",dev)
ds=TiledFeatureDataset(Path("features/usf_base_s1_ps4_res20"),"train")
dl=DataLoader(ds,batch_size=64,shuffle=True,num_workers=2,pin_memory=True)
head=ConcatPrePostPerPixelHead(768,3,4).to(dev)
opt=torch.optim.AdamW(head.parameters(),lr=1e-3)
lf=nn.CrossEntropyLoss(ignore_index=-1)
for i,(feat,label) in enumerate(dl):
    feat,label=feat.to(dev),label.to(dev)
    print(f"batch{i} feat nan={torch.isnan(feat).any().item()} max|f|={feat.abs().max().item():.1f} "
          f"label uniq={torch.unique(label).tolist()}")
    logits=head(feat)
    loss=lf(logits,label)
    print(f"   logits nan={torch.isnan(logits).any().item()} loss={loss.item()}")
    opt.zero_grad(); loss.backward()
    gnan=any(p.grad is not None and torch.isnan(p.grad).any() for p in head.parameters())
    print("   grad nan=",gnan)
    opt.step()
    if i>=3: break
print("done")
