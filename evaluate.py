"""Ignite FID (2048 pooled features) and IS, with explicit pixel handling.

Uses torchvision ImageNet Inception-v3, NOT the TensorFlow/clean-fid network.
Consequently these FID values must not be compared to clean-fid benchmarks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import inception_v3,Inception_V3_Weights
from torchvision.transforms.functional import pil_to_tensor,normalize
from ignite.metrics import FID,InceptionScore
from train import Generator,ROOT

def resize(batch):
    # Identical quantization and PIL bilinear resizing for real and fake images.
    imgs=batch.clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
    return torch.stack([pil_to_tensor(Image.fromarray(im).resize((299,299),Image.Resampling.BILINEAR)).float()/255 for im in imgs])

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoints',nargs='+',default=['initial.pt','generator_epoch_10.pt']); a=ap.parse_args()
    torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'; torch.backends.cudnn.benchmark=True
    model=inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    pooled=[]
    model.avgpool.register_forward_hook(lambda m,i,o: pooled.append(o.flatten(1).cpu()))
    def features(batch):
        inp=resize(batch).to(device)
        inp=normalize(inp,[.485,.456,.406],[.229,.224,.225])
        logits=model(inp); feat=pooled.pop()
        return feat,logits.softmax(1).cpu()
    ids=np.load(ROOT/'split.npz')['test']; data=np.load(ROOT/'images.npy',mmap_mode='r'); n=len(ids); bs=50
    cache=ROOT/'real_inception.npz'
    if cache.exists():
        c=np.load(cache); rf=torch.from_numpy(c['features']); rp=torch.from_numpy(c['probabilities'])
    else:
        fs=[]; ps=[]
        for j in range(0,n,bs):
            real=torch.from_numpy(np.array(data[ids[j:j+bs]])).permute(0,3,1,2).float()/255
            f,p=features(real); fs.append(f); ps.append(p)
            if j%500==0: print('real',j,'/',n,flush=True)
        rf=torch.cat(fs); rp=torch.cat(ps); np.savez(cache,features=rf.numpy(),probabilities=rp.numpy())
    def score(ff,fp):
        fid=FID(num_features=2048,feature_extractor=nn.Identity(),device=device)
        ins=InceptionScore(num_features=1000,feature_extractor=nn.Identity(),device='cpu')
        fid.reset(); ins.reset()
        for j in range(0,n,bs): fid.update((ff[j:j+bs].to(device),rf[j:j+bs].to(device))); ins.update(fp[j:j+bs])
        # Standard ten-split IS in addition to Ignite's whole-set score.
        splits=[]
        for p in fp.chunk(10):
            kl=(p*(p.clamp_min(1e-16).log()-p.mean(0).clamp_min(1e-16).log())).sum(1)
            splits.append(kl.mean().exp().item())
        return {'FID':float(fid.compute()),'IS_global':float(ins.compute()),'IS_10split_mean':float(np.mean(splits)),'IS_10split_std':float(np.std(splits))}
    results={}
    path=ROOT/'metrics.json'
    if path.exists(): results=json.loads(path.read_text())
    for ckname in a.checkpoints:
        torch.manual_seed(1285); torch.cuda.manual_seed_all(1285)
        g=Generator().to(device).eval(); ck=torch.load(ROOT/ckname,map_location=device,weights_only=False); g.load_state_dict(ck['G'])
        fs=[]; ps=[]; start=time.time()
        for j in range(0,n,bs):
            fake=(g(torch.randn(min(bs,n-j),100,1,1,device=device))+1)/2
            f,p=features(fake); fs.append(f); ps.append(p)
            if j%500==0: print(ckname,j,'/',n,flush=True)
        ff=torch.cat(fs); fp=torch.cat(ps); np.savez(ROOT/f'features_epoch_{ck["epoch"]:02d}.npz',features=ff.numpy(),probabilities=fp.numpy())
        results[ckname]=score(ff,fp); results[ckname]['seconds']=time.time()-start
        path.write_text(json.dumps(results,indent=2)); print(ckname,results[ckname],flush=True)
    ins=InceptionScore(num_features=1000,feature_extractor=nn.Identity()); ins.reset(); ins.update(rp)
    results['protocol']={'real_count':n,'fake_count':n,'real_IS_global':float(ins.compute()),'feature_dimensions':2048,'weights':'torchvision Inception_V3_Weights.IMAGENET1K_V1','resize':'64x64 RGB -> uint8 round -> PIL bilinear 299x299','normalization':'ImageNet mean/std; torchvision transform_input=True','evaluation_seed':1285,'split_seed':285,'FID_implementation':'ignite.metrics.FID with identity feature extractor on avgpool features','IS_implementation':'ignite.metrics.InceptionScore with identity feature extractor on softmax probabilities'}
    path.write_text(json.dumps(results,indent=2)); print('DONE',results,flush=True)
if __name__=='__main__': main()
