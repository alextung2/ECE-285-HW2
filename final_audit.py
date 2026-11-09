"""Independent numerical and provenance checks of the completed experiment."""
import csv,hashlib,json,io
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.models import inception_v3,Inception_V3_Weights
from torchvision.transforms.functional import normalize
from train import Generator,Discriminator,ROOT
from evaluate import resize

def main():
    torch.set_num_threads(4); threadpool_limits(4); torch.backends.cudnn.benchmark=True
    results={}; cfg=json.loads((ROOT/'config.json').read_text()); manifest=json.loads((ROOT/'manifest.json').read_text()); metrics=json.loads((ROOT/'metrics.json').read_text())
    data=np.load(ROOT/'images.npy',mmap_mode='r'); split=np.load(ROOT/'split.npz')
    seen={}; unique=[]; duplicate=[]
    for i,img in enumerate(data):
        h=hashlib.sha256(img.tobytes()).hexdigest()
        if h in seen: duplicate.append({'index':i,'same_as':seen[h]})
        else: seen[h]=i; unique.append(i)
    expected=np.array(unique); np.random.default_rng(285).shuffle(expected)
    assert np.array_equal(expected[:5000],split['test']) and np.array_equal(expected[5000:],split['train'])
    assert duplicate==manifest['duplicates_at_64px'] and not manifest['errors']
    assert len(set(split['test']) & set(split['train']))==0
    assert len(data)==81444 and len(unique)==80694
    results['dataset']={'images':len(data),'unique_crops':len(unique),'duplicates':len(duplicate),'training':len(split['train']),'held_out':len(split['test']),'split_reproduced':True,'exact_crop_leakage':False}
    with (ROOT/'losses.csv').open() as f: rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows]==list(range(5910))
    assert all(sum(int(r['epoch'])==e for r in rows)==591 for e in range(1,11))
    assert all(np.isfinite(float(r[k])) for r in rows for k in ['D_loss','G_loss','D_real','D_fake_before','D_fake_after'])
    assert sum(p.numel() for p in Generator().parameters())==cfg['G_parameters']==3576704
    assert sum(p.numel() for p in Discriminator().parameters())==cfg['D_parameters']==2765568
    results['training']={'epochs':10,'steps':len(rows),'finite_logs':True,'parameter_counts_match':True}
    real=np.load(ROOT/'real_inception.npz'); rf=real['features'].astype(np.float64); rp=real['probabilities'].astype(np.float64)
    mu_r=rf.mean(0); cov_r=np.cov(rf,rowvar=False); wr,ur=eigh(cov_r); root=(ur*np.sqrt(np.maximum(wr,0)))@ur.T
    scores={}
    for e,key in [(0,'initial.pt'),(1,'generator_epoch_01.pt'),(5,'generator_epoch_05.pt'),(10,'generator_epoch_10.pt')]:
        f=np.load(ROOT/f'features_epoch_{e:02d}.npz'); ff=f['features'].astype(np.float64); p=f['probabilities'].astype(np.float64)
        assert ff.shape==(5000,2048) and p.shape==(5000,1000)
        assert np.isfinite(ff).all() and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
        mu_f=ff.mean(0); cov_f=np.cov(ff,rowvar=False); middle=root@cov_f@root
        eig=eigh((middle+middle.T)/2,eigvals_only=True)
        fid=float(((mu_f-mu_r)**2).sum()+np.trace(cov_f)+np.trace(cov_r)-2*np.sqrt(np.maximum(eig,0)).sum())
        def ins(x): return float(np.exp(np.mean(np.sum(x*(np.log(np.maximum(x,1e-16))-np.log(np.maximum(x.mean(0),1e-16))),axis=1))))
        score=ins(p); parts=[ins(x) for x in np.array_split(p,10)]
        m=metrics[key]
        assert abs(fid-m['FID'])<1e-3,(key,fid,m['FID'])
        assert abs(score-m['IS_global'])<1e-5
        assert abs(np.mean(parts)-m['IS_10split_mean'])<1e-5 and abs(np.std(parts)-m['IS_10split_std'])<1e-5
        scores[key]={'independent_FID':fid,'FID_difference':fid-m['FID'],'independent_IS':score,'IS_difference':score-m['IS_global']}
        print(key,scores[key],flush=True)
    assert abs(ins(rp)-metrics['protocol']['real_IS_global'])<1e-5
    results['independent_metrics']=scores
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ck=torch.load(ROOT/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==10 and ck['step']==5910
    g=Generator().to(device).eval(); g.load_state_dict(ck['G']); fixed=ck['fixed'].to(device)
    image_checks={}
    for epoch in [1,10]:
        saved=torch.load(ROOT/f'generator_epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False); g.load_state_dict(saved['G'])
        with torch.no_grad(): img=(g(fixed)+1)/2
        grid=make_grid(img,nrow=8).mul(255).add(.5).clamp(0,255).permute(1,2,0).byte().cpu().numpy()
        original=np.asarray(Image.open(ROOT/f'samples_epoch_{epoch:02d}.png').convert('RGB'))
        delta=np.abs(grid.astype(int)-original.astype(int)); assert delta.max()<=1
        image_checks[f'epoch_{epoch}']={'max_pixel_difference':int(delta.max()),'mean_pixel_difference':float(delta.mean())}
    true_real=torch.from_numpy(np.array(data[split['test'][:64]])).permute(0,3,1,2).float()/255
    grid=make_grid(true_real,nrow=8).mul(255).add(.5).clamp(0,255).permute(1,2,0).byte().numpy()
    assert np.array_equal(grid,np.asarray(Image.open(ROOT/'real_samples.png').convert('RGB')))
    results['image_provenance']={'generated':image_checks,'held_out_grid_exact_match':True}
    # Re-extract selected held-out features to rule out an incorrect/stale cache.
    model=inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval(); captured=[]
    model.avgpool.register_forward_hook(lambda m,i,o:captured.append(o.flatten(1).cpu().numpy()))
    sample_ids=np.array([0,137,999,2048,4999]); batch=torch.from_numpy(np.array(data[split['test'][sample_ids]])).permute(0,3,1,2).float()/255
    with torch.no_grad(): logits=model(normalize(resize(batch).to(device),[.485,.456,.406],[.229,.224,.225])); probs=logits.softmax(1).cpu().numpy()
    feat=captured.pop(); fd=float(np.max(np.abs(feat-real['features'][sample_ids]))); pd=float(np.max(np.abs(probs-real['probabilities'][sample_ids])))
    reference=real['features'][sample_ids]
    relative_l2=np.linalg.norm(feat-reference,axis=1)/np.linalg.norm(reference,axis=1)
    cosine=np.sum(feat*reference,axis=1)/(np.linalg.norm(feat,axis=1)*np.linalg.norm(reference,axis=1))
    probability_l1=np.abs(probs-real['probabilities'][sample_ids]).sum(1)
    # Benchmark-selected cuDNN kernels and different batch sizes need not give
    # bit-identical Inception outputs. Require close representations, and
    # record the actual differences instead of claiming an exact cache match.
    assert relative_l2.max()<.005 and cosine.min()>.99999
    assert probability_l1.max()<.02
    results['real_feature_cache']={'rechecked_examples':len(sample_ids),'max_feature_difference':fd,'max_probability_difference':pd,'max_relative_L2_error':float(relative_l2.max()),'minimum_cosine_similarity':float(cosine.min()),'max_probability_L1_error':float(probability_l1.max()),'comparison':'tolerant to cuDNN benchmark/batch-size numerical variation; not bit-identical'}
    results['passed']=True
    (ROOT/'final_audit.json').write_text(json.dumps(results,indent=2)); print('ALL AUDIT CHECKS PASSED',flush=True)
if __name__=='__main__': main()
