"""A reproducible unconditional DCGAN. Run from any working directory."""
import argparse,csv,json,random,time,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

ROOT=Path(__file__).resolve().parent
class Generator(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]
        channels=[100,512,256,128,64,3]
        for i,(a,b) in enumerate(zip(channels[:-1],channels[1:])):
            layers.append(nn.ConvTranspose2d(a,b,4,1 if i==0 else 2,0 if i==0 else 1,bias=False))
            layers.extend([nn.BatchNorm2d(b),nn.ReLU(True)] if i<4 else [nn.Tanh()])
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; channels=[3,64,128,256,512,1]
        for i,(a,b) in enumerate(zip(channels[:-1],channels[1:])):
            layers.append(nn.Conv2d(a,b,4,1 if i==4 else 2,0 if i==4 else 1,bias=False))
            if 0<i<4: layers.append(nn.BatchNorm2d(b))
            if i<4: layers.append(nn.LeakyReLU(.2,True))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).flatten()
def init(m):
    if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)): nn.init.normal_(m.weight,0,.02)
    if isinstance(m,nn.BatchNorm2d): nn.init.normal_(m.weight,1,.02); nn.init.zeros_(m.bias)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    seed=285; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    images=torch.from_numpy(np.load(ROOT/'images.npy')).permute(0,3,1,2).contiguous().to(device)
    split=np.load(ROOT/'split.npz'); ids=torch.tensor(split['train'],device=device)
    G=Generator().to(device); D=Discriminator().to(device); G.apply(init); D.apply(init)
    go=torch.optim.Adam(G.parameters(),lr=.0002,betas=(.5,.999)); do=torch.optim.Adam(D.parameters(),lr=.0002,betas=(.5,.999))
    loss=nn.BCEWithLogitsLoss(); fixed=torch.randn(64,100,1,1,device=device); batch=128; start_epoch=0; step=0
    if a.resume:
        ck=torch.load(ROOT/'checkpoint.pt',map_location=device,weights_only=False)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D']); go.load_state_dict(ck['go']); do.load_state_dict(ck['do']); start_epoch=ck['epoch']; step=ck['step']; fixed=ck['fixed'].to(device)
        torch.set_rng_state(ck['rng_cpu'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng_cuda']])
    config={'seed':seed,'batch_size':batch,'latent_dim':100,'learning_rate':.0002,'betas':[.5,.999],'epochs':a.epochs,'train_count':len(ids),'test_count':len(split['test']),'G_parameters':sum(p.numel() for p in G.parameters()),'D_parameters':sum(p.numel() for p in D.parameters()),'torch':torch.__version__,'device':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),'drop_last':True,'precision':'float32','cudnn_benchmark':True}
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)); print(config,flush=True)
    if not a.resume:
        G.eval()
        with torch.no_grad(): save_image((G(fixed)+1)/2,ROOT/'samples_epoch_00.png',nrow=8)
        torch.save({'G':G.state_dict(),'epoch':0},ROOT/'initial.pt')
    save_image(images[torch.tensor(split['test'][:64],device=device)].float()/255,ROOT/'real_samples.png',nrow=8)
    with (ROOT/'losses.csv').open('a' if a.resume else 'w',newline='') as f:
        w=csv.writer(f)
        if not a.resume: w.writerow(['epoch','step','D_loss','G_loss','D_real','D_fake_before','D_fake_after'])
        start=time.time()
        for epoch in range(start_epoch+1,a.epochs+1):
            G.train(); D.train(); order=ids[torch.randperm(len(ids),device=device)]; vals=[]; et=time.time()
            for j in range(0,len(order)-batch+1,batch):
                real=images[order[j:j+batch]].float().div_(127.5).sub_(1)
                do.zero_grad(set_to_none=True)
                z=torch.randn(batch,100,1,1,device=device); fake=G(z)
                dr=D(real); df=D(fake.detach()); dl=loss(dr,torch.ones_like(dr))+loss(df,torch.zeros_like(df)); dl.backward(); do.step()
                go.zero_grad(set_to_none=True)
                for p in D.parameters(): p.requires_grad_(False)
                dg=D(fake); gl=loss(dg,torch.ones_like(dg)); gl.backward(); go.step()
                for p in D.parameters(): p.requires_grad_(True)
                row=[epoch,step,dl.item(),gl.item(),dr.sigmoid().mean().item(),df.sigmoid().mean().item(),dg.sigmoid().mean().item()]; w.writerow(row); vals.append(row[2:]); step+=1
                if step%100==0: f.flush(); print('epoch',epoch,'step',step,'D/G',round(row[2],3),round(row[3],3),'elapsed',round(time.time()-start),flush=True)
            f.flush(); G.eval()
            with torch.no_grad(): save_image((G(fixed)+1)/2,ROOT/f'samples_epoch_{epoch:02d}.png',nrow=8)
            ck={'G':G.state_dict(),'D':D.state_dict(),'go':go.state_dict(),'do':do.state_dict(),'epoch':epoch,'step':step,'fixed':fixed.cpu(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            torch.save(ck,ROOT/'checkpoint.pt')
            if epoch in [1,5,10,20]: torch.save({'G':G.state_dict(),'epoch':epoch},ROOT/f'generator_epoch_{epoch:02d}.pt')
            print('EPOCH DONE',epoch,'seconds',time.time()-et,'means',np.mean(vals,axis=0).tolist(),flush=True)
if __name__=='__main__': main()
