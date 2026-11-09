"""Generate new samples using the trained GAN checkpoint."""
import argparse
import torch
from torchvision.utils import save_image
from train import Generator,ROOT

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='generator_epoch_10.pt'); p.add_argument('--seed',type=int,default=2285); p.add_argument('--count',type=int,default=64); p.add_argument('--output',default='new_samples.png'); a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(a.seed)
    g=Generator().to(device).eval(); ck=torch.load(ROOT/a.checkpoint,map_location=device,weights_only=False); g.load_state_dict(ck['G'])
    with torch.no_grad(): images=(g(torch.randn(a.count,100,1,1,device=device))+1)/2
    save_image(images.clamp(0,1),ROOT/a.output,nrow=8)
    print(ROOT/a.output)
if __name__=='__main__': main()
