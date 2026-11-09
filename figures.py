"""Figures computed from the experiment's actual outputs."""
import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT

def main():
    with (ROOT/'losses.csv').open() as f: rows=list(csv.DictReader(f))
    x=np.array([int(r['step'])+1 for r in rows]); g=np.array([float(r['G_loss']) for r in rows]); d=np.array([float(r['D_loss']) for r in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.5,1.65),layout='constrained')
    for v,col,label in [(g,'#b45309','Generator'),(d,'#0369a1','Discriminator')]:
        ax.plot(x,v,color=col,alpha=.13,lw=.4)
        k=100; ax.plot(x[k-1:],np.convolve(v,np.ones(k)/k,mode='valid'),color=col,lw=1.2,label=label)
    ax.set(xlabel='Optimization step',ylabel='BCE loss'); ax.legend(frameon=False,ncol=2,loc='upper right'); ax.grid(alpha=.16)
    fig.savefig(ROOT/'loss_curve.pdf'); fig.savefig(ROOT/'loss_curve.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(6.5,2.15),layout='constrained')
    for ax,name,label in zip(axs,['real_samples.png','samples_epoch_01.png','samples_epoch_10.png'],['Held-out real images','Generated: epoch 1','Generated: epoch 10']):
        im=Image.open(ROOT/name).convert('RGB'); tiles=[]
        for i in range(16):
            r,c=divmod(i,8); tiles.append(im.crop((2+c*66,2+r*66,2+c*66+64,2+r*66+64)))
        grid=Image.new('RGB',(4*64,4*64),'white')
        for i,im in enumerate(tiles): grid.paste(im,((i%4)*64,(i//4)*64))
        ax.imshow(grid); ax.set_title(label,fontsize=9); ax.axis('off')
    fig.savefig(ROOT/'comparison.pdf'); fig.savefig(ROOT/'comparison.png',dpi=180); plt.close(fig)
    print('Figures saved')
if __name__=='__main__': main()
