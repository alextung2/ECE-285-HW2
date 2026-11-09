"""Verify/decode the assigned Kaggle archive and cache RGB center crops."""
import concurrent.futures as cf
import io,json,zipfile,time,hashlib,threading,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repair',action='store_true'); args=ap.parse_args()
    start=time.time()
    with zipfile.ZipFile(ROOT/'wikiart.zip') as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png')) and not n.startswith('__MACOSX'))
        out=np.lib.format.open_memmap(ROOT/'images.npy',mode='r+' if args.repair else 'w+',dtype=np.uint8,shape=(len(names),64,64,3))
        errors=[]; local=threading.local()
        targets=list(enumerate(names))
        if args.repair:
            previous=json.loads((ROOT/'manifest.json').read_text())
            targets=[(e['index'],e['name']) for e in previous['errors']]
        def decode(item):
            i,n=item
            try:
                # Separate handles prevent races in concurrent buffered ZIP reads.
                if not hasattr(local,'archive'): local.archive=zipfile.ZipFile(ROOT/'wikiart.zip')
                with Image.open(io.BytesIO(local.archive.read(n))) as im:
                    im=im.convert('RGB'); w,h=im.size
                    # torchvision Resize(64) followed by CenterCrop(64).
                    nw,nh=(64,int(64*h/w)) if w<=h else (int(64*w/h),64)
                    im=im.resize((nw,nh),Image.Resampling.BILINEAR)
                    x,y=round((nw-64)/2),round((nh-64)/2)
                    out[i]=np.asarray(im.crop((x,y,x+64,y+64)))
                return None
            except Exception as e: return {'index':i,'name':n,'error':str(e)}
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            for i,e in enumerate(pool.map(decode,targets)):
                if e: errors.append(e)
                if (i+1)%5000==0: print('decoded',i+1,'/',len(names),'seconds',round(time.time()-start),flush=True)
        out.flush()
        bad={e['index'] for e in errors}; seen={}; unique=[]; duplicates=[]
        for i in range(len(names)):
            if i in bad: continue
            h=hashlib.sha256(out[i].tobytes()).hexdigest()
            if h in seen: duplicates.append({'index':i,'same_as':seen[h]})
            else: seen[h]=i; unique.append(i)
        valid=np.array(unique)
        rng=np.random.default_rng(285); rng.shuffle(valid)
        np.savez(ROOT/'split.npz',test=valid[:5000],train=valid[5000:])
        (ROOT/'manifest.json').write_text(json.dumps({'source':'https://www.kaggle.com/datasets/steubk/wikiart','names':names,'errors':errors,'duplicates_at_64px':duplicates,'seed':285,'test_count':5000,'train_count':len(valid)-5000},indent=2))
        print('DONE',len(names),'error_count',len(errors),'duplicate_count',len(duplicates),'seconds',time.time()-start,flush=True)
if __name__=='__main__': main()
