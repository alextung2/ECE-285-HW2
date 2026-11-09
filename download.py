import requests,time,pathlib,sys
p=pathlib.Path(__file__).resolve().parent/'wikiart.zip'; t=time.time(); n=0
if p.exists() and p.stat().st_size==33662671673:
 print('Complete assigned archive already exists:',p); sys.exit(0)
with requests.get('https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart',stream=True,timeout=120) as r:
 r.raise_for_status()
 with p.open('wb') as f:
  for b in r.iter_content(8*1024*1024):
   f.write(b); n+=len(b)
   if n//(1024**3) != (n-len(b))//(1024**3): print(f'{n/1e9:.1f} GB, {n/1e6/(time.time()-t):.1f} MB/s',flush=True)
print('done',n,time.time()-t,flush=True)
