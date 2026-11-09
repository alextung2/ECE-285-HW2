"""Wait for the active download, then prepare, train and evaluate."""
import pathlib,subprocess,sys,time,zipfile
r=pathlib.Path(__file__).resolve().parent
while True:
    p=r/'wikiart.zip'
    if p.exists() and p.stat().st_size==33662671673:
        try:
            with zipfile.ZipFile(p) as z: print('Archive ready:',len(z.namelist()),flush=True)
            break
        except zipfile.BadZipFile: pass
    time.sleep(5)
stages=[['prepare.py'],['train.py','--epochs','10'],['evaluate.py','--checkpoints','initial.pt','generator_epoch_01.pt','generator_epoch_05.pt','generator_epoch_10.pt'],['figures.py']]
if '--skip-prepare' in sys.argv: stages=stages[1:]
for args in stages:
    print('RUN',args,flush=True)
    with (r/(args[0]+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(r/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=r)
print('Pipeline complete',flush=True)
