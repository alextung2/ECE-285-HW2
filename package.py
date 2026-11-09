"""Package report, source, logs, split and generator checkpoints; exclude data."""
import pathlib,zipfile,shutil,json
r=pathlib.Path(__file__).resolve().parent
out=r.parent/'output'; (out/'pdf').mkdir(parents=True,exist_ok=True)
shutil.copy2(r/'report.pdf',out/'pdf'/'Homework_2_Completed_Report.pdf')
shutil.copy2(r/'samples_epoch_10.png',out/'Generated_Paintings.png')
allowed={'.py','.tex','.sty','.json','.csv','.md','.txt','.png','.pdf','.pt','.log'}
files=[p for p in r.iterdir() if p.is_file() and (p.suffix in allowed or p.name in ['split.npz','.gitignore']) and p.name not in ['checkpoint.pt','compile_console.txt'] and (p.suffix!='.log' or p.name in ['train.py.log','evaluate.py.log'])]
target=out/'Homework_2_Reproducibility.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files: z.write(p,'Homework_2/'+p.name)
print(target,'bytes',target.stat().st_size,'files',len(files))
print(out/'pdf'/'Homework_2_Completed_Report.pdf')
