# ECE 285 Homework 2: WikiArt DCGAN

Code repository: https://github.com/alextung2/ECE-285-HW2

This directory contains an actual training experiment on the assignment's
`steubk/wikiart` Kaggle archive. Generated examples come from the implemented
GAN, not an image generation service. The report is built from saved logs and
evaluation outputs. No Gradescope submission is performed.

The completed run used 75,694 training images and 5,000 held-out images after
removing 750 identical 64x64 crops from 81,444 successfully decoded images.
Ten epochs produced 5,910 updates. Final torchvision-feature FID was
95.29655; whole-set IS was 2.89162; ten-split IS was 2.86096 +/- 0.09317.
The final report is exactly two pages and was rendered for visual inspection.

## Reproduce

Use Python 3.12. A CUDA-enabled PyTorch install is recommended. The original
machine uses a CUDA 11.8 build with its installed NVIDIA driver.

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python download.py
python prepare.py
python train.py --epochs 10
python evaluate.py --checkpoints initial.pt generator_epoch_01.pt generator_epoch_05.pt generator_epoch_10.pt
python figures.py
python report_values.py
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Run the LaTeX commands from this directory. Scripts resolve their input/output
paths relative to themselves.
The archive is approximately 33.7 GB. The preprocessed cache is about 1 GB.
The data and Inception cache are excluded from the portable submission package.
Generate additional images with `python generate.py --seed 2285`.

## Experiment details and audit trail

- `prepare.py` converts images to RGB, resizes the shorter side to 64 with PIL
  bilinear interpolation, and center-crops 64x64. Corrupt images are logged.
  Identical 64x64 crops are deduplicated before splitting. This is not a
  near-duplicate or artist-disjoint split.
- `manifest.json` records all archive paths, failures, duplicate mappings and
  split counts. `split.npz` stores the exact train/test indices (seed 285).
- `train.py` implements the two networks, normal initialization, alternating
  non-saturating BCE optimization, fixed-noise snapshots, and checkpoints.
  `losses.csv` contains every optimization step. `config.json` records settings.
- `evaluate.py` uses a disjoint set of 5,000 real images and 5,000 generated
  images per checkpoint. Evaluation seed 1285 is reset for each checkpoint.
  It uses the assignment-linked Ignite metric classes with explicitly supplied
  2048-dimensional pooled features for FID and softmax probabilities for IS.
  All image inputs are restored to [0,1], quantized identically, resized with
  PIL bilinear interpolation, and normalized for pretrained Inception-v3.
- FID is lower-is-better; IS is higher-is-better. The whole-set Ignite IS and
  ten-split IS are both saved. Split standard deviation is not a confidence
  interval across training runs. These are torchvision-Inception FID results,
  not TensorFlow/clean-fid benchmark values. IS uses ImageNet classes and does
  not directly measure artistic quality.
- Training uses one seed. cuDNN benchmarking is enabled, so bit-for-bit CUDA
  reproducibility is not guaranteed. A resumed run restores PyTorch CPU/CUDA
  random states and optimizer states.

## Primary references

- Assigned dataset: https://www.kaggle.com/datasets/steubk/wikiart
- Assigned evaluation tutorial: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- DCGAN: https://arxiv.org/abs/1511.06434
- Ignite FID API: https://docs.pytorch.org/ignite/generated/ignite.metrics.FID.html

The tutorial's last verbal statement reverses metric directions; this project
uses the correct directions. It also avoids the default Ignite FID feature
extractor (1000 class probabilities) by explicitly using pooled image features.
