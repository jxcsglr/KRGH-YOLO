# KRGH-YOLO

Official implementation of **KRGH-YOLO**, a YOLO11s-based detector for
fine-grained classroom behavior detection.

KRGH-YOLO adds two lightweight modules while leaving the original box-regression
and DFL-based decoding paths unchanged:

- **Key-Region Guided Fusion (KRGFusion)** recalibrates the two top-down neck
  concatenation features using local and oriented strip context.
- **Head-Hand-Aware Classification Attention (HHCA)** adapts only the P3 and P4
  classification branches using complementary posture and hand/object-interaction
  cues. It is not an explicit head or hand detector and needs no part labels.

This repository contains only the final YOLO11s-based implementation and its
training utilities. Earlier exploratory model families are not included.

## Repository layout

```text
KRGH-YOLO/
  krgh_yolo/
    krgfusion.py       # KRGFusion and neck injection
    hhca.py             # HHCA, CBAM control, and classification injection
    trainer.py          # Ultralytics trainer factory
    data.py             # SCB/SCSB dataset YAML generation
  train.py              # train + best-checkpoint validation + JSON summary
  validate.py           # standalone validation
  predict.py            # image/video inference
  scripts/              # one-command Linux and PowerShell wrappers
  tests/                # dataset-independent module tests
```

The `experiments/ultralytics_baselines/` package contains only two compatibility
aliases. They allow checkpoints trained in the original research workspace to be
loaded after moving them into this repository.

## Installation

Python 3.11, PyTorch 2.5.1, CUDA 12.4, and Ultralytics 8.3.165 were used for the
paper experiments. A clean environment can be prepared with:

```bash
conda create -n krgh-yolo python=3.11 -y
conda activate krgh-yolo

# Optional: install the PyTorch build matching your CUDA runtime first.
# See https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Dataset format

The one-command wrappers expect a standard YOLO detection dataset:

```text
dataset_root/
  images/
    train/
    val/
  labels/
    train/
    val/
```

SCB and SCSB class names are built in. For another dataset, pass an Ultralytics
dataset YAML directly or provide `--class-names` in class-ID order. Dataset images
and labels are not distributed in this repository.

## One-command training

The defaults reproduce the final model recipe: pretrained YOLO11s, KRGFusion at
neck layers 12/15, HHCA on P3/P4 classification branches, AdamW, 300 epochs,
640 x 640 input, batch size 8, and seed 22.

SCB-Dataset3-U on Linux/WSL:

```bash
bash scripts/train_scb.sh /path/to/SCB-Dataset3-U 0
```

SCSB individual6:

```bash
bash scripts/train_scsb.sh /path/to/SCSB_yolo_individual6 0
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 `
  -DatasetRoot "D:\datasets\SCB-Dataset3-U" -Preset scb -Device 0
```

The equivalent direct command is:

```bash
python train.py /path/to/SCB-Dataset3-U --preset scb --device 0
```

Run a configuration check without importing PyTorch or starting training:

```bash
python train.py /path/to/SCB-Dataset3-U --preset scb --dry-run
```

Each completed run contains `weights/best.pt`, standard Ultralytics artifacts,
and `summary.json` with the protocol, metrics, parameter count, FLOPs, inserted
modules, package versions, and checkpoint provenance.

## Validation and inference

```bash
python validate.py runs/krgh-yolo/EXPERIMENT/weights/best.pt \
  /path/to/SCB-Dataset3-U --preset scb --device 0

python predict.py runs/krgh-yolo/EXPERIMENT/weights/best.pt \
  /path/to/images --device 0 --conf 0.25
```

Both current checkpoints and checkpoints produced under the historical
`experiments.ultralytics_baselines.*` module paths are supported.

## Ablations

The same entry point covers the paper's controlled variants:

```bash
# YOLO11s baseline
python train.py DATA --fusion none --head none --name baseline

# KRGFusion only
python train.py DATA --fusion krgf --head none --name krgf_only

# HHCA only
python train.py DATA --fusion none --head hhca --name hhca_only

# Internal KRGFusion branches
python train.py DATA --head none --krgf-context local --name krgf_local
python train.py DATA --head none --krgf-context strip --name krgf_strip

# Internal HHCA branches
python train.py DATA --fusion none --hhca-branch posture --name hhca_posture
python train.py DATA --fusion none --hhca-branch hand_object --name hhca_hand_object

# CBAM control at the same classification scales
python train.py DATA --fusion krgf --head cbam --name krgf_cbam
```

Insertion positions are controlled with `--krgf-layers` and `--hhca-scales`.
Pass `--help` to see every option.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q krgh_yolo experiments train.py validate.py predict.py
```

## Reproducibility notes

- Metrics may vary with random seed, CUDA kernels, package versions, and dataset
  preparation. Keep `summary.json` with every reported result.
- `best.pt` is selected by Ultralytics validation fitness
  (`0.1 * mAP50 + 0.9 * mAP50:95`) and is re-evaluated before summary export.
- Model weights, datasets, and generated `runs/` directories are intentionally
  excluded from Git.

## Publish to GitHub

Create an empty repository named `KRGH-YOLO` on GitHub, then run the following
commands from this directory:

```bash
git init
git add .
git commit -m "Initial KRGH-YOLO release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/KRGH-YOLO.git
git push -u origin main
```

Before making the repository public, replace `YOUR_ACCOUNT`, add the paper's
BibTeX entry when it is available, and choose an explicit software license. A
license is deliberately not selected here because that legal choice belongs to
the project authors.
