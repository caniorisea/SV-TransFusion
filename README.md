# SV-TransFusion: Sparse Voxel–Query Interaction for LiDAR 3D Object Detection

This repository contains the implementation of **SV-TransFusion**, a LiDAR-only 3D object detector built on **OpenPCDet v0.5.0**.

> **Status:** The accompanying manuscript is **under review**.  

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
  - [SVQI: Sparse Voxel–Query Interaction](#svqi-sparse-voxelquery-interaction)
  - [QCD: Query-based Contrastive Denoising (training-only)](#qcd-query-based-contrastive-denoising-training-only)
- [Environment](#environment)
- [Installation](#installation)
- [Data Preparation (nuScenes)](#data-preparation-nuscenes)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference Notes](#inference-notes)
- [Repository Structure](#repository-structure)
- [Compatibility](#compatibility)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **SVQI (Sparse Voxel–Query Interaction):** Enables object queries to interact with **non-empty sparse 3D voxels** to recover vertical/structural cues that may be weakened by BEV flattening.
- **QCD (Query-based Contrastive Denoising):** A **training-only** strategy combining denoising queries and a contrastive objective to improve robustness and separability of learned query representations.
- **OpenPCDet-based:** Implemented following **OpenPCDet v0.5.0** conventions for dataset handling, training, and evaluation.

---

## Method Overview

SV-TransFusion follows a query-based detection paradigm commonly used in BEV-based LiDAR detectors, and extends query refinement with direct interaction between queries and sparse 3D voxel features.

### SVQI: Sparse Voxel–Query Interaction

Many BEV-based pipelines aggregate 3D voxel features along the height axis to build a 2D BEV representation. SVQI complements this by allowing each query to attend to a **local** set of **non-empty** sparse voxels around its current predicted 3D center (updated layer-by-layer in the decoder).

**Core ideas:**
- **Dynamic voxel indexing:** For each query, retrieve a local neighborhood of non-empty voxels near the query center to avoid global attention over all voxels.
- **Geometry-aware alignment:** Incorporate **relative position** (query center to voxel location) via a learnable positional encoding so attention can reason about fine-grained 3D structure.
- **Sparse cross-attention:** Perform attention only on retrieved **non-empty** voxels, avoiding invalid sampling in empty space.

### QCD: Query-based Contrastive Denoising (training-only)

Transformer-style bipartite matching can be unstable early in training. QCD introduces an auxiliary training mechanism:
- Generate **noise-corrupted** queries from ground-truth boxes (e.g., center jittering and size scaling).
- Use **attention masks** to prevent information leakage between denoising groups and standard matching queries.
- Apply a **contrastive loss** on query embeddings to encourage separability between classes/instances.

**QCD is removed during inference** and does not introduce inference-time branches.

---

## Environment

Reference versions for this repository:

- **Python:** 3.8
- **PyTorch:** 1.11.1
- **CUDA:** Required (use a PyTorch 1.11.1 build matching your CUDA toolkit/driver)
- **Base framework:** OpenPCDet **v0.5.0**
- **Dataset:** nuScenes (LiDAR-only)

> **Tip:** Build and runtime issues are most commonly caused by mismatched PyTorch/CUDA versions. Keep them consistent.

---

## Installation

### 1) Clone
```bash
cd sv-transfusion
```

### 2) Create conda environment
```bash
conda create -n svtransfusion python=3.8 -y
conda activate svtransfusion
```

### 3) Install PyTorch 1.11.1
Install PyTorch **1.11.1** from the official PyTorch channel for your CUDA version.

### 4) Install dependencies and compile extensions
This repository follows OpenPCDet v0.5.0 installation patterns.

```bash
pip install -r requirements.txt
python setup.py develop
```

**If you see errors compiling CUDA ops:**
- Check `nvcc` is available (`nvcc --version`).
- Ensure your CUDA toolkit matches the PyTorch wheel CUDA version.
- Ensure your GPU driver is compatible with the CUDA version.

---

## Data Preparation (nuScenes)

### 1) Download nuScenes
Download nuScenes from the official source and place it under:

```text
data/nuscenes/
  v1.0-trainval/
  samples/
  sweeps/
  maps/
  ...
```
(Keep the nuScenes folder structure as required by the nuScenes devkit.)

### 2) Generate OpenPCDet info files
Use OpenPCDet-style preprocessing to create infos.

Example command (path/module name may vary slightly depending on repo layout):
```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset create_nuscenes_infos \
  --data_path ./data/nuscenes \
  --version v1.0-trainval
```

If your repository provides a wrapper under `tools/`, prefer that wrapper.

---

## Training

Training is driven by YAML config files under `configs/`.

Typical config (example name):
- `configs/nuscenes/sv_transfusion.yaml`

The config usually controls:
- Enabling/disabling **SVQI**
- Enabling/disabling **QCD**
- SVQI neighborhood parameters (e.g., radius, sampling)
- QCD noise parameters and denoising group settings
- Loss weights (matching / denoising / contrastive)
- Standard OpenPCDet training settings (optimizer, schedule, augmentation, etc.)

### Run training
```bash
python tools/train.py \
  --cfg_file configs/nuscenes/sv_transfusion.yaml
```

For distributed training, use the launch scripts provided by this repository (if included), following OpenPCDet v0.5.0 conventions.

---

## Evaluation

### Evaluate a checkpoint
```bash
python tools/test.py \
  --cfg_file configs/nuscenes/sv_transfusion.yaml \
  --ckpt /path/to/checkpoint.pth
```

Evaluation outputs are produced using the nuScenes evaluation pipeline integrated in OpenPCDet-style code.

---

## Inference Notes

- **QCD is training-only** and is disabled/removed during inference.
- Inference uses the standard query initialization and query refinement pipeline configured in the YAML.
- If SVQI is enabled in the config, inference includes SVQI-based query–voxel interaction.

---

## Repository Structure

A typical layout (exact structure may vary):

```text
sv-transfusion/
  configs/                 # experiment configs
  pcdet/                   # OpenPCDet core (v0.5.0 style)
  tools/                   # train/test scripts
  data/
    nuscenes/              # dataset (not included)
  README.md
```

---

## Compatibility

- Designed to be compatible with **OpenPCDet v0.5.0**
- Reference runtime: **Python 3.8** + **PyTorch 1.11.1**

---

## Citation

The manuscript is under review. Please check back for an updated BibTeX entry.

---

## Acknowledgements

This project is built on top of **OpenPCDet** and uses the **nuScenes** dataset and devkit.# SV-TransFusion
