
# TMELFNet: Topo-Morphological Edge Logits Field for 3D Dental Proposal Generation and Instance Segmentation

A unified tooth instance segmentation framework based on edge learning, topological clustering, and patch-wise segmentation, with modular design and easy configuration.

## Overview
This project implements a modular tooth instance segmentation pipeline, which integrates edge prediction, topological clustering, and mask generation. The framework supports flexible training, inference, and evaluation for each component, and can be used as a plug-and-play boundary optimization module for existing tooth instance segmentation models.

The pipeline is built with **Hydra + YAML** for flexible configuration management, and uses PyTorch, PyTorch Geometric, Open3D, and PyTorch Lightning as core dependencies.

## Project Structure
```
├── configs/            # Hydra configuration files (core parameters)
├── dataset/            # Data loading and preprocessing module
├── model/              # Our proposed models (core implementation)
│   ├── _EdgeSegNet/    # Edge prediction network
│   ├── _TMELFNet /     # PatchSegNet + TMC module (inference-only)
│   └── InstanceSegNets/ # Tooth instance segmentation network
├── model2/             # Existing baseline models (Pointnet++, DGCNN, etc.)
├── requirements.txt    # Project dependencies
└── datasets/           # Datasets for training and evaluation
└── ...
```

## Core Modules
Our framework consists of four key components:

1. **ELFNet**
   - Predicts tooth boundaries (BG, TAE, TTE)
   - Lightweight and easy to train
   - Configured via `Module_BinDTSegNet.yaml`

2. **TMC (Topological Clustering Module)**
   - Clusters edge predictions to obtain tooth centroids
   - **No training required**, inference-only
   - Takes ELFNet outputs as input

3. **PatchSegNet**
   - Generates high-precision tooth instance masks
   - Trained with slightly perturbed ground-truth centroids
   - Supports flexible downsampling rates (≥16000 points recommended)

4. **InstanceSegNet**
   - Final tooth instance prediction network
   - Can be replaced with DTSegnet, PTSegnet, or other models

### Pipeline Flow
```
ELFNet → [BG, TAE, TTE] → TMC → [predicted centroids] → PatchSegNet → [instance masks] → InstanceSegNet
```

The combination **ELFNet + TMC + PatchSegNet** acts as a **plug-and-play boundary optimization module** for boosting instance segmentation performance.

## Dependencies
Main dependencies:
- torch
- torch_geometric
- open3d
- pytorch_lightning
- hydra-core
- numpy
- scipy

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Training
The framework supports separate training of each trainable module.

### 1. Train InstanceSegNet
Use existing models such as DTSegnet or PTSegnet. Training scripts and configurations are provided in the repository.

### 2. Train ELFNet
- Simple and fast training
- Configure parameters in `configs/Module_BinDTSegNet.yaml`
- Save trained weights for inference

### 3. Train PatchSegNet
- Trained with slightly jittered `gt_centroids`
- Adjust downsampling rate based on hardware performance
- Recommended minimum points: 16000

### 4. TMC
- **No training needed**
- Only used during inference

## Inference
Run prediction using trained models:
1. ELFNet: Load weights and predict edge labels
2. TMC: Take ELFNet outputs and predict cluster centroids
3. PatchSegNet: Take centroids and generate instance masks
4. Assign labels to instance masks with post-processing

Each module provides independent inference interfaces.

## Evaluation
Supports evaluation for all components:
- Instance segmentation metrics
- ELFNet boundary segmentation metrics
- TMC clustering metrics
- PatchSegNet binary segmentation metrics

## Usage
All modules support **script-based execution** and **code-based API calls** for training, inference, and evaluation.

Configure experiments easily via YAML files in the `configs/` directory using Hydra.

## Acknowledgements
This project references and re-implements code from:
- Teeth3DS
- Tooth Group Network(TGN)
- CrossTooth
- TADPM
- ToothInstanceNet
- etc.

We appreciate the contributions of the above works.


If our work is helpful to your research, please kindly cite our paper:
```
@ARTICLE{11644129,
  author={Zhuang, Jiafu and Chen, Lanxiang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Topo-Morphological Edge Logits Field for 3D Dental Proposal Generation and Instance Segmentation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2026.3721080}}
```

