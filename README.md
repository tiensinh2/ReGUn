# Reference-Guided Machine Unlearning 

Official implementation for the paper: **"Reference-Guided Machine Unlearning"** (Currently under review).

---
## Abstract

Machine unlearning aims to remove the influence of specific data from trained models while preserving general utility. Existing approximate unlearning methods often rely on performance-degradation heuristics, such as loss maximization or random labeling. However, these signals can be poorly conditioned, leading to unstable optimization and harming the model's generalization. We argue that unlearning should instead prioritize distributional indistinguishability, aligning the model’s behavior on forget data with its behavior on truly unseen data. Motivated by this, we propose Reference-Guided Unlearning ReGUn, a framework that leverages a disjoint held-out dataset to provide a principled, class-conditioned reference for distillation. We demonstrate that, across various model architectures, natural image datasets, and varying forget fractions, ReGUn consistently outperforms standard approximate baselines, achieving a superior forgetting-utility trade-off.

---

## Overview

**ReGUn** is a machine unlearning approach that aligns model behavior on forget data with predictions from a disjoint held-out reference set.


This codebase implements ReGUn and provides a **unified evaluation pipeline** for comparing multiple machine unlearning methods on standard vision benchmarks:

- **Datasets**: CIFAR-10, CIFAR-100 (auto-downloaded), Tiny-ImageNet (must be downloaded and placed in data directory)
- **Models**: ResNet, Swin Transformer, Vision Transformer (ViT)
- **Unlearning Methods**: ReGUn, NegGrad, NegGrad+, Finetune, l1-sparse, SCRUB, SalUn, SSD, Amun, LUR

All configurations are managed via [Hydra](https://hydra.cc/) (see `conf/` directory).

---

## Getting Started

Install dependencies `requirements.txt` (requires Python 3.8+ and PyTorch >= 2.0).

**Weights & Biases**: The pipeline integrates with [wandb](https://wandb.ai/) for experiment tracking. Login with `wandb login` or set `WANDB_MODE=offline` to disable cloud syncing.

### Configuration

All experiments are configured via YAML files in the `conf/` directory:

- **`conf/config.yaml`**: Main configuration with defaults
- **`conf/data/`**: Dataset configurations (CIFAR-10, CIFAR-100, Tiny-ImageNet)
- **`conf/model/`**: Model architectures (ResNet, Swin, ViT)
- **`conf/unlearn/`**: Unlearning method hyperparameters

*Note: The default settings are for ResNet on CIFAR. Other models and datasets require config adjustments (e.g. num_classes, model_stem, ...)*

---

## Pipeline: Three-Step Execution

The evaluation pipeline consists of three sequential scripts that must be run in order:

### Step 1: Generate Reference Models (`run1_reference.py`)

**Purpose**: Train multiple reference models on the retain split for robust membership inference evaluation (RMIA)(typically run 4 times with different random seeds).

**Example Usage**:
```bash
# Run this 4 times with different --run-idx values (1, 2, 3, 4)
for idx in 1 2 3 4; do
  python run1_reference.py --run-idx $idx data=cifar10 model=resnet
done
```

**Output**: Saved reference models in `$CACHE_DIR/models/`

---

### Step 2: Train Base and Retrained Models (`run2_base.py`)

**Purpose**: Train a base model on the full training set and a retrained model from scratch on only the retain split.

**Example Usage**:
```bash
python run2_base.py data=cifar10 model=resnet
```

**Output**: 
- Base model: `$CACHE_DIR/models/*_base.ckpt`
- Retrained model: `$CACHE_DIR/models/*_retrained.ckpt`

---

### Step 3: Run Unlearning and Evaluate (`run3_unlearning.py`)

**Purpose**: Load the base model, apply the specified unlearning method, and evaluate it.

**Example Usage**:
```bash
# Run ReGUn
python run3_unlearning.py data=cifar10 model=resnet unlearn=regun

# Or other methods
python run3_unlearning.py data=cifar10 model=resnet unlearn=neggrad
python run3_unlearning.py data=cifar10 model=resnet unlearn=salun
```

**Output**: Results and its logs in `$OUTPUTS_DIR/`

---

## Running on SLURM Clusters

For compute clusters using SLURM and Singularity/Apptainer:

### 1. Build Container

```bash
apptainer build mul_env.sif mul_env.def
```

*Note: You may need to adjust the base image or CUDA/PyTorch versions in `mul_env.def`*

### 2. Set Environment Variables

```bash
export PROJ_DIR=$PWD
export IMG=/path/to/mul_env.sif
export DATASET_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUTS_ROOT=/path/to/outputs
```

### 3. Submit Jobs

```bash
# Step 1: Reference models (submit 4 jobs)
for idx in 1 2 3 4; do
  sbatch --export=ALL,PROJ_DIR=$PROJ_DIR,IMG=$IMG,DATASET_ROOT=$DATASET_ROOT,CACHE_ROOT=$CACHE_ROOT,OUTPUTS_ROOT=$OUTPUTS_ROOT \
    run_slurm.sbatch run1 --run-idx $idx data=cifar10 model=resnet
done

# Step 2: Base training
sbatch --export=ALL,PROJ_DIR=$PROJ_DIR,IMG=$IMG,DATASET_ROOT=$DATASET_ROOT,CACHE_ROOT=$CACHE_ROOT,OUTPUTS_ROOT=$OUTPUTS_ROOT \
  run_slurm.sbatch run2 data=cifar10 model=resnet

# Step 3: Unlearning
sbatch --export=ALL,PROJ_DIR=$PROJ_DIR,IMG=$IMG,DATASET_ROOT=$DATASET_ROOT,CACHE_ROOT=$CACHE_ROOT,OUTPUTS_ROOT=$OUTPUTS_ROOT \
  run_slurm.sbatch run3 data=cifar10 model=resnet unlearn=regun
```

---
