# DUFlow Minimal Reproduction

This folder extracts the smallest runnable training/evaluation subset from the working directory for the paper:

> **GPU-Based Real-Time Topology-Preserving Crack Segmentation**

The code implements the original dual-branch model available in this workspace: a directional-gCRU discriminative branch plus a conditional Flow Matching branch with multi-scale feature fusion. It intentionally excludes datasets, checkpoints, predictions, ablations, reviewer-only controls, AETF/FMS2 extensions, VMamba variants, and generated deployment artifacts.

## 1. Contents

```text
DUFlow_minimal_repro/
├── flow_matching/
│   ├── __init__.py
│   ├── dataset_loader.py
│   └── dual_branch_model_parallel.py
├── train/
│   ├── __init__.py
│   └── train_dual_branch_parallel.py
├── test/
│   ├── __init__.py
│   ├── eval_dual_branch_parallel.py
│   └── metrics.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. Dataset links

### FIND (main dataset used by the paper)

- Dataset title: **Fused Image Dataset for convolutional neural Network-based crack Detection (FIND)**
- Zenodo record: https://doi.org/10.5281/zenodo.6383044
- Concept DOI: https://doi.org/10.5281/zenodo.6383043

The paper uses the fused/5D modality and the official split:

- official Train: 2,000 samples
- official Test: 500 samples
- the official Train set is split reproducibly into 1,800 training and 200 validation samples
- the 500-image official Test set is not used for model selection

### CRACK500 (fine-tuning and cross-dataset evaluation)

- Project/data entry: https://github.com/fyangneil/pavement-crack-detection

The paper follows an 8:1:1 protocol (400/50/50) for its CRACK500 adaptation experiment. This differs from the commonly distributed original 250/50/200 split, so the exact split manifest must be recorded when reproducing the paper.

### DeepCrack (zero-shot evaluation)

- Official project/data entry: https://github.com/yhlleo/DeepCrack

### CSF-50K (zero-shot evaluation)

The paper cites:

> Asadi et al., “Controllable flow matching synthesis for generalizable crack segmentation with a 50k image-mask benchmark,” *Automation in Construction* 187, 106938 (2026).

The supplied paper does not provide a verified public download URL for CSF-50K. To avoid publishing an unverified link, obtain the dataset from the cited paper's official data-availability statement or the authors' repository.

## 3. Expected FIND layout

After downloading FIND, arrange the selected modality as follows:

```text
dataset/
├── Train/
│   ├── 5d/
│   │   └── *.png
│   └── mask/
│       └── *.bmp
└── Test/
    ├── 5d/
    │   └── *.png
    └── mask/
        └── *.bmp
```

The loader validates equal image/mask counts and checks filename-stem pairing. If the downloaded archive uses a different naming convention, rename or reorganize it without changing the official Train/Test membership.

## 4. Environment

Python 3.9 or newer is recommended. Install PyTorch for your own CUDA/CPU platform first, then install the remaining packages:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

For a CUDA-specific PyTorch command, use the official selector:
https://pytorch.org/get-started/locally/

## 5. Train

The paper reports 60 epochs, batch size 8, AdamW, initial learning rate 1e-4, cosine annealing, input size 256×256, and threshold 0.5. The extracted training script provides the core model and deterministic 90/10 split; use the following command as the paper-oriented starting point:

```bash
python -m train.train_dual_branch_parallel \
  --data-root dataset \
  --epochs 60 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --image-size 256 \
  --threshold 0.5 \
  --amp \
  --output-dir checkpoints/duflow_find
```

The script writes:

- `split_manifest.json`
- `best.pt`
- `last.pt`
- optional periodic checkpoints
- `results.json`

Important: the current workspace implementation uses `DualBranchLoss = flow MSE + weight_cal × (BCE + Dice)` with default `weight_cal=10`. The paper text describes a richer final topology/boundary/router objective, while several implementation details are not fully specified in the PDF. Therefore, this extracted folder reproduces the smallest runnable implementation present in the workspace, not a claim that every equation in the paper is reconstructed solely from the PDF.

## 6. Evaluate

The paper selects one-step Euler inference. Evaluate a checkpoint with:

```bash
python -m test.eval_dual_branch_parallel \
  --data-root dataset \
  --test-split Test \
  --checkpoint checkpoints/duflow_find/best.pt \
  --output-dir eval_results/duflow_find \
  --sample-steps 1 \
  --threshold 0.5
```

The evaluator reports dataset-level micro Dice and IoU for:

- `cal_branch`: discriminative branch
- `flow_branch`: Flow Matching branch
- `best_one`: ground-truth-visible per-image oracle, diagnostic only
- `oracle_global`: ground-truth-visible global oracle, diagnostic only

`best_one` and `oracle_global` are **not deployable model outputs** and must not be reported as ordinary inference results.

The script also saves binary predictions under the selected output directory.

## 7. Quick smoke checks

Compile all Python modules:

```bash
python -m compileall flow_matching train test
```

Instantiate the model and run a small CPU forward pass:

```bash
python -c "import torch; from flow_matching.dual_branch_model_parallel import DualBranchParallelFlowModel; m=DualBranchParallelFlowModel(in_channels=10, base_channels=8); x=torch.randn(1,10,32,32); t=torch.zeros(1); v,p=m(x,t); print(v.shape,p.shape)"
```

Expected output:

```text
torch.Size([1, 3, 32, 32]) torch.Size([1, 1, 32, 32])
```

## 8. Reproducibility notes

- The training script uses seed `20260902` by default and writes exact split indices.
- The official Test split is evaluated only after validation-based checkpoint selection.
- Flow inference begins from random noise; keep the evaluation seed and Euler step count in every result record.
- The evaluator's Dice/IoU are dataset-level micro metrics. Do not compare them directly with per-image macro averages unless the aggregation rule is matched.
- The PDF does not fully specify every implementation detail, including some topology/boundary reductions, the complete router architecture, all data augmentation, and all optimizer settings. Preserve the code/config used for each published result.
- Pretrained weights are intentionally excluded because the local checkpoints are hundreds of megabytes. Publish them separately via a GitHub Release, Zenodo, or a model hub, with SHA-256 and model configuration.

## 9. Before publishing to GitHub

This extracted folder is not initialized as a Git repository. To create a standalone repository, run `git init` inside `DUFlow_minimal_repro` after reviewing the files.

1. Confirm the copyright holder and add an appropriate `LICENSE` file. No license is added here because it must not be guessed.
2. Do not commit datasets, checkpoints, predictions, ONNX/TensorRT engines, or local environment files.
3. If weights are released, document their URL, SHA-256, epoch, model class, split seed, and checkpoint schema.
4. Cite both the paper and each dataset according to their original licenses and citation instructions.

## 10. Scope

Included:

- directional gCRU dual-branch model
- FIND loader
- deterministic training/validation split
- training and checkpoint selection
- CAL and Euler Flow evaluation
- Dice/IoU metrics

Excluded:

- dataset files and weights
- topology-boundary debate/VMamba extensions found elsewhere in the workspace
- nondirectional ablation
- AETF/FMS2 and feature-codec experiments
- reviewer control experiments
- Jetson/ONNX/TensorRT generated artifacts
- third-party baseline source trees
