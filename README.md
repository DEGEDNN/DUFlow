# DUFlow

Official implementation of **DUFlow**, a dual-branch generative–discriminative framework for pavement crack segmentation.

DUFlow combines a discriminative U-Net branch, conditional Flow Matching, directional gCRU modules, multi-scale cross-branch feature fusion, and topology–boundary debate routing for topology-preserving crack segmentation.

## Highlights

- Dual independent encoders
- Four-directional gCRU for crack continuity modeling
- Conditional Flow Matching
- Encoder- and decoder-level multi-scale feature fusion
- Topology–Boundary Debate Routing
- One-step Euler inference

## Architecture

DUFlow contains two complementary branches:

1. **Discriminative branch**
   - U-Net encoder–decoder
   - four-directional gCRU enhanced skip features
   - produces discriminative crack prediction \(p_d\)

2. **Flow Matching branch**
   - independent encoder
   - time-conditioned decoder
   - receives multi-scale structural guidance from the discriminative branch
   - produces Flow Matching prediction \(p_g\)

The two predictions are adaptively combined by the debate router:

\[
p_f = w_d p_d + w_g p_g,
\qquad
w_d + w_g = 1.
\]

During training, topology and boundary teachers provide routing supervision. They are removed during inference, so the deployed router does not require ground truth.

## Repository Structure

```text
DUFlow/
├── data/
├── models/
├── eval_cross_dataset.py
└── README.md
```

Main components:

- `models/duflow_model.py` — DUFlow network
- `models/duflow_wrapper.py` — inference utilities
- `data/` — dataset loading and preprocessing
- `eval_cross_dataset.py` — cross-dataset evaluation

## Installation

```bash
git clone https://github.com/DEGEDNN/DUFlow.git
cd DUFlow

conda create -n duflow python=3.10 -y
conda activate duflow

pip install torch torchvision
pip install numpy opencv-python scipy scikit-image tqdm
```

## Datasets

Experiments use the **FIND** and **CRACK500** pavement crack datasets.

### FIND

FIND (Fused Image dataset for convolutional neural Network-based crack Detection) contains pixel-level crack annotations and multiple pavement imaging modalities, including intensity, range, filtered range, and fused images.

Dataset: https://doi.org/10.5281/zenodo.6383044

If you use FIND, please cite:

```bibtex
@dataset{zhou2022find,
  author    = {Zhou, Shanglian and Canchila, Carlos and Song, Wei},
  title     = {Fused Image dataset for convolutional neural Network-based crack Detection (FIND)},
  year      = {2022},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.6383044}
}
```

### CRACK500

CRACK500 is a large-scale pavement crack segmentation dataset derived from high-resolution road images with pixel-level annotations.

If you use CRACK500, please cite the original works associated with the dataset:

```bibtex
@inproceedings{zhang2016road,
  title     = {Road Crack Detection Using Deep Convolutional Neural Network},
  author    = {Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
  pages     = {3708--3712},
  year      = {2016},
  publisher = {IEEE}
}

@article{yang2019feature,
  title   = {Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author  = {Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal = {arXiv preprint arXiv:1901.06340},
  year    = {2019}
}
```

## Evaluation

Example:

```bash
python eval_cross_dataset.py \
    --weight /path/to/model.pth \
    --data_root /path/to/dataset \
    --dataset crack500 \
    --device cuda
```

For one-step Flow Matching inference:

```bash
python eval_cross_dataset.py \
    --weight /path/to/model.pth \
    --data_root /path/to/dataset \
    --dataset generic \
    --sample_steps 1 \
    --device cuda
```

## Core Method

The main components of DUFlow are:

- **Directional gCRU** for long-range crack continuity
- **Independent discriminative and Flow Matching encoders**
- **E3/E4 encoder-level structural guidance**
- **D1–D4 decoder-level multi-scale fusion**
- **Time-conditioned Flow Matching**
- **Topology–Boundary Debate Routing**
- **Adaptive final prediction fusion**

## Paper

The DUFlow manuscript is currently under review / preparation. Citation information will be added after publication.

## License

Please add the appropriate open-source `LICENSE` file before public release.

## Contact

For questions about the implementation, please open an issue in this repository.
