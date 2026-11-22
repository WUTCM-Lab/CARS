# CARS Official Release

## CARS: Conflict-Aware Relation Self-Distillation for Unsupervised Multimodal Change Detection

This repository contains the official implementation of CARS.

CARS learns compact structural relations from multi-level features and addresses the conflict between cross-modal alignment and change preservation through reliability-aware pseudo-anchor mining, conflict-aware relation self-distillation, and adaptive relation-aware change scoring.

The source code is available at: https://github.com/WUTCM-Lab/CARS

## Method Components

### 1. Multi-Level Compact Relation Encoder (MCRE)

MCRE converts multi-level feature maps into a small set of learnable relation tokens and constructs compact relation matrices to represent within-image structural dependencies.

### 2. Reliability-Aware Pseudo-Anchor Mining and Conflict-Aware Relation Self-Distillation

An exponential moving average (EMA) teacher provides stable relation targets and confidence cues to identify likely unchanged and high-discrepancy candidate regions. CARS aligns cross-temporal relations on likely unchanged candidates while preserving relation discrepancies on high-discrepancy candidates. Conflict-aware weighting balances these two objectives.

### 3. Adaptive Relation-Aware Change Scoring (ARCS)

ARCS estimates the quality of feature and relation discrepancy cues from unlabeled score statistics and adaptively combines them to produce the final change-intensity map.

## Inference

Only the student branch is required for inference. The EMA teacher is used to provide training targets and is not needed to compute the final change map.

```text
Multimodal Bi-temporal Image Pair
              |
      Shared Student Backbone
              |
 Multi-Level Compact Relation Encoder
              |
   Feature Discrepancy + Relation Discrepancy
              |
 Adaptive Relation-Aware Change Scoring
              |
       Change-Intensity Map
              |
        Otsu Thresholding
              |
          Change Map
```

## Repository Structure

```text
CARS_Codes/

├── model/
│   └── CARS.py
│
├── utils/
│   ├── data.py
│   └── Evaluation.py
│
├── data/
│   ├── Flood/
│   │   ├── T1.ppm
│   │   ├── T2.ppm
│   │   └── GT.ppm
│   ├── Italy/
│   │   ├── Italy_1.bmp
│   │   ├── Italy_2.bmp
│   │   └── Italy_gt.bmp
│   ├── Mexico/
│   │   ├── im1.bmp
│   │   ├── im2.bmp
│   │   └── im3.bmp
│   └── Yellow/
│       ├── yellow_C_1.bmp
│       ├── yellow_C_2.bmp
│       └── yellow_C_gt.bmp
│
├── train_CARS_relation_ARCS.py
└── README.md
```

The included image pairs correspond to Italy, Gloucester (stored in `data/Flood/`), Yellow River, and Mexico. Italy, Gloucester, and Yellow River are heterogeneous pairs; Mexico is included as a same-sensor control pair.

## Requirements

The implementation uses Python, PyTorch, torchvision, OpenCV, NumPy, SciPy, scikit-learn, and tqdm. Install PyTorch and torchvision versions compatible with your CUDA environment, then install the remaining dependencies:

```bash
pip install opencv-python numpy scipy scikit-learn tqdm
```

A pinned `requirements.txt` is not included in the current release.

## Running

From the repository root (`CARS_Codes/`), the default command trains and evaluates CARS on the included Yellow River pair:

```bash
python train_CARS_relation_ARCS.py
```

To run another included pair, specify its image and ground-truth paths:

**Italy**

```bash
python train_CARS_relation_ARCS.py \
    --data_name italy \
    --t1_path ./data/Italy/Italy_1.bmp \
    --t2_path ./data/Italy/Italy_2.bmp \
    --gt_path ./data/Italy/Italy_gt.bmp
```

**Gloucester**

```bash
python train_CARS_relation_ARCS.py \
    --data_name gloucester2 \
    --t1_path ./data/Flood/T1.ppm \
    --t2_path ./data/Flood/T2.ppm \
    --gt_path ./data/Flood/GT.ppm
```

**Mexico**

```bash
python train_CARS_relation_ARCS.py \
    --data_name Mexico \
    --t1_path ./data/Mexico/im1.bmp \
    --t2_path ./data/Mexico/im2.bmp \
    --gt_path ./data/Mexico/im3.bmp
```

The main training and evaluation settings can be changed through command-line arguments. Run the following command to list the available options:

```bash
python train_CARS_relation_ARCS.py --help
```

The script automatically uses `cuda:0` when CUDA is available and otherwise runs on the CPU. The output root can be changed with `--vision_path`.

## Current Release

Included:

- CARS model implementation, including MCRE and the EMA student-teacher framework
- Training and evaluation pipeline with conflict-aware relation self-distillation and ARCS
- Evaluation utilities
- Four image pairs used in the paper

Not included:

- Pinned environment configuration (`requirements.txt`)
- Separate inference-only entry script
- Standalone configuration files (parameters are provided through command-line arguments)

## Citation

```bibtex
@article{zhang2026cars,
  title   = {CARS: Conflict-Aware Relation Self-Distillation for Unsupervised Multimodal Change Detection},
  author  = {Zhang, Bo and Chen, Yaxiong and Gong, Tengfei and Xiong, Shili},
  journal = {},
  year    = {2026},
  note    = {}
}
```
