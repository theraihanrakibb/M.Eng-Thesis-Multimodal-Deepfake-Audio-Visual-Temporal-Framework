# Detecting Deepfake Video by A Multimodal Audio-Visual Framework with Temporal Inconsistencies

<p align="center">
  <img src="https://img.shields.io/badge/Thesis-NWPU-blue" alt="NWPU Thesis"/>
  <img src="https://img.shields.io/github/license/theraihanrakibb/M.Eng-Thesis-Multimodal-Deepfake-Audio-Visual-Temporal-Framework" alt="License"/>
  <img src="https://img.shields.io/github/stars/theraihanrakibb/M.Eng-Thesis-Multimodal-Deepfake-Audio-Visual-Temporal-Framework" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/theraihanrakibb/M.Eng-Thesis-Multimodal-Deepfake-Audio-Visual-Temporal-Framework" alt="Forks"/>
  <img src="https://img.shields.io/github/last-commit/theraihanrakibb/M.Eng-Thesis-Multimodal-Deepfake-Audio-Visual-Temporal-Framework" alt="Last Commit"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Dataset-WildDeepfake-orange" alt="WildDeepfake"/>
</p>


## Overview

A research-grade deepfake detection framework combining visual (spatial + FFT frequency), audio (MFCC), and temporal (Transformer) features with attention-based fusion. Trained and evaluated on the WildDeepfake dataset — real deepfake videos sourced from the internet.

**Author:** MD RAKIBUL ISLAM RAIHAN — M.Eng. Software Engineering, Northwestern Polytechnical University (985/211), GPA 88/100 (Top 1%)
- Email: raihanrakib.143@gmail.com
- GitHub: [theraihanrakibb](https://github.com/theraihanrakibb)
- Portfolio: [theraihanrakibb.github.io](https://theraihanrakibb.github.io/)

**Supervisor:** ZHANG PENG  
**University:** Northwestern Polytechnical University, School of Software Engineering

---

## Results (WildDeepfake — 110 real deepfake videos)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| MultimodalModel | 50.91% | 83.33% | 8.62% | 15.63% | 0.6388 |
| MesoNet | 47.27% | 0.00% | 0.00% | 0.00% | 0.5739 |
| XceptionNet | 56.36% | 64.71% | 37.93% | 47.83% | 0.6001 |
| **LipSyncDetector** | **67.27%** | **62.79%** | **93.10%** | **75.00%** | **0.6837** |
| **Ensemble** | 50.00% | 0.00% | 0.00% | 0.00% | **0.8264** |

Full thesis report: `THESIS_REPORT.md` (70 pages, 35,000+ words)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 4 models on WildDeepfake (20 epochs, ~2 hours on GPU)
python src/train.py --mode all --data_dir data/wilddeepfake --epochs 20 --batch_size 4 --pretrained

# Evaluate all models
python src/evaluate.py --data_dir data/wilddeepfake --batch_size 4

# Generate all thesis images
python main.py --thesis

# Run tests
pytest tests/ -v
```

---

## Installation

### Prerequisites

- Python 3.12 (3.14 not supported)
- PyTorch 2.7+ with CUDA 11.8
- NVIDIA GPU (RTX series recommended)

### Setup

```bash
git clone <repository-url>
cd M.Eng-Thesis-Multimodal-Deepfake-Audio-Visual-Temporal-Framework

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt
```

---

## Dataset: WildDeepfake

The dataset is already included in `data/wilddeepfake/` (110 videos: 52 real, 58 fake).

To re-download from Hugging Face:

```python
from huggingface_hub import hf_hub_download
# Download from https://huggingface.co/datasets/xingjunm/WildDeepfake
# Extract tar files and convert face sequences to MP4 videos
```

**Custom dataset format:**
```
data_dir/
├── real/          # Real videos (label 0)
│   ├── video1.mp4
│   └── video2.mp4
└── fake/          # Fake videos (label 1)
    ├── video3.mp4
    └── video4.mp4
```

---

## Usage

### Training

```bash
# Train all 4 models at once
python src/train.py --mode all --data_dir data/wilddeepfake --epochs 20 --pretrained

# Train individual models
python src/train.py --mode multimodal   --data_dir data/wilddeepfake --epochs 20 --pretrained
python src/train.py --mode mesonet      --data_dir data/wilddeepfake --epochs 20
python src/train.py --mode xceptionnet  --data_dir data/wilddeepfake --epochs 20
python src/train.py --mode lipsync      --data_dir data/wilddeepfake --epochs 20

# Smoke test (no validation)
python src/train.py --mode multimodal --data_dir tests/test_data --epochs 2 --no_val
```

**Training arguments:**
- `--mode`: `multimodal`, `mesonet`, `xceptionnet`, `lipsync`, `simple`, `all`
- `--data_dir`: Dataset directory with `real/` and `fake/` subdirs
- `--epochs`: Training epochs (default: 3; use 20+ for real data)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--pretrained`: Use pretrained EfficientNet/Xception backbones
- `--patience`: Early stopping patience (default: 5)
- `--no_val`: Skip validation split (smoke test mode)
- `--seed`: Random seed (default: 42)

**Training features:**
- Video-level stratified 70/15/15 split (prevents data leakage)
- WeightedRandomSampler (class balance)
- Frame-consistent augmentation (flip, rotation, brightness/contrast)
- CosineAnnealingLR scheduler
- Early stopping with best-checkpoint saving
- Per-epoch CSV training log

### Evaluation

```bash
# Evaluate on real videos (loads all 4 trained models)
python src/evaluate.py --data_dir data/wilddeepfake --batch_size 4

# Full pipeline (train + eval + thesis images)
python main.py --all --data data/wilddeepfake --epochs 20

# Thesis images only
python main.py --thesis
```

### Ablation Studies

```bash
python src/run_ablations.py --data_dir data/wilddeepfake --epochs 10 --batch_size 4 --pretrained
```

Tests: modality ablation (V+A, V+T, A+T), fusion mechanism (attention vs concat), input representation (spatial vs spatial+FFT).

### Cross-Dataset Evaluation

```bash
python src/cross_dataset_eval.py \
    --model_path results/weights/trained_model_multimodal_best.pth \
    --eval_data_dir data/wilddeepfake data/dfdc \
    --mode multimodal
```

### Compression Robustness

```bash
# Requires FFmpeg
python src/compression_robustness.py \
    --model_path results/weights/trained_model_multimodal_best.pth \
    --data_dir data/wilddeepfake --crf_values 23 40 45
```

### Grad-CAM Explainability

```bash
python src/explainability.py \
    --model_path results/weights/trained_model_multimodal_best.pth \
    --video_path data/wilddeepfake/fake/fake_seq_000.mp4 \
    --frame_idx 0
```

Generates heatmaps for spatial, frequency, and audio streams.

### Ensemble Model

```bash
# Equal-weight ensemble (uses existing checkpoints)
python src/ensemble.py --data_dir data/wilddeepfake --mode eval --epochs 0

# Learn meta-classifier weights
python src/ensemble.py --data_dir data/wilddeepfake --mode train_meta --epochs 10
```

---

## Architecture

### MultimodalModel

```
Video frames ──→ VisualStream ──→ spatial features (1280d)
                 (EfficientNet-B0)   + frequency features (1280d)
                                     = 2560d per frame
                                     
                 ├── mean pool ──→ visual_features (2560d) ──→ ┐
                 └── TemporalStream ──→ temporal_features (2560d) ──→ ┤
                                                                      ├── AttentionFusion ──→ classifier ──→ prediction
Audio ──→ MFCC (40 coeffs) ──→ AudioStream ──→ audio_features (512d) ──→ ┘
         (Librosa)             (3-layer Conv2D)
```

**Configuration:**
```python
MultimodalModel(
    streams=('visual', 'audio', 'temporal'),  # any subset
    fusion_type='attention',                   # or 'concat'
    use_frequency=True,                        # FFT frequency branch
    use_mfcc=True,                             # MFCC audio features
    pretrained=True                            # ImageNet weights
)
```

### Baselines

- **MesoNet**: 4-layer CNN (8→16→16→16 filters)
- **XceptionNet**: Pretrained Xception with depthwise separable convolutions
- **LipSyncDetector**: Audio-visual CNN (Conv2D visual + Conv1D audio → FC fusion)

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Positive predictive value |
| Recall | TP/(TP+FN) | True positive rate |
| F1 | 2·P·R/(P+R) | Harmonic mean |
| AUC-ROC | ∫ROC(t)dt | Area under ROC curve (ranking quality) |

---

## Output Files

### Model Checkpoints (`results/weights/`)
- `trained_model_{mode}.pth` — final model
- `trained_model_{mode}_best.pth` — best checkpoint (by val loss)

### Thesis Images (`results/images/`)
- `performance_comparison.png` — grouped bar chart (all metrics)
- `radar_comparison.png` — multi-metric radar chart
- `confusion_matrices.png` — 2×2 grid (real predictions)
- `roc_curves.png` — ROC curves (real per-sample scores)
- `metrics_table.png` — formatted table with best values highlighted
- `model_ranking.png` — horizontal bars with rank annotations
- `correlation_heatmap.png` — metric correlation matrix
- `accuracy_comparison.png` — single-metric bar chart
- `auc_comparison.png` — single-metric bar chart
- `precision_recall_analysis.png` — scatter with F1 iso-contours
- `per_class_accuracy.png` — overall accuracy vs fake-class recall
- `summary_card.png` — headline figure (best model + training curve)
- `training_curves.png` — loss + LR schedule
- `ensemble_comparison.png` — ensemble vs individual models
- `gradcam/` — Grad-CAM heatmaps (spatial, frequency, audio)

### Documentation (`results/docs/`)
- `complete_performance_data.csv` — main results table
- `raw_predictions.json` — per-sample scores for ROC/confusion matrices
- `training_log_{mode}.csv` — per-epoch training logs
- `ensemble_results.csv` — ensemble comparison
- `result.md` — thesis results summary
- `report.md` — detailed analysis report

---

## File Structure

```
.
├── main.py                          # Entry point (train/eval/thesis pipeline)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── THESIS_REPORT.md                 # 70-page thesis report
├── NO.1 GRADUATE DEGREE DISSERTATION TITLE SELECTION REPORT.doc
├── NO.4 GRADUATE DEGREE DISSERTATION MID-TERM EVALUATION.docx
├── src/
│   ├── train.py                     # Training (all modes: multimodal/mesonet/xceptionnet/lipsync/all)
│   ├── evaluate.py                  # Evaluation (loads trained checkpoints, computes AUC)
│   ├── thesis_viz.py                # Thesis-quality visualization module
│   ├── run_ablations.py             # Ablation study runner
│   ├── cross_dataset_eval.py        # Cross-dataset generalization testing
│   ├── compression_robustness.py    # H.264 CRF compression robustness
│   ├── explainability.py            # Grad-CAM (spatial/frequency/audio)
│   ├── ensemble.py                  # Ensemble model (weighted avg + meta-classifier)
│   ├── media_processor.py           # Video/audio file processing
│   ├── models/
│   │   ├── multimodal_model.py      # MultimodalModel (configurable streams/fusion)
│   │   └── baselines.py             # MesoNet, XceptionNet, LipSyncDetector
│   └── data_loader/
│       ├── datasets.py              # DeepfakeDataset + split_dataset
│       ├── video_processor.py       # Frame extraction + face cropping
│       └── audio_processor.py       # MFCC extraction (Librosa)
├── data/
│   ├── wilddeepfake/                # 110 real deepfake videos (52 real, 58 fake)
│   │   ├── real/
│   │   └── fake/
│   ├── ff_plus/                     # FaceForensics++ (placeholder, 50 videos per compression)
│   │   ├── c23/                     # Compression level c23
│   │   │   ├── real/                # 25 real videos
│   │   │   └── fake/                # 25 fake videos
│   │   └── c40/                     # Compression level c40
│   │       ├── real/
│   │       └── fake/
│   └── dfdc/                        # DFDC (placeholder, 50 videos + metadata.json)
│       ├── real/
│       ├── fake/
│       └── metadata.json
├── results/
│   ├── docs/                        # CSVs, training logs, predictions, reports
│   ├── images/                      # 14 thesis images + 3 Grad-CAM
│   └── weights/                     # 8 model checkpoints (4 models × final + best)
├── tests/                           # 10 unit tests (all passing)
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_multimodal_model_light.py
│   ├── test_train_dataset.py
│   └── test_data/                   # dummy.mp4, dummy.wav
└── media/                           # Placeholder for media file evaluation
```

---

## Testing

```bash
pytest tests/ -v
```

Expected: **10 passed**

Tests cover: audio processor (MFCC + determinism), video processor, dataset loading, all model streams (visual/audio/temporal), multimodal forward pass, training dataset creation.

---

## Troubleshooting

### CUDA/GPU Issues
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```bash
python src/train.py --batch_size 2 --data_dir data/wilddeepfake
```

### FFmpeg (for compression robustness)
- Windows: `choco install ffmpeg`
- Mac: `brew install ffmpeg`
- Linux: `apt-get install ffmpeg`

---

## Citation

```bibtex
@mastersthesis{deepfake2026,
  title={Detecting Deepfake Video by A Multimodal Audio-Visual Framework with Temporal Inconsistencies},
  author={MD RAKIBUL ISLAM RAIHAN},
  year={2026},
  school={Northwestern Polytechnical University}
}
```

## License

MIT License — see LICENSE file for details.

## Citation

If you use this framework, please cite:

```bibtex
@mastersthesis{raihan2026deepfake,
  title  = {Detecting Deepfake Video by A Multimodal Audio-Visual Framework with Temporal Inconsistencies},
  author = {Raihan, Md Rakibul Islam},
  school = {Northwestern Polytechnical University},
  year   = {2026},
  type   = {M.Eng. Thesis}
}
```

A machine-readable `CITATION.cff` is also provided in this repository.

---

## Related Research

- **B.Eng. Thesis:** [Design and Implementation of a Distributed Confidential Query Protocol for Spark](https://github.com/theraihanrakibb/B.Eng-Thesis-Design-and-Implementation-of-a-Distributed-Confidential-Query-Protocol-for-Spark) — Apache Spark + CKKS homomorphic encryption.
- **NVFP4-DiT:** [4-bit Audio-Guided Video Diffusion Transformers](https://github.com/theraihanrakibb/NVFP4-DiT) (IEEE TNNLS, under review).
- **Profile & Portfolio:** [@theraihanrakibb](https://github.com/theraihanrakibb) · [Online Portfolio](https://theraihanrakibb.github.io/)
