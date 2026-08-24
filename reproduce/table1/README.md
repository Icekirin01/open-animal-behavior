# Table 1 — Mouse Behavior Classification

This experiment compares Kinetics-400-pretrained Video Swin-T and TimeSformer backbones on five mouse social behaviors using 3-fold cross-validation. Both models use the same MLP classification head.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

The commands below assume the repository is stored in `/content/drive/MyDrive/reproduce`. Put the downloaded checkpoints in `table1/` and rename them as shown, or change only `--model_path` to the real filename.

### TimeSformer — Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/table1

!python eval_timesformer.py \
    --model_path /content/drive/MyDrive/reproduce/table1/table1_timesformer_fold1.pth \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --save_cm /content/drive/MyDrive/reproduce/table1/timesformer_fold1_cm.png
```

### Video Swin-T — Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/table1

!python eval_swin3d.py \
    --model_path /content/drive/MyDrive/reproduce/table1/table1_swin3d_fold1.pth \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --save_cm /content/drive/MyDrive/reproduce/table1/swin3d_fold1_cm.png
```

For Fold 2 or Fold 3, use that fold's checkpoint and change `--test_folds` to `2` or `3`.

## Training Code

These are complete Fold 1 examples. The checkpoints and training log are written directly to Google Drive.

### TimeSformer — Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/table1

!python train_timesformer.py \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --train_folds 3 2 \
    --seed 2025 \
    --model_save_dir /content/drive/MyDrive/reproduce/table1/checkpoints/timesformer
```

### Video Swin-T — Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/table1

!python train_swin3d.py \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --train_folds 3 2 \
    --seed 2025 \
    --model_save_dir /content/drive/MyDrive/reproduce/table1/checkpoints/swin3d
```

Use these fold pairs for the other runs:

| Run | `--test_folds` | `--train_folds` |
|---|---:|---|
| Fold 1 | `1` | `3 2` |
| Fold 2 | `2` | `1 3` |
| Fold 3 | `3` | `1 2` |

## Seed Design

`--seed` controls model initialization, augmentation and data-loader randomness. Use the same seed when comparing Swin-T with TimeSformer. To report stability across runs, repeat all three folds with additional seeds while keeping the fold assignments unchanged.

GPU kernels, mixed precision and multi-worker loading can still cause small run-to-run differences even with the same seed. Exact paper results therefore require the corresponding pretrained checkpoint.

## Arguments

### Training

| Argument | Swin-T default | TimeSformer default | Description |
|---|---:|---:|---|
| `--base_video_dir` | `data/videos` | `data/videos` | Root containing fold folders `1/`, `2/`, `3/` |
| `--label_dir` | `data/labels` | `data/labels` | One-hot frame-label CSV directory |
| `--test_folds` | `1` | `1` | Held-out fold(s) |
| `--train_folds` | `3 2` | `3 2` | Training fold(s) |
| `--seed` | `2025` | `2025` | General random seed |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation steps |
| `--num_epochs` | `5` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | `3e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | `16` / `4` | Temporal window and stride |
| `--model_save_dir` | `checkpoints/swin3d` | `checkpoints/timesformer` | Output directory |
| `--use_class_weights` | off | off | Enable inverse-frequency weights |
| `--hf_model` | — | `facebook/timesformer-base-finetuned-k400` | Hugging Face backbone |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | `.pth` checkpoint to load |
| `--base_video_dir` | `data/videos` | Root containing the test fold |
| `--label_dir` | `data/labels` | Label CSV directory |
| `--test_folds` | `1` | Fold(s) to evaluate |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Optional temporal smoothing |
| `--save_cm` | none | Confusion-matrix image path |

## Behavior Classes

The default classes are `Aggression`, `Investigation`, `Allo-groom`, `Standing` and `Other`. `Self-groom` and `Chasing` exist in the seven-column one-hot CSV files but are excluded by default.

## Data and Trained Models

Videos and pretrained checkpoints are available upon request. Each label CSV contains one row per video frame and one column per behavior.

## Method Details

- A 16-frame window moves through each video with stride 4. Training uses the majority label inside each window.
- Evaluation averages the probabilities of all windows covering each frame, then optionally applies temporal smoothing.
- Both backbones receive eight frames. TimeSformer uniformly samples them from the 16-frame window.
- Training augmentation uses random Gaussian blur and temporal dropout.
- Evaluation reports per-class F1/AP, macro F1, mAP and a confusion matrix.
