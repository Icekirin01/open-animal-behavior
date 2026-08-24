# Figure 4 — Mouse Video-Level Data Efficiency

This experiment measures how Video Swin-T and TimeSformer performance changes when only a fixed fraction of the mouse training videos is used, while retaining the same 3-fold test protocol.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

### TimeSformer — 75% data, video seed 42, Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/figure4

!python eval_timesformer_ratio.py \
    --model_path /content/drive/MyDrive/reproduce/figure4/figure4_vitb_fold1_0.75_42.pth \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --save_cm /content/drive/MyDrive/reproduce/figure4/timesformer_ratio075_fold1_cm.png
```

### Video Swin-T — 75% data, video seed 42, Fold 1

Put the Swin-T checkpoint in the shown location, or replace its filename with the real one.

```bash
%cd /content/drive/MyDrive/reproduce/figure4

!python eval_swin3d_ratio.py \
    --model_path /content/drive/MyDrive/reproduce/figure4/figure4_swin3d_fold1_0.75_42.pth \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --save_cm /content/drive/MyDrive/reproduce/figure4/swin3d_ratio075_fold1_cm.png
```

Evaluation does not need `--train_data_ratio`; that value is already represented by the checkpoint being loaded.

## Training Code

### TimeSformer — 75% data, video seed 42, Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/figure4

!python train_timesformer_ratio.py \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --train_folds 3 2 \
    --train_data_ratio 0.75 \
    --video_split_seed 42 \
    --seed 2025 \
    --model_save_dir /content/drive/MyDrive/reproduce/figure4/checkpoints/timesformer_ratio
```

### Video Swin-T — 75% data, video seed 42, Fold 1

```bash
%cd /content/drive/MyDrive/reproduce/figure4

!python train_swin3d_ratio.py \
    --base_video_dir /content/drive/MyDrive/traindata/kuo/video/train/crop/resized \
    --label_dir /content/drive/MyDrive/traindata/kuo/label_interval \
    --test_folds 1 \
    --train_folds 3 2 \
    --train_data_ratio 0.75 \
    --video_split_seed 42 \
    --seed 2025 \
    --model_save_dir /content/drive/MyDrive/reproduce/figure4/checkpoints/swin3d_ratio
```

For Fold 2 use `--test_folds 2 --train_folds 1 3`; for Fold 3 use `--test_folds 3 --train_folds 1 2`. Repeat the full set for every data ratio and video-selection seed used in the figure.

## Seed Design

- `--seed` controls model initialization, augmentation and data-loader randomness. Keep it equal when comparing backbones.
- `--video_split_seed` controls which training videos are retained when `--train_data_ratio < 1.0`. Use the experiment seeds `42`, `123` and `999`.
- The fold folders define train/test membership and must not change between ratios.

Even with fixed seeds, GPU nondeterminism, mixed precision and multi-worker loading can cause small metric differences. Use the paper checkpoint when exact reproduction is required.

## Arguments

### Training

| Argument | Default | Description |
|---|---:|---|
| `--base_video_dir` | `data/videos` | Root containing fold folders |
| `--label_dir` | `data/labels` | One-hot frame-label CSV directory |
| `--test_folds` | `1` | Held-out fold(s) |
| `--train_folds` | `3 2` | Training fold(s) |
| `--train_data_ratio` | `1.0` | Fraction of training videos retained |
| `--video_split_seed` | `42` | Video-subsampling seed |
| `--seed` | `2025` | General random seed |
| `--batch_size` | `8` | Batch size |
| `--accumulation_steps` | `2` | Gradient accumulation |
| `--num_epochs` | `5` | Training epochs |
| `--base_lr` | Swin-T: `3.8e-5`; TimeSformer: `3e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--model_save_dir` | model-specific | Checkpoint/log directory |
| `--hf_model` | `facebook/timesformer-base-finetuned-k400` | TimeSformer only |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Ratio-specific `.pth` checkpoint |
| `--base_video_dir` | `data/videos` | Root containing the held-out fold |
| `--label_dir` | `data/labels` | Label CSV directory |
| `--test_folds` | `1` | Fold(s) to evaluate |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Confusion-matrix image path |

## 3-Fold Cross-Validation

| Run | `--test_folds` | `--train_folds` |
|---|---:|---|
| Fold 1 | `1` | `3 2` |
| Fold 2 | `2` | `1 3` |
| Fold 3 | `3` | `1 2` |

## Output Naming Convention

```text
{model}_train_{folds}_val_{folds}_ratio{pct}_vseed{seed}_ep{N}_f1_{val}_map_{val}.pth
```

Example: `swin3d_train_3_2_val_1_ratio75_vseed42_ep5_f1_0.6123_map_0.7456.pth`. At 100% data, the `_vseed` suffix is omitted.
