# Figure 6 — Cross-Domain Swin3D

This experiment evaluates a Video Swin-T checkpoint on a different mouse dataset and compares zero-shot transfer with Kinetics-400 or custom-checkpoint fine-tuning.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

### Evaluate the provided Figure 6 checkpoint

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure6

!python crossdomain_eval.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/kinetics400_ratio0.5_42.pth \
    --test_video_dir /content/drive/MyDrive/xxx/new_domain/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/new_domain/labels/test/ \
    --output_dir /content/drive/MyDrive/xxx/results/crossdomain \
    --save_cm \
    --save_results /content/drive/MyDrive/xxx/results/crossdomain/results.json
```

### Zero-shot evaluation without additional training

Replace `mouse_source_pretrained.pth` with the source-domain checkpoint you want to transfer.

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure6

!python zeroshot_eval.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/mouse_source_pretrained.pth \
    --test_video_dir /content/drive/MyDrive/xxx/new_domain/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/new_domain/labels/test/ \
    --output_dir /content/drive/MyDrive/xxx/results/zeroshot \
    --seed 2025 \
    --save_cm
```

## Training Code

The paths below are Google Drive examples. Replace `xxx` with the folders you created for the new-domain videos, labels and checkpoints.

### Fine-tune from Kinetics-400 using 50% of the videos

Omitting `--pretrained_model_path` makes the script initialize from torchvision Kinetics-400 weights.

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure6

!python crossdomain_train.py \
    --train_video_dir /content/drive/MyDrive/xxx/new_domain/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/new_domain/labels/train/ \
    --train_data_ratio 0.5 \
    --video_split_seed 42 \
    --seed 2025 \
    --val_split_seed 1337 \
    --base_lr 3.8e-5 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/figure6/kinetics400_ratio05
```

### Fine-tune from a custom source-domain checkpoint

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure6

!python crossdomain_train.py \
    --pretrained_model_path /content/drive/MyDrive/xxx/checkpoints/mouse_source_pretrained.pth \
    --train_video_dir /content/drive/MyDrive/xxx/new_domain/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/new_domain/labels/train/ \
    --train_data_ratio 0.5 \
    --video_split_seed 42 \
    --seed 2025 \
    --val_split_seed 1337 \
    --base_lr 1e-5 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/figure6/custom_ratio05
```

## Seed Design

- `--seed` controls model initialization, augmentation and shuffle. Change it for repeated training runs.
- `--val_split_seed` controls train/validation assignment. Keep it fixed at `1337` for fair comparison.
- `--video_split_seed` controls which videos are retained when `--train_data_ratio < 1.0`; use the same value when comparing initialization methods.
- `zeroshot_eval.py` also accepts `--seed`, but performs no training.

## Arguments

### `crossdomain_train.py`

| Argument | Default | Description |
|---|---:|---|
| `--pretrained_model_path` | none | Location of a custom `.pth` checkpoint on Google Drive; omit for Kinetics-400 initialization |
| `--train_video_dir` | required | Google Drive folder containing new-domain training videos |
| `--train_label_dir` | required | Google Drive folder containing new-domain training-label CSV files |
| `--train_data_ratio` | `1.0` | Fraction of training videos retained |
| `--video_split_seed` | `42` | Video-subsampling seed |
| `--seed` | `2025` | General random seed |
| `--val_split_seed` | `1337` | Fixed validation-split seed |
| `--validation_ratio` | `0.2` | Validation fraction |
| `--batch_size` | `8` | Batch size |
| `--accumulation_steps` | `2` | Gradient accumulation |
| `--num_epochs` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | Learning rate; typically lower for a custom checkpoint |
| `--model_save_dir` | `checkpoints/crossdomain` | Google Drive folder used to save checkpoints and logs |

### Evaluation scripts

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Location of the `.pth` checkpoint on Google Drive |
| `--test_video_dir` | required | Google Drive folder containing test videos |
| `--test_label_dir` | required | Google Drive folder containing test-label CSV files |
| `--output_dir` | zero-shot: `results/zeroshot`; cross-domain: model folder | Google Drive folder used to save evaluation outputs |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | off | Save confusion-matrix images |
| `--save_results` | none | Google Drive path for the JSON file; `crossdomain_eval.py` only |

### Google Drive Path Examples

```text
--model_path /content/drive/MyDrive/xxx/checkpoints/model.pth
--pretrained_model_path /content/drive/MyDrive/xxx/checkpoints/source_model.pth
--train_video_dir /content/drive/MyDrive/xxx/new_domain/videos/train
--train_label_dir /content/drive/MyDrive/xxx/new_domain/labels/train
--test_video_dir /content/drive/MyDrive/xxx/new_domain/videos/test
--test_label_dir /content/drive/MyDrive/xxx/new_domain/labels/test
--output_dir /content/drive/MyDrive/xxx/results
```

## Scripts

| Script | Purpose |
|---|---|
| `zeroshot_eval.py` | Evaluate a source checkpoint directly, with no training |
| `crossdomain_train.py` | Fine-tune from Kinetics-400 or a custom checkpoint |
| `crossdomain_eval.py` | Evaluate a trained cross-domain checkpoint |

## Output Naming Convention

```text
{mode}_{ratio}{vseed}_seed{seed}_ep{N}_f1_{val}_map_{val}.pth
```

Examples: `kinetics400_ratio100_seed123_ep3_f1_0.7500_map_0.8000.pth` and `crossdomain_ratio50_vseed42_seed123_ep5_f1_0.6800_map_0.7200.pth`.

## Behavior Classes

Five of the seven CSV columns are used: `Aggression` (0), `Investigation` (1), `Allo-groom` (2), `Standing` (4) and `Other` (6). Frames labeled `Self-groom` (3) or `Chasing` (5) are skipped.

## Model Initialization Comparison

| Mode | `--pretrained_model_path` | Backbone/head source | Typical LR |
|---|---|---|---:|
| K400 fine-tune | omitted | Kinetics-400 backbone, random MLP head | `3.8e-5` |
| Custom fine-tune | checkpoint path | Backbone and head from checkpoint | about `1e-5` |
| Zero-shot | checkpoint path | Backbone and head from checkpoint; no updates | N/A |
