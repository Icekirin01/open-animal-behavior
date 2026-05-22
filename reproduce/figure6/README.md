# Cross-Domain Swin3D — Zero-Shot & Fine-tuning

Evaluate or fine-tune a Video Swin-T model on a new dataset with different behavior classes.
Supports zero-shot evaluation, Kinetics-400 fine-tuning, and custom pretrained fine-tuning with data efficiency experiments.

## Requirements

```bash
pip install torch torchvision decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Scripts

| Script | Description |
|---|---|
| `zeroshot_eval.py` | Load a pretrained model → evaluate directly on test set (no training) |
| `crossdomain_train.py` | Fine-tune on a new dataset (K400 or custom pretrained init) → auto-evaluate best epoch on test set |
| `crossdomain_eval.py` | Evaluate a trained checkpoint on a test set (standalone, no training) |

## Arguments

### zeroshot_eval.py

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to pretrained `.pth` checkpoint |
| `--test_video_dir` | *(required)* | Test video directory |
| `--test_label_dir` | *(required)* | Test label directory |
| `--output_dir` | `results/zeroshot` | Output directory for results |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | off | Save confusion matrix images |

### crossdomain_train.py

| Argument | Default | Description |
|---|---|---|
| `--pretrained_model_path` | *(none)* | Path to custom pretrained `.pth`. If omitted, uses **Kinetics-400** weights. |
| `--seed` | `2025` | General seed (model init, augmentation, shuffle) |
| `--val_split_seed` | `1337` | Val split seed (**FIXED** across experiments) |
| `--video_split_seed` | `42` | Seed for subsampling training videos |
| `--train_video_dir` | *(required)* | Training video directory |
| `--train_label_dir` | *(required)* | Training label directory |
| `--model_save_dir` | `checkpoints/crossdomain` | Checkpoint directory |
| `--train_data_ratio` | `1.0` | Fraction of training videos to use (1.0=all, 0.5=50%) |
| `--validation_ratio` | `0.2` | Fraction for validation |
| `--batch_size` | `8` | Batch size |
| `--accumulation_steps` | `2` | Gradient accumulation steps |
| `--num_epochs` | `5` | Number of epochs |
| `--base_lr` | `3.8e-5` | Learning rate (use `~1e-5` for custom pretrained, `~3.8e-5` for K400) |
| `--use_class_weights` | off | Enable inverse-frequency class weights |

### crossdomain_eval.py

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to `.pth` checkpoint |
| `--test_video_dir` | *(required)* | Test video directory |
| `--test_label_dir` | *(required)* | Test label directory |
| `--output_dir` | *(model dir)* | Output directory |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | off | Save confusion matrix images |
| `--save_results` | *(none)* | Path to save JSON results |

## Seed Design

- **`--seed`** (general): controls model init, augmentation, shuffle. **Change** across runs.
- **`--val_split_seed`** (val split): controls train/val assignment. **Keep FIXED** (default 1337).
- **`--video_split_seed`** (data subsampling): controls which videos are selected when `--train_data_ratio < 1.0`.

## Usage

### Zero-Shot Evaluation (no training)

```bash
python zeroshot_eval.py \
    --model_path checkpoints/fulltrain_all3folds_ep5.pth \
    --test_video_dir data/new/videos/test \
    --test_label_dir data/new/labels/test \
    --save_cm
```

### Fine-tune from Kinetics-400 (no custom model)

```bash
# Full data
python crossdomain_train.py --seed 123 \
    --train_video_dir data/new/videos/train \
    --train_label_dir data/new/labels/train \
    --base_lr 3.8e-5

# 50% data
python crossdomain_train.py --seed 123 \
    --train_data_ratio 0.5 --video_split_seed 42 \
    --train_video_dir data/new/videos/train \
    --train_label_dir data/new/labels/train \
    --base_lr 3.8e-5
```

### Fine-tune from Custom Pretrained

```bash
# Full data
python crossdomain_train.py --seed 123 \
    --pretrained_model_path checkpoints/fulltrain_all3folds_ep5.pth \
    --train_video_dir data/new/videos/train \
    --train_label_dir data/new/labels/train \
    --base_lr 1e-5

# 50% data
python crossdomain_train.py --seed 123 \
    --pretrained_model_path checkpoints/fulltrain_all3folds_ep5.pth \
    --train_data_ratio 0.5 --video_split_seed 42 \
    --train_video_dir data/new/videos/train \
    --train_label_dir data/new/labels/train \
    --base_lr 1e-5
```

### Evaluation

```bash
python crossdomain_eval.py \
    --model_path checkpoints/crossdomain/crossdomain_ratio100_seed123_ep3_f1_0.75_map_0.80.pth \
    --test_video_dir data/new/videos/test \
    --test_label_dir data/new/labels/test \
    --save_cm --save_results results.json
```

## Output Naming Convention

```
{mode}_{ratio}{vseed}_seed{seed}_ep{N}_f1_{val}_map_{val}.pth
```

Examples:
- `kinetics400_ratio100_seed123_ep3_f1_0.7500_map_0.8000.pth` — K400 init, 100% data
- `crossdomain_ratio50_vseed42_seed123_ep5_f1_0.6800_map_0.7200.pth` — Custom pretrained, 50% data

## Behavior Classes

5 classes selected from 7 in the label CSVs:

| Selected | Original Column | Notes |
|---|---|---|
| `Aggression` | 0 | Rare behavior |
| `Investigation` | 1 | |
| `Allo-groom` | 2 | Rare behavior |
| `Standing` | 4 | |
| `Other` | 6 | |

Excluded: `Self-groom` (col 3), `Chasing` (col 5) — frames with these labels are skipped.

## Model Init Comparison

| Mode | `--pretrained_model_path` | Backbone init | MLP head | Typical LR |
|---|---|---|---|---|
| K400 | *(omitted)* | Kinetics-400 (torchvision) | Random | `3.8e-5` |
| Custom | checkpoint path | Custom pretrained weights | From checkpoint | `1e-5` |
| Zero-shot | checkpoint path | From checkpoint | From checkpoint | N/A |
