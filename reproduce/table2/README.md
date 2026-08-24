# Table 2 — CalMS21 and CRIM13

This experiment trains and evaluates a Kinetics-400-pretrained Video Swin-T with an MLP head on the CalMS21 and CRIM13 mouse-behavior datasets.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

### CalMS21

```bash
%cd /content/drive/MyDrive/reproduce/table2

!python eval_calms21.py \
    --model_path /content/drive/MyDrive/reproduce/table2/CalMS21_123.pth \
    --test_video_dir /content/drive/MyDrive/traindata/video_dataset/test/resized/ \
    --test_label_dir /content/drive/MyDrive/traindata/label_dataset/test/ \
    --save_cm /content/drive/MyDrive/reproduce/table2/calms21_cm.png \
    --save_results /content/drive/MyDrive/reproduce/table2/calms21_results.json
```

### CRIM13

The command assumes CRIM13 is under `/content/drive/MyDrive/traindata/crim13/`. Change the three CRIM13 paths if your Drive layout differs.

```bash
%cd /content/drive/MyDrive/reproduce/table2

!python eval_crim13.py \
    --model_path /content/drive/MyDrive/reproduce/table2/CRIM13_123.pth \
    --test_video_dir /content/drive/MyDrive/traindata/crim13/video_dataset/test/ \
    --test_label_dir /content/drive/MyDrive/traindata/crim13/label_dataset/ \
    --save_cm /content/drive/MyDrive/reproduce/table2/crim13_cm.png \
    --save_results /content/drive/MyDrive/reproduce/table2/crim13_results.json
```

## Training Code

### CalMS21

```bash
%cd /content/drive/MyDrive/reproduce/table2

!python train_calms21.py \
    --train_video_dir /content/drive/MyDrive/traindata/video_dataset/train/resized/ \
    --train_label_dir /content/drive/MyDrive/traindata/label_dataset/train/ \
    --seed 123 \
    --val_split_seed 1337 \
    --model_save_dir /content/drive/MyDrive/reproduce/table2/checkpoints/calms21
```

### CRIM13

```bash
%cd /content/drive/MyDrive/reproduce/table2

!python train_crim13.py \
    --train_video_dir /content/drive/MyDrive/traindata/crim13/video_dataset/train/ \
    --train_label_dir /content/drive/MyDrive/traindata/crim13/label_dataset/ \
    --seed 123 \
    --val_split_seed 1337 \
    --model_save_dir /content/drive/MyDrive/reproduce/table2/checkpoints/crim13
```

To repeat the experiment, rerun each training command with `--seed 1337` and `--seed 2025`; do not change `--val_split_seed`.

## Seed Design

- `--seed` controls model initialization, augmentation and data-loader shuffle. Change it across repeated runs (`123`, `1337`, `2025`).
- `--val_split_seed` controls which videos enter training and validation. Keep it fixed at `1337` so the dataset split stays identical across model seeds.

## Arguments

### Training

| Argument | CalMS21 default | CRIM13 default | Description |
|---|---:|---:|---|
| `--train_video_dir` | `data/calms21/videos/train` | `data/crim13/videos/train` | Training-video directory |
| `--train_label_dir` | `data/calms21/labels/train` | `data/crim13/labels` | Training-label directory |
| `--seed` | `123` | `123` | General random seed |
| `--val_split_seed` | `1337` | `1337` | Fixed validation-split seed |
| `--validation_ratio` | `0.15` | `0.15` | Fraction of videos used for validation |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation steps |
| `--num_epochs` | `5` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | `2e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | `16` / `4` | Temporal window and stride |
| `--use_class_weights` | off | off | Enable inverse-frequency weights |
| `--model_save_dir` | `checkpoints/calms21` | `checkpoints/crim13` | Checkpoint/log directory |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | `.pth` checkpoint |
| `--test_video_dir` | dataset-specific | Test-video directory |
| `--test_label_dir` | dataset-specific | Test-label directory |
| `--batch_size` | `1` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Confusion-matrix PNG path |
| `--save_results` | none | Metrics JSON path |

## Output Naming Convention

```text
{dataset}_seed{general}_vsplit{val_split}_ep{N}_f1_{val}_map_{val}.pth
```

Examples: `calms21_seed123_vsplit1337_ep3_f1_0.8388_map_0.9005.pth` and `crim13_seed2025_vsplit1337_ep5_f1_0.4521_map_0.3876.pth`.

## Dataset Differences

| | CalMS21 | CRIM13 |
|---|---|---|
| Classes | 4: Attack, Investigation, Mount, Other | 13: category1–category13 |
| Label filename | `{video}.csv` | `{video}_one_hot.csv` |
| Split strategy | Stratified by rare Attack behavior | Random split with all classes represented in validation |
| Default learning rate | `3.8e-5` | `2e-5` |
