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
%cd /content/drive/MyDrive/xxx/reproduce/table2

!python eval_calms21.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/CalMS21_123.pth \
    --test_video_dir /content/drive/MyDrive/xxx/calms21/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/calms21/labels/test/ \
    --save_cm /content/drive/MyDrive/xxx/results/calms21_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/calms21_results.json
```

### CRIM13

The paths are examples only. Replace `xxx` and the following folder names with the locations you created in Google Drive.

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table2

!python eval_crim13.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/CRIM13_123.pth \
    --test_video_dir /content/drive/MyDrive/xxx/crim13/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/crim13/labels/ \
    --save_cm /content/drive/MyDrive/xxx/results/crim13_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/crim13_results.json
```

## Training Code

### CalMS21

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table2

!python train_calms21.py \
    --train_video_dir /content/drive/MyDrive/xxx/calms21/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/calms21/labels/train/ \
    --seed 123 \
    --val_split_seed 1337 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/table2/calms21
```

### CRIM13

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table2

!python train_crim13.py \
    --train_video_dir /content/drive/MyDrive/xxx/crim13/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/crim13/labels/ \
    --seed 123 \
    --val_split_seed 1337 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/table2/crim13
```

To repeat the experiment, rerun each training command with `--seed 1337` and `--seed 2025`; do not change `--val_split_seed`.

## Seed Design

- `--seed` controls model initialization, augmentation and data-loader shuffle. Change it across repeated runs (`123`, `1337`, `2025`).
- `--val_split_seed` controls which videos enter training and validation. Keep it fixed at `1337` so the dataset split stays identical across model seeds.

## Arguments

### Training

| Argument | CalMS21 default | CRIM13 default | Description |
|---|---:|---:|---|
| `--train_video_dir` | `data/calms21/videos/train` | `data/crim13/videos/train` | Google Drive folder containing training videos |
| `--train_label_dir` | `data/calms21/labels/train` | `data/crim13/labels` | Google Drive folder containing training-label CSV files |
| `--seed` | `123` | `123` | General random seed |
| `--val_split_seed` | `1337` | `1337` | Fixed validation-split seed |
| `--validation_ratio` | `0.15` | `0.15` | Fraction of videos used for validation |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation steps |
| `--num_epochs` | `5` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | `2e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | `16` / `4` | Temporal window and stride |
| `--use_class_weights` | off | off | Enable inverse-frequency weights |
| `--model_save_dir` | `checkpoints/calms21` | `checkpoints/crim13` | Google Drive folder used to save checkpoints and logs |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Location of the `.pth` checkpoint on Google Drive |
| `--test_video_dir` | dataset-specific | Google Drive folder containing test videos |
| `--test_label_dir` | dataset-specific | Google Drive folder containing test-label CSV files |
| `--batch_size` | `1` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Google Drive path where the confusion-matrix PNG is saved |
| `--save_results` | none | Google Drive path where the metrics JSON is saved |

### Google Drive Path Examples

```text
--model_path /content/drive/MyDrive/xxx/checkpoints/model.pth
--train_video_dir /content/drive/MyDrive/xxx/videos/train
--train_label_dir /content/drive/MyDrive/xxx/labels/train
--test_video_dir /content/drive/MyDrive/xxx/videos/test
--test_label_dir /content/drive/MyDrive/xxx/labels/test
--model_save_dir /content/drive/MyDrive/xxx/outputs
```

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
