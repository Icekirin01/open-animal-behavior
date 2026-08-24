# Table 3 — Fly Copulation Classification

This experiment compares Kinetics-400-pretrained Video Swin-T and TimeSformer models on four fly-copulation behavior classes using a stratified video-level validation split.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

The paths below are Google Drive examples. Replace `xxx` with the folder structure you created in your own Drive.

### TimeSformer

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table3

!python eval_fly_timesformer.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/table3_timesformer.pth \
    --test_video_dir /content/drive/MyDrive/xxx/fly/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/fly/labels/test/ \
    --save_cm /content/drive/MyDrive/xxx/results/timesformer_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/timesformer_results.json
```

### Video Swin-T

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table3

!python eval_fly_swin3d.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/table3_swin3d.pth \
    --test_video_dir /content/drive/MyDrive/xxx/fly/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/fly/labels/test/ \
    --save_cm /content/drive/MyDrive/xxx/results/swin3d_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/swin3d_results.json
```

## Training Code

### TimeSformer

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table3

!python train_fly_timesformer.py \
    --train_video_dir /content/drive/MyDrive/xxx/fly/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/fly/labels/train/ \
    --seed 1337 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/table3/fly_timesformer
```

### Video Swin-T

```bash
%cd /content/drive/MyDrive/xxx/reproduce/table3

!python train_fly_swin3d.py \
    --train_video_dir /content/drive/MyDrive/xxx/fly/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/fly/labels/train/ \
    --seed 123 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/table3/fly_swin3d
```

## Seed Design

| Seed | Default | Purpose |
|---|---|---|
| `--seed` | Swin-T: `123`; TimeSformer: `1337` | Model initialization, augmentation and shuffle |
| `--val_split_seed` | `123` | Validation split; keep fixed across comparisons |

Change `--seed` for repeated runs, but keep `--val_split_seed 123` so every run sees the same train/validation videos.

## Arguments

### Training

| Argument | Swin-T default | TimeSformer default | Description |
|---|---:|---:|---|
| `--train_video_dir` | `data/fly/videos/train` | same | Google Drive folder containing training videos |
| `--train_label_dir` | `data/fly/labels/train` | same | Google Drive folder containing training-label CSV files |
| `--seed` | `123` | `1337` | General random seed |
| `--val_split_seed` | `123` | `123` | Fixed validation-split seed |
| `--validation_ratio` | `0.15` | `0.15` | Validation fraction |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation |
| `--num_epochs` | `5` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | `3e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | `16` / `4` | Temporal window and stride |
| `--model_save_dir` | `checkpoints/fly_swin3d` | `checkpoints/fly_timesformer` | Google Drive folder used to save checkpoints and logs |
| `--use_class_weights` | off | off | Enable inverse-frequency weights |
| `--hf_model` | — | `facebook/timesformer-base-finetuned-k400` | Hugging Face backbone |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Location of the `.pth` checkpoint on Google Drive |
| `--test_video_dir` | `data/fly/videos/test` | Google Drive folder containing test videos |
| `--test_label_dir` | `data/fly/labels/test` | Google Drive folder containing test-label CSV files |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Google Drive path where the confusion-matrix PNG is saved |
| `--save_results` | none | Google Drive path where the metrics JSON is saved |

### Google Drive Path Examples

```text
--model_path /content/drive/MyDrive/xxx/checkpoints/model.pth
--train_video_dir /content/drive/MyDrive/xxx/fly/videos/train
--train_label_dir /content/drive/MyDrive/xxx/fly/labels/train
--test_video_dir /content/drive/MyDrive/xxx/fly/videos/test
--test_label_dir /content/drive/MyDrive/xxx/fly/labels/test
--model_save_dir /content/drive/MyDrive/xxx/outputs
```

## Behavior Remapping

| Original class | Selected class |
|---|---|
| `wing_extension` | `wing_extension` |
| `circle` | `circle` |
| `copul_attempt` | `others` |
| `copulation` | `copulation` |
| `others` | `others` |

## Output Naming Convention

```text
fly_{model}_seed{X}_vsplit{Y}_ep{N}_f1_{val}_map_{val}.pth
```

Examples: `fly_swin3d_seed123_vsplit123_ep3_f1_0.7234_map_0.8123.pth` and `fly_timesformer_seed1337_vsplit123_ep5_f1_0.6987_map_0.7890.pth`.

## Difference from Figure 5 Ratio Scripts

Table 3 always trains on all selected training videos and chooses the best checkpoint by validation macro F1 over all classes. Figure 5 adds `--train_data_ratio` and `--video_split_seed`, and selects by macro F1 excluding `others`.
