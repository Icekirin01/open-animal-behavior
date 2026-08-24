# Figure 5 — Fly Video-Level Data Efficiency

This experiment measures Video Swin-T and TimeSformer performance at different fractions of the fly training videos. Video selection is stratified before the normal train/validation split.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

Put the Figure 5 checkpoints in `/content/drive/MyDrive/reproduce/figure5/` and rename them as shown, or change only the checkpoint filename.

### TimeSformer — 75% data, video seed 42

```bash
%cd /content/drive/MyDrive/reproduce/figure5

!python eval_fly_timesformer_ratio.py \
    --model_path /content/drive/MyDrive/reproduce/figure5/figure5_timesformer_ratio0.75_42.pth \
    --test_video_dir "/content/drive/My Drive/IMG test IMG/video_dataset/test/" \
    --test_label_dir "/content/drive/My Drive/IMG test IMG/label_dataset/test/" \
    --save_cm /content/drive/MyDrive/reproduce/figure5/timesformer_ratio075_cm.png \
    --save_results /content/drive/MyDrive/reproduce/figure5/timesformer_ratio075_results.json
```

### Video Swin-T — 75% data, video seed 42

```bash
%cd /content/drive/MyDrive/reproduce/figure5

!python eval_fly_swin3d_ratio.py \
    --model_path /content/drive/MyDrive/reproduce/figure5/figure5_swin3d_ratio0.75_42.pth \
    --test_video_dir "/content/drive/My Drive/IMG test IMG/video_dataset/test/" \
    --test_label_dir "/content/drive/My Drive/IMG test IMG/label_dataset/test/" \
    --save_cm /content/drive/MyDrive/reproduce/figure5/swin3d_ratio075_cm.png \
    --save_results /content/drive/MyDrive/reproduce/figure5/swin3d_ratio075_results.json
```

Evaluation does not accept a ratio argument; the loaded checkpoint determines which training ratio is being evaluated.

## Training Code

### TimeSformer — 75% data, video seed 42

```bash
%cd /content/drive/MyDrive/reproduce/figure5

!python train_fly_timesformer_ratio.py \
    --train_video_dir "/content/drive/My Drive/IMG test IMG/video_dataset/train/" \
    --train_label_dir "/content/drive/My Drive/IMG test IMG/label_dataset/train/" \
    --train_data_ratio 0.75 \
    --video_split_seed 42 \
    --seed 2025 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/reproduce/figure5/checkpoints/fly_timesformer_ratio
```

### Video Swin-T — 75% data, video seed 42

```bash
%cd /content/drive/MyDrive/reproduce/figure5

!python train_fly_swin3d_ratio.py \
    --train_video_dir "/content/drive/My Drive/IMG test IMG/video_dataset/train/" \
    --train_label_dir "/content/drive/My Drive/IMG test IMG/label_dataset/train/" \
    --train_data_ratio 0.75 \
    --video_split_seed 42 \
    --seed 2025 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/reproduce/figure5/checkpoints/fly_swin3d_ratio
```

Change `--train_data_ratio` and `--video_split_seed` together for the other Figure 5 conditions. The best checkpoint is selected by validation macro F1 excluding `others`.

## Seed Design

- `--seed` controls model initialization, augmentation and shuffle; the default is `2025` for both backbones.
- `--val_split_seed` controls the train/validation split after video subsampling. Keep it fixed at `123`.
- `--video_split_seed` controls which videos are retained at partial ratios. Change this across subsampling repeats, for example `42`, `123` and `999`.

## Arguments

### Training

| Argument | Default | Description |
|---|---:|---|
| `--train_video_dir` | `data/fly/videos/train` | Training-video directory |
| `--train_label_dir` | `data/fly/labels/train` | Training-label directory |
| `--train_data_ratio` | `1.0` | Fraction of training videos retained |
| `--video_split_seed` | `42` | Video-subsampling seed |
| `--seed` | `2025` | General random seed |
| `--val_split_seed` | `123` | Fixed validation-split seed |
| `--validation_ratio` | `0.15` | Validation fraction |
| `--batch_size` | `8` | Batch size |
| `--accumulation_steps` | `2` | Gradient accumulation |
| `--num_epochs` | `5` | Training epochs |
| `--base_lr` | `3.8e-5` | Learning rate |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--model_save_dir` | model-specific | Checkpoint/log directory |
| `--use_class_weights` | off | Enable inverse-frequency weights |

### Evaluation

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Ratio-specific `.pth` checkpoint |
| `--test_video_dir` | `data/fly/videos/test` | Test-video directory |
| `--test_label_dir` | `data/fly/labels/test` | Test-label directory |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Confusion-matrix PNG path |
| `--save_results` | none | Metrics JSON path |

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
fly_{model}_{ratio_tag}{_vseed}_seed{seed}_ep{N}_f1no_{val}_map_{val}.pth
```

Examples: `fly_swin3d_ratio75_vseed42_seed2025_ep3_f1no_0.7234_map_0.8123.pth` and `fly_timesformer_ratio100_seed2025_ep5_f1no_0.6987_map_0.7890.pth`. At 100% data, the `_vseed` suffix is omitted.
