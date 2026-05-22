# Fly Copulation — Video-Level Data Efficiency Analysis

Video Swin-T / TimeSformer (K400 pretrained) + MLP Head + CE Loss.
Stratified video-level split + video-level subsampling for data efficiency experiments.

Best model selected by **val F1-macro excluding 'others'**.

## Requirements

```bash
pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Behavior Remapping

Original 5 classes → Selected 4 classes:

| Original | → Selected |
|---|---|
| wing_extension | wing_extension |
| circle | circle |
| copul_attempt | **others** (merged) |
| copulation | copulation |
| others | others |

## Arguments

| Argument | Swin3D Default | TimeSformer Default | Description |
|---|---|---|---|
| `--train_data_ratio` | `1.0` | `1.0` | Fraction of training videos to use |
| `--video_split_seed` | `42` | `42` | RNG seed for video subsampling |
| `--seed` | `2025` | `2025` | General seed (model init, augmentation) |
| `--val_split_seed` | `123` | `123` | Val split seed (**FIXED**) |
| `--base_lr` | `3.8e-5` | `3.8e-5` | Learning rate |
| `--model_save_dir` | `checkpoints/fly_swin3d_ratio` | `checkpoints/fly_timesformer_ratio` | Checkpoint dir |

## Training

```bash
# Swin3D
python train_fly_swin3d_ratio.py \
  --train_video_dir /path/to/videos/train \
  --train_label_dir /path/to/labels/train \
  --train_data_ratio 0.75 --video_split_seed 42

# TimeSformer
python train_fly_timesformer_ratio.py \
  --train_video_dir /path/to/videos/train \
  --train_label_dir /path/to/labels/train \
  --train_data_ratio 0.75 --video_split_seed 42
```

## Evaluation

```bash
# Swin3D
python eval_fly_swin3d_ratio.py \
  --model_path checkpoints/fly_swin3d_ratio/model.pth \
  --test_video_dir /path/to/videos/test \
  --test_label_dir /path/to/labels/test

# TimeSformer
python eval_fly_timesformer_ratio.py \
  --model_path checkpoints/fly_timesformer_ratio/model.pth \
  --test_video_dir /path/to/videos/test \
  --test_label_dir /path/to/labels/test
```

## Output Naming Convention

```
fly_{model}_{ratio_tag}{_vseed}{_seed}_ep{N}_f1no_{val}_map_{val}.pth
```

Examples:
- `fly_swin3d_ratio75_vseed42_seed2025_ep3_f1no_0.7234_map_0.8123.pth`
- `fly_timesformer_ratio100_seed2025_ep5_f1no_0.6987_map_0.7890.pth`

At `ratio=100`, the `_vseed` suffix is omitted.
