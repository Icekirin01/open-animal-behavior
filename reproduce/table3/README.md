# Fly Copulation — Stratified Val Split (No Data Efficiency Ratio)

Video Swin-T / TimeSformer (K400 pretrained) + MLP Head + CE Loss.
Stratified video-level split by rare behavior (copulation).

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

## Seed Design

Two separate seeds for different purposes:

| Seed | Default | Purpose |
|---|---|---|
| `--seed` | Swin3D: `123`, TimeSformer: `1337` | Model init, augmentation, shuffle |
| `--val_split_seed` | `123` (FIXED) | Validation split — **never change** |

Change `--seed` for multiple runs; keep `--val_split_seed` fixed for fair comparison.

## Training

```bash
# Swin3D
python train_fly_swin3d.py \
  --train_video_dir /path/to/videos/train \
  --train_label_dir /path/to/labels/train \
  --seed 123

# TimeSformer
python train_fly_timesformer.py \
  --train_video_dir /path/to/videos/train \
  --train_label_dir /path/to/labels/train \
  --seed 1337
```

## Evaluation

```bash
# Swin3D
python eval_fly_swin3d.py \
  --model_path checkpoints/fly_swin3d/model.pth \
  --test_video_dir /path/to/videos/test \
  --test_label_dir /path/to/labels/test

# TimeSformer
python eval_fly_timesformer.py \
  --model_path checkpoints/fly_timesformer/model.pth \
  --test_video_dir /path/to/videos/test \
  --test_label_dir /path/to/labels/test
```

## Output Naming Convention

```
fly_{model}_seed{X}_vsplit{Y}_ep{N}_f1_{val}_map_{val}.pth
```

Examples:
- `fly_swin3d_seed123_vsplit123_ep3_f1_0.7234_map_0.8123.pth`
- `fly_timesformer_seed1337_vsplit123_ep5_f1_0.6987_map_0.7890.pth`

## Differences from Ratio Scripts

| | This version | Ratio version |
|---|---|---|
| `--train_data_ratio` | ❌ | ✅ |
| `--video_split_seed` | ❌ | ✅ |
| Best model selection | val F1-macro (all) | val F1-macro (no others) |
| Purpose | Standard training | Data efficiency analysis |
