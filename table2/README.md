# CalMS21 & CRIM13 — Video Swin-T Training & Evaluation

Video Swin-T (Kinetics-400 pretrained) + MLP Head + Cross Entropy Loss.
Stratified video-level split with fixed val split seed for fair comparison across training seeds.

## Requirements

```bash
pip install torch torchvision decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Arguments

### Training (shared by both datasets)

| Argument | CalMS21 Default | CRIM13 Default | Description |
|---|---|---|---|
| `--seed` | `123` | `123` | General seed (model init, augmentation, shuffle) |
| `--val_split_seed` | `1337` | `1337` | Val split seed (**FIXED** across experiments) |
| `--train_video_dir` | `data/calms21/videos/train` | `data/crim13/videos/train` | Training video directory |
| `--train_label_dir` | `data/calms21/labels/train` | `data/crim13/labels` | Training label directory |
| `--batch_size` | `8` | `8` | Batch size |
| `--accumulation_steps` | `2` | `2` | Gradient accumulation steps |
| `--num_epochs` | `5` | `5` | Number of epochs |
| `--base_lr` | `3.8e-5` | `2e-5` | Learning rate |
| `--validation_ratio` | `0.15` | `0.15` | Fraction of videos for validation |
| `--use_class_weights` | off | off | Enable inverse-frequency class weights |
| `--model_save_dir` | `checkpoints/calms21` | `checkpoints/crim13` | Checkpoint directory |

## Seed Design

Two separate seeds serve different purposes:

- **`--seed`** (general): controls model initialization, augmentation randomness, dataloader shuffle. **Change this** across runs (123, 1337, 2025) to test training stability.
- **`--val_split_seed`** (val split): controls which videos go to train vs val. **Keep this FIXED** (default 1337) so all runs use the same train/val split for fair comparison.

## Training

### CalMS21 (4 classes: Attack, Investigation, Mount, Other)

```bash
python train_calms21.py --seed 123
python train_calms21.py --seed 1337
python train_calms21.py --seed 2025
```

### CRIM13 (13 classes: category1–category13)

```bash
python train_crim13.py --seed 123
python train_crim13.py --seed 1337
python train_crim13.py --seed 2025
```

## Evaluation

```bash
# CalMS21
python eval_calms21.py --model_path checkpoints/calms21/model.pth

# CRIM13
python eval_crim13.py --model_path checkpoints/crim13/model.pth
```

## Output Naming Convention

```
{dataset}_seed{general}_vsplit{val_split}_ep{N}_f1_{val}_map_{val}.pth
```

Examples:
- `calms21_seed123_vsplit1337_ep3_f1_0.8388_map_0.9005.pth`
- `crim13_seed2025_vsplit1337_ep5_f1_0.4521_map_0.3876.pth`

## Dataset Differences

| | CalMS21 | CRIM13 |
|---|---|---|
| Classes | 4 | 13 |
| Label file | `{video}.csv` | `{video}_one_hot.csv` |
| Split strategy | Stratified by rare behavior (Attack) | Random + ensure all classes in val |
| Default LR | `3.8e-5` | `2e-5` |
