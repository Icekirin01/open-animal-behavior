# Figure 7 — Open-to-Lab Fly TimeSformer Transfer

This is a two-stage TimeSformer experiment: first train on an open fly dataset, then use that checkpoint to fine-tune on the lab fly dataset and evaluate on the lab test set.

## Requirements

Run this cell once in Colab:

```bash
!pip install torch torchvision transformers decord numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Evaluation Code

### Evaluate the fine-tuned lab checkpoint

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure7

!python eval_fly_timesformer_finetune.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/fly_timesformer_ft_ratio25_vseed42_ft_full_ep2_f1no_0.5559_map_0.7001.pth \
    --test_video_dir /content/drive/MyDrive/xxx/lab_fly/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/lab_fly/labels/test/ \
    --save_cm /content/drive/MyDrive/xxx/results/finetune_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/finetune_results.json
```

### Evaluate the stage-one open-dataset checkpoint

The paths below are Google Drive examples. Replace `xxx` with the locations of your open-dataset files.

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure7

!python eval_fly_timesformer_open.py \
    --model_path /content/drive/MyDrive/xxx/checkpoints/figure7_open_pretrained.pth \
    --test_video_dir /content/drive/MyDrive/xxx/open_fly/videos/test/ \
    --test_label_dir /content/drive/MyDrive/xxx/open_fly/labels/test/ \
    --save_cm /content/drive/MyDrive/xxx/results/open_cm.png \
    --save_results /content/drive/MyDrive/xxx/results/open_results.json
```

## Training Code

Run stage 1 first. After it finishes, copy the best checkpoint path printed by the script into stage 2 as `--pretrained_model_path`.

### Stage 1 — Train TimeSformer on the open dataset

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure7

!python train_fly_timesformer_open.py \
    --train_video_dir /content/drive/MyDrive/xxx/open_fly/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/open_fly/labels/train/ \
    --seed 2025 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/figure7/open
```

### Stage 2 — Fine-tune on 25% of the lab videos

The example below assumes the selected stage-one checkpoint was renamed to `figure7_open_pretrained.pth`.

```bash
%cd /content/drive/MyDrive/xxx/reproduce/figure7

!python train_fly_timesformer_finetune.py \
    --pretrained_model_path /content/drive/MyDrive/xxx/checkpoints/figure7_open_pretrained.pth \
    --finetune_strategy full \
    --train_video_dir /content/drive/MyDrive/xxx/lab_fly/videos/train/ \
    --train_label_dir /content/drive/MyDrive/xxx/lab_fly/labels/train/ \
    --train_data_ratio 0.25 \
    --video_split_seed 42 \
    --seed 2025 \
    --val_split_seed 123 \
    --model_save_dir /content/drive/MyDrive/xxx/outputs/figure7/finetune_ratio025
```

Use `--reinit_head` only when you intentionally want to discard the stage-one MLP head and transfer the backbone alone.

## Seed Design

- `--seed` controls model initialization, augmentation and shuffle. The default is `2025` in both stages.
- `--val_split_seed` controls train/validation assignment. Keep it fixed at `123`.
- `--video_split_seed` is used only during lab fine-tuning and controls which videos are retained when `--train_data_ratio < 1.0`; the Figure 7 example uses `42`.
- When comparing fine-tuning strategies, keep the pretrained checkpoint, selected lab videos and validation split identical.

## Arguments

### Stage 1: `train_fly_timesformer_open.py`

| Argument | Default | Description |
|---|---:|---|
| `--train_video_dir` | `data/fly_open/videos/train` | Google Drive folder containing open-dataset training videos |
| `--train_label_dir` | `data/fly_open/labels/train` | Google Drive folder containing open-dataset training-label CSV files |
| `--seed` | `2025` | General random seed |
| `--val_split_seed` | `123` | Fixed validation-split seed |
| `--validation_ratio` | `0.15` | Validation fraction |
| `--batch_size` | `8` | Batch size |
| `--accumulation_steps` | `2` | Gradient accumulation |
| `--num_epochs` | `5` | Training epochs |
| `--base_lr` | `3e-5` | Learning rate |
| `--model_save_dir` | `checkpoints/fly_timesformer_open` | Google Drive folder used to save stage-one checkpoints and logs |

### Stage 2: `train_fly_timesformer_finetune.py`

| Argument | Default | Description |
|---|---:|---|
| `--pretrained_model_path` | required | Location of the stage-one `.pth` checkpoint on Google Drive |
| `--finetune_strategy` | `full` | `full`, `head_only` or `gradual` |
| `--reinit_head` | off | Reinitialize the MLP head before fine-tuning |
| `--train_video_dir` | `data/fly/videos/train` | Google Drive folder containing lab training videos |
| `--train_label_dir` | `data/fly/labels/train` | Google Drive folder containing lab training-label CSV files |
| `--train_data_ratio` | `0.75` | Fraction of lab videos retained |
| `--video_split_seed` | `42` | Lab video-subsampling seed |
| `--seed` | `2025` | General random seed |
| `--val_split_seed` | `123` | Fixed validation-split seed |
| `--base_lr` | `3.8e-5` | Fine-tuning learning rate |
| `--model_save_dir` | `checkpoints/fly_timesformer_finetune` | Google Drive folder used to save fine-tuned checkpoints and logs |

### Evaluation scripts

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | required | Location of an open or fine-tuned `.pth` checkpoint on Google Drive |
| `--test_video_dir` | dataset-specific | Google Drive folder containing test videos |
| `--test_label_dir` | dataset-specific | Google Drive folder containing test-label CSV files |
| `--batch_size` | `8` | Evaluation batch size |
| `--window_size` / `--stride` | `16` / `4` | Temporal window and stride |
| `--smooth_window_size` | `1` | Temporal smoothing window |
| `--save_cm` | none | Google Drive path where the confusion-matrix PNG is saved |
| `--save_results` | none | Google Drive path where the metrics JSON is saved |

### Google Drive Path Examples

```text
--model_path /content/drive/MyDrive/xxx/checkpoints/model.pth
--pretrained_model_path /content/drive/MyDrive/xxx/checkpoints/open_model.pth
--train_video_dir /content/drive/MyDrive/xxx/lab_fly/videos/train
--train_label_dir /content/drive/MyDrive/xxx/lab_fly/labels/train
--test_video_dir /content/drive/MyDrive/xxx/lab_fly/videos/test
--test_label_dir /content/drive/MyDrive/xxx/lab_fly/labels/test
--model_save_dir /content/drive/MyDrive/xxx/outputs
```

## Workflow

1. Train on the open dataset with `train_fly_timesformer_open.py`.
2. Select its best validation checkpoint.
3. Pass that checkpoint to `train_fly_timesformer_finetune.py`.
4. Evaluate the resulting lab checkpoint with `eval_fly_timesformer_finetune.py`.

## Behavior Remapping

| Original class | Selected class |
|---|---|
| `wing_extension` | `wing_extension` |
| `circle` | `circle` |
| `copul_attempt` | `others` |
| `copulation` | `copulation` |
| `others` | `others` |

## Output Naming Convention

Stage one produces names such as `fly_timesformer_open_seed2025_vsplit123_ep5_f1_..._map_....pth`. Stage two encodes the data ratio, video seed and strategy, for example `fly_timesformer_ft_ratio25_vseed42_ft_full_ep2_f1no_0.5559_map_0.7001.pth`.
