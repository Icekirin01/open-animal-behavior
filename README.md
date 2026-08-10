# CREAC — Cross-species Retrainable Animal behavior Classifier

Video-based classification of animal social behaviors using video transformers
(Video Swin-T and TimeSformer) pretrained on Kinetics-400 and fine-tuned on your
own recordings. No pose estimation, no programming — the training and inference
tools run as point-and-click notebooks on a free Google Colab GPU.

- **Models & datasets** (pretrained weights, lab videos, annotations): [huggingface.co/yiheng266/animal-social-models](https://huggingface.co/yiheng266/animal-social-models)
- **Code, GUIs & paper reproduction**: this repository

---

## Run it in Colab (no installation)

Both notebooks open directly in Colab — click the link, and everything installs
in the first cell.

| Notebook | What it does | Open in Colab |
|---|---|---|
| **`inference.ipynb`** | Load a trained model and classify your videos → export CSV or BORIS event logs | [Open](https://colab.research.google.com/github/yiheng226/CREAC/blob/main/inference.ipynb) |
| **`training.ipynb`** | Fine-tune a pretrained model on your own labeled videos | [Open](https://colab.research.google.com/github/yiheng226/CREAC/blob/main/training.ipynb) |

---

## What you need to prepare

**For inference** — just videos:

- A folder of video files (`.mp4`, `.avi`, …), e.g. on your Google Drive.

That's it. Pick a pretrained model from HuggingFace (or upload your own), point
the notebook at the folder, and run.

**For training (fine-tuning)** — videos **plus** frame-level labels:

- A **video folder** (one file per recording).
- A **label folder** with one annotation file per video, in either format:
  - **BORIS** event logs exported from [BORIS](https://www.boris.unito.it/), or
  - **one-hot CSV** — one row per frame, one column per behavior (1 = active).
- Video and label files are matched automatically by filename, so keep the base
  names the same (`IMG3_1.mp4` ↔ `IMG3_1.csv`).

Example files for both label formats are provided in this repository. Inside the
training notebook you then choose which labels to **keep, merge, or exclude**,
and whether to reuse the pretrained classification head or start a fresh one.

---

## What each notebook walks you through

**Inference** — (1) select a model from HuggingFace or a local folder →
(2) point at a video folder → (3) run batch or single-video inference, with a
live ethogram, frame scrubber, and per-behavior statistics → (4) export per-frame
CSV or BORIS event logs.

**Training** — (1) pick a pretrained backbone → (2) load videos + labels and
preview the annotation timeline → (3) map your labels onto training classes →
(4) set hyperparameters and train, with live per-epoch **F1 / mAP** and, after
training, **Per-epoch / Threshold / Ethogram** validation views → (5) checkpoints
save automatically with a config file, ready to use in the inference notebook.

---

## Where everything lives

| What | Where |
|---|---|
| Training & inference GUIs (notebooks) | this repo — `training.ipynb`, `inference.ipynb` |
| Per-figure code + plotted-data CSVs (paper reproduction) | this repo — [`reproduce/`](reproduce/) |
| Core library (models, inference, config) | this repo — `models.py`, `inference.py`, `config_utils.py`, … |
| Pretrained model weights (mouse & fly) | HuggingFace — [`animal-social-models`](https://huggingface.co/yiheng266/animal-social-models) |
| Lab video recordings + frame-level annotations | HuggingFace — [`animal-social-models`](https://huggingface.co/yiheng266/animal-social-models) |
| Public benchmarks (CalMS21, CRIM13) | original sources (see paper) |

---

## Reproducing the paper (for reviewers)

All results are reproducible from [`reproduce/`](reproduce/), organized **by
figure**. Each figure's subfolder has its own README with the exact commands,
and contains the training and evaluation scripts for that experiment plus the
CSV files with every plotted data point.

```bash
pip install -r requirements.txt
```

Covered experiments: backbone comparison, per-behavior data efficiency
(video-level subsampling), cross-domain transfer, and the public-benchmark
evaluations.

> **A note on exact numbers.** Because of GPU nondeterminism, mixed-precision
> training, and multi-worker data loading, retraining can differ slightly from
> the reported metrics even with a fixed seed. To reproduce the paper's numbers
> exactly, use the released checkpoints on HuggingFace (or request them).

The graphical notebooks are for *prospective* use — training new models and
classifying new videos — and do not themselves reproduce the paper; the
standalone scripts in `reproduce/` serve that purpose.

---


## License

Released under the MIT License — see [LICENSE](LICENSE).
