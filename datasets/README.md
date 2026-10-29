# Datasets for DLCSPDY — PPE Object Detection

This folder contains three YOLO-format datasets used for training and evaluating DLCSPDY. Each dataset follows the standard Ultralytics layout:

```
<dataset>/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Configs with relative paths for reproduction are in [`configs/datasets/`](../configs/datasets/).

---

## PPE2025

| Property | Value |
|----------|-------|
| **Classes (4)** | `Helmet`, `NoHelmet`, `NoVest`, `Vest` |
| **Train / Val / Test** | 1,829 / 228 / 228 images |
| **Annotation format** | YOLO (normalized `class x_center y_center width height`) |
| **Source** | Kaggle — PPE detection dataset |
| **Config** | [`configs/datasets/ppe2025.yaml`](../configs/datasets/ppe2025.yaml) |

### Class semantics

- **Helmet** — Worker wearing a safety helmet
- **NoHelmet** — Worker without a helmet
- **Vest** — Worker wearing a high-visibility vest
- **NoVest** — Worker without a vest

### Split policy

Images are partitioned into `train`, `valid`, and `test` folders as provided by the dataset publisher. Use `valid` during training for model selection; report final metrics on `test`.

---

## CHVG

| Property | Value |
|----------|-------|
| **Classes (8)** | `blue`, `glass`, `head`, `person`, `red`, `vest`, `white`, `yellow` |
| **Train / Val / Test** | 1,358 / 169 / 171 images |
| **Annotation format** | YOLOv11 |
| **License** | MIT |
| **Source** | [Roboflow — CHVG Conversion v3](https://universe.roboflow.com/scalersai/chvg-conversion/dataset/3) |
| **Config** | [`configs/datasets/chvg.yaml`](../configs/datasets/chvg.yaml) |

### Pre-processing (Roboflow export)

- Auto-orientation (EXIF stripped)
- Resize to **416×416** (stretch)

### Split policy

Standard `train` / `valid` / `test` split from Roboflow export (total 1,699 images before folder assignment).

---

## CPPE

| Property | Value |
|----------|-------|
| **Classes (4)** | `gloves`, `harness`, `helmet`, `shoes` |
| **Train / Val / Test** | 2,545 / 474 / 161 images |
| **Annotation format** | YOLO |
| **License** | CC BY 4.0 |
| **Source** | [Roboflow — CPPE v2](https://universe.roboflow.com/genius-rtnnb/cppe/dataset/2) |
| **Config** | [`configs/datasets/cppe.yaml`](../configs/datasets/cppe.yaml) |

### Pre-processing (Roboflow export)

- Auto-orientation (EXIF stripped)
- Resize to **640×640** (fit within)

### Split policy

Standard `train` / `valid` / `test` split from Roboflow export (total 3,180 images).

---

## Usage in training / evaluation

```bash
# Train on each dataset
python train.py --data configs/datasets/ppe2025.yaml
python train.py --data configs/datasets/chvg.yaml --name chvg
python train.py --data configs/datasets/cppe.yaml --name cppe

# Evaluate on test split
python val.py --weights runs/detect/train/weights/best.pt --data configs/datasets/ppe2025.yaml --split test
```

---

## Downloading datasets

- **PPE2025**: Download from Kaggle and extract into `datasets/PPE2025/`. Do **not** commit API credentials; use `~/.kaggle/kaggle.json` locally.
- **CHVG / CPPE**: Exported from Roboflow (see `README.roboflow.txt` in each folder) or download from the Roboflow Universe links above.

---

## Label format

Each image in `images/` has a corresponding `.txt` file in `labels/` with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width and height.
