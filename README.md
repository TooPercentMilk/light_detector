# light_detector
Detects traffic lights from in-vehicle video footage.

---

## Project Architecture

```
Frame Input
    │
    ▼
┌─────────────┐
│ Preprocessor │  Resize, normalize, channel swap
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   Detector   │  YOLOX — locates traffic lights in the frame
│ (yolox_tl)  │  Output: bounding boxes + confidence scores
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Tracker    │  ByteTrack — associates detections across frames
│              │  Output: tracked objects with persistent IDs
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  RoiRefiner  │  Crops + pads each tracked box from the full frame
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ StateClassifier  │  CNN — classifies each crop as red/yellow/green/off
│ (tl_state_cls)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ TemporalSmoother │  Majority-vote over a sliding window to reduce flicker
└──────┬───────────┘
       │
       ▼
┌──────────┐
│ MapGate  │  (Optional) Filters lights irrelevant to the ego lane using HD-map priors
└──────┬───┘
       │
       ▼
┌─────────────┐
│ Postprocess │  Assembles final TrafficLight objects
│  + Viz      │  Draws overlays for debug/display
└─────────────┘
```

### Module map

| Path | Responsibility |
|---|---|
| `src/adas_perception/traffic_light/node.py` | Pipeline orchestrator (`TrafficLightNode`) |
| `src/adas_perception/traffic_light/preprocess.py` | Frame normalization |
| `src/adas_perception/traffic_light/detector/` | YOLOX wrapper + trainer/exporter |
| `src/adas_perception/traffic_light/tracker/` | ByteTrack wrapper |
| `src/adas_perception/traffic_light/state/` | ROI crop, CNN classifier, temporal smoother |
| `src/adas_perception/traffic_light/fusion/` | HD-map gate (no-op without map data) |
| `src/adas_perception/traffic_light/postprocess.py` | Output assembly |
| `src/adas_perception/traffic_light/viz/` | Overlay rendering |
| `src/adas_perception/traffic_light/schemas.py` | Shared data classes (`Detection`, `TrackedObject`, `TrafficLight`, `LightState`) |
| `src/adas_perception/traffic_light/config.py` | Typed config dataclasses + YAML loader |
| `configs/default.yaml` | Default runtime configuration |

---

## Weights

Model weights are stored in `weights/` (not committed to git):

| File | Description |
|---|---|
| `weights/yolox_tl.pth` | YOLOX fine-tuned on traffic lights (required for inference) |
| `weights/tl_state_classifier.pth` | CNN state classifier — red/yellow/green/off (required for inference) |
| `weights/yolox_s.pth` *(or `_m`, `_l`)* | Generic YOLOX pretrained checkpoint used as the starting point for fine-tuning |

Download pretrained YOLOX backbone weights from the
[YOLOX releases page](https://github.com/Megvii-BaseDetection/YOLOX/releases)
and place them in `weights/` before running training.

---

## Setup

```bash
# 1. Install project + model dependencies
pip install -e ".[models]"

# 2. (Optional) Clone standalone ByteTrack — only needed if not using pip-installed YOLOX
python setup_external.py
```

---

## Usage

```bash
python -m adas_perception.traffic_light.node \
    --config configs/default.yaml \
    --image-dir data/frames/ \
    --image-size 960
```

Detector training/evaluation use 960x960 by default. Override the square
detector/preprocessor size when comparing runs:

```bash
python scripts/train.py --target detector --dataset data/coco_tl --det-image-size 1280
python scripts/evaluate.py --config configs/val_best.yaml --dataset data/coco_tl --image-size 1280
```
