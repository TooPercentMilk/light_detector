# Known Issues & Fixes

A log of issues encountered during development and their resolutions.

---

## 1. `ImportError: cannot import name 'COCODataset' from 'yolox.data'`

**Context:** Running detector evaluation via `scripts/evaluate.py`.

**Cause:** The evaluator (`detector/evaluator.py`) imported `COCODataset` from `yolox.data`, but the bundled ByteTrack fork of YOLOX does not include that class — it only ships MOT-related datasets.

**Fix:** Replaced the `COCODataset` dependency with a lightweight `_COCOValDataset` class in `evaluator.py` that uses `pycocotools.coco.COCO` directly and applies the same letterbox preprocessing YOLOX expects.

---

## 2. `PermissionError` when deleting temp file on Windows

**Context:** Detector evaluation crashes in `evaluator.py` at `Path(tmp_path).unlink()`.

**Cause:** `tempfile.mkstemp` returns a raw file descriptor. The original code opened a second handle via `open()` without closing the fd, leaving a dangling OS handle that Windows locks. `pycocotools` may also retain a handle during `loadRes`.

**Fix:** Used `os.fdopen(tmp_fd, "w")` to properly close the file descriptor, and wrapped the `unlink` in a `try/except OSError` to tolerate Windows file-locking.

---

## 3. Pipeline produces 0 detections (end-to-end) despite working detector eval

**Context:** COCO mAP evaluation showed reasonable detection performance (mAP@50 ≈ 0.46), but the end-to-end pipeline reported 0 predicted objects on the same validation set.

**Cause:** The `Preprocessor` applied ImageNet-style normalization (divide by 255, subtract mean, divide by std), producing values centered around 0 in the range [-2, 2]. However, YOLOX expects raw float32 pixel values in the **[0, 255]** range with letterbox padding. The mismatched input distribution caused all detection confidence scores to fall below the threshold.

The standalone detector evaluation worked because it used its own YOLOX-compatible preprocessing (via `_COCOValDataset`), bypassing `Preprocessor` entirely.

**Fix:** Rewrote `Preprocessor` to use YOLOX-compatible preprocessing:
- Letterbox resize (preserve aspect ratio, pad with value 114)
- Float32 in [0, 255] — no `/255` normalization or ImageNet mean/std subtraction

The state classifier was unaffected because it applies its own normalization internally in `classify()`.

---

## 4. `AttributeError: module 'numpy' has no attribute 'float'`

**Context:** Pipeline crashes inside ByteTrack's `byte_tracker.py` and `matching.py` when the tracker processes detections.

**Cause:** ByteTrack was written for older NumPy versions that supported `np.float` as an alias for Python's `float`. NumPy 1.20 deprecated this alias and NumPy 1.24+ removed it entirely.

**Fix:** Replaced all `np.float` usages with `np.float64` in:
- `external/ByteTrack/yolox/tracker/byte_tracker.py`
- `external/ByteTrack/yolox/tracker/matching.py`

---

## 5. Pipeline produces 14,804 predictions but 0 TP matches (coordinate mismatch)

**Context:** After fixing issues 3 & 4, the pipeline ran successfully and produced ~14,804 detections across the validation set, but IoU matching against ground truth yielded 0 true positives.

**Cause:** The `Preprocessor` letterbox-resizes images to 640×640 before feeding them to YOLOX. The detector produces bounding boxes in this 640×640 letterbox coordinate space. However, the ground truth annotations are in **original image coordinates** (e.g. 1280×960). Without rescaling the predicted boxes back, IoU between predictions and GT was always near zero.

Additionally, the classifier was cropping ROIs from the original image using letterbox-space coordinates, which would extract the wrong (or empty) region.

**Fix:** Modified `Preprocessor.__call__` to return `tuple[torch.Tensor, float]` — the preprocessed tensor and the letterbox scale factor `r`. Then in `TrafficLightNode.process_frame`, added rescaling after postprocess:

```python
tensor, scale = self.preprocessor(image)
# ... detect + postprocess ...
for det in detections:
    det.bbox = det.bbox / scale
```

This maps detection boxes back to original image coordinates before tracking and ROI classification.
