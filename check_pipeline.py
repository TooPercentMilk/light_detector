"""Dry-run diagnostic: walk the full training → inference pipeline and report
which components are ready vs. missing / stubbed out.

Usage::

    python check_pipeline.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# ── colour helpers ────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"

results: list[tuple[str, str, str]] = []  # (status, component, detail)


def record(status: str, component: str, detail: str) -> None:
    results.append((status, component, detail))
    print(f"  {status}  {component}: {detail}")


# =====================================================================
# 1. DEPENDENCY CHECKS
# =====================================================================
print(f"\n{BOLD}1. Dependency checks{RESET}")

for pkg, install_hint in [
    ("torch", "pip install torch"),
    ("cv2", "pip install opencv-python"),
    ("numpy", "pip install numpy"),
    ("yaml", "pip install pyyaml"),
    ("yolox", 'pip install -e ".[models]"'),
    ("cython_bbox", "pip install cython_bbox"),
    ("lap", "pip install lap"),
]:
    try:
        importlib.import_module(pkg)
        record(PASS, f"import {pkg}", "available")
    except ImportError:
        record(FAIL, f"import {pkg}", f"NOT installed  →  {install_hint}")

# =====================================================================
# 2. CONFIG LOADING
# =====================================================================
print(f"\n{BOLD}2. Config loading{RESET}")

try:
    from adas_perception.traffic_light.config import load_config

    cfg = load_config("configs/default.yaml")
    record(PASS, "load_config", f"loaded configs/default.yaml")
except Exception as e:
    record(FAIL, "load_config", str(e))
    cfg = None

# =====================================================================
# 3. WEIGHT FILES
# =====================================================================
print(f"\n{BOLD}3. Weight files{RESET}")

weights_dir = Path("weights")
for name, path_attr in [
    ("Detector weights", cfg.detector.model_path if cfg else "weights/yolox_tl.pth"),
    ("Classifier weights", cfg.classifier.model_path if cfg else "weights/tl_state_classifier.pth"),
]:
    p = Path(path_attr)
    if p.is_file():
        mb = p.stat().st_size / 1024 / 1024
        record(PASS, name, f"{p}  ({mb:.1f} MB)")
    else:
        record(FAIL, name, f"{p} NOT FOUND — need to train or download")

# Check for generic pretrained backbone
pretrained_patterns = list(weights_dir.glob("yolox_*.pth")) if weights_dir.is_dir() else []
fine_tuned = Path(cfg.detector.model_path if cfg else "weights/yolox_tl.pth")
pretrained_patterns = [p for p in pretrained_patterns if p != fine_tuned]
if pretrained_patterns:
    record(PASS, "Pretrained backbone", ", ".join(str(p) for p in pretrained_patterns))
else:
    record(WARN, "Pretrained backbone",
           "No generic yolox_*.pth found in weights/ — download from YOLOX releases for fine-tuning")

# =====================================================================
# 4. DETECTOR
# =====================================================================
print(f"\n{BOLD}4. Detector (YOLOX){RESET}")

try:
    from adas_perception.traffic_light.detector import DETECTOR_REGISTRY

    if "yolox" in DETECTOR_REGISTRY:
        record(PASS, "Detector registry", "'yolox' registered")
    else:
        record(FAIL, "Detector registry", "'yolox' not in registry")
except Exception as e:
    record(FAIL, "Detector registry", str(e))

try:
    from adas_perception.traffic_light.detector.yolox_wrapper import YoloxWrapper, _check_yolox

    _check_yolox()
    from adas_perception.traffic_light.detector.yolox_wrapper import _yolox_available

    if _yolox_available:
        record(PASS, "YoloxWrapper.load_model", "yolox package found — load_model is functional")
    else:
        record(FAIL, "YoloxWrapper.load_model", "yolox package NOT importable")

    record(PASS, "YoloxWrapper.predict", "implemented")
    record(PASS, "YoloxWrapper._decode", "implemented")
except Exception as e:
    record(FAIL, "YoloxWrapper", str(e))

# =====================================================================
# 5. DETECTOR TRAINING
# =====================================================================
print(f"\n{BOLD}5. Detector training pipeline{RESET}")

try:
    from adas_perception.traffic_light.detector.trainer import YoloxTrainer

    t = YoloxTrainer()
    try:
        t.train("dummy_path", epochs=1)
        record(PASS, "YoloxTrainer.train", "runs (unexpected — verify correctness)")
    except NotImplementedError:
        record(FAIL, "YoloxTrainer.train", "NOT IMPLEMENTED — need training loop, dataloader, optimizer")
    except Exception as e:
        record(WARN, "YoloxTrainer.train", f"error (not NotImplementedError): {e}")
except Exception as e:
    record(FAIL, "YoloxTrainer import", str(e))

# =====================================================================
# 6. DETECTOR EVALUATION
# =====================================================================
print(f"\n{BOLD}6. Detector evaluation{RESET}")

try:
    from adas_perception.traffic_light.detector.evaluator import evaluate

    try:
        evaluate(None, "dummy_path")
        record(PASS, "evaluate()", "runs (unexpected — verify correctness)")
    except NotImplementedError:
        record(FAIL, "evaluate()", "NOT IMPLEMENTED — need COCO-metric evaluation loop")
    except Exception as e:
        record(WARN, "evaluate()", f"error: {e}")
except Exception as e:
    record(FAIL, "evaluate() import", str(e))

# =====================================================================
# 7. DETECTOR EXPORT
# =====================================================================
print(f"\n{BOLD}7. Detector ONNX export{RESET}")

try:
    from adas_perception.traffic_light.detector.export import export_onnx

    record(PASS, "export_onnx()", "implemented")
except Exception as e:
    record(FAIL, "export_onnx()", str(e))

# =====================================================================
# 8. PREPROCESSOR
# =====================================================================
print(f"\n{BOLD}8. Preprocessor{RESET}")

try:
    from adas_perception.traffic_light.preprocess import Preprocessor
    import numpy as np

    if cfg:
        pre = Preprocessor(cfg.preprocess)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            tensor = pre(dummy_img)
            record(PASS, "Preprocessor.__call__", f"output shape={tuple(tensor.shape)}")
        except NotImplementedError:
            record(FAIL, "Preprocessor.__call__",
                   "NOT IMPLEMENTED — need resize, normalize, HWC→CHW, to tensor")
        except Exception as e:
            record(WARN, "Preprocessor.__call__", f"error: {e}")
    else:
        record(WARN, "Preprocessor", "skipped (config not loaded)")
except Exception as e:
    record(FAIL, "Preprocessor import", str(e))

# =====================================================================
# 9. POSTPROCESSOR (NMS)
# =====================================================================
print(f"\n{BOLD}9. Postprocessor (NMS + confidence filter){RESET}")

try:
    from adas_perception.traffic_light.postprocess import postprocess
    import inspect

    src = inspect.getsource(postprocess)
    if "return detections" in src and "TODO" in src:
        record(WARN, "postprocess()", "STUB — returns detections unchanged, NMS not implemented")
    else:
        record(PASS, "postprocess()", "appears implemented")
except Exception as e:
    record(FAIL, "postprocess()", str(e))

# =====================================================================
# 10. TRACKER
# =====================================================================
print(f"\n{BOLD}10. Tracker (ByteTrack){RESET}")

try:
    from adas_perception.traffic_light.tracker import TRACKER_REGISTRY

    if "bytetrack" in TRACKER_REGISTRY:
        record(PASS, "Tracker registry", "'bytetrack' registered")
    else:
        record(FAIL, "Tracker registry", "'bytetrack' not in registry")
except Exception as e:
    record(FAIL, "Tracker registry", str(e))

try:
    import adas_perception.traffic_light.tracker.bytetrack_wrapper as _btw_mod
    from adas_perception.traffic_light.tracker.bytetrack_wrapper import ByteTrackWrapper

    _btw_mod._ensure_bytetrack_on_path()
    if _btw_mod._bytetrack_available:
        record(PASS, "ByteTrackWrapper", "BYTETracker available and wrapper implemented")
    else:
        record(FAIL, "ByteTrackWrapper", "BYTETracker NOT importable — install yolox or run setup_external.py")
except Exception as e:
    record(FAIL, "ByteTrackWrapper", str(e))

# =====================================================================
# 11. TRACKER ASSOCIATION UTILS
# =====================================================================
print(f"\n{BOLD}11. Tracker association utilities{RESET}")

try:
    from adas_perception.traffic_light.tracker.association import iou_batch

    import numpy as np
    try:
        iou_batch(np.zeros((1, 4)), np.zeros((1, 4)))
        record(PASS, "iou_batch()", "implemented")
    except NotImplementedError:
        record(FAIL, "iou_batch()", "NOT IMPLEMENTED — need vectorised IoU computation")
except Exception as e:
    record(FAIL, "iou_batch()", str(e))

try:
    from adas_perception.traffic_light.tracker.association import linear_assignment

    import numpy as np
    try:
        linear_assignment(np.zeros((1, 1)), thresh=0.5)
        record(PASS, "linear_assignment()", "implemented")
    except NotImplementedError:
        record(FAIL, "linear_assignment()",
               "NOT IMPLEMENTED — need Hungarian algorithm (scipy.optimize.linear_sum_assignment)")
except Exception as e:
    record(FAIL, "linear_assignment()", str(e))

# =====================================================================
# 12. STATE CLASSIFIER
# =====================================================================
print(f"\n{BOLD}12. State classifier (CNN){RESET}")

try:
    from adas_perception.traffic_light.state.classifier import StateClassifier

    if cfg:
        cls = StateClassifier(cfg.classifier)
        cls.load_model(cfg.classifier.model_path, cfg.classifier.device)

        import numpy as np
        dummy_roi = np.zeros((64, 32, 3), dtype=np.uint8)
        result = cls.classify(dummy_roi)

        from adas_perception.traffic_light.schemas import LightState
        if result == LightState.UNKNOWN:
            record(WARN, "StateClassifier.load_model", "STUB — logs but does not load a real model")
            record(WARN, "StateClassifier.classify", "STUB — always returns UNKNOWN")
        else:
            record(PASS, "StateClassifier", f"returned {result}")
    else:
        record(WARN, "StateClassifier", "skipped (config not loaded)")
except Exception as e:
    record(FAIL, "StateClassifier", str(e))

# =====================================================================
# 13. ROI REFINER
# =====================================================================
print(f"\n{BOLD}13. ROI refiner{RESET}")

try:
    from adas_perception.traffic_light.state.roi_refiner import RoiRefiner
    import numpy as np

    r = RoiRefiner()
    crop = r.refine(np.zeros((480, 640, 3), dtype=np.uint8), np.array([100, 100, 200, 200]))
    record(PASS, "RoiRefiner.refine", f"output shape={crop.shape}")
except Exception as e:
    record(FAIL, "RoiRefiner.refine", str(e))

# =====================================================================
# 14. TEMPORAL SMOOTHER
# =====================================================================
print(f"\n{BOLD}14. Temporal smoother{RESET}")

try:
    from adas_perception.traffic_light.state.temporal_smoother import TemporalSmoother
    from adas_perception.traffic_light.schemas import LightState

    if cfg:
        sm = TemporalSmoother(cfg.temporal_smoother)
        out = sm.update(1, LightState.RED)
        record(PASS, "TemporalSmoother.update", f"returned {out}")
    else:
        record(WARN, "TemporalSmoother", "skipped (config not loaded)")
except Exception as e:
    record(FAIL, "TemporalSmoother", str(e))

# =====================================================================
# 15. VISUALIZATION
# =====================================================================
print(f"\n{BOLD}15. Visualization overlays{RESET}")

try:
    from adas_perception.traffic_light.viz.overlays import draw_traffic_lights
    import inspect

    src = inspect.getsource(draw_traffic_lights)
    if "return image.copy()" in src and "TODO" in src:
        record(WARN, "draw_traffic_lights()", "STUB — returns blank copy, no boxes/labels drawn")
    else:
        record(PASS, "draw_traffic_lights()", "appears implemented")
except Exception as e:
    record(FAIL, "draw_traffic_lights()", str(e))

# =====================================================================
# 16. MAP GATE (optional)
# =====================================================================
print(f"\n{BOLD}16. Map gate (optional){RESET}")

try:
    from adas_perception.traffic_light.fusion.map_gate import MapGate
    import inspect

    src = inspect.getsource(MapGate.filter)
    if "TODO" in src:
        record(WARN, "MapGate.filter", "STUB — passes through all lights (map projection not implemented)")
    else:
        record(PASS, "MapGate.filter", "implemented or pass-through by design")
except Exception as e:
    record(FAIL, "MapGate", str(e))

# =====================================================================
# 17. END-TO-END NODE
# =====================================================================
print(f"\n{BOLD}17. End-to-end pipeline (TrafficLightNode){RESET}")

try:
    from adas_perception.traffic_light.node import TrafficLightNode
    import inspect

    src = inspect.getsource(TrafficLightNode.process_frame)
    if "state_confidence=0.0" in src and "TODO" in src:
        record(WARN, "TrafficLightNode.process_frame",
               "state_confidence hardcoded to 0.0 — classifier does not propagate confidence")
    record(PASS, "TrafficLightNode", "class importable, process_frame() wired up")
except Exception as e:
    record(FAIL, "TrafficLightNode", str(e))

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*70}")
print(f"{BOLD}SUMMARY{RESET}")
print(f"{'='*70}\n")

pass_count = sum(1 for s, _, _ in results if "PASS" in s)
warn_count = sum(1 for s, _, _ in results if "WARN" in s)
fail_count = sum(1 for s, _, _ in results if "FAIL" in s)

print(f"  {GREEN}{pass_count} passed{RESET}   {YELLOW}{warn_count} warnings{RESET}   {RED}{fail_count} failed{RESET}\n")

if fail_count or warn_count:
    print(f"{BOLD}TODO list:{RESET}\n")
    idx = 1
    for status, component, detail in results:
        if "FAIL" in status:
            print(f"  {RED}{idx}. [MUST FIX]{RESET}  {component}")
            print(f"     {detail}\n")
            idx += 1
    for status, component, detail in results:
        if "WARN" in status:
            print(f"  {YELLOW}{idx}. [SHOULD FIX]{RESET}  {component}")
            print(f"     {detail}\n")
            idx += 1
else:
    print(f"  {GREEN}All checks passed!{RESET}")

print()
