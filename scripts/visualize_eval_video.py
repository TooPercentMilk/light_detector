"""Render full evaluation clips as an annotated tracking video.

The script defaults to the held-out COCO ``test`` split and the best runtime
configuration in ``configs/val_best.yaml``. Frames are processed in clip order,
ByteTrack state is reset at clip boundaries, and every frame from each selected
clip is written to one stitched output video.

Usage::

    python scripts/visualize_eval_video.py

    python scripts/visualize_eval_video.py \
        --split test \
        --clip daySequence2 \
        --output runs/viz/daySequence2_tracks.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adas_perception.traffic_light.config import PipelineConfig, load_config
from adas_perception.traffic_light.node import TrafficLightNode
from adas_perception.traffic_light.viz.overlays import draw_traffic_lights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _project_path(path: str | Path) -> Path:
    """Resolve repo-relative paths while still honoring absolute paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return ROOT / candidate


def _clip_name(file_name: str, split: str) -> str:
    name = Path(file_name).name
    if "--" in name:
        return name.split("--", 1)[0]
    return split


def _load_split_clips(dataset: Path, split: str) -> dict[str, list[Path]]:
    """Return frame paths grouped by clip name for a COCO-style split."""
    split_dir = dataset / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split image directory not found: {split_dir}")

    ann_path = dataset / "annotations" / f"instances_{split}.json"
    clips: dict[str, list[Path]] = defaultdict(list)

    if ann_path.is_file():
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        missing = 0
        images = sorted(coco.get("images", []), key=lambda img: img.get("file_name", ""))
        for image_info in images:
            file_name = image_info.get("file_name")
            if not file_name:
                continue
            frame_path = split_dir / file_name
            if not frame_path.is_file():
                missing += 1
                continue
            clips[_clip_name(file_name, split)].append(frame_path)

        if missing:
            logger.warning("Skipped %d annotation entries with missing image files", missing)
    else:
        logger.warning("Annotation file not found; scanning image directory: %s", ann_path)
        for frame_path in sorted(
            p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            clips[_clip_name(frame_path.name, split)].append(frame_path)

    return dict(sorted(clips.items()))


def _configure_devices(config: PipelineConfig, device: str | None) -> None:
    if device:
        config.detector.device = device
        config.classifier.device = device
        return

    detector_device = str(config.detector.device)
    classifier_device = str(config.classifier.device)
    if not detector_device.startswith("cuda") and not classifier_device.startswith("cuda"):
        return

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; using CPU for detector and classifier")
        config.detector.device = "cpu"
        config.classifier.device = "cpu"


def _resolve_config_paths(config: PipelineConfig) -> None:
    config.detector.model_path = str(_project_path(config.detector.model_path))
    config.classifier.model_path = str(_project_path(config.classifier.model_path))


def _require_runtime_files(config: PipelineConfig) -> None:
    required = {
        "detector weights": Path(config.detector.model_path),
        "classifier weights": Path(config.classifier.model_path),
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required runtime file(s) missing:\n" + "\n".join(missing))


def _reset_clip_state(node: TrafficLightNode) -> None:
    node.tracker.reset()
    if hasattr(node.smoother, "reset"):
        node.smoother.reset()


def _draw_frame_label(frame, clip_name: str, frame_idx: int, total_frames: int) -> None:
    label = f"{clip_name}  {frame_idx + 1}/{total_frames}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (8, 8), (18 + tw, 18 + th + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (13, 13 + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _open_writer(output_path: Path, codec: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {output_path}. "
            f"Try a different --codec, e.g. mp4v or XVID."
        )
    return writer


def render_video(
    config: PipelineConfig,
    clips: dict[str, list[Path]],
    clip_names: list[str],
    output_path: Path,
    fps: float,
    codec: str,
    output_size: tuple[int, int] | None,
    draw_frame_label: bool,
    max_frames_per_clip: int | None,
    progress_every: int,
) -> None:
    node = TrafficLightNode(config)

    writer: cv2.VideoWriter | None = None
    writer_size = output_size
    resize_warned = False
    total_written = 0
    total_detections = 0
    start = time.perf_counter()

    for clip_number, clip_name in enumerate(clip_names, start=1):
        _reset_clip_state(node)
        frame_paths = clips[clip_name]
        if max_frames_per_clip is not None:
            frame_paths = frame_paths[:max_frames_per_clip]

        logger.info(
            "Processing clip %d/%d: %s (%d frames)",
            clip_number,
            len(clip_names),
            clip_name,
            len(frame_paths),
        )

        for frame_id, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning("Could not read %s; skipping", frame_path)
                continue

            lights = node.process_frame(image, frame_id)
            total_detections += len(lights)
            annotated = draw_traffic_lights(image, lights)

            if draw_frame_label:
                _draw_frame_label(annotated, clip_name, frame_id, len(frame_paths))

            if writer_size is None:
                height, width = annotated.shape[:2]
                writer_size = (width, height)

            if (annotated.shape[1], annotated.shape[0]) != writer_size:
                if not resize_warned:
                    logger.warning(
                        "Input frame sizes differ; resizing all output frames to %dx%d",
                        writer_size[0],
                        writer_size[1],
                    )
                    resize_warned = True
                annotated = cv2.resize(annotated, writer_size, interpolation=cv2.INTER_AREA)

            if writer is None:
                writer = _open_writer(output_path, codec, fps, writer_size)

            writer.write(annotated)
            total_written += 1

            if progress_every > 0 and total_written % progress_every == 0:
                elapsed = max(time.perf_counter() - start, 1e-6)
                logger.info(
                    "Wrote %d frames (%.2f fps effective, %d detections)",
                    total_written,
                    total_written / elapsed,
                    total_detections,
                )

    if writer is None:
        raise RuntimeError("No readable frames were processed; video was not created")

    writer.release()
    elapsed = max(time.perf_counter() - start, 1e-6)
    logger.info(
        "Done. Wrote %d frames to %s (%.2f fps effective, %d detections)",
        total_written,
        output_path,
        total_written / elapsed,
        total_detections,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize full evaluation clips as one ByteTrack-annotated video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/val_best.yaml", help="Best pipeline config YAML")
    parser.add_argument("--dataset", default="data/coco_tl", help="COCO-format dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to render")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--clip", action="append", help="Clip name to render; repeat to stitch selected clips")
    parser.add_argument("--list-clips", action="store_true", help="List available clips and exit")
    parser.add_argument("--device", default=None, help="Override device for both detector and classifier")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS; defaults to tracker frame_rate")
    parser.add_argument("--codec", default="mp4v", help="Four-character OpenCV video codec")
    parser.add_argument("--output-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--draw-frame-label", action="store_true", help="Draw clip/frame text in the top-left corner")
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Debug option; omit to render full clips",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Log progress every N written frames")
    args = parser.parse_args(argv)

    if len(args.codec) != 4:
        parser.error("--codec must be exactly four characters")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_frames_per_clip is not None and args.max_frames_per_clip <= 0:
        parser.error("--max-frames-per-clip must be positive")
    if args.output_size is not None and (args.output_size[0] <= 0 or args.output_size[1] <= 0):
        parser.error("--output-size values must be positive")

    config_path = _project_path(args.config)
    dataset_path = _project_path(args.dataset)

    clips = _load_split_clips(dataset_path, args.split)
    if not clips:
        raise RuntimeError(f"No frames found for split '{args.split}' in {dataset_path}")

    if args.list_clips:
        for name, frames in clips.items():
            print(f"{name}: {len(frames)} frames")
        return

    config = load_config(config_path)
    _resolve_config_paths(config)
    _configure_devices(config, args.device)
    _require_runtime_files(config)

    clip_names = args.clip or sorted(clips)
    unknown = [name for name in clip_names if name not in clips]
    if unknown:
        parser.error(f"Unknown clip(s): {', '.join(unknown)}. Use --list-clips to inspect available clips.")

    fps = args.fps if args.fps is not None else float(config.tracker.frame_rate)
    output_size = tuple(args.output_size) if args.output_size is not None else None
    output_path = args.output or Path(f"runs/viz/eval_{args.split}_bytetrack.mp4")

    logger.info("Config: %s", config_path)
    logger.info("Detector weights: %s", config.detector.model_path)
    logger.info("Classifier weights: %s", config.classifier.model_path)
    logger.info("Split: %s | clips: %s", args.split, ", ".join(clip_names))

    render_video(
        config=config,
        clips=clips,
        clip_names=clip_names,
        output_path=output_path,
        fps=fps,
        codec=args.codec,
        output_size=output_size,
        draw_frame_label=args.draw_frame_label,
        max_frames_per_clip=args.max_frames_per_clip,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
