"""Unified training script for the traffic-light detection pipeline.

Trains the YOLOX detector, the state classifier, or both sequentially.

Usage::

    # Train both (detector first, then classifier)
    python scripts/train.py --target both --dataset data/coco_tl

    # Detector only
    python scripts/train.py --target detector --dataset data/coco_tl --det-epochs 50

    # Detector only, with augmentations disabled
    python scripts/train.py --target detector --dataset data/coco_tl \
        --det-no-augment

    # Detector only, using only images that contain at least one traffic light
    python scripts/train.py --target detector --dataset data/coco_tl \
        --det-positive-images-only

    # Detector only, append runtime hflip samples for images with 8-24px lights
    python scripts/train.py --target detector --dataset data/coco_tl \
        --det-small-light-flip

    # Detector only, train on only the top half of each image
    python scripts/train.py --target detector --dataset data/coco_tl \
        --det-top-half-only

    # Resume detector training from latest full checkpoint
    python scripts/train.py --target detector --dataset data/coco_tl \
        --det-resume runs/train/detector/latest.pth --det-epochs 80

    # Classifier only
    python scripts/train.py --target classifier --dataset data/coco_tl --cls-epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
DEFAULT_DETECTOR_IMAGE_SIZE = 960


def _infer_detector_num_classes(dataset_path: str | Path) -> int | None:
    ann_path = Path(dataset_path) / "annotations" / "instances_train.json"
    if not ann_path.is_file():
        return None
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco.get("categories", [])
    if not categories:
        return None
    return len(categories)


def train_detector(args: argparse.Namespace) -> Path:
    """Train the YOLOX traffic-light detector. Returns the final weights path."""
    from adas_perception.traffic_light.detector.trainer import YoloxTrainer

    output_dir = args.output_dir / "detector"
    logger.info("=" * 60)
    logger.info("DETECTOR TRAINING")
    logger.info("=" * 60)

    pretrained = args.det_pretrained
    if args.det_resume:
        logger.info("Resuming detector training from checkpoint: %s", args.det_resume)

    dev = args.device or ("cuda" if _cuda_available() else "cpu")
    det_num_classes = args.det_num_classes
    if det_num_classes is None:
        det_num_classes = _infer_detector_num_classes(args.dataset) or 1
        logger.info("Inferred detector class count from dataset: %d", det_num_classes)
    mosaic_prob = 1.0
    scale_jitter_range = (0.5, 1.5)
    flip_prob = 0.5
    if args.det_hsv_only:
        mosaic_prob = 0.0
        scale_jitter_range = (1.0, 1.0)
        flip_prob = 0.0

    trainer = YoloxTrainer(
        model_config={
            "num_classes": det_num_classes,
            "input_size": args.det_input_size,
            "device": dev,
            "pretrained_ckpt": pretrained,
            "resume_ckpt": args.det_resume,
            "exp_name": args.det_exp_name,
            "data_num_workers": args.num_workers,
            "conf_threshold": 0.05,
            "nms_threshold": 0.45,
            "mosaic_prob": mosaic_prob,
            "scale_jitter_range": scale_jitter_range,
            "flip_prob": flip_prob,
            "small_light_flip": args.det_small_light_flip,
            "small_light_flip_range": tuple(args.det_small_light_flip_range),
            "top_crop_only": args.det_top_crop_only,
            "top_crop_fraction": args.det_top_crop_fraction,
        },
        output_dir=str(output_dir),
    )

    # Auto-detect classifier weights for per-epoch e2e accuracy monitoring.
    classifier_best = args.output_dir / "classifier" / "best.pth"
    classifier_config = None
    if classifier_best.is_file():
        classifier_config = {"model_path": str(classifier_best), "device": dev}
        logger.info("Will monitor e2e accuracy using classifier: %s", classifier_best)

    trainer.train(
        dataset_path=args.dataset,
        epochs=args.det_epochs,
        batch_size=args.det_batch_size,
        lr=args.det_lr,
        patience=args.det_patience,
        no_mosaic_epochs=args.det_no_mosaic_epochs,
        augment=not args.det_no_augment,
        positive_images_only=args.det_positive_images_only,
        val_every=args.det_val_every,
        classifier_config=classifier_config,
    )

    final = output_dir / "yolox_tl.pth"
    if args.install_weights and final.is_file():
        dst = WEIGHTS_DIR / "yolox_tl.pth"
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy2(final, dst)
        logger.info("Installed detector weights -> %s", dst)

    return final


def train_classifier(args: argparse.Namespace) -> Path:
    """Train the MobileNetV3 state classifier. Returns the final weights path."""
    from adas_perception.traffic_light.state.trainer import ClassifierTrainer

    output_dir = args.output_dir / "classifier"
    logger.info("=" * 60)
    logger.info("CLASSIFIER TRAINING")
    logger.info("=" * 60)

    trainer = ClassifierTrainer(
        num_classes=args.cls_num_classes,
        input_size=tuple(args.cls_input_size),
        output_dir=str(output_dir),
        pretrained_backbone=not args.cls_no_pretrained,
    )
    trainer.train(
        dataset_path=args.dataset,
        epochs=args.cls_epochs,
        batch_size=args.cls_batch_size,
        lr=args.cls_lr,
        device=args.device,
        num_workers=args.num_workers,
        patience=args.cls_patience,
        resume_checkpoint=args.cls_resume,
    )

    final = output_dir / "tl_state_classifier.pth"
    if args.install_weights and final.is_file():
        dst = WEIGHTS_DIR / "tl_state_classifier.pth"
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy2(final, dst)
        logger.info("Installed classifier weights -> %s", dst)

    return final


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train traffic-light detector, classifier, or both",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- global ----
    parser.add_argument(
        "--target",
        choices=["detector", "classifier", "both"],
        required=True,
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to COCO-format dataset directory (e.g. data/coco_tl)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    parser.add_argument(
        "--install-weights",
        action="store_true",
        help="Copy final weights into weights/ after training",
    )

    # ---- detector ----
    det = parser.add_argument_group("detector")
    det.add_argument("--det-epochs", type=int, default=50)
    det.add_argument("--det-batch-size", type=int, default=8)
    det.add_argument("--det-lr", type=float, default=1e-3)
    det.add_argument(
        "--det-num-classes",
        type=int,
        default=None,
        help="Detector class count; defaults to len(dataset annotations categories)",
    )
    det.add_argument(
        "--det-image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        help="Square detector/preprocessor image size",
    )
    det.add_argument(
        "--det-input-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Detector/preprocessor input size; overrides --det-image-size",
    )
    det.add_argument("--det-pretrained", default="weights/yolox_m.pth", help="Backbone checkpoint for fine-tuning")
    det.add_argument("--det-resume", default=None, help="Resume training from checkpoint (overrides --det-pretrained)")
    det.add_argument("--det-exp-name", default="yolox-m", help="YOLOX experiment variant (must match --det-pretrained)")
    det.add_argument("--det-patience", type=int, default=10, help="Early stopping patience (0 = disabled)")
    det.add_argument("--det-no-mosaic-epochs", type=int, default=50, help="Disable mosaic for the final N epochs (0 = always on)")
    det.add_argument(
        "--det-no-augment",
        action="store_true",
        help="Disable standard stochastic detector augmentation (mosaic, scale jitter, HSV jitter, and random flips)",
    )
    det.add_argument(
        "--det-hsv-only",
        action="store_true",
        help="Use detector HSV jitter only; disable mosaic, scale jitter/crop, and random flips",
    )
    det.add_argument(
        "--det-positive-images-only",
        action="store_true",
        help="Use only training images that contain at least one traffic-light annotation",
    )
    det.add_argument(
        "--det-small-light-flip",
        action="store_true",
        help=(
            "Append runtime horizontal-flip virtual samples for train images "
            "with at least one traffic light in the configured size range"
        ),
    )
    det.add_argument(
        "--det-small-light-flip-range",
        nargs=2,
        type=float,
        default=[8.0, 24.0],
        metavar=("MIN_PX", "MAX_PX"),
        help="Inclusive/exclusive sqrt(area) pixel range for --det-small-light-flip",
    )
    det.add_argument(
        "--det-val-every",
        type=int,
        default=1,
        help="Run val mAP evaluation every N epochs for early stopping (0 = use train loss instead)",
    )
    det.add_argument(
        "--det-top-half-only",
        "--det-top-40-only",
        "--det-top-third-only",
        dest="det_top_crop_only",
        action="store_true",
        help="Train with detector inputs cropped to the top half of each image and padded to the YOLO input size",
    )
    det.add_argument(
        "--det-top-crop-fraction",
        type=float,
        default=0.5,
        help="Image-height fraction kept when --det-top-half-only is enabled",
    )

    # ---- classifier ----
    cls = parser.add_argument_group("classifier")
    cls.add_argument("--cls-epochs", type=int, default=30)
    cls.add_argument("--cls-batch-size", type=int, default=64)
    cls.add_argument("--cls-lr", type=float, default=3e-4)
    cls.add_argument("--cls-num-classes", type=int, default=4)
    cls.add_argument("--cls-input-size", nargs=2, type=int, default=[32, 64], metavar=("W", "H"))
    cls.add_argument("--cls-no-pretrained", action="store_true", help="Skip ImageNet-pretrained backbone")
    cls.add_argument("--cls-resume", default=None, help="Resume classifier training from checkpoint")
    cls.add_argument("--cls-patience", type=int, default=5, help="Early stopping patience (0 = disabled)")

    args = parser.parse_args(argv)
    if args.det_image_size <= 0:
        parser.error("--det-image-size must be positive")
    if args.det_input_size is None:
        args.det_input_size = [args.det_image_size, args.det_image_size]
    elif any(v <= 0 for v in args.det_input_size):
        parser.error("--det-input-size values must be positive")
    if args.det_num_classes is not None and args.det_num_classes <= 0:
        parser.error("--det-num-classes must be positive")
    if args.det_small_light_flip_range[0] < 0:
        parser.error("--det-small-light-flip-range MIN_PX must be non-negative")
    if args.det_small_light_flip_range[1] <= args.det_small_light_flip_range[0]:
        parser.error("--det-small-light-flip-range MAX_PX must be greater than MIN_PX")
    if not 0.0 < args.det_top_crop_fraction <= 1.0:
        parser.error("--det-top-crop-fraction must be in the range (0, 1]")

    if args.target in ("detector", "both"):
        train_detector(args)

    if args.target in ("classifier", "both"):
        train_classifier(args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
