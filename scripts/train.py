"""Unified training script for the traffic-light detection pipeline.

Trains the YOLOX detector, the state classifier, or both sequentially.

Usage::

    # Train both (detector first, then classifier)
    python scripts/train.py --target both --dataset data/coco_tl

    # Detector only
    python scripts/train.py --target detector --dataset data/coco_tl --det-epochs 50

    # Classifier only
    python scripts/train.py --target classifier --dataset data/coco_tl --cls-epochs 30
"""

from __future__ import annotations

import argparse
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


def train_detector(args: argparse.Namespace) -> Path:
    """Train the YOLOX traffic-light detector. Returns the final weights path."""
    from adas_perception.traffic_light.detector.trainer import YoloxTrainer

    output_dir = args.output_dir / "detector"
    logger.info("=" * 60)
    logger.info("DETECTOR TRAINING")
    logger.info("=" * 60)

    trainer = YoloxTrainer(
        model_config={
            "num_classes": args.det_num_classes,
            "input_size": args.det_input_size,
            "device": args.device or ("cuda" if _cuda_available() else "cpu"),
            "pretrained_ckpt": args.det_pretrained,
            "exp_name": args.det_exp_name,
            "data_num_workers": args.num_workers,
        },
        output_dir=str(output_dir),
    )
    trainer.train(
        dataset_path=args.dataset,
        epochs=args.det_epochs,
        batch_size=args.det_batch_size,
        lr=args.det_lr,
        patience=args.det_patience,
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
    det.add_argument("--det-batch-size", type=int, default=16)
    det.add_argument("--det-lr", type=float, default=1e-3)
    det.add_argument("--det-num-classes", type=int, default=1)
    det.add_argument("--det-input-size", nargs=2, type=int, default=[640, 640], metavar=("W", "H"))
    det.add_argument("--det-pretrained", default="weights/yolox_m.pth", help="Backbone checkpoint for fine-tuning")
    det.add_argument("--det-exp-name", default="yolox-m", help="YOLOX experiment variant (must match --det-pretrained)")
    det.add_argument("--det-patience", type=int, default=5, help="Early stopping patience (0 = disabled)")

    # ---- classifier ----
    cls = parser.add_argument_group("classifier")
    cls.add_argument("--cls-epochs", type=int, default=30)
    cls.add_argument("--cls-batch-size", type=int, default=64)
    cls.add_argument("--cls-lr", type=float, default=3e-4)
    cls.add_argument("--cls-num-classes", type=int, default=4)
    cls.add_argument("--cls-input-size", nargs=2, type=int, default=[32, 64], metavar=("W", "H"))
    cls.add_argument("--cls-no-pretrained", action="store_true", help="Skip ImageNet-pretrained backbone")
    cls.add_argument("--cls-patience", type=int, default=5, help="Early stopping patience (0 = disabled)")

    args = parser.parse_args(argv)

    if args.target in ("detector", "both"):
        train_detector(args)

    if args.target in ("classifier", "both"):
        train_classifier(args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
