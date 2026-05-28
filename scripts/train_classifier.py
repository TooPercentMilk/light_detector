"""Train the traffic-light state classifier (MobileNetV3-Small).

Usage::

    python scripts/train_classifier.py --dataset data/coco_tl --epochs 30
    python scripts/train_classifier.py --dataset data/coco_tl --epochs 50 --batch-size 128 --lr 3e-4 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train traffic-light state classifier")
    parser.add_argument("--dataset", required=True, help="Path to COCO-format dataset (e.g. data/coco_tl)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/train_classifier")
    parser.add_argument("--input-size", nargs=2, type=int, default=[32, 64], metavar=("W", "H"))
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet-pretrained backbone")
    parser.add_argument("--resume", default=None, help="Resume training from checkpoint")
    args = parser.parse_args(argv)

    from adas_perception.traffic_light.state.trainer import ClassifierTrainer

    trainer = ClassifierTrainer(
        num_classes=4,
        input_size=tuple(args.input_size),
        output_dir=args.output_dir,
        pretrained_backbone=not args.no_pretrained,
    )
    trainer.train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
