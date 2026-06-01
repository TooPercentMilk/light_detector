"""Inspect COCO images by title with ground-truth bounding boxes.

By default this opens an interactive prompt. Enter an image filename such as
``daySequence1--00000.jpg`` or the stem ``daySequence1--00000`` to display the
image with GT boxes drawn.

Examples:

    python scripts/inspect_image_ground_truth.py

    python scripts/inspect_image_ground_truth.py --split train

    python scripts/inspect_image_ground_truth.py --image daySequence1--00000.jpg

    python scripts/inspect_image_ground_truth.py --image daySequence1--00000.jpg \
        --no-display --output-dir runs/inspect_gt
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TAG_TO_COLOR: dict[str, tuple[int, int, int]] = {
    "go": (0, 190, 0),
    "goLeft": (0, 190, 0),
    "goForward": (0, 190, 0),
    "stop": (0, 0, 255),
    "stopLeft": (0, 0, 255),
    "warning": (0, 210, 255),
    "warningLeft": (0, 210, 255),
}
DEFAULT_BOX_COLOR = (255, 170, 0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int | None = None
    height: int | None = None


def _safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)


def _clean_title(value: str) -> str:
    return value.strip().strip("\"'")


def _default_annotation_path(dataset: Path, split: str) -> Path:
    return dataset / "annotations" / f"instances_{split}.json"


def _default_image_dir(dataset: Path, split: str) -> Path:
    return dataset / split


def _load_coco(
    annotation_path: Path,
) -> tuple[list[ImageRecord], dict[int, list[dict[str, Any]]], dict[int, str]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images: list[ImageRecord] = []
    for image in payload.get("images", []):
        images.append(
            ImageRecord(
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                width=int(image["width"]) if image.get("width") is not None else None,
                height=int(image["height"]) if image.get("height") is not None else None,
            )
        )

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in payload.get("categories", [])
    }

    return images, annotations_by_image, categories


def _title_keys(file_name: str) -> set[str]:
    path = Path(file_name)
    return {
        file_name.casefold(),
        path.name.casefold(),
        path.stem.casefold(),
    }


def _build_title_index(images: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    index: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        for key in _title_keys(image.file_name):
            index[key].append(image)
    return index


def _resolve_image_title(
    title: str,
    title_index: dict[str, list[ImageRecord]],
) -> tuple[ImageRecord | None, str | None]:
    query = _clean_title(title)
    if not query:
        return None, "Enter an image title, filename, or stem."

    matches = title_index.get(query.casefold(), [])
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        options = ", ".join(image.file_name for image in matches[:8])
        return None, f"Ambiguous image title; matches include: {options}"
    return None, None


def _suggest_titles(title: str, images: list[ImageRecord], limit: int = 8) -> list[str]:
    query = _clean_title(title).casefold()
    if not query:
        return []

    contains = [
        image.file_name
        for image in images
        if query in image.file_name.casefold() or query in Path(image.file_name).stem.casefold()
    ]
    if contains:
        return contains[:limit]

    lowered_to_title = {image.file_name.casefold(): image.file_name for image in images}
    close = difflib.get_close_matches(query, list(lowered_to_title), n=limit, cutoff=0.25)
    return [lowered_to_title[value] for value in close]


def _image_path(image_dir: Path, file_name: str) -> Path:
    direct_path = image_dir / file_name
    if direct_path.is_file():
        return direct_path

    basename_path = image_dir / Path(file_name).name
    if basename_path.is_file():
        return basename_path

    return direct_path


def _clip_box_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x, y, w, h = (float(v) for v in bbox[:4])
    x1 = max(0, min(image_width - 1, int(round(x))))
    y1 = max(0, min(image_height - 1, int(round(y))))
    x2 = max(0, min(image_width - 1, int(round(x + w))))
    y2 = max(0, min(image_height - 1, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _annotation_tag(annotation: dict[str, Any]) -> str:
    attributes = annotation.get("attributes")
    if isinstance(attributes, dict):
        tag = attributes.get("lisa_tag")
        if tag:
            return str(tag)
    return ""


def _annotation_label(annotation: dict[str, Any], categories: dict[int, str]) -> str:
    tag = _annotation_tag(annotation)
    category = categories.get(int(annotation.get("category_id", -1)), "object")
    label = tag or category
    if annotation.get("id") is not None:
        label = f"{label} #{annotation['id']}"
    return label


def _draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    if not text:
        return

    image_height, image_width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    x, y = anchor
    y = max(text_height + baseline + 4, y)
    right = min(image_width - 1, x + text_width + 6)
    top = max(0, y - text_height - baseline - 5)
    bottom = min(image_height - 1, y + baseline + 2)

    cv2.rectangle(image, (x, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_ground_truth(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    categories: dict[int, str],
) -> np.ndarray:
    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]
    thickness = max(1, round(min(image_width, image_height) / 500))

    for annotation in annotations:
        box = _clip_box_xywh(annotation["bbox"], image_width, image_height)
        if box is None:
            continue

        tag = _annotation_tag(annotation)
        color = TAG_TO_COLOR.get(tag, DEFAULT_BOX_COLOR)
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        _draw_label(annotated, _annotation_label(annotation, categories), (x1, y1 - 4), color)

    return annotated


def _save_overlay(image: np.ndarray, image_record: ImageRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(Path(image_record.file_name).stem)}__gt.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


def _show_image(window_title: str, image: np.ndarray) -> None:
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, image)
    logger.info("Press any key in the image window to continue.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)


def inspect_image(
    title: str,
    images: list[ImageRecord],
    title_index: dict[str, list[ImageRecord]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    image_dir: Path,
    output_dir: Path | None,
    display: bool,
) -> bool:
    image_record, error = _resolve_image_title(title, title_index)
    if error is not None:
        logger.error(error)
        return False
    if image_record is None:
        logger.error("Image title not found: %s", _clean_title(title))
        suggestions = _suggest_titles(title, images)
        if suggestions:
            logger.info("Closest matches: %s", ", ".join(suggestions))
        return False

    path = _image_path(image_dir, image_record.file_name)
    image = cv2.imread(str(path))
    if image is None:
        logger.error("Could not read image file: %s", path)
        return False

    annotations = annotations_by_image.get(image_record.image_id, [])
    annotated = draw_ground_truth(image, annotations, categories)
    logger.info(
        "%s | image_id=%d | GT boxes=%d",
        image_record.file_name,
        image_record.image_id,
        len(annotations),
    )

    if output_dir is not None:
        saved_path = _save_overlay(annotated, image_record, output_dir)
        logger.info("Saved overlay: %s", saved_path)

    if display:
        try:
            _show_image(f"GT {image_record.file_name}", annotated)
        except cv2.error as exc:
            logger.error("OpenCV could not display the image window: %s", exc)
            return False

    return True


def _prompt_loop(
    images: list[ImageRecord],
    title_index: dict[str, list[ImageRecord]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    image_dir: Path,
    output_dir: Path | None,
    display: bool,
) -> int:
    print(f"Loaded {len(images)} images from {image_dir}")
    print("Enter an image title/filename/stem, or q to quit.")

    while True:
        try:
            title = input("image> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        title = title.strip()
        if title.casefold() in {"q", "quit", "exit"}:
            return 0
        if not title:
            continue

        inspect_image(
            title=title,
            images=images,
            title_index=title_index,
            annotations_by_image=annotations_by_image,
            categories=categories,
            image_dir=image_dir,
            output_dir=output_dir,
            display=display,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prompt for a COCO image title and display it with GT bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/coco_tl"), help="COCO dataset root")
    parser.add_argument("--split", default="val", help="Dataset split to inspect")
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=None,
        help="COCO annotation JSON. Defaults to dataset/annotations/instances_<split>.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Image directory. Defaults to dataset/<split>",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image title/filename/stem to inspect once. If omitted, starts the prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save annotated overlays.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV window. Useful when only saving overlays.",
    )
    args = parser.parse_args(argv)

    annotation_path = args.ann_file or _default_annotation_path(args.dataset, args.split)
    image_dir = args.image_dir or _default_image_dir(args.dataset, args.split)

    if not annotation_path.is_file():
        parser.error(f"Annotation file not found: {annotation_path}")
    if not image_dir.is_dir():
        parser.error(f"Image directory not found: {image_dir}")

    display = not args.no_display
    output_dir = args.output_dir
    if not display and output_dir is None:
        output_dir = Path("runs/inspect_gt")

    images, annotations_by_image, categories = _load_coco(annotation_path)
    title_index = _build_title_index(images)

    logger.info("Annotations: %s", annotation_path)
    logger.info("Images: %s", image_dir)

    if args.image is not None:
        return 0 if inspect_image(
            title=args.image,
            images=images,
            title_index=title_index,
            annotations_by_image=annotations_by_image,
            categories=categories,
            image_dir=image_dir,
            output_dir=output_dir,
            display=display,
        ) else 1

    return _prompt_loop(
        images=images,
        title_index=title_index,
        annotations_by_image=annotations_by_image,
        categories=categories,
        image_dir=image_dir,
        output_dir=output_dir,
        display=display,
    )


if __name__ == "__main__":
    sys.exit(main())
