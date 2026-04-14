"""Quick validation of the converted COCO dataset."""
import json
from collections import Counter
from pathlib import Path

out = Path("data/coco_tl")

for split in ["train", "val", "test"]:
    with open(out / "annotations" / f"instances_{split}.json") as f:
        d = json.load(f)

    tags = Counter(a["attributes"]["lisa_tag"] for a in d["annotations"])
    n_img = len(d["images"])
    n_ann = len(d["annotations"])
    img_ids_with_ann = set(a["image_id"] for a in d["annotations"])
    n_neg = n_img - len(img_ids_with_ann)

    # Count actual image files
    img_dir = out / split
    n_files = len(list(img_dir.glob("*.jpg")))

    print(f"=== {split} ===")
    print(f"  JSON images: {n_img}, disk files: {n_files}, match: {n_img == n_files}")
    print(f"  Annotations: {n_ann}")
    print(f"  Negative (unannotated) images: {n_neg}")
    print(f"  Tags: {dict(tags)}")

    # Spot check
    if d["annotations"]:
        a = d["annotations"][0]
        print(f"  Sample: id={a['id']}, img_id={a['image_id']}, bbox={a['bbox']}, tag={a['attributes']['lisa_tag']}")

    # Bbox validity
    bad = [a for a in d["annotations"] if a["bbox"][2] <= 0 or a["bbox"][3] <= 0]
    print(f"  Invalid bboxes: {len(bad)}")

    # ID uniqueness
    img_ids = [i["id"] for i in d["images"]]
    ann_ids = [a["id"] for a in d["annotations"]]
    print(f"  Unique image IDs: {len(set(img_ids))} / {len(img_ids)}")
    print(f"  Unique ann IDs: {len(set(ann_ids))} / {len(ann_ids)}")
    print()
