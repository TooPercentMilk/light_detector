"""Clone / update external repos that are not pip-installable.

Usage::

    python setup_external.py          # clone if missing, pull if present
    python setup_external.py --clean  # remove all external repos

----

**Do I need this?**

ByteTrack (the tracker) is already bundled inside the pip-installable YOLOX
package (``yolox.tracker.byte_tracker``), so if you have run::

    pip install -e ".[models]"

then ``setup_external.py`` is **not required** for normal inference.

Clone the external ByteTrack repo only if you need the *latest standalone*
ByteTrack version, which may contain improvements not yet merged into YOLOX.
The ``bytetrack_wrapper`` will always prefer the pip-installed version and
only fall back to ``external/ByteTrack`` if the pip package is absent.

----

Repos are cloned into ``external/<name>`` and pinned to a specific ref so
builds are reproducible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXTERNAL_DIR = Path(__file__).resolve().parent / "external"

# ── repos to manage ──────────────────────────────────────────────────
# Each entry: (directory name, git URL, commit/tag to pin)
REPOS: list[tuple[str, str, str]] = [
    (
        "ByteTrack",
        "https://github.com/ifzhang/ByteTrack.git",
        "main",  # pin to a specific commit hash for reproducibility
    ),
]


def _run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def clone_or_update() -> None:
    EXTERNAL_DIR.mkdir(exist_ok=True)

    for name, url, ref in REPOS:
        dest = EXTERNAL_DIR / name

        if dest.exists():
            print(f"[update] {name} — pulling latest for ref '{ref}' ...")
            _run(["git", "fetch", "--all"], cwd=str(dest))
            _run(["git", "checkout", ref], cwd=str(dest))
            _run(["git", "pull", "--ff-only"], cwd=str(dest))
        else:
            print(f"[clone]  {name} — cloning into external/{name} ...")
            _run(["git", "clone", url, str(dest)])
            _run(["git", "checkout", ref], cwd=str(dest))

        print(f"  ✓ {name} ready at {dest}\n")


def clean() -> None:
    for name, _, _ in REPOS:
        dest = EXTERNAL_DIR / name
        if dest.exists():
            print(f"[clean]  removing {dest} ...")
            shutil.rmtree(dest)

    # Remove the external dir itself if empty
    if EXTERNAL_DIR.exists() and not any(EXTERNAL_DIR.iterdir()):
        EXTERNAL_DIR.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage external repos.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all cloned external repos.",
    )
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        clone_or_update()


if __name__ == "__main__":
    main()
