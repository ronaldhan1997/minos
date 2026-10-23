"""Locate this tuning project and the Minos checkout it calls.

The tuner lives in ``minos_subnet/minos_tuning``. The parent directory is the
miner and validator. ``MINOS_REPO_ROOT`` overrides that when the checkout is
somewhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def minos_root() -> Path:
    """Directory that contains ``start-miner.sh`` and ``templates/``."""
    override = os.environ.get("MINOS_REPO_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        parent = project_root().parent
        if (parent / "start-miner.sh").is_file() and (parent / "templates").is_dir():
            root = parent
        else:
            root = (parent / "minos_subnet").resolve()
    if not (root / "start-miner.sh").is_file() or not (root / "templates").is_dir():
        raise RuntimeError(
            f"Minos checkout not found at {root}. "
            "Set MINOS_REPO_ROOT to the directory that contains start-miner.sh."
        )
    return root


def ensure_minos_importable() -> Path:
    """Put the Minos checkout on ``sys.path`` so ``templates`` and ``utils`` import."""
    root = minos_root()
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root
