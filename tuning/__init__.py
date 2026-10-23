"""Practice-mode Optuna tuner for Minos GATK HaplotypeCaller.

This package calls ``bash start-miner.sh --practice``. It does not submit a
live round and it does not write ``configs/gatk.conf``.
"""

from minos_layout import ensure_minos_importable

ensure_minos_importable()

__version__ = "0.1.0"
