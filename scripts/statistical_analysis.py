"""Compatibility wrapper for statistical comparison analysis.

Usage:
    python scripts/statistical_analysis.py results/<experiment_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_results import main


if __name__ == "__main__":
    main()
