"""Compatibility entry point for the corrected ordered-annotation evaluator.

The old image_id-based evaluator was invalid for RC-3D parts 2-4.
Its historical source is frozen in audit/failure_analysis/original_source.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rc3d_author_order import main

if __name__ == '__main__':
    main()
