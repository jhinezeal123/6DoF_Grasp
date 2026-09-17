#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octo.data.stage_a_statistics import save_stage_a_statistics_bundle
from octo.robot.stage_a_visualization import save_raw_dataset_visualization_bundle


def _parse_args():
    ap = argparse.ArgumentParser(description="Compute/export canonical Stage-A normalization statistics, raw dataset summary, and optional 3D visualization")
    ap.add_argument("--builder_data_dir", required=True, help="TFDS builder data dir")
    ap.add_argument("--output_dir", default=None, help="Optional output directory for exported JSON files")
    ap.add_argument("--force_recompute", action="store_true", help="Force recompute even if cached statistics JSON exists")
    ap.add_argument("--config", default=None, help="Optional robot config JSON for 3D visualization export")
    ap.add_argument("--skip_visualize_3d", action="store_true", help="Skip 3D export even when --config is provided")
    return ap.parse_args()


def main():
    args = _parse_args()
    exported = save_stage_a_statistics_bundle(
        builder_data_dir=args.builder_data_dir,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
    )
    print(f"Statistics JSON: {exported['statistics_json']}")
    print(f"Summary JSON: {exported['summary_json']}")
    if args.config and not args.skip_visualize_3d:
        builder_data_dir = Path(args.builder_data_dir).resolve()
        raw_dir = builder_data_dir / "downloads" / "manual" / "raw"
        viz = save_raw_dataset_visualization_bundle(
            robot_config_path=args.config,
            raw_dir=raw_dir,
            output_dir=args.output_dir or builder_data_dir,
        )
        print(f"3D visualization PNG: {viz['visualization_png']}")
        print(f"3D visualization stats JSON: {viz['statistics_json']}")


if __name__ == "__main__":
    main()
