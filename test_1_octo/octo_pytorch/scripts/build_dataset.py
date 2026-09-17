#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow_datasets as tfds

import octo.data.stage_a_tfds  # noqa: F401
from octo.data.stage_a_statistics import save_stage_a_statistics_bundle
from octo.robot.stage_a_visualization import save_raw_dataset_visualization_bundle

# refferences:
# /home/ceec/khoanhd/VLAR_Lab/bridge_data_robot
# /home/ceec/khoanhd/VLAR_Lab/bridge_data_v2
# /home/ceec/khoanhd/VLAR_Lab/droid
# /home/ceec/khoanhd/VLAR_Lab/droid_policy_learning
# /home/ceec/khoanhd/VLABenchMark_Lab/openvla


def _parse_args():
    ap = argparse.ArgumentParser(description="Build TFDS/RLDS dataset + statistics + optional 3D visualization from raw episodes")
    ap.add_argument("--raw_dir", required=True, help="Directory containing raw/episode_xxxxxx folders")
    ap.add_argument("--builder_data_dir", required=True, help="Directory where TFDS builder outputs will be written")
    ap.add_argument(
        "--builder_name",
        default="myarm_stage_a_dataset",
        help=(
            "TFDS builder name (optionally with config). "
            "Stage-A configs: myarm_stage_a_dataset (default=commanded), "
            "myarm_stage_a_dataset/commanded_no_noops, myarm_stage_a_dataset/commanded_invert_gripper, "
            "myarm_stage_a_dataset/commanded_invert_gripper_no_noops, myarm_stage_a_dataset/achieved, myarm_stage_a_dataset/achieved_no_noops."
        ),
    )
    ap.add_argument("--overwrite", action="store_true", help="Remove existing builder data dir before prepare")
    ap.add_argument("--skip_statistics", action="store_true", help="Skip dataset statistics export")
    ap.add_argument("--force_recompute_statistics", action="store_true", help="Force recompute stats")
    ap.add_argument("--summary_only", action="store_true", help="Only export statistics/summary from an existing builder dir")
    ap.add_argument("--config", default=None, help="Optional robot config JSON. When provided, export a 3D PNG + JSON using workspace bounds.")
    ap.add_argument("--skip_visualize_3d", action="store_true", help="Skip 3D PNG/JSON export even when --config is provided")
    return ap.parse_args()


def main():
    args = _parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    builder_data_dir = Path(args.builder_data_dir).resolve()
    manual_raw_dir = builder_data_dir / "downloads" / "manual" / "raw"

    if not args.summary_only:
        manual_dir = builder_data_dir / "downloads" / "manual"
        if args.overwrite and builder_data_dir.exists():
            shutil.rmtree(builder_data_dir)
        manual_raw_dir.mkdir(parents=True, exist_ok=True)
        for child in list(manual_raw_dir.iterdir()):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        for episode_dir in sorted(raw_dir.glob("episode_*")):
            target = manual_raw_dir / episode_dir.name
            try:
                target.symlink_to(episode_dir, target_is_directory=True)
            except Exception:
                shutil.copytree(episode_dir, target)
        builder = tfds.builder(args.builder_name, data_dir=str(builder_data_dir))
        builder.download_and_prepare()
        print(f"Prepared TFDS dataset: {builder.info.full_name}")
        print(f"Builder data dir: {builder_data_dir}")

    if not args.skip_statistics:
        exported = save_stage_a_statistics_bundle(
            builder_data_dir=builder_data_dir,
            builder_name=args.builder_name,
            force_recompute=args.force_recompute_statistics,
        )
        print(f"Statistics JSON: {exported['statistics_json']}")
        print(f"Summary JSON: {exported['summary_json']}")

    if args.config and not args.skip_visualize_3d:
        viz = save_raw_dataset_visualization_bundle(
            robot_config_path=args.config,
            raw_dir=manual_raw_dir if manual_raw_dir.exists() else raw_dir,
            output_dir=builder_data_dir,
        )
        print(f"3D visualization PNG: {viz['visualization_png']}")
        print(f"3D visualization stats JSON: {viz['statistics_json']}")


if __name__ == "__main__":
    main()
