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


def _parse_args():
    ap = argparse.ArgumentParser(description="Convert Stage-A raw episodes into a TFDS/RLDS dataset")
    ap.add_argument("--raw_dir", required=True, help="Directory containing raw/episode_xxxxxx folders")
    ap.add_argument("--builder_data_dir", required=True, help="Directory where TFDS builder outputs will be written")
    ap.add_argument("--overwrite", action="store_true", help="Remove existing builder data dir before prepare")
    ap.add_argument("--skip_statistics", action="store_true", help="Skip exporting canonical Stage-A statistics JSON after TFDS prepare")
    ap.add_argument("--force_recompute_statistics", action="store_true", help="Force recompute dataset statistics even if cached JSON exists")
    return ap.parse_args()


def main():
    args = _parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    builder_data_dir = Path(args.builder_data_dir).resolve()
    manual_dir = builder_data_dir / "downloads" / "manual"
    manual_raw_dir = manual_dir / "raw"

    if args.overwrite and builder_data_dir.exists():
        shutil.rmtree(builder_data_dir)
    manual_raw_dir.mkdir(parents=True, exist_ok=True)

    if manual_raw_dir.exists():
        for child in manual_raw_dir.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    for episode_dir in sorted(raw_dir.glob("episode_*")):
        target = manual_raw_dir / episode_dir.name
        if target.exists():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        try:
            target.symlink_to(episode_dir, target_is_directory=True)
        except Exception:
            shutil.copytree(episode_dir, target)

    builder = tfds.builder("myarm_stage_a_dataset", data_dir=str(builder_data_dir))
    builder.download_and_prepare()
    print(f"Prepared TFDS dataset: {builder.info.full_name}")
    print(f"Builder data dir: {builder_data_dir}")

    if not args.skip_statistics:
        exported = save_stage_a_statistics_bundle(
            builder_data_dir=builder_data_dir,
            force_recompute=args.force_recompute_statistics,
        )
        print(f"Statistics JSON: {exported['statistics_json']}")
        print(f"Summary JSON: {exported['summary_json']}")


if __name__ == "__main__":
    main()
