#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octo.robot.stage_a_naming import parse_episode_index
from octo.robot.stage_a_visualization import save_waypoint_visualization_bundle


def _parse_args():
    ap = argparse.ArgumentParser(description="Visualize Stage-A waypoint folders in 3D and export statistics JSON")
    ap.add_argument("--config", required=True, help="Robot/camera JSON config with workspace bounds")
    ap.add_argument("--waypoint_dir", required=True, help="Directory containing normalized waypoint JSON files")
    ap.add_argument("--output_dir", required=True, help="Directory where PNG + JSON outputs will be written")
    ap.add_argument("--waypoint_glob", default="episode_*_wpts.json", help="Waypoint file glob. Files must still follow episode_xxxxxx_wpts.json")
    ap.add_argument("--start_episode", "--strat_episode", dest="start_episode", default=None, help="Inclusive start episode selector")
    ap.add_argument("--end_episode", default=None, help="Inclusive end episode selector")
    return ap.parse_args()


def main():
    args = _parse_args()
    exported = save_waypoint_visualization_bundle(
        robot_config_path=args.config,
        waypoint_dir=args.waypoint_dir,
        output_dir=args.output_dir,
        waypoint_glob=args.waypoint_glob,
        start_episode=parse_episode_index(args.start_episode),
        end_episode=parse_episode_index(args.end_episode),
    )
    print(f"3D visualization PNG: {exported['visualization_png']}")
    print(f"3D statistics JSON: {exported['statistics_json']}")


if __name__ == "__main__":
    main()
