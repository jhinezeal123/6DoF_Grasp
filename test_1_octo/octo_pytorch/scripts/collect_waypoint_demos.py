#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octo.robot.cameras import MultiCameraRig
from octo.robot.collector import StageAAbortRequested, collect_stage_a_episode, load_absolute_waypoints
from octo.robot.config import RobotPipelineConfig
from octo.robot.mock_robot import MockMyArmM750Robot
from octo.robot.myarm_m750 import MyArmM750Robot
from octo.robot.stage_a_debug import StageADebugPreview, StageADebugPreviewConfig
from octo.robot.stage_a_naming import format_episode_id, parse_episode_index, parse_waypoint_episode_index


DEPRECATED_VERBOSE_HELP = argparse.SUPPRESS

# python scripts/collect_waypoint_demos.py --config ./scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json --plan_json ./scripts/datasets/myarm_stage_a_collection_plan.example.json --debug --start_episode 000101 --end_episode 000110 --overwrite
def _parse_args():
    ap = argparse.ArgumentParser(description="Collect Stage-A episodes from a directory of normalized waypoint JSON files")
    ap.add_argument("--config", required=True, help="Robot/camera JSON config")
    ap.add_argument("--plan_json", required=True, help="Collection plan JSON")
    ap.add_argument("--mock", action="store_true", help="Force mock robot backend")
    ap.add_argument("--debug", action="store_true", help="Enable debug logs + preview during collection")
    ap.add_argument("--verbose", dest="debug", action="store_true", help=DEPRECATED_VERBOSE_HELP)
    ap.add_argument("--start_episode", "--strat_episode", dest="start_episode", default=None, help="Inclusive start episode selector, e.g. 12 or episode_000012")
    ap.add_argument("--end_episode", default=None, help="Inclusive end episode selector, e.g. 20 or episode_000020")
    ap.add_argument("--overwrite", action="store_true", help="Delete existing episode_xxxxxx folders before recording")
    ap.add_argument("--preview_max_height", type=int, default=240, help="Preview max height when --debug is enabled")
    return ap.parse_args()


def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_robot(cfg: RobotPipelineConfig, force_mock: bool):
    backend = cfg.robot.backend.lower()
    if force_mock or backend in {"mock", "mock_myarm", "mock_myarm_m750"}:
        return MockMyArmM750Robot(cfg.robot, visualize=False)
    if backend in {"myarm", "myarm_m750"}:
        return MyArmM750Robot(cfg.robot)
    raise ValueError(f"Unsupported robot backend: {cfg.robot.backend!r}")


def _resolve_modalities(plan: dict, cfg: RobotPipelineConfig) -> Sequence[str]:
    capture_modalities = plan.get("capture_modalities")
    if isinstance(capture_modalities, list) and capture_modalities:
        return [str(x) for x in capture_modalities]
    return [name for name, cam_cfg in cfg.cameras.items() if bool(cam_cfg.enabled)]


def _make_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("collect_waypoint_demos")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def _collect_waypoint_files(
    waypoint_dir: Path,
    waypoint_glob: str,
    *,
    start_episode: Optional[int],
    end_episode: Optional[int],
) -> List[Tuple[int, Path]]:
    matched: List[Tuple[int, Path]] = []
    invalid: List[Path] = []
    for path in waypoint_dir.glob(waypoint_glob):
        idx = parse_waypoint_episode_index(path)
        if idx is None:
            invalid.append(path)
            continue
        if start_episode is not None and idx < start_episode:
            continue
        if end_episode is not None and idx > end_episode:
            continue
        matched.append((idx, path))
    if invalid:
        raise ValueError(
            "All waypoint JSON files must follow the normalized name 'episode_xxxxxx_wpts.json'. "
            f"Invalid files: {[p.name for p in sorted(invalid)]}"
        )
    matched.sort(key=lambda x: x[0])
    return matched


def _prepare_episode_dir(raw_root: Path, episode_id: str, overwrite: bool) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    episode_dir = raw_root / episode_id
    if episode_dir.exists():
        if overwrite:
            shutil.rmtree(episode_dir)
        else:
            raise FileExistsError(
                f"Episode folder already exists: {episode_dir}. Re-run with --overwrite to replace it."
            )
    return episode_dir


def _make_debug_callback(
    *,
    debug: bool,
    episode_id: str,
    include_modalities: Sequence[str],
    preview_max_height: int,
):
    if not debug:
        return None, None
    preview = StageADebugPreview(
        StageADebugPreviewConfig(
            enabled_modalities=list(include_modalities),
            max_height=preview_max_height,
            window_name=f"stage_a_collect_{episode_id}",
        )
    )

    def _callback(step_info: dict) -> bool:
        target = np.asarray(step_info["target_7d"], dtype=np.float64)
        achieved = np.asarray(step_info["achieved_7d"], dtype=np.float64)
        pos_err_mm = 1000.0 * np.linalg.norm(achieved[:3] - target[:3])
        rot_err_deg = np.degrees(np.linalg.norm(achieved[3:6] - target[3:6]))
        status_lines = [
            (
                f"target xyz=({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m | "
                f"rpy=({target[3]:+.3f}, {target[4]:+.3f}, {target[5]:+.3f}) rad | grip={target[6]:.2f}"
            ),
            (
                f"achvd  xyz=({achieved[0]:+.3f}, {achieved[1]:+.3f}, {achieved[2]:+.3f}) m | "
                f"rpy=({achieved[3]:+.3f}, {achieved[4]:+.3f}, {achieved[5]:+.3f}) rad | grip={achieved[6]:.2f}"
            ),
            f"tracking error: pos={pos_err_mm:.1f} mm | rot={rot_err_deg:.1f} deg",
        ]
        return preview.show_step(
            episode_id=episode_id,
            step_idx=int(step_info["step_idx"]),
            frames=step_info["frames"],
            frame_meta=step_info.get("frame_meta"),
            status_lines=status_lines,
        )

    return preview, _callback


def main():
    args = _parse_args()
    cfg = RobotPipelineConfig.from_json(args.config)
    plan = _load_json(args.plan_json)
    dataset_root = Path(plan["dataset_root"]).resolve()
    raw_root = dataset_root / "raw"
    waypoint_dir = Path(plan["waypoint_dir"]).resolve()
    waypoint_glob = str(plan.get("waypoint_glob", "episode_*_wpts.json"))
    start_episode = parse_episode_index(args.start_episode)
    end_episode = parse_episode_index(args.end_episode)
    if start_episode is not None and end_episode is not None and start_episode > end_episode:
        raise ValueError("start_episode must be <= end_episode")

    dry_runs = int(plan.get("dry_runs_per_waypoint", 0))
    record_runs = int(plan.get("record_runs_per_waypoint", 1))
    control_hz = float(plan.get("control_hz", 5.0))
    segment_time_override = plan.get("segment_time_s", None)
    pause_after_dry_runs = bool(plan.get("pause_after_dry_runs", False))
    reset_to_home = bool(plan.get("reset_to_home", True))
    initial_settle_s = float(plan.get("initial_settle_s", cfg.robot.reset_sleep_s))
    include_modalities = list(_resolve_modalities(plan, cfg))
    logger = _make_logger(args.debug)

    waypoint_files = _collect_waypoint_files(
        waypoint_dir,
        waypoint_glob,
        start_episode=start_episode,
        end_episode=end_episode,
    )
    if not waypoint_files:
        raise FileNotFoundError(
            f"No normalized waypoint files matched {waypoint_glob!r} in {waypoint_dir} for the requested range."
        )

    if record_runs > 1 and not args.overwrite:
        logger.warning(
            "record_runs_per_waypoint=%d with fixed episode numbering will fail on the second pass unless --overwrite is used.",
            record_runs,
        )

    robot = _build_robot(cfg, args.mock)
    cameras = MultiCameraRig(cfg.cameras)
    stop_collection = False

    try:
        cameras.start()
        robot.connect()
        robot.configure_for_policy()
        logger.info(
            "Matched %d waypoint files in %s | range=[%s, %s] | cameras=%s",
            len(waypoint_files),
            waypoint_dir,
            format_episode_id(start_episode) if start_episode is not None else "first",
            format_episode_id(end_episode) if end_episode is not None else "last",
            include_modalities,
        )
        if args.debug:
            logger.info("Debug preview enabled. Window: q/ESC abort. SSH/headless: type 'q' + Enter in terminal (or Ctrl+C).")

        for queue_idx, (episode_idx, waypoint_path) in enumerate(waypoint_files):
            if stop_collection:
                break
            episode_id = format_episode_id(episode_idx)
            waypoints, segment_time_s, wp_meta = load_absolute_waypoints(waypoint_path)
            if segment_time_override is not None:
                segment_time_s = float(segment_time_override)
            instruction = str(wp_meta.get("instruction", plan.get("default_instruction", waypoint_path.stem.replace("_", " "))))
            logger.info(
                "Waypoint file: %s | episode=%s | instruction=%s | dry_runs=%d | record_runs=%d",
                waypoint_path.name,
                episode_id,
                instruction,
                dry_runs,
                record_runs,
            )

            preview, step_callback = _make_debug_callback(
                debug=args.debug,
                episode_id=episode_id,
                include_modalities=include_modalities,
                preview_max_height=args.preview_max_height,
            )
            try:
                aborted_episode = False
                for dry_idx in range(dry_runs):
                    logger.info("Dry run %d/%d for %s", dry_idx + 1, dry_runs, episode_id)
                    tmp_dir = dataset_root / ".dry_runs" / episode_id / f"run_{dry_idx:03d}"
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    try:
                        collect_stage_a_episode(
                            robot=robot,
                            cameras=cameras,
                            instruction=instruction,
                            raw_episode_dir=tmp_dir,
                            absolute_waypoints=waypoints,
                            control_hz=control_hz,
                            segment_time_s=segment_time_s,
                            logger=logger,
                            initial_settle_s=initial_settle_s,
                            reset_to_home=reset_to_home,
                            waypoint_source=str(waypoint_path),
                            include_modalities=include_modalities,
                            run_kind="dry_run",
                            extra_meta={"collection_plan": str(Path(args.plan_json).resolve())},
                            step_callback=step_callback,
                        )
                    except (KeyboardInterrupt, StageAAbortRequested):
                        logger.warning("Dry run aborted during %s. No data was saved.", episode_id)
                        stop_collection = True
                        aborted_episode = True
                        break
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                if dry_runs > 0 and not aborted_episode and reset_to_home:
                    input(f"Dry runs finished for {episode_id}. Press Enter to return home... ")
                    try:
                        robot.reset_to_home()
                    except Exception:
                        logger.warning("Failed to reset robot to home pose after dry runs for %s", episode_id, exc_info=True)

                if dry_runs > 0 and pause_after_dry_runs and not aborted_episode:
                    input(f"Dry runs finished for {episode_id}. Arrange objects, then press Enter to record... ")

                for rec_idx in range(record_runs):
                    if aborted_episode:
                        break
                    episode_dir = _prepare_episode_dir(raw_root, episode_id, args.overwrite)
                    logger.info("Recording run %d/%d -> %s", rec_idx + 1, record_runs, episode_dir.name)
                    try:
                        collect_stage_a_episode(
                            robot=robot,
                            cameras=cameras,
                            instruction=instruction,
                            raw_episode_dir=episode_dir,
                            absolute_waypoints=waypoints,
                            control_hz=control_hz,
                            segment_time_s=segment_time_s,
                            logger=logger,
                            initial_settle_s=initial_settle_s,
                            reset_to_home=reset_to_home,
                            waypoint_source=str(waypoint_path),
                            include_modalities=include_modalities,
                            run_kind="record",
                            extra_meta={"collection_plan": str(Path(args.plan_json).resolve())},
                            step_callback=step_callback,
                        )
                    except (KeyboardInterrupt, StageAAbortRequested):
                        logger.warning("Collection aborted during %s. Removing incomplete episode folder: %s", episode_id, episode_dir)
                        shutil.rmtree(episode_dir, ignore_errors=True)
                        stop_collection = True
                        break
                    time.sleep(float(plan.get("sleep_between_recordings_s", 0.0)))
            finally:
                if preview is not None:
                    preview.close()

            if reset_to_home:
                input(f"Finished {episode_id}. Press Enter to return home... ")
                try:
                    robot.reset_to_home()
                except Exception:
                    logger.warning("Failed to reset robot to home pose after %s", episode_id, exc_info=True)

            if stop_collection:
                break
            if queue_idx < len(waypoint_files) - 1:
                input(f"Press Enter to continue to the next episode... ")
    finally:
        try:
            cameras.stop()
        finally:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
