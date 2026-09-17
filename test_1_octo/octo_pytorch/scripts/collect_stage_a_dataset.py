#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

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


def _parse_args():
    ap = argparse.ArgumentParser(description="Collect one raw Stage-A episode from a normalized waypoint JSON file")
    ap.add_argument("--config", required=True, help="Robot/camera JSON config")
    ap.add_argument("--waypoints_json", required=True, help="JSON file containing absolute 7D waypoints")
    ap.add_argument("--instruction", default=None, help="Optional instruction override stored for the episode")
    ap.add_argument("--dataset_root", required=True, help="Root directory for Stage-A dataset (raw will live under dataset_root/raw)")
    ap.add_argument("--episode_id", default=None, help="Optional episode id, e.g. episode_000001. Defaults to waypoint filename if normalized.")
    ap.add_argument("--control_hz", type=float, default=5.0, help="Dense logging / control rate")
    ap.add_argument("--segment_time_s", type=float, default=None, help="Override segment interpolation time")
    ap.add_argument("--capture_modalities", default=None, help="Comma-separated list like primary,wrist")
    ap.add_argument("--dry_runs", type=int, default=0, help="Execute dry-runs before recording")
    ap.add_argument("--pause_after_dry_runs", action="store_true", help="Pause for Enter after dry-runs")
    ap.add_argument("--mock", action="store_true", help="Use mock robot backend regardless of config")
    ap.add_argument("--skip_reset", action="store_true", help="Skip reset-to-home before the episode")
    ap.add_argument("--overwrite", action="store_true", help="Delete existing target episode folder before recording")
    ap.add_argument("--debug", action="store_true", help="Enable debug logs + camera preview while collecting")
    ap.add_argument("--verbose", dest="debug", action="store_true", help=DEPRECATED_VERBOSE_HELP)
    ap.add_argument("--preview_max_height", type=int, default=360, help="Preview max height when --debug is enabled")
    return ap.parse_args()


def _setup_logger(dataset_root: Path, debug: bool) -> logging.Logger:
    logger = logging.getLogger("stage_a_collect")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(raw_dir / "collector.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _build_robot(cfg: RobotPipelineConfig, force_mock: bool):
    backend = cfg.robot.backend.lower()
    if force_mock or backend in {"mock", "mock_myarm", "mock_myarm_m750"}:
        return MockMyArmM750Robot(cfg.robot, visualize=False)
    if backend in {"myarm", "myarm_m750"}:
        return MyArmM750Robot(cfg.robot)
    raise ValueError(f"Unsupported robot backend: {cfg.robot.backend!r}")


def _parse_modalities(value: Optional[str], cfg: RobotPipelineConfig) -> Sequence[str]:
    if value:
        items = [x.strip() for x in value.split(",") if x.strip()]
        if items:
            return items
    return [name for name, cam_cfg in cfg.cameras.items() if bool(cam_cfg.enabled)]


def _resolve_episode_id(waypoints_json: Path, explicit_episode_id: Optional[str]) -> str:
    derived_idx = parse_waypoint_episode_index(waypoints_json)
    if explicit_episode_id is not None:
        idx = parse_episode_index(explicit_episode_id)
        if idx is None:
            raise ValueError(f"Invalid episode_id: {explicit_episode_id!r}")
        if derived_idx is not None and int(idx) != int(derived_idx):
            raise ValueError(
                f"episode_id={format_episode_id(idx)} does not match waypoint filename episode={format_episode_id(derived_idx)}"
            )
        return format_episode_id(idx)
    if derived_idx is not None:
        return format_episode_id(derived_idx)
    raise ValueError(
        "waypoints_json must follow the normalized name 'episode_xxxxxx_wpts.json' unless --episode_id is provided."
    )


def _ensure_target_episode_dir(dataset_root: Path, episode_id: str, overwrite: bool) -> Path:
    raw_root = dataset_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    episode_dir = raw_root / episode_id
    if episode_dir.exists():
        if overwrite:
            shutil.rmtree(episode_dir)
        else:
            raise FileExistsError(
                f"Target episode folder already exists: {episode_dir}. Re-run with --overwrite to replace it."
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
    dataset_root = Path(args.dataset_root).resolve()
    logger = _setup_logger(dataset_root, args.debug)
    waypoints_json = Path(args.waypoints_json).resolve()

    waypoints, file_segment_time_s, wp_meta = load_absolute_waypoints(waypoints_json)
    segment_time_s = float(args.segment_time_s) if args.segment_time_s is not None else float(file_segment_time_s)
    instruction = str(args.instruction or wp_meta.get("instruction") or waypoints_json.stem.replace("_", " "))
    include_modalities = list(_parse_modalities(args.capture_modalities, cfg))
    episode_id = _resolve_episode_id(waypoints_json, args.episode_id)
    logger.info(
        "Loaded %d absolute waypoints from %s | episode_id=%s | segment_time_s=%.3f | cameras=%s",
        len(waypoints),
        waypoints_json,
        episode_id,
        segment_time_s,
        include_modalities,
    )

    robot = _build_robot(cfg, args.mock)
    cameras = MultiCameraRig(cfg.cameras)
    preview, step_callback = _make_debug_callback(
        debug=args.debug,
        episode_id=episode_id,
        include_modalities=include_modalities,
        preview_max_height=args.preview_max_height,
    )

    try:
        cameras.start()
        logger.info("Started cameras: %s", cameras.available_modalities())
        robot.connect()
        robot.configure_for_policy()
        logger.info("Connected robot backend=%s", cfg.robot.backend)
        if args.debug:
            logger.info("Debug preview enabled. Window: q/ESC abort. SSH/headless: type 'q' + Enter in terminal (or Ctrl+C).")

        aborted = False
        for dry_idx in range(max(0, int(args.dry_runs))):
            logger.info("Dry run %d/%d for %s", dry_idx + 1, args.dry_runs, episode_id)
            tmp_dir = dataset_root / ".dry_runs" / episode_id / f"run_{dry_idx:03d}"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                collect_stage_a_episode(
                    robot=robot,
                    cameras=cameras,
                    instruction=instruction,
                    raw_episode_dir=tmp_dir,
                    absolute_waypoints=waypoints,
                    control_hz=args.control_hz,
                    segment_time_s=segment_time_s,
                    logger=logger,
                    initial_settle_s=float(cfg.robot.reset_sleep_s),
                    reset_to_home=not args.skip_reset,
                    waypoint_source=str(waypoints_json),
                    include_modalities=include_modalities,
                    run_kind="dry_run",
                    extra_meta={"collector_mode": "single_episode"},
                    step_callback=step_callback,
                )
            except (KeyboardInterrupt, StageAAbortRequested):
                logger.warning("Dry run aborted by operator. No data was saved.")
                aborted = True
                break
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        if args.dry_runs > 0 and args.pause_after_dry_runs and not aborted:
            input(f"Dry runs finished for {episode_id}. Arrange objects, then press Enter to record... ")

        if not aborted:
            episode_dir = _ensure_target_episode_dir(dataset_root, episode_id, args.overwrite)
            try:
                collect_stage_a_episode(
                    robot=robot,
                    cameras=cameras,
                    instruction=instruction,
                    raw_episode_dir=episode_dir,
                    absolute_waypoints=waypoints,
                    control_hz=args.control_hz,
                    segment_time_s=segment_time_s,
                    logger=logger,
                    initial_settle_s=float(cfg.robot.reset_sleep_s),
                    reset_to_home=not args.skip_reset,
                    waypoint_source=str(waypoints_json),
                    include_modalities=include_modalities,
                    run_kind="record",
                    extra_meta={"collector_mode": "single_episode"},
                    step_callback=step_callback,
                )
            except (KeyboardInterrupt, StageAAbortRequested):
                logger.warning("Collection aborted. Removing incomplete episode folder: %s", episode_dir)
                shutil.rmtree(episode_dir, ignore_errors=True)
                aborted = True

            if not aborted:
                logger.info("Stage-A raw episode saved under %s", episode_dir)
    finally:
        if preview is not None:
            preview.close()
        try:
            cameras.stop()
        finally:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
