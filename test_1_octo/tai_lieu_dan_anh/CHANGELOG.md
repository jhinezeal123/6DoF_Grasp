# Changelog

## v0_2_5
- Add: normalized Stage-A episode/waypoint naming workflow using `episode_xxxxxx` folders and `episode_xxxxxx_wpts.json` files.
- Add: ranged waypoint collection via `--start_episode/--end_episode` (and `--strat_episode` alias) while preserving one-to-one episode index matching.
- Add: `--overwrite` support for Stage-A collectors to delete and replace existing `episode_xxxxxx` folders safely.
- Update: Stage-A collectors now use `--debug` instead of `--verbose`, show live primary/wrist preview when enabled, and allow operator abort with `q`/`ESC` without saving the interrupted episode.
- Add: `scripts/visualize_waypoints_3d.py` plus reusable 3D waypoint/raw-dataset visualization utilities with workspace-aware statistics export.
- Update: `build_dataset.py` and `compute_stage_a_statistics.py` can now emit 3D PNG/JSON trajectory summaries when `--config` is provided.

## v0.2.3

* Add: exact Bridge-style SE(3) control semantics via `se3_integration_mode=bridge_exact`, with explicit `hardware_measured_enabled`, `pivot_source`, and `compose_base` config fields.
* Add: comparison control modes `measured_compose` and `target_only` so Jetson rollout experiments can A/B different measured-vs-target propagation behaviors without touching code.
* Update: `BridgeStyleActionIntegrator` now logs target-before-step, compose-base transform, effective pivot source, and measured pose participation while keeping translation clamping in transform space.
* Update: `MyArmM750Robot` and `MockMyArmM750Robot` now expose richer debug snapshots with target-before/after, measured-before/after, pivot position, integration mode, and resync reason for easier controller diagnosis.
* Add: rollout preview now supports `preview_layout=side_by_side` and shows both primary + wrist cameras with labels/timestamps, independent of whether wrist is consumed by the model.
* Update: example robot configs now default to `bridge_exact`, enable wrist camera preview, and use the new periodic resync keys.
* Add: regression tests for v0.2.3 transform integrator modes in `tests/test_transform_integrator_v023.py`.

## v0.2.1

* Add: explicit control-period tuning in `RobotConfig` via `control_period_s`, plus firmware pacing knobs `post_command_sleep_s` and `hardware_read_settle_s` for Jetson single-device rollout tuning.
* Update: `MyArmM750Robot` now supports Bridge-like neutral orientation handling for proprio through `neutral_orientation_rpy_rad` and `proprio_orientation_mode=relative_to_neutral`.
* Add: hardware reset readback sync and periodic resync diagnostics with geometric pose error metrics based on transforms instead of naive Euler subtraction.
* Add: `octo.robot.debug_logger.RobotDebugLogger` to save CSV + PNG traces for previous target transform, measured hardware pose, tracking error, and camera freshness on Jetson.
* Update: rollout runner prints blocking/control-period/resync settings and logs target-vs-measured errors every step for direct tuning on-device.
* Update: example robot configs now enable Bridge-like relative orientation, debug traces, and more realistic resync defaults for v0.2.1.

## v0.2.0

* Add: `octo.robot.transform_integrator.BridgeStyleActionIntegrator` for Bridge-like SE(3) target propagation using `next_transform = delta_transform @ previous_target_transform`.
* Update: `MyArmM750Robot.apply_action()` no longer integrates actions with naive `current_pose + delta`; it now composes rigid transforms and still uses MyArm Cartesian `write_coords(...)` at the transport layer.
* Update: `MockMyArmM750Robot` mirrors the same Bridge-style action semantics, so dry-runs and visualization match the real backend more closely.
* Add: camera preprocessing controls in `CameraConfig` for crop, flip, rotation, and stale-frame checks, moving the camera pipeline closer to Bridge's per-topic image processing.
* Add: optional periodic hardware resync hooks for MyArm target state recovery without rebuilding the target from measured pose on every step.
* Update: robot config examples now expose the new Bridge-style integration and camera preprocessing options.

## v0.0.5

* Fix: `octo.model.components.action_heads_pt` no longer references undefined `TokenGroup` in type annotations (now `TokenGroupPt` + postponed annotations), preventing import-time crash when instantiating `DiffusionActionHeadPt`.

## v0.0.4

* Fix: goal-conditioned `create_tasks()` no longer creates numpy pad masks (now torch tensors), preventing `pad_mask_dict` stack errors.
* Fix: remove unused torchvision import in tokenizers to avoid hard dependency/ABI issues on Jetson.
* Fix: runners now force local-repo imports (avoid accidentally using an older editable install).
* Fix: runners set `TRANSFORMERS_NO_TF/NO_FLAX/NO_JAX` to avoid TensorFlow/XLA side-effect logs.
* Update: `runner_inference_sample.py` now generates `pad_mask_dict` for observations and supports `--goal_wrist`.
* Add: `scripts/runner_realtime_camera.py` – OpenCV/V4L2 realtime camera runner with window=2 rollout-style action streaming.

## v0.0.3

* Fix: Jetson smoke test now uses `OctoModelPt.forward(..., transformer_only=True)` (no `run_transformer`).
* Fix: `HFTokenizer` is now torch-only friendly (no Flax/JAX import), and can optionally produce embeddings via PyTorch `AutoModel`.
* Add: `scripts/runner_inference_sample.py` – minimal end-to-end runner for torch checkpoints.
* Update: `requirements_torch.txt` no longer requires `torchvision`.

## v0.0.2

* Torch-only inference: move TF/JAX/Flax/Orbax imports behind a lazy loader for JAX checkpoint conversion.
* Add: JAX→Torch conversion script and Jetson smoke test.

## v0_2_4
- merged Stage-A / TFDS-RLDS dataset build pipeline from the finetune repo into the infer repo
- added PyTorch `train_pt.py` plus RLDS worker factory and faster `TorchRLDSDataset` wrapper
- added Stage-A dataset tools: raw episode collection, TFDS prepare, statistics export, dataset view
- preserved v0_2_3 robot control semantics/configs (`bridge_exact`, measured pivot, side-by-side preview)
- added configurable multi-episode waypoint collection workflow with dry-runs before recording
- added clearer train/finetune configs with dataloader / TF threading knobs to address GPU starvation
