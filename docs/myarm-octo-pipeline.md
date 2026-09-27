# Octo Torch Robot Pipeline for MyArm M750

This pipeline mirrors the structure of Octo's real-robot evaluation flow:

- a Gym-compatible robot environment
- camera abstraction separated from robot control
- a history wrapper for observation windows
- temporal ensembling or receding-horizon execution for action chunks
- a torch-only policy runner suitable for Jetson deployment

## Added modules

- `octo/robot/config.py`
  - JSON-driven configuration for robot, workspace, cameras, and rollout.
- `octo/robot/cameras.py`
  - OpenCV camera streams with MJPG / 1280x720 / 60 FPS support.
  - Primary and wrist cameras can be enabled or disabled independently.
- `octo/robot/myarm_m750.py`
  - Real MyArm M750 driver based on `pymycobot.MyArmMControl`.
  - Public policy interface uses SI units (`m`, `rad`).
  - Hardware commands are converted internally to vendor units (`mm`, `deg`).
  - Uses coordinate firmware with:
    - base frame (`reference_frame = 0`)
    - MoveL (`movement_type = 1`)
    - TCP / tool end type (`end_type = 1`)
- `octo/robot/mock_robot.py`
  - Safe mock backend with the same delta-TCP action interface as the real robot.
  - Optional 3D visualization of TCP motion in the robot base frame.
- `octo/robot/env.py`
  - Gym env that couples the robot backend with the camera rig.
- `octo/robot/policy.py`
  - Torch Octo inference adapter that fills missing modalities from the checkpoint example batch.
- `scripts/run_octo_robot_eval.py`
  - Main entrypoint for both real and mock deployment.
- `scripts/robot_configs/*.json`
  - Example configs for the real robot and the mock backend.

## Action convention

The env expects a 7D action in SI units:

`[dx, dy, dz, droll, dpitch, dyaw, gripper_open]`

- `dx, dy, dz` are TCP deltas in meters.
- `droll, dpitch, dyaw` are TCP orientation deltas in radians.
- `gripper_open` is interpreted as an open fraction in `[0, 1]`.

Internally:

- the delta pose is clipped by per-step safety limits
- the target absolute TCP pose is clamped into the configured workspace
- the final absolute pose is sent to MyArm via `write_coords()` in `mm` and `deg`

## Workspace convention

Workspace bounds are absolute bounds in the robot base frame.

Edit them in the JSON config:

- `translation_min_m`
- `translation_max_m`
- `rotation_min_rad`
- `rotation_max_rad`

This matches the style of Octo robot eval where the robot-side env owns the safety envelope.

## Running

### Real robot

```bash
python scripts/run_octo_robot_eval.py \
  --config scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json \
  --text "pick up the coke can"
```

### Mock backend with live cameras and 3D TCP visualization

```bash
python scripts/run_octo_robot_eval.py \
  --config scripts/robot_configs/mock_myarm_primary_wrist_720p_mjpg.json \
  --text "pick up the coke can"
```

### Goal-conditioned rollout

```bash
python scripts/run_octo_robot_eval.py \
  --config scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json \
  --goal_primary ./assets/goal_primary.png
```

### Language + goal together

```bash
python scripts/run_octo_robot_eval.py \
  --config scripts/robot_configs/myarm_m750_primary_wrist_720p_mjpg.json \
  --text "place the object on the plate" \
  --goal_primary ./assets/goal_primary.png
```

## Notes for Jetson

- The runner is torch-only and disables TF / Flax / JAX imports.
- Camera capture is independent from robot control so you can swap webcams or camera indices in JSON only.
- If your checkpoint does not use a wrist stream, keep `wrist.enabled=false` or leave it enabled; the policy adapter will ignore unneeded modalities.
- If your checkpoint expects a modality that is disabled in hardware, the adapter pads it safely from the example batch.

## Recommended first bring-up sequence

1. Run the mock config with the real cameras.
2. Verify the 3D TCP path looks reasonable for your prompt.
3. Tighten workspace bounds.
4. Switch to the real MyArm backend.
5. Start with low `move_speed` and small `max_step_translation_m` / `max_step_rotation_rad`.
