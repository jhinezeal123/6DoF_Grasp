# Simulation grasp validation

This workflow tests the 6DoF grasp path in MuJoCo without opening a serial
port, starting ROS, or constructing a physical robot driver.

## Limits

- cam2.jpg is a perception replay only. It has no synchronized robot joints
  and its camera intrinsics are estimated. Its grasp is never transformed into
  the robot base frame.
- The synthetic scene uses a 25 mm blue cube at 1200 kg/m3 on a 60 mm narrow
  pedestal. The pedestal clears the stock long finger collision boxes. Cube,
  material, pedestal, lighting and wrist camera pose are simulation
  assumptions, not measurements of the object or camera in cam2.jpg.
- The scene excludes the overlapping upper_arm_link/forearm_link collision
  proxies at the elbow. All other robot self contacts and robot/world contacts
  are checked; grasp contact is permitted only between finger pads and cube.
- The current GraspPose output has not passed the geometric end-to-end checks.
  The upstream VGN Grasp type leaves the grasp-frame definition open, while
  its simulator uses rotation column 3 as the approach direction. This harness
  assumes the same +Z approach axis and measures the simulated tool-to-grasp
  offset. No unverified quaternion rotation is applied in the adapter.
- Passing these checks establishes simulation behavior only. Real robot
  validation still needs object measurements, camera calibration and
  synchronized image/joint samples across poses.

## Run on KTMT

Keep the photo private and outside Git. The perception worker is pinned at
666c7eb608c5315ea252fd02b3f5446c39198ee6.

    cd /workspace/6DoF_Grasp/grasp_pipeline_repo
    bash scripts/worker.sh start
    cd /workspace/6DoF_Grasp/htc_sim_grasp_validation
    MUJOCO_GL=egl .venv/bin/python -m pytest -q
    MUJOCO_GL=egl .venv/bin/python tools/sim_grasp_validation.py \
      --mode all \
      --socket /workspace/6DoF_Grasp/grasp_pipeline_repo/.runtime/worker.sock \
      --photo /path/to/private/cam2.jpg

Artifacts go under .local_data/sim_grasp_validation/, which Git ignores.
The report lists the synthetic cube, pedestal, table and camera mount dimensions,
visual RGBA, friction values and lighting; it also stores hashes of both XML
models. Each simulation case stores a rendered image, synchronized joint/object state,
seed, scene and robot-model hashes, candidate, IK/path results and video.
During actuator motion, `simulation_trace` samples simulation time, cube height,
six joint angles and active contacts every 30 ms, including the final state on
failure. The projection check compares the projected cube center with the
segmentation-rendered cube pixel centroid.

Modes are photo, oracle, e2e, and all. Oracle candidates come only from the
known synthetic cube geometry. Only e2e uses the worker candidate from the
rendered camera image.

## Volume frame and depth source

Two options exist because the pipeline's own fallbacks hide the two defects this
harness needs to separate.

`--volume gravity` builds `T_cam_volume` from the known camera pose with the
volume axes on the robot base axes, so Z points up along gravity, and passes it
to the worker as `camera_from_volume`/`T_cam_volume`. VGN was trained on
gravity-aligned volumes and its own simulator builds them that way; the pipeline
default is camera-aligned and turns a top-down approach into a horizontal one.
`--volume auto` keeps that default and is the baseline. The gravity volume is
anchored at the known capture-time object centre, so it isolates the volume frame
and does not by itself represent an RGB-only pipeline.

`--depth-source sim` runs the pinned pipeline in its own virtualenv through
`src/m750/sim/adapters/grasppose_bridge.py`, which replaces the monocular depth
adapter with the depth map rendered by MuJoCo. The mask, TSDF, VGN and grasp
decoding are unchanged. Use it to separate a depth failure from a grasp failure;
the default `--depth-source worker` measures the deployed path.

The worker must be started with `YOLOE_CONF=0.05` for these checks. The detector
localises the cube in all ten views - the proposed box centre sits 22 px from the
true silhouette centre in every one - but scores 0.099-0.139 in four of them,
under the artifact's 0.20 operating point.

When a planned candidate fails during execution, e2e resets the scene and plans
the next-ranked candidate, the way a robot retries. `perception.attempts_used`
records how many attempts a case needed.

## Acceptance checklist

- The existing software tests pass and the simulation module does not import a
  robot driver or ROS.
- The cam2.jpg replay succeeds 3/3 times with finite pose values, valid
  rotation and width at most 69 mm; report has no joint state and marks K as
  estimated.
- Pinocchio FK and MuJoCo differ by at most 2 mm and 2 degrees. Rendered cube
  projection differs by at most 2 pixels. Each IK waypoint is within 3 mm and
  3 degrees.
- Run the same 10 cases for B and C (five cube locations by two light levels).
  C counts as passed only if all 10 grasp, lift the cube at least 50 mm and hold
  for one second.
- Oversized grasps, worker errors, unreachable IK and unsafe swept paths stop
  before actuator commands. Unexpected robot self collisions, table contacts
  and non-finger cube contacts fail the case.

## Failure handling

1. If photo replay fails, check worker status, prompt ID, K and socket protocol.
2. If oracle B fails, inspect frame conversion, tool-to-grasp offset, FK/IK,
   joint limits, swept contacts and contact forces.
3. If B passes but C fails before grasp, inspect the saved image, worker
   detections, camera pose, lighting and object appearance. Record the measured
   perception-to-ground-truth error as a sim-to-image gap.
4. If the worker grasp passes IK/path checks but the cube is not lifted, compare
   its predicted pose to the known cube state and inspect finger contact,
   friction, actuator tracking and the lift video.

## Latest KTMT result (2026-09-30)

The combined report is in `.local_data/sim_grasp_validation/report.json`.
Oracle B and end-to-end C ran on 6DoF commit
`50cfbdf6afc585ca90a804890c727d1beb7de16a`; the private `cam2.jpg` replay
was refreshed on `bb6deceb484d26ad31b3f172b6992a1f1d6225d2` after adding
explicit scene metadata to the report. That metadata-only change did not change
simulation physics. Both used perception worker commit
`666c7eb608c5315ea252fd02b3f5446c39198ee6`. The source image is kept at
`.local_data/private/cam2.jpg`, which Git ignores.

- Software checks: 44 passed with `MUJOCO_GL=egl`. Negative checks reject
  too-wide grasps, worker failure, unreachable IK and swept-path collision
  before `execute` is called.
- A, private `cam2.jpg` replay: 3/3 valid worker results; width 33.9 mm.
  The saved report records no joint state, marks K estimated and does not
  assert a base-frame grasp.
- B, geometry oracle: 10/10 grasp-and-lift successes. Each held for 1 second
  and lifted at least 55.2 mm. Maximum FK difference was below 2 mm/2 degrees;
  maximum rendered-centre reprojection error was 0.324 px.
- C, rendered perception to grasp-and-lift: 0/10. The worker found no grasp in
  four cases (03-06). In the other six, the closest reported candidate position
  errors per image were 36.1, 37.9, 39.3, 41.7, 83.0 and 92.0 mm. Most candidates
  failed IK or swept-path checks. In case 08, rank 4 passed those checks but
  differed from the known top-grasp orientation by 97.6 degrees; the attempt
  displaced the cube off its pedestal, and the cube fell to the table. This was
  not a successful grasp.
- The final scene configuration gave 3/5 detections at the 0.8 light multiplier
  and 6/10 across both light levels. At 0.8, a 32-degree camera FOV with brighter
  ambient light, gray table and brighter cube gave 0/5; a darker table and
  ambient-light trial at 42.2 degrees also gave 0/5. The original configuration
  was retained because it detected the most cases.

All 20 simulation videos, rendered images, synchronized joint/object states,
camera matrices, grasps and planning details are saved under
`.local_data/sim_grasp_validation/`. Result C does not meet the 10/10 gate.
These results establish neither real-camera calibration nor readiness to grasp
the object in `cam2.jpg`; the hardware validation gate remains closed.

## KTMT result 2026-10-01: gate met on both B and C

Run with `--mode all --volume gravity --depth-source sim` and the worker started
as `YOLOE_CONF=0.05 bash scripts/worker.sh start`. Report:
`.local_data/gate_final/report.json`.

Three defects were separated before this run. The volume frame accounted for the
horizontal approaches: on identical exact depth, a camera-aligned volume gave
`approach_b = [-0.51, 0.84, 0.20]` with 115 mm position error, a gravity-aligned
one gave `[-0.11, -0.16, -0.98]` with 15 mm. The Lite-Mono depth was
anti-correlated with truth on this domain (Pearson -0.43), so `--depth-source sim`
feeds the depth MuJoCo renders. The detector's operating point was below four of
the ten views (0.099-0.139 against 0.20) even though the box it proposed was
correct in all ten.

- A, private `cam2.jpg` replay: 3/3 valid worker results; width 33.9 mm, score
  0.9636.
- B, geometry oracle: 10/10, each lifted at least 55.2 mm and held 1 second.
- C, rendered perception to grasp-and-lift: 10/10, each lifted at least 56.0 mm
  and held 1 second. Nine cases succeeded on rank 1. Case 04 needed two attempts:
  rank 1 was a phantom candidate 132.5 mm from the cube and closed on nothing, and
  rank 2 at 31.9 mm lifted 56.5 mm.

Caveats. Depth still comes from the simulator, so this does not establish that the
monocular path can source metric depth on this domain; the Depth Anything V2 metric
branch reaches Pearson +0.93 against truth but carries a 6.3x scale bias at the
object and 10.5x at the table. The gravity volume is anchored at the known object
centre rather than at a perceived one. VGN still ranks a phantom candidate first
in case 04 within 0.1% of the correct one, which a quality threshold cannot
separate; the retry hides that rather than fixing it.
