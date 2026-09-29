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
    MUJOCO_GL=egl .venv/bin/python tools/sim_grasp_validation.py       --mode all --socket /workspace/6DoF_Grasp/grasp_pipeline_repo/.runtime/worker.sock       --photo /path/to/private/cam2.jpg

Artifacts go under .local_data/sim_grasp_validation/, which Git ignores.
Each simulation case stores a rendered image, camera matrix, synchronized
joint/object state, seed, scene and robot-model hashes, camera matrix,
candidate, IK/path results, contacts, cube height and video. The projection
check compares the projected cube center with the
segmentation-rendered cube pixel centroid.

Modes are photo, oracle, e2e, and all. Oracle candidates come only from the
known synthetic cube geometry. Only e2e uses the worker candidate from the
rendered camera image.

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
