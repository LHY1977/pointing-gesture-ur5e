# Pointing-Gesture Human–Robot Interaction (Simulation)

Reproduction of a pointing-gesture HRI method in simulation, using a webcam
(MediaPipe hand tracking) to control a simulated UR5e robot arm (MuJoCo).
The operator points at a target; after a confirmation gesture, the robot's
end effector moves to and points at the indicated location.

Course project for **Robotics (01PEEQW)**, M.Sc. Mechatronic Engineering,
Politecnico di Torino.

## How it works

The system runs as a real-time control loop (~30 Hz) with three layers:

1. **Perception** — MediaPipe Hand Landmarker detects 21 hand keypoints.
   The index finger (landmarks 5 → 8) gives the pointing direction; the
   thumb is used for a confirmation gesture.
2. **Mapping** — the pointing direction sets the target on the table (XY),
   and the hand's vertical image position sets the target height (Z).
   A monocular webcam cannot estimate depth reliably, so direction is used
   instead of a literal 3-D ray projection.
3. **Control** — Closed-Loop Inverse Kinematics (CLIK) with the geometric
   Jacobian. The end effector is driven to the target and oriented to point
   at it (6-DOF). Damped least-squares is used to stay stable near
   kinematic singularities.

## Requirements

- Python 3.10+
- A webcam

Install the dependencies:

```bash
pip install mujoco mediapipe opencv-python numpy
```

## Run

```bash
python vision_mujoco_final.py
```

Two windows open: the webcam view and the MuJoCo simulation.

**Controls:**
- Point with your index finger to move the yellow preview marker.
- Move your hand up/down to change the target height.
- Make a "pistol" gesture (extend the thumb) to confirm — the red target
  locks and the robot moves to point at it.

## Files

| File | Description |
|------|-------------|
| `vision_mujoco_final.py` | Main program: perception, mapping, gesture confirmation, CLIK control |
| `intersection.py` | Ray–plane intersection, the paper's original targeting geometry |

## Notes and limitations

- The controller is reactive (CLIK), not planning-based: it does not plan a
  collision-free path or model the tool geometry. Targets are limited to a
  comfortable region of the workspace.
- The UR5e model carries no tool, so tool collisions are not simulated.
- Perception was adapted to a commodity webcam, which is why direction-based
  mapping is used instead of the paper's depth-based ray projection.
