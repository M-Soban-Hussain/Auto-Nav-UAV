"""
depth_nav.py
Autonomous depth-camera navigation for an AirSim multirotor.

- You control forward THRUST only, with W (speed up) / S (slow down / brake).
- The script autonomously computes roll/pitch/yaw (via body-frame velocity + yaw rate)
  so the drone steers itself toward DEST_X, DEST_Y, DEST_Z while avoiding obstacles
  (buildings, cars, etc.) using the front depth camera.
- The nose of the drone always points in the direction it is currently moving,
  because forward speed is commanded purely in the body X axis (vy = 0) while
  yaw is turned independently.

Requires only libraries you already installed in your setup steps:
    numpy, opencv-python, keyboard, msgpack-rpc-python (airsim)
No new installs are needed. If you ever get "No module named 'cv2'" run:
    py -m pip install opencv-python
(same for the others, using the commands from your Steps doc.)

Run it the same way as your manual script:
    python depth_nav.py
(with the AirSim environment already running and PYTHONPATH set to your PythonClient folder,
 exactly like in your existing WASD script.)
"""

import sys
import time
import math
import numpy as np
import keyboard
import cv2  # only used for the optional depth debug window

# EDIT THIS to match your PythonClient path, same as your manual control script
sys.path.append(r"D:\Soban\AirSim-main\PythonClient")
import airsim

# ============================================================
# USER-CONFIGURABLE PARAMETERS - edit these for your map/run
# ============================================================
DEST_X = -57.85                # target X, NED meters, relative to takeoff point
DEST_Y = -245.23                # target Y, NED meters
DEST_Z = -8.0                # target altitude, NED (negative = up). -8 = ~8m above ground
ARRIVAL_RADIUS = 3.0         # meters - stop when this close to the destination

CAMERA_NAME = "0"            # front camera in AirSim's default multirotor rig
CAMERA_FOV_DEG = 90.0        # must match the camera FOV in your settings.json
N_SECTORS = 9                # number of horizontal depth "slices" to evaluate
MAX_DEPTH_M = 40.0           # depth values beyond this are clipped (treated as "open")
SAFE_DISTANCE_M = 6.0        # sectors closer than this are penalized heavily
EMERGENCY_STOP_M = 3.0       # if the dead-center sector is closer than this, force-brake

GOAL_WEIGHT = 0.55           # how strongly it prefers heading toward the destination
SAFETY_WEIGHT = 0.45         # how strongly it prefers open space

MAX_SPEED = 8.0              # m/s ceiling for W-controlled thrust
MIN_SPEED = 0.0              # m/s floor (set negative if you want reverse braking)
SPEED_STEP = 0.4             # m/s added/removed per loop tick while W/S is held

MAX_YAW_RATE_DEG = 45.0      # deg/s cap on autonomous turning
YAW_KP = 1.8                 # proportional gain: yaw error (rad) -> yaw rate

TARGET_ALT_KP = 2.5          # proportional gain for altitude hold
DT = 0.05                    # control loop period (s)
SHOW_DEPTH_WINDOW = True     # live cv2 window of the depth map (debugging aid)
# ============================================================


def normalize_angle(a):
    """Wrap an angle (radians) to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def get_depth_sectors(client):
    """
    Pull one depth frame and reduce it to N_SECTORS horizontal readings.
    Returns (sector_depths, sector_angles_rad) or (all_clear, None) if the
    image wasn't ready this tick.
    """
    req = airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True, False)
    responses = client.simGetImages([req])
    resp = responses[0]

    if resp.width == 0 or resp.height == 0:
        return np.full(N_SECTORS, MAX_DEPTH_M), None

    depth = np.array(resp.image_data_float, dtype=np.float32).reshape(resp.height, resp.width)
    depth = np.clip(depth, 0.0, MAX_DEPTH_M)

    # focus on a horizontal band around the image center (drone's own altitude level)
    row_lo = int(resp.height * 0.35)
    row_hi = int(resp.height * 0.65)
    band = depth[row_lo:row_hi, :]

    sector_width = resp.width // N_SECTORS
    sector_depths = np.zeros(N_SECTORS, dtype=np.float32)
    for i in range(N_SECTORS):
        c0 = i * sector_width
        c1 = resp.width if i == N_SECTORS - 1 else (i + 1) * sector_width
        chunk = band[:, c0:c1]
        # 10th percentile instead of true min -> a bit less jumpy from single noisy pixels
        sector_depths[i] = np.percentile(chunk, 10)

    if SHOW_DEPTH_WINDOW:
        vis = (depth / MAX_DEPTH_M * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imshow("depth", vis)
        cv2.waitKey(1)

    half_fov = math.radians(CAMERA_FOV_DEG) / 2.0
    sector_angles = np.array([
        -half_fov + (i + 0.5) * (2 * half_fov / N_SECTORS)
        for i in range(N_SECTORS)
    ])
    return sector_depths, sector_angles


def choose_heading(sector_depths, sector_angles, current_yaw, goal_bearing):
    """Blend obstacle clearance with the destination bearing to pick a world-frame heading."""
    best_score = -1e9
    best_rel_angle = 0.0
    center_idx = len(sector_depths) // 2

    for d, rel_angle in zip(sector_depths, sector_angles):
        world_angle = normalize_angle(current_yaw + rel_angle)
        goal_align = math.cos(normalize_angle(world_angle - goal_bearing))  # 1 = straight at goal

        safety = d / MAX_DEPTH_M
        if d < SAFE_DISTANCE_M:
            safety -= 1.5  # strong penalty for tight sectors

        score = SAFETY_WEIGHT * safety + GOAL_WEIGHT * goal_align
        if score > best_score:
            best_score = score
            best_rel_angle = rel_angle

    center_blocked = sector_depths[center_idx] < EMERGENCY_STOP_M
    return normalize_angle(current_yaw + best_rel_angle), center_blocked


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()

    state = client.getMultirotorState()
    target_alt_z = state.kinematics_estimated.position.z_val
    if DEST_Z != 0:
        client.moveToZAsync(DEST_Z, 2.0).join()
        target_alt_z = DEST_Z

    speed = 0.0
    print("READY.  W = speed up, S = slow down/brake, ESC = land & quit.")
    print(f"Destination (NED): ({DEST_X}, {DEST_Y}, {DEST_Z})")

    try:
        while True:
            if keyboard.is_pressed("esc"):
                print("ESC pressed - landing.")
                break

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(orientation)

            dx = DEST_X - pos.x_val
            dy = DEST_Y - pos.y_val
            dist_to_goal = math.hypot(dx, dy)

            if dist_to_goal < ARRIVAL_RADIUS:
                print("Destination reached.")
                break

            goal_bearing = math.atan2(dy, dx)

            sector_depths, sector_angles = get_depth_sectors(client)
            if sector_angles is None:
                desired_heading, center_blocked = goal_bearing, False
            else:
                desired_heading, center_blocked = choose_heading(
                    sector_depths, sector_angles, yaw, goal_bearing
                )

            # -------- thrust from W/S --------
            if keyboard.is_pressed("w"):
                speed = min(MAX_SPEED, speed + SPEED_STEP)
            if keyboard.is_pressed("s"):
                speed = max(MIN_SPEED, speed - SPEED_STEP)

            forward_speed = speed
            if center_blocked:
                forward_speed = min(forward_speed, 0.0)  # hard brake, keep turning to clear the path

            # -------- yaw control --------
            yaw_error = normalize_angle(desired_heading - yaw)
            yaw_rate_rad = max(-math.radians(MAX_YAW_RATE_DEG),
                                min(math.radians(MAX_YAW_RATE_DEG), YAW_KP * yaw_error))
            yaw_rate_deg = math.degrees(yaw_rate_rad)

            # -------- altitude hold --------
            vz = (target_alt_z - pos.z_val) * TARGET_ALT_KP

            client.moveByVelocityBodyFrameAsync(
                forward_speed, 0.0, vz, DT,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg)
            )

            time.sleep(DT)

    except KeyboardInterrupt:
        pass
    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        if SHOW_DEPTH_WINDOW:
            cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
