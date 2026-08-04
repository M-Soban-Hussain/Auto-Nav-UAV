import sys
import time
import keyboard
import math

sys.path.append(r"D:\Soban\AirSim-main\PythonClient")
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("READY")

alt_hold = False
target_z = 0

DT = 0.03
VEL = 20.0

# 🔥 TRIPLED YAW SPEED (NOW DIRECT RATE)
YAW_SPEED = 75.0   # rad/sec (fast)

print("""
SPACE = altitude hold
W/S = altitude adjust
A/D = yaw
ARROWS = movement
""")

try:
    while True:

        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        z = pos.z_val

        # -----------------------
        # ALT HOLD TOGGLE
        # -----------------------
        if keyboard.is_pressed("space"):
            alt_hold = not alt_hold
            time.sleep(0.2)

            if alt_hold:
                target_z = z
                print("ALT HOLD ON")
            else:
                print("ALT HOLD OFF")

        # -----------------------
        # ALTITUDE CONTROL
        # -----------------------
        vz = 0
        if alt_hold:
            if keyboard.is_pressed("w"):
                target_z -= 0.3
            if keyboard.is_pressed("s"):
                target_z += 0.3

            vz = (target_z - z) * 3.0

        # -----------------------
        # YAW CONTROL (FIXED + RESPONSIVE)
        # -----------------------
        yaw_rate = 0.0

        if keyboard.is_pressed("a"):
            yaw_rate = -YAW_SPEED
        elif keyboard.is_pressed("d"):
            yaw_rate = YAW_SPEED

        # -----------------------
        # MOVEMENT
        # -----------------------
        vx = 0
        vy = 0

        if keyboard.is_pressed("up"):
            vx = VEL
        elif keyboard.is_pressed("down"):
            vx = -VEL

        if keyboard.is_pressed("right"):
            vy = VEL
        elif keyboard.is_pressed("left"):
            vy = -VEL

        vx_body = vx
        vy_body = vy

        # -----------------------
        # APPLY CONTROL
        # -----------------------
        client.moveByVelocityBodyFrameAsync(
            vx_body,
            vy_body,
            vz,
            DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )

        time.sleep(DT)

except KeyboardInterrupt:
    print("Landing...")

finally:
    client.landAsync()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Stopped")