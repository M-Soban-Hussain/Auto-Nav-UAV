import sys
import time

# Change this to your AirSim PythonClient folder
sys.path.append(r"D:\Soban\AirSim-main\PythonClient")

import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

try:
    while True:
        pos = client.getMultirotorState().kinematics_estimated.position
        print(
            f"\rX: {pos.x_val:.2f}   Y: {pos.y_val:.2f}   Z: {pos.z_val:.2f}",
            end="",
            flush=True,
        )
        time.sleep(0.05)
except KeyboardInterrupt:
    print()
