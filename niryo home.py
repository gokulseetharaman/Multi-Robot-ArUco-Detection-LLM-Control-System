from pyniryo import *
import math
import sys
import time

# ---------- User-configurable ----------
NIRYO_IP = "192.168.1.15"

robot = NiryoRobot("192.168.1.15")
robot.clear_collision_detected()

robot.calibrate_auto()
robot.move(JointsPosition(0,0,0,0,-1.5,0))
pose = robot.get_pose()  # meters + radians (roll/pitch/yaw)
print(pose)
