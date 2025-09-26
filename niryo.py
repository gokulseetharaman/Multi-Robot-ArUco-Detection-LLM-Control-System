from pyniryo import NiryoRobot, PoseObject as Pose
import math, time

IP = "192.168.1.15"
deg = math.radians

robot = NiryoRobot(IP)
robot.calibrate_auto()
robot.clear_collision_detected()

robot.update_tool()

# Move to pick pose and grasp
robot.release_with_tool()
time.sleep(2)
robot.move(Pose(x=0.335, y=-0.044, z=0.3, roll=0, pitch=1.5, yaw=0))# Ensure tool is open
robot.move(Pose(x=0.335, y=0.044, z=0.1, roll=0, pitch=1.5, yaw=0))# Ensure tool is open
robot.grasp_with_tool()

robot.move(Pose(x=0.335, y=0.044, z=0.3, roll=0, pitch=1.5, yaw=-0))
time.sleep(3)
robot.move(Pose(x=0.208, y=0.238, z=0.3, roll=0, pitch=1.5, yaw=0))# Ensure tool is open
robot.move(Pose(x=0.208, y=0.238, z=0.1, roll=0, pitch=1.5, yaw=0))# Ensure tool is open
robot.release_with_tool()

robot.move(Pose(x=0.335, y=0.044, z=0.3, roll=0, pitch=1.5, yaw=0))# Ensure tool is open

# Move to place pose and release

# robot.release_with_tool()