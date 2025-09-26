# prompt.py
def llm1_safety_prompt(user_input: str) -> str:
    prompt = f"""
You are a safety and ethics Robotics AI assistant.

Task:
- it will obey all the asimilov's laws of robotics
- Review the following user input for any potential safety, ethical, or legal issues.
- If the input is safe and has no issues, respond with "True".
- If the input has any issues, respond with "False".
- return ONLY "True" or "False" with no additional text.

User input: "{user_input}"
"""
    return prompt

# ........................................................................................................................

def build_joint_planning_prompt(
        robot_class: str,
        user_task: str,
        aruco_base: dict,
        robots_current_joints: dict | None = None,
        robots_current_cartesians: dict | None = None,
        eqx: str = "",
        eqy: str = "",
) -> str:
    robots_current_joints = robots_current_joints or {"UR": [], "Kinova": [], "Niryo": []}
    robots_current_cartesians = robots_current_cartesians or {"UR": None, "Kinova": None, "Niryo": None}

    # Helper functions
    def f2(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "0.00"

    def fmt_xyz3(xyz):
        if not xyz:
            return ["0.00", "0.00", "0.00"]
        return [f2(v) for v in xyz]

    # Build joints block
    rcj_block = (
        "  {\n"
        f'    "UR":     [{",".join(str(j) for j in robots_current_joints.get("UR", []))}],\n'
        f'    "Kinova": [{",".join(str(j) for j in robots_current_joints.get("Kinova", []))}],\n'
        f'    "Niryo":  [{",".join(str(j) for j in robots_current_joints.get("Niryo", []))}]\n'
        "  }"
    )

    # Build cartesians block
    rcc_items = []
    for name in ["UR", "Kinova", "Niryo"]:
        pose = robots_current_cartesians.get(name)
        if pose:
            xyz = fmt_xyz3(pose.get("xyz_m"))
            rpy = fmt_xyz3(pose.get("rxyz_deg"))
            rcc_items.append(
                f'    "{name}": {{"xyz_m":[{",".join(xyz)}], "rxyz_deg":[{",".join(rpy)}]}}'
            )
        else:
            rcc_items.append(f'    "{name}": null')
    rcc_block = "{\n" + ",\n".join(rcc_items) + "\n  }"

    # Build the prompt
    prompt = f"""You are a robotics planning assistant. Compute ONLY the robot's cartesian-space waypoints.

ArUco ID mapping:
- 0 = table
- 1 = niryo
- 2 = Kinova
- 3 = ur
- Any other detected marker corresponds to a task target.

Instructions:
- Analyze the task and its subtasks, and generate a sequence of cartesian-space waypoints for the specified robot to accomplish the task.
- Use robot_class to determine which robot to plan for.
- Use aruco_base to understand the position and orientation of the table and robots.
- Waypoints must be collision-aware and reachable for the chosen robot.
- Keep poses as [x, y, z, rx, ry, rz] with meters and degrees (right-handed, RXYZ intrinsic).

TASK BREAKDOWN:
Two main task types:

1. PICK:
   - approach the part
   - descent 
   - close the gripper
   - ascent

2. PICK AND PLACE:
   - approach the part
   - descent 
   - close the gripper
   - ascent
   - move to target
   - descent 
   - open the gripper
   - ascent

ROBOT-SPECIFIC PARAMETERS(dont change the valuse only calculate X and Y):
- RX, RY, RZ values are static (only X and Y change based on target position)
- Z values change for ascent/descent operations

For Kinova: rx=180, ry=0, rz=90; z_ascent=0.23, z_descent=0.04
For Niryo:  rx=0.00, ry=1.50, rz=0.00; z_ascent=0.3, z_descent=0.1
For UR:     rx=0.0, ry=3.1416, rz=0.0; z_ascent=0.5, z_descent=0.250

COORDINATE TRANSFORMATION:
- X and Y coordinates should be calculated using the calibration equations provided
- Substitute target world coordinates (w_x, w_y) from aruco_base into the equations:
  - X = {eqx} (substitute w_x and w_y from target marker position)
  - Y = {eqy} (substitute w_x and w_y from target marker position)
- Where w_x and w_y are the world X and Y coordinates of the target marker from aruco_base

INPUTS:
- robot_class: {robot_class}
- user_task: "{user_task}"
- aruco_base: {aruco_base}
- calibration_equations: eqx="{eqx}", eqy="{eqy}"
- robots_current_joints:
{rcj_block}
- robots_current_cartesians:
{rcc_block}

OUTPUT FORMAT (valid JSON):
{{
  "waypoints_cartesian": [
    {{"name": "approach", "pose": [calculated_x, calculated_y, z_ascent, rx, ry, rz]}},
    {{"name": "descent", "pose": [calculated_x, calculated_y, z_descent, rx, ry, rz]}},
    {{"name": "close_gripper", "pose": [calculated_x, calculated_y, z_descent, rx, ry, rz]}},
    {{"name": "ascent", "pose": [calculated_x, calculated_y, z_ascent, rx, ry, rz]}}
  ]
}}
"""

    return prompt

# ..........................................................................................................................
def generate_robot_program_prompt(waypoints: str, robot_type: str = "auto") -> str:


    base_instructions = """You are an expert robotics programmer. Given a user request, write a complete Python program to control a robot arm. The program must:
- Include all necessary imports, connection setup, and commands to perform the task.
- Be well-structured with clear functions and comments explaining each step.
- Handle exceptions robustly (connection errors, motion errors, missing frames).
- Ensure a clean disconnection from the robot in finally blocks.
- Validate inputs and add brief logging prints for each major action.
"""

    # Robot-specific code templates
    robot_templates = {
        "niryo": """
For Niryo robots, use the pyniryo library:

```python
from pyniryo import *
import math
import sys
import time

# ---------- User-configurable ----------
NIRYO_IP = "192.168.1.15"

try:
    robot = NiryoRobot(NIRYO_IP)
    robot.calibrate_auto()
    robot.move(JointsPosition(0, 0, 0, 0, -1.5, 0))

    # Your waypoint implementation here

finally:
    if 'robot' in locals():
        robot.close_connection()
```
""",

        "kinova": """
For Kinova robots, use the Kortex API and update waypoints in the steps variable:

```python
import sys
import os
import threading
import time
import math

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

TIMEOUT = 30000

def get_kinova_current_joints(base_cyclic):
    fb = base_cyclic.RefreshFeedback()
    return [act.position for act in fb.actuators[:7]]

def get_kinova_current_cartesian(base_cyclic):
    fb = base_cyclic.RefreshFeedback()
    return [
        fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z,
        fb.base.tool_pose_theta_x, fb.base.tool_pose_theta_y, fb.base.tool_pose_theta_z,
    ]

def wait_for_action_end(base):
    event = threading.Event()
    def handler(notification):
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            event.set()
    sub_id = base.OnNotificationActionTopic(handler, Base_pb2.NotificationOptions())
    event.wait(TIMEOUT)
    base.Unsubscribe(sub_id)

def move_to_cartesian_pose(base, xyz_m, rxyz_deg, frame=Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE):
    action = Base_pb2.Action()
    action.name = "reach_pose"

    rp = action.reach_pose
    pose = rp.target_pose

    if hasattr(pose, "reference_frame"):
        pose.reference_frame = frame
    elif hasattr(rp, "reference_frame"):
        rp.reference_frame = frame

    pose.x, pose.y, pose.z = xyz_m
    pose.theta_x, pose.theta_y, pose.theta_z = rxyz_deg

    base.ExecuteAction(action)
    wait_for_action_end(base)

def control_gripper(base, open_gripper):
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.value = 0.0 if open_gripper else 1.0
    base.SendGripperCommand(cmd)
    time.sleep(1.0)

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    args = utilities.parseConnectionArguments()

    # Update this steps variable with your waypoints
    steps = [
        ({"xyz": [0.88, -0.134, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        # Add more waypoints here
    ]

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        try:
            j = get_kinova_current_joints(base_cyclic)
            t = get_kinova_current_cartesian(base_cyclic)
            print("Start joints (deg):", [f"{v:.2f}" for v in j])
            print("Start TCP (m,rad):", [f"{v:.4f}" for v in t])
        except Exception as e:
            print("Feedback read error:", e)

        for i, (pose_cmd, open_state) in enumerate(steps, 1):
            xyz = pose_cmd["xyz"]
            rxyz_deg = pose_cmd["rxyz_deg"]
            print(f"Step {i}: Move to XYZ={xyz}, RXYZ(deg)={rxyz_deg} | Gripper: {'Open' if open_state else 'Closed'}")
            move_to_cartesian_pose(base, xyz, rxyz_deg)
            control_gripper(base, open_gripper=open_state)

            tcp = get_kinova_current_cartesian(base_cyclic)
            print("Actual TCP now (m,rad):", [f"{v:.4f}" for v in tcp])

        print("Cartesian sequence complete.")

if __name__ == "__main__":
    main()
```
""",

        "ur": """
For Universal Robots (UR), use socket communication and update waypoints in the waypoints_cartesian variable:

```python
import socket
import math

UR5_IP = "192.168.1.13"
UR5_PORT = 30002
GRIPPER_IP = "192.168.1.13"
GRIPPER_PORT = 63352

def get_ur_current_joints(ip, port=30003):
    ur_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ur_socket.connect((ip, port))
    data = ur_socket.recv(1108)
    ur_socket.close()
    joints = []
    for i in range(6):
        idx = 252 + i * 8
        joints.append(float.fromhex(data[idx:idx+8].hex()))
    return joints

def send_ur5(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR5_IP, UR5_PORT))
        s.sendall((cmd + "\\n").encode('utf-8'))

def send_gripper(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((GRIPPER_IP, GRIPPER_PORT))
        s.sendall((cmd + "\\n").encode('utf-8'))
        try:
            resp = s.recv(1024)
            print(f"Gripper response: {resp.decode().strip()}")
        except Exception:
            pass

def main():
    # Update this waypoints_cartesian variable with your waypoints
    waypoints_cartesian = [
        # Example: [x, y, z, rx, ry, rz]
        # Add your waypoints here
    ]

    try:
        current_joints = get_ur_current_joints(UR5_IP)
        print(f"Current joints: {current_joints}")

        for i, waypoint in enumerate(waypoints_cartesian):
            print(f"Moving to waypoint {i+1}: {waypoint}")
            # Implement movement commands here

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

only the python code no additional text.
"""
    }

    # Auto-detect robot type from waypoints if not specified
    if robot_type == "auto":
        waypoints_lower = waypoints.lower()
        if "niryo" in waypoints_lower or "pyniryo" in waypoints_lower:
            robot_type = "niryo"
        elif "kinova" in waypoints_lower or "kortex" in waypoints_lower:
            robot_type = "kinova"
        elif "ur" in waypoints_lower or "universal" in waypoints_lower:
            robot_type = "ur"
        else:
            robot_type = "niryo"  # default fallback

    # Build the complete prompt
    full_prompt = base_instructions + "\n"

    if robot_type in robot_templates:
        full_prompt += robot_templates[robot_type]
    else:
        # Generic template if robot type not recognized
        full_prompt += "\nImplement the robot control using the appropriate library for your robot platform.\n"

    full_prompt += f"\nThe given waypoints are:\n\"\"\"{waypoints}\"\"\"\n"
    full_prompt += "\nUse these waypoints to implement the complete robot control program following the template above."

    return full_prompt