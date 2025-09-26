import sys
import os
import threading
import time
import math

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

TIMEOUT = 30000

# ------------- Helpers -------------


def get_kinova_current_joints(base_cyclic):
    fb = base_cyclic.RefreshFeedback()
    return [act.position for act in fb.actuators[:7]]

def get_kinova_current_cartesian(base_cyclic):
    fb = base_cyclic.RefreshFeedback()
    # Kortex cyclic feedback exposes TCP pose on fb.base
    return [
        fb.base.tool_pose_x,
        fb.base.tool_pose_y,
        fb.base.tool_pose_z,
        fb.base.tool_pose_theta_x,
        fb.base.tool_pose_theta_y,
        fb.base.tool_pose_theta_z,
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

    rp   = action.reach_pose          # ConstrainedPose
    pose = rp.target_pose             # Pose (no reference_frame in your SDK)

    # If your SDK exposes reference_frame on target_pose or on reach_pose, set it; otherwise skip.
    if hasattr(pose, "reference_frame"):
        pose.reference_frame = frame
    elif hasattr(rp, "reference_frame"):
        rp.reference_frame = frame
    # else: defaults to BASE frame in many Kortex releases

    pose.x, pose.y, pose.z = xyz_m
    pose.theta_x = rxyz_deg[0]
    pose.theta_y = rxyz_deg[1]
    pose.theta_z = rxyz_deg[2]

    # (optional) simple speed constraint if available in your proto
    # if hasattr(rp, "constraint") and hasattr(rp.constraint, "oneof_type") and hasattr(rp.constraint.oneof_type, "speed"):
    #     rp.constraint.oneof_type.speed.translation = 0.2
    #     rp.constraint.oneof_type.speed.orientation = 15.0

    base.ExecuteAction(action)
    wait_for_action_end(base)


def control_gripper(base, open_gripper):
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.value = 0.0 if open_gripper else 1.0   # 0.0=open, 1.0=closed (typical)
    base.SendGripperCommand(cmd)
    time.sleep(1.0)

# ------------- Demo sequence (Cartesian) -------------
def main():
    # add project root to path (same as your joint script)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    args = utilities.parseConnectionArguments()

    # Each step: ({"xyz":[x,y,z], "rxyz_deg":[rx,ry,rz]}, gripper_open_bool)
    # Example pick pattern near front of robot; adjust to your workspace
    steps = [
        # ({"xyz":[0.1,  0.05, 0.15], "rxyz_deg":[180, 0, 90 ]}, True),   # approach above target (open)
        ({"xyz": [0.514, 0.189, 0.20], "rxyz_deg": [180, 0, 90]}, True),  #
        ({"xyz": [0.514, -0.325, 0.04], "rxyz_deg": [180, 0, 90]}, True),  #
        ({"xyz": [0.514, -0.325, 0.04], "rxyz_deg": [180, 0, 90]}, False),  #
        ({"xyz": [0.514, 0.189, 0.20], "rxyz_deg": [180, 0, 90]}, False),  #
    ]

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Optional: print current joint + tcp before start
        try:
            j = get_kinova_current_joints(base_cyclic)
            t = get_kinova_current_cartesian(base_cyclic)
            print("Start joints (deg):", [f"{v:.2f}" for v in j])
            print("Start TCP   (m,rad):", [f"{v:.4f}" for v in t])
        except Exception as e:
            print("Feedback read error (safe to continue):", e)

        for i, (pose_cmd, open_state) in enumerate(steps, 1):
            xyz = pose_cmd["xyz"]
            rxyz_deg = pose_cmd["rxyz_deg"]
            print(f"\nStep {i}: Move to XYZ={xyz}, RXYZ(deg)={rxyz_deg} | Gripper: {'Open' if open_state else 'Closed'}")
            move_to_cartesian_pose(base, xyz, rxyz_deg)
            control_gripper(base, open_gripper=open_state)

            # Read back actual tcp
            tcp = get_kinova_current_cartesian(base_cyclic)
            print("  Actual TCP now (m,rad):", [f"{v:.4f}" for v in tcp])

        print("\nCartesian sequence complete.")

if __name__ == "__main__":
    main()
