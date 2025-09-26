import socket
import numpy as np
from pyniryo import NiryoRobot
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
import sys
import os
import  struct
import utilities

def get_niryo_current_joints(ip):
    try:
        robot = NiryoRobot(ip)
        joints = robot.get_joints()
        robot.close_connection()
        return [f"{j:.2f}" for j in joints]   # <-- formatted
    except Exception as e:
        print(f"Niryo error: {e}")
        return [0.00, -0.60, 0.90, 0.00, 0.85, 0.00]




def get_ur_current_joints(ip: str, port: int = 30003, timeout: float = 1.0):
    """
    Read UR joint positions (radians) from the Secondary Interface (port 30003).
    Handles big-endian doubles and variable packet sizes across firmware.
    Returns list of 6 strings formatted to .2f.
    """
    def recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def parse_at_offsets(buf, offsets):
        for off in offsets:
            if off + 8*6 <= len(buf):
                vals = struct.unpack_from("!6d", buf, off)  # big-endian 6 doubles
                if all(-7.0 <= v <= 7.0 for v in vals):
                    return [f"{v:.2f}" for v in vals]
        return None

    def scan_for_plausible(buf):
        # fall back: scan every 8 bytes for 6 consecutive big-endian doubles in [-7,7]
        end = len(buf) - 8*6
        for off in range(0, end, 8):
            try:
                vals = struct.unpack_from("!6d", buf, off)
            except struct.error:
                break
            if all(-7.0 <= v <= 7.0 for v in vals):
                return [f"{v:.2f}" for v in vals]
        return None

    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            # read 4-byte big-endian length header
            hdr = recvall(s, 4)
            if len(hdr) != 4:
                raise RuntimeError("Incomplete header from UR.")
            pkt_len = struct.unpack("!i", hdr)[0]
            body = recvall(s, pkt_len - 4)
            if len(body) != pkt_len - 4:
                raise RuntimeError(f"Incomplete packet: expected {pkt_len-4}, got {len(body)}")

            buf = hdr + body  # keep header so offsets like 252 match legacy docs

            # Try known offsets first (varies by firmware/model)
            # 252 is common on CB3 for q_actual; include other nearby candidates.
            joints = parse_at_offsets(buf, offsets=[252, 244, 260, 444, 440, 480])
            if joints is None:
                joints = scan_for_plausible(buf)

            if joints is None:
                raise RuntimeError("Could not locate joint block in packet.")

            return joints

    except Exception as e:
        print(f"UR error: {e}")
        # Safe, consistent fallback (strings with .2f)
        fallback = [-1.20, -0.90, -0.50, -1.60, 1.05, 0.20]
        return [f"{v:.2f}" for v in fallback]



def get_kinova_current_joints(base_cyclic):
    try:
        feedback = base_cyclic.RefreshFeedback()
        return [f"{act.position:.2f}" for act in feedback.actuators[:7]]
    except Exception as e:
        print(f"Kinova error: {e}")
        return [0.35, -0.70, 1.10, -1.30, 0.85, 0.25, 0.00]


def get_all_robot_joints(base_cyclic: BaseCyclicClient,
                         niryo_ip: str = "192.168.1.15",
                         ur_ip: str = "192.168.1.13"):
    niryo_joints  = get_niryo_current_joints(niryo_ip)
    ur_joints     = get_ur_current_joints(ur_ip)
    kinova_joints = get_kinova_current_joints(base_cyclic)

    # Do not print here; just return data
    return {
        "niryo":  niryo_joints,
        "ur":     ur_joints,
        "kinova": kinova_joints
    }

