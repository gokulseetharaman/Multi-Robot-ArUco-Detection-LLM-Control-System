import socket
import json
import math
import time
import struct
import re
from contextlib import closing
from pyniryo import NiryoRobot
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

# ---------- small helpers ----------
def rad2deg(vals):
    return [v * 180.0 / math.pi for v in vals]

def _print_pose(name, xyz, rxyz_deg):
    x, y, z = xyz
    rx, ry, rz = rxyz_deg
    print(f"{name} TCP pose:")
    print(f"  Position (m):  X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    print(f"  Rotation (°):  Rx={rx:.2f}, Ry={ry:.2f}, Rz={rz:.2f}\n")

def _pick_local_ip_for_peer(peer_ip: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 80))
        return s.getsockname()[0]
    finally:
        s.close()

# =========================================================
# NIRYO — Cartesian via pyniryo
# =========================================================
def get_niryo_current_cartesian(ip: str):
    """
    Returns: (xyz_m, rxyz_deg)
    """
    robot = None
    try:
        robot = NiryoRobot(ip)
        pose = robot.get_pose()  # meters + radians (roll/pitch/yaw)
        if hasattr(pose, "roll"):
            rpy_rad = [pose.roll, pose.pitch, pose.yaw]
        else:
            rpy_rad = [pose.rX, pose.rY, pose.rZ]
        xyz = (pose.x, pose.y, pose.z)
        rxyz_deg = rad2deg(rpy_rad)
        return xyz, rxyz_deg
    finally:
        try:
            if robot:
                robot.close_connection()
        except Exception:
            pass

# =========================================================
# UR — Cartesian via reverse socket (preferred) + realtime (30003) fixed offsets
# =========================================================
def _rad2deg(vals):
    return [v * 180.0 / math.pi for v in vals]

# ---------- 1) Reverse-socket method (preferred) ----------
def get_ur_current_cartesian_reverse(ip: str,
                                     primary_ports=(30002, 30001),
                                     reverse_port: int = 55001,
                                     timeout: float = 8.0,
                                     retries: int = 2,
                                     verbose: bool = True):
    """
    Asks the robot (via Primary/Secondary interface) to connect back and send get_actual_tcp_pose().
    Accepts 'p[...]', '[...]', or '(...)'. Returns (xyz_m, rxyz_deg).
    """
    def parse_line(line: str):
        cleaned = line.replace("(", "[").replace(")", "]").strip()
        if cleaned.startswith("p["):
            cleaned = cleaned[1:]
        m = re.search(r"\[([^\]]+)\]", cleaned)
        if not m:
            raise RuntimeError(f"UR pose parse failed: {line!r}")
        vals = [float(x.strip()) for x in m.group(1).split(",")]
        if len(vals) < 6:
            raise RuntimeError(f"UR pose length invalid: {vals!r}")
        xyz = vals[:3]
        rxyz_deg = _rad2deg(vals[3:6])
        return xyz, rxyz_deg

    local_ip = _pick_local_ip_for_peer(ip)
    if verbose:
        print(f"[UR] Reverse method: local_ip={local_ip}, reverse_port={reverse_port}, try_ports={primary_ports}")

    last_err = None
    for prim in primary_ports:
        for attempt in range(1, retries + 1):
            if verbose:
                print(f"[UR] Attempt {attempt}/{retries} via Primary port {prim} ...")
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((local_ip, reverse_port))
                srv.listen(1)
                srv.settimeout(timeout)

                urscript = f"""
def send_pose_back():
  p = get_actual_tcp_pose()
  socket_open("{local_ip}", {reverse_port})
  socket_send_line(to_str(p))
  socket_close()
end
send_pose_back()
"""
                # send script to robot
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as cli:
                    cli.settimeout(timeout)
                    cli.connect((ip, prim))
                    cli.sendall(urscript.encode("utf-8"))
                    time.sleep(0.2)

                # accept reverse connection
                try:
                    conn, addr = srv.accept()
                    if verbose:
                        print(f"[UR] Robot connected back from {addr}")
                except socket.timeout as e:
                    last_err = e
                    if verbose:
                        print(f"[UR] Reverse accept timeout on {prim}: {e}")
                    continue

                with conn:
                    conn.settimeout(timeout)
                    buf = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        if b"\n" in buf:
                            break

                line = buf.decode("utf-8", errors="ignore").strip()
                if verbose:
                    print(f"[UR] Raw reverse line: {line!r}")
                return parse_line(line)

            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[UR] Reverse attempt failed on port {prim}: {e}")
                time.sleep(0.3)
            finally:
                try:
                    srv.close()
                except Exception:
                    pass

    raise RuntimeError(f"UR reverse connection failed after tries on {primary_ports}: {last_err}")

# ---------- 2) Realtime stream (30003) — fixed offsets from your 2nd script ----------
def _recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)

def get_ur_current_cartesian_rt_fixed(ip: str, port: int = 30003, timeout: float = 2.0):
    """
    Reads one realtime packet and extracts tool pose at known offsets:
    - TCP pose (x,y,z,rx,ry,rz) starts at byte 444 (6 doubles)
    Returns (xyz_m, rxyz_deg).
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(timeout)
        s.connect((ip, port))
        # 4-byte big-endian total length, followed by body
        hdr = _recvall(s, 4)
        if len(hdr) != 4:
            raise RuntimeError("UR RT header incomplete")
        pkt_len = struct.unpack("!i", hdr)[0]
        body = _recvall(s, pkt_len - 4)
        if len(body) != (pkt_len - 4):
            raise RuntimeError("UR RT packet incomplete")

    # Fixed offset parse (matches your working snippet)
    base = body  # body only (header removed). Offset 444 is from full packet start (hdr+body),
                 # but since we've removed 4 bytes of header, use 444-4 = 440 here.
    start = 440  # 444 total offset minus the 4-byte length header
    vals = []
    for i in range(6):
        idx = start + i * 8
        vals.append(struct.unpack_from('!d', base, idx)[0])

    x, y, z, rx, ry, rz = vals
    return [x, y, z], _rad2deg([rx, ry, rz])

def get_ur_current_joints_rt_fixed(ip: str, port: int = 30003, timeout: float = 2.0):
    """
    Optional: joints via fixed offset (6 doubles starting at byte 252 in full packet → 248 in body).
    Returns list[6] in radians.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(timeout)
        s.connect((ip, port))
        hdr = _recvall(s, 4)
        if len(hdr) != 4:
            raise RuntimeError("UR RT header incomplete")
        pkt_len = struct.unpack("!i", hdr)[0]
        body = _recvall(s, pkt_len - 4)
        if len(body) != (pkt_len - 4):
            raise RuntimeError("UR RT packet incomplete")

    start = 248  # 252 total offset minus 4-byte header
    joints = []
    for i in range(6):
        idx = start + i * 8
        joints.append(struct.unpack_from('!d', body, idx)[0])
    return joints

# ---------- 3) Wrapper that tries reverse then fixed-offset realtime ----------
def get_ur_current_cartesian(ip: str,
                             primary_ports=(30002, 30001),
                             reverse_port: int = 55001,
                             timeout: float = 8.0,
                             verbose: bool = True):
    """
    Tries reverse socket first; if it fails, uses fixed-offset realtime (30003).
    Returns (xyz_m, rxyz_deg).
    """
    try:
        return get_ur_current_cartesian_reverse(ip,
                                                primary_ports=primary_ports,
                                                reverse_port=reverse_port,
                                                timeout=timeout,
                                                verbose=verbose)
    except Exception as e1:
        if verbose:
            print(f"[UR] Reverse failed → trying realtime (fixed offsets): {e1}")
        # realtime fallback
        return get_ur_current_cartesian_rt_fixed(ip, port=30003, timeout=2.0)

# =========================================================
# KINOVA — Cartesian via BaseCyclic feedback (Kortex SDK)
# =========================================================
def get_kinova_current_cartesian(base_cyclic: BaseCyclicClient):
    """
    Returns: (xyz_m, rxyz_deg) from tool_pose_* fields (meters + degrees).
    """
    fb = base_cyclic.RefreshFeedback()
    xyz = (fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z)
    rxyz_deg = (fb.base.tool_pose_theta_x, fb.base.tool_pose_theta_y, fb.base.tool_pose_theta_z)
    return xyz, rxyz_deg

# =========================================================
# All robots — convenience aggregator
# =========================================================
FAKE_VALUES = {
    "niryo":  {"xyz_m": (0.30, 0.00, 0.20), "rxyz_deg": (180.0, 0.0, 90.0)},
    "ur":     {"xyz_m": (0.40, 0.10, 0.20), "rxyz_deg": (180.0, -10.0, 90.0)},
    "kinova": {"xyz_m": (0.20, -0.10, 0.30), "rxyz_deg": (170.0, 5.0, 85.0)},
}

def get_all_robot_cartesians(niryo_ip: str,
                             ur_ip: str,
                             base_cyclic: BaseCyclicClient,
                             ur_primary_ports=(30002, 30001),
                             ur_reverse_port: int = 55001,
                             verbose: bool = True):
    out = {}

    # Niryo
    try:
        n_xyz, n_rdeg = get_niryo_current_cartesian(niryo_ip)
        _print_pose("Niryo", n_xyz, n_rdeg)
        out["niryo"] = {"xyz_m": n_xyz, "rxyz_deg": n_rdeg}
    except Exception as e:
        print(f"Niryo: ERROR → {e}  → returning FAKE values")
        out["niryo"] = FAKE_VALUES["niryo"]

    # UR
    try:
        u_xyz, u_rdeg = get_ur_current_cartesian(
            ur_ip,
            primary_ports=ur_primary_ports,   # <-- FIXED: was primary_port
            reverse_port=ur_reverse_port,
            verbose=verbose
        )
        _print_pose("UR", u_xyz, u_rdeg)
        out["ur"] = {"xyz_m": u_xyz, "rxyz_deg": u_rdeg}
    except Exception as e:
        print(f"UR: ERROR → {e}  → returning FAKE values")
        out["ur"] = FAKE_VALUES["ur"]

    # Kinova
    try:
        k_xyz, k_rdeg = get_kinova_current_cartesian(base_cyclic)
        _print_pose("Kinova", k_xyz, k_rdeg)
        out["kinova"] = {"xyz_m": k_xyz, "rxyz_deg": k_rdeg}
    except Exception as e:
        print(f"Kinova: ERROR → {e}  → returning FAKE values")
        out["kinova"] = FAKE_VALUES["kinova"]

    return out

# =========================================================
# Simple main that RETURNS ALL and PRINTS
# =========================================================
def fetch_all_cartesian(niryo_ip: str,
                        ur_ip: str,
                        base_cyclic: BaseCyclicClient,
                        ur_primary_ports=(30002, 30001),
                        ur_reverse_port: int = 55001,
                        verbose: bool = False):
    """
    Fetch Cartesian poses from Niryo, UR, and Kinova.
    Returns dict: { "niryo": {...}, "ur": {...}, "kinova": {...} }
    Each entry has keys 'xyz_m' (tuple) and 'rxyz_deg' (tuple/list).
    """
    results = get_all_robot_cartesians(
        niryo_ip=niryo_ip,
        ur_ip=ur_ip,
        base_cyclic=base_cyclic,
        ur_primary_ports=ur_primary_ports,
        ur_reverse_port=ur_reverse_port,
        verbose=verbose
    )

    # Pretty print
    print("=== SUMMARY (meters / degrees) ===")
    for name in ("niryo", "ur", "kinova"):
        xyz = results[name]["xyz_m"]
        rxyz = results[name]["rxyz_deg"]
        _print_pose(name.capitalize(), xyz, rxyz)

    return results


# if __name__ == "__main__":
#     # Build a Kortex session for Kinova feedback
#     import utilities
#     import os
#     import sys
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#
#     args = utilities.parseConnectionArguments()
#     with utilities.DeviceConnection.createTcpConnection(args) as router:
#         base_cyclic = BaseCyclicClient(router)
#
#         # IPs — change if needed
#         niryo_ip = "192.168.1.15"
#         ur_ip    = "192.168.1.13"
#
#         all_cart = get_all_robot_cartesians(
#             niryo_ip=niryo_ip,
#             ur_ip=ur_ip,
#             base_cyclic=base_cyclic,
#             ur_primary_ports=(30002, 30001),
#             ur_reverse_port=55001,
#             verbose=False
#         )
#
#         # Pretty print single-line + dict echo
#         print("=== SUMMARY (meters / degrees) ===")
#         for name in ("niryo", "ur", "kinova"):
#             xyz = all_cart[name]["xyz_m"]
#             rxyz = all_cart[name]["rxyz_deg"]
#             _print_pose(name.capitalize(), xyz, rxyz)
#
#         print("Raw dict:\n", all_cart)
