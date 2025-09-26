import socket
import math
import time

UR5_IP = "192.168.1.13"
UR5_PORT = 30002
GRIPPER_IP = "192.168.1.13"
GRIPPER_PORT = 63352

import socket

def get_ur_current_joints(ip, port=30003):
    # 30003 is RTDE interface for most UR robots (adjust if different)
    ur_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ur_socket.connect((ip, port))
    data = ur_socket.recv(1108)  # 1108 bytes is typical for UR data packet
    ur_socket.close()
    # Joint values are 6 doubles starting from byte 252 (RTDE/secondary interface)
    joints = []
    for i in range(6):
        # Each double is 8 bytes
        idx = 252 + i * 8
        joints.append(float.fromhex(data[idx:idx+8].hex()))
    return joints


def send_ur5(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR5_IP, UR5_PORT))
        s.sendall((cmd + "\n").encode('utf-8'))

def send_gripper(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as g:
        g.connect((UR5_IP, GRIPPER_PORT))
        g.sendall(b"SET SPE 255\n")
        g.sendall((cmd + '\n').encode('utf-8'))
        resp = g.recv(1024).decode().strip()
        print(f"Gripper | Sent: {cmd} | Resp: {resp}")

#
# send_ur5("movej(p[0, 0, 00, 0.0, 0, 0.0], a=0.05, v=0.25)")
# # #1
send_gripper("SET POS 0")

send_ur5("movep(p[0.340, -0.449, 0.400, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")
time.sleep(3)
send_ur5("movep(p[0.400, -0.680, 0.250, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")
time.sleep(5)
send_gripper("SET POS 500")
send_ur5("movep(p[0.340, -0.449, 0.400, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")
time.sleep(3)
send_ur5("movep(p[-0.082, -0.741, 0.500, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")
time.sleep(5)
send_ur5("movep(p[-0.082, -0.741, 0.250, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")
time.sleep(5)
send_gripper("SET POS 0")
time.sleep(3)
send_ur5("movep(p[0.340, -0.449, 0.400, 0.0, 3.1416, 0.0], a=0.05, v=0.25)")



#2
