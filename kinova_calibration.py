from __future__ import annotations
import os
# ---------- make numpy/opencv deterministic ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import math
import hashlib
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List

# External deps you already have
import aruco_core as ac
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

np.random.seed(42)
cv2.setNumThreads(1)

# ---------------- Patch for RouterClient ----------------
def patch_router_client():
    original_init = RouterClient.__init__
    def new_init(self, transport, error_callback=None):
        self.notificationService = {}
        original_init(self, transport, error_callback)
    RouterClient.__init__ = new_init
patch_router_client()

# ---------------- Configuration ----------------
TIMEOUT = 60000
SETTLE_SEC = 20
CONNECTION_RETRIES = 3
RECONNECT_DELAY = 2
DEFAULT_IP = "192.168.1.10"
TCP_PORT = 10000
USERNAME = "admin"
PASSWORD = "admin"
REQUIRE_ALL_STEPS = True  # add near other constants


# Camera Config (from your ac module + constants here)
CAM_A_INDEX = ac.CAM_A_INDEX
CAM_B_INDEX = 2
CAP_W       = ac.CAP_W
CAP_H       = ac.CAP_H
HFOV_A_DEG  = ac.HFOV_A_DEG
HFOV_B_DEG  = ac.HFOV_B_DEG
MARKER_ID   = 2
MARKER_LEN_M = ac.MARKER_LEN_M
DICT_ID      = ac.DICT_ID
WORLD_ANCHOR_TRY_ORDER = ac.WORLD_ANCHOR_TRY_ORDER

# === ArUco & anchor determinism (edit these to match your setup) ===
EXPECTED_MARKER_LEN_M = MARKER_LEN_M        # set literal if you want, e.g., 0.040
EXPECTED_DICT_ID      = DICT_ID             # set literal if you want, e.g., cv2.aruco.DICT_4X4_50
FORCE_ANCHOR_CAMERA   = "B"                 # choose "A" or "B" and stick to it
ANCHOR_ERR_MAX_PX     = 0.5                 # gate for accepting the anchor
ANCHOR_TOL_MM         = 1.0                 # tolerate <=1 mm drift vs. first anchor in this pose
ANCHOR_TOL_DEG        = 0.5                 # tolerate <=0.5° drift vs. first anchor in this pose

# Capture parameters (deterministic)
TARGET_VALID_FRAMES = 40       # exact number of accepted frames per pose
REPROJ_THRESHOLD_PX = 0.5      # accept frame if marker reproj err below this
PER_POSE_TIMEOUT_S  = 10.0     # hard stop (safeguard) per pose, but fixed-count drives capture

# ------------- Helpers -------------
def _hash(arr) -> str:
    return hashlib.sha1(np.asarray(arr, float).tobytes()).hexdigest()[:10]

def intrinsics_from_fov(w: int, h: int, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = np.deg2rad(hfov_deg)
    fx = w / (2 * np.tan(fov_x / 2))
    fy = fx
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    D = np.zeros((5,), dtype=float)  # 1-D distortion vector
    return K, D

def compute_reprojection_error(corners, rvec, tvec, K, D, marker_length):
    try:
        obj_points = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)
        img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        img_points = img_points.reshape(-1, 2)
        corners = corners.reshape(-1, 2)
        err = np.mean(np.sqrt(np.sum((img_points - corners) ** 2, axis=1)))
        return float(err)
    except Exception:
        return float('inf')

# ---------- Strict intrinsics loader (NO fallback) ----------
def load_calibration_npz_strict(path: str):
    with np.load(path) as data:
        def _pick(d, *keys):
            for k in keys:
                if k in d: return np.array(d[k], float)
            return None
        K = _pick(data, "camera_matrix","K","mtx","intrinsic_matrix","cameraMatrix","M1","arr_0")
        D = _pick(data, "dist_coeffs","distCoeffs","D","dist","D1","arr_1")
        if K is None or D is None:
            raise KeyError(f"{path}: missing K/D keys; has {data.files}")
        if K.shape != (3,3):
            raise ValueError(f"{path}: bad K shape {K.shape}")
        return K, np.ravel(D).astype(float)

def try_lock_camera_settings(cap: cv2.VideoCapture):
    # Best-effort; different backends behave differently. Ignore failures.
    try:
        # Some backends: 0.25 = manual exposure, 0.75 = auto
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)   # tune to your lighting
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    except Exception:
        pass

def initialize_cameras():
    # Sanity locks
    assert abs(MARKER_LEN_M - EXPECTED_MARKER_LEN_M) < 1e-9, \
        f"MARKER_LEN_M mismatch: {MARKER_LEN_M} vs EXPECTED_MARKER_LEN_M={EXPECTED_MARKER_LEN_M}"
    assert DICT_ID == EXPECTED_DICT_ID, "DICT_ID mismatch—check your aruco dictionary"

    print(f"Opening cameras: A={CAM_A_INDEX}, B={CAM_B_INDEX}")
    capA = ac.open_cam(CAM_A_INDEX, CAP_W, CAP_H)
    capB = ac.open_cam(CAM_B_INDEX, CAP_W, CAP_H)
    if capA is None or capB is None:
        raise RuntimeError("Failed to open cameras")

    # Lock camera settings (best effort)
    try_lock_camera_settings(capA)
    try_lock_camera_settings(capB)

    # Verify frames come through
    for cap, name in [(capA, 'A'), (capB, 'B')]:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Camera {name} not providing valid frames")
        print(f"Camera {name}: {frame.shape} OK")

    calibA_path = os.getenv("CALIB_A_PATH", "calib_cam0.npz")
    calibB_path = os.getenv("CALIB_B_PATH", "calib_cam2.npz")

    # STRICT: do not fallback to HFOV (avoid silent scale drift)
    K_A, D_A = load_calibration_npz_strict(calibA_path)
    K_B, D_B = load_calibration_npz_strict(calibB_path)

    # Fingerprints for sanity
    print("K_A hash:", _hash(K_A), "D_A hash:", _hash(D_A))
    print("K_B hash:", _hash(K_B), "D_B hash:", _hash(D_B))

    return capA, capB, K_A, D_A, K_B, D_B

# ---------- Anchor pose helpers ----------
def rvec_to_rot(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3,1))
    return R

def rotation_angle_deg(R):
    tr = np.clip((np.trace(R)-1)/2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def anchor_pose_delta(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    a,b: dict with 'rvec','tvec' (from detect_aruco)
    returns (d_trans_mm, d_angle_deg)
    """
    Ra = rvec_to_rot(a['rvec']); Rb = rvec_to_rot(b['rvec'])
    dR = Ra.T @ Rb
    d_ang = rotation_angle_deg(dR)
    ta = np.asarray(a['tvec'], float).reshape(3); tb = np.asarray(b['tvec'], float).reshape(3)
    d_trans_mm = float(np.linalg.norm(tb - ta) * 1000.0)
    return d_trans_mm, d_ang

# ---------- Anchor selection (force single camera + freeze) ----------
def _anchor_info(res, K, D, anchor_id, marker_len):
    if str(anchor_id) not in res['markers']:
        return None
    m = res['markers'][str(anchor_id)]
    err = m.get("reproj_err_px", float('inf'))
    if np.isinf(err) and all(k in m for k in ("corners","rvec","tvec")):
        err = compute_reprojection_error(m['corners'], m['rvec'], m['tvec'], K, D, marker_len)
    return {"err": float(err), "m": m}

def capture_once(frameA: np.ndarray, frameB: np.ndarray, K_A: np.ndarray, D_A: np.ndarray,
                 K_B: np.ndarray, D_B: np.ndarray, capture_idx: int) -> Dict[str, Any]:

    resA = ac.detect_aruco(frameA, K_A, D_A, MARKER_LEN_M, DICT_ID)
    resB = ac.detect_aruco(frameB, K_B, D_B, MARKER_LEN_M, DICT_ID)

    anchor_id = WORLD_ANCHOR_TRY_ORDER[0]
    infoA = _anchor_info(resA, K_A, D_A, anchor_id, MARKER_LEN_M)
    infoB = _anchor_info(resB, K_B, D_B, anchor_id, MARKER_LEN_M)

    anchor_valid = False
    anchor_pose   = None

    # FORCE single camera for anchor determinism
    chosen = infoB if FORCE_ANCHOR_CAMERA == "B" else infoA
    if chosen is not None and chosen["err"] < ANCHOR_ERR_MAX_PX:
        anchor_valid = True
        anchor_pose  = {"rvec": chosen["m"].get("rvec"), "tvec": chosen["m"].get("tvec")}
        # remove the anchor from the non-chosen camera so fusion uses just one origin
        if FORCE_ANCHOR_CAMERA == "B" and str(anchor_id) in resA['markers']:
            del resA['markers'][str(anchor_id)]
        if FORCE_ANCHOR_CAMERA == "A" and str(anchor_id) in resB['markers']:
            del resB['markers'][str(anchor_id)]
    else:
        anchor_valid = False

    # Flag anchors
    ac.ensure_world_anchor(resA, WORLD_ANCHOR_TRY_ORDER)
    ac.ensure_world_anchor(resB, WORLD_ANCHOR_TRY_ORDER)

    # Fuse
    fused = ac.fuse_two_results(resA, resB)

    # Build outputs
    per_id, aruco_base = {}, {}
    for mid in sorted(fused.keys(), key=int):
        m = fused[mid]
        x, y, z = map(float, m["xyz_world"])
        per_id[int(mid)] = {
            "world_xyz": (x, y, z),
            "src": m.get("src"),
            "error_px": float(m.get("reproj_err_px", 0.0)),
            "cands": [
                {"src": c["src"], "cam_xyz": tuple(map(float, c["cam_xyz"]))}
                for c in m.get("cands", [])
            ],
        }
        aruco_base[str(mid)] = [x, y, z]

    return {
        "capture_idx": capture_idx,
        "world_anchor": {
            "anchor_valid": anchor_valid,
            "pose": anchor_pose,  # used for freeze check
            "source": FORCE_ANCHOR_CAMERA if anchor_valid else None,
        },
        "per_id": per_id,
        "aruco_base": aruco_base,
    }

# ---------- Deterministic per-pose averaging with anchor freeze ----------
def median_of_means(arr: np.ndarray, chunks: int = 5) -> float:
    n = len(arr)
    k = (n // chunks) * chunks
    if k == 0:  # fallback
        return float(np.median(arr))
    parts = arr[:k].reshape(chunks, -1).mean(axis=1)
    return float(np.median(parts))

def average_world_xy(capA: cv2.VideoCapture, capB: cv2.VideoCapture, K_A: np.ndarray, D_A: np.ndarray,
                     K_B: np.ndarray, D_B: np.ndarray, step_idx: int) -> Optional[Tuple[float, float]]:
    xs, ys, zs, raw = [], [], [], []
    frame_count = 0
    t0 = time.time()
    first_anchor_pose = None
    print(f"Collecting marker {MARKER_ID} data for step {step_idx}...")

    while len(xs) < TARGET_VALID_FRAMES and (time.time() - t0) < PER_POSE_TIMEOUT_S:
        okA, frameA = capA.read()
        okB, frameB = capB.read()
        if not (okA and okB):
            continue

        result = capture_once(frameA, frameB, K_A, D_A, K_B, D_B, frame_count)
        frame_count += 1

        if not result["world_anchor"]["anchor_valid"]:
            continue

        # Anchor freeze: lock to first valid anchor pose in this pose
        if first_anchor_pose is None:
            first_anchor_pose = result["world_anchor"]["pose"]
        else:
            d_mm, d_deg = anchor_pose_delta(first_anchor_pose, result["world_anchor"]["pose"])
            if (d_mm > ANCHOR_TOL_MM) or (d_deg > ANCHOR_TOL_DEG):
                # reject this frame due to anchor drift
                continue

        if MARKER_ID in result["per_id"]:
            w_x, w_y, w_z = result["per_id"][MARKER_ID]["world_xyz"]
            reproj_err = result["per_id"][MARKER_ID]["error_px"]
            if reproj_err < REPROJ_THRESHOLD_PX:
                xs.append(w_x); ys.append(w_y); zs.append(w_z)
                raw.append((w_x, w_y, w_z, reproj_err))

    if len(xs) < TARGET_VALID_FRAMES:
        print(f"[WARN] Only {len(xs)}/{TARGET_VALID_FRAMES} valid frames captured at step {step_idx}")
        if len(xs) < 5:  # give up if too few
            return None

    xs, ys, zs = np.array(xs, float), np.array(ys, float), np.array(zs, float)
    mean_x = median_of_means(xs, chunks=5)
    mean_y = median_of_means(ys, chunks=5)
    mean_z = median_of_means(zs, chunks=5)

    # Save raw per-step (deterministic content)
    # if raw:
    #     np.savetxt(f"raw_detections_step_{step_idx}.csv", np.array(raw),
    #                delimiter=",", header="world_x,world_y,world_z,reproj_err_px", comments="")
    print(f"Step {step_idx} avg: ({mean_x:.4f}, {mean_y:.4f}, {mean_z:.4f}) from {len(xs)} frames")
    return mean_x, mean_y

# ---------- Robot helpers ----------
def deg2rad(d): return d * math.pi / 180.0

def get_kinova_current_joints(base_cyclic):
    try:
        fb = base_cyclic.RefreshFeedback()
        return [act.position for act in fb.actuators[:7]]
    except Exception:
        return None

def get_kinova_current_cartesian(base_cyclic):
    try:
        fb = base_cyclic.RefreshFeedback()
        return [
            fb.base.tool_pose_x,
            fb.base.tool_pose_y,
            fb.base.tool_pose_z,
            fb.base.tool_pose_theta_x,
            fb.base.tool_pose_theta_y,
            fb.base.tool_pose_theta_z,
        ]
    except Exception:
        return None

def check_connection(base):
    try:
        base.GetArmState()
        return True
    except Exception:
        return False

def wait_for_action_end(base_cyclic, target_pose, tolerance=0.005, timeout=30):
    start = time.time()
    stable_count = 0
    required_stable_readings = 3
    while time.time() - start < timeout:
        try:
            fb = base_cyclic.RefreshFeedback()
            current = [
                fb.base.tool_pose_x,
                fb.base.tool_pose_y,
                fb.base.tool_pose_z,
                fb.base.tool_pose_theta_x,
                fb.base.tool_pose_theta_y,
                fb.base.tool_pose_theta_z,
            ]
            pos_err = sum(abs(c - t) for c, t in zip(current[:3], target_pose[:3]))
            ang_err = sum(abs(c - t) for c, t in zip(current[3:], target_pose[3:]))
            if pos_err < tolerance and ang_err < deg2rad(2):
                stable_count += 1
                if stable_count >= required_stable_readings:
                    print(f"Reached target pose in {time.time() - start:.2f}s")
                    return
            else:
                stable_count = 0
        except Exception:
            pass
        time.sleep(0.05)
    print(f"[WARN] Motion may not have completed within {timeout}s")

def move_to_cartesian_pose(base, base_cyclic, xyz_m, rxyz_deg, frame=Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE):
    print(f"Moving to: XYZ={xyz_m}, RXYZ(deg)={rxyz_deg}")
    action = Base_pb2.Action()
    action.name = "reach_pose"
    rp = action.reach_pose
    pose = rp.target_pose
    pose.x, pose.y, pose.z = xyz_m
    pose.theta_x, pose.theta_y, pose.theta_z = rxyz_deg
    base.ExecuteAction(action)
    target_pose = list(xyz_m) + rxyz_deg
    wait_for_action_end(base_cyclic, target_pose)

def control_gripper(base, open_gripper):
    try:
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.value = 0.0 if open_gripper else 1.0
        base.SendGripperCommand(cmd)
        time.sleep(1.0)
        print(f"Gripper {'opened' if open_gripper else 'closed'}")
    except Exception as e:
        print(f"[WARN] Gripper control error: {e}")

def connect_to_robot(ip=DEFAULT_IP, port=TCP_PORT, username=USERNAME, password=PASSWORD):
    for attempt in range(CONNECTION_RETRIES):
        try:
            print(f"Connection attempt {attempt + 1}/{CONNECTION_RETRIES} to {ip}:{port}")
            transport = TCPTransport(); transport.connect(ip, port)
            router = RouterClient(transport, lambda kEx: print(f"Router error: {kEx}"))
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = username
            session_info.password = password
            session_info.session_inactivity_timeout = 600000
            session_info.connection_inactivity_timeout = 60000
            session_manager = SessionManager(router); session_manager.CreateSession(session_info)
            base = BaseClient(router); base_cyclic = BaseCyclicClient(router)
            if check_connection(base):
                print("Robot connection established")
                return transport, router, session_manager, base, base_cyclic
            raise ConnectionError("Connection check failed")
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            try: transport.disconnect()
            except Exception: pass
            if attempt < CONNECTION_RETRIES - 1:
                time.sleep(RECONNECT_DELAY)
    raise ConnectionError(f"Failed to connect after {CONNECTION_RETRIES} attempts")




# ---------- Model fitting (deterministic, regularized) ----------
def fit_affine_ridge(world_xy: np.ndarray, robot_xy: np.ndarray, alpha: float = 1e-4):
    # Build design with explicit bias column; standardize only x,y
    Xw = np.asarray(world_xy, float)
    Y  = np.asarray(robot_xy, float)
    ones = np.ones((Xw.shape[0], 1), dtype=float)
    X = np.hstack([Xw, ones])   # [x_w, y_w, 1]

    mu = X[:, :2].mean(axis=0)
    sd = X[:, :2].std(axis=0) + 1e-12
    Xn = X.copy()
    Xn[:, :2] = (X[:, :2] - mu) / sd  # do NOT standardize bias

    I = np.eye(Xn.shape[1])
    I[-1, -1] = 0.0  # do not penalize bias
    W = np.linalg.solve(Xn.T @ Xn + alpha * I, Xn.T @ Y)  # (3x2)

    Ax = W[0,:] / sd[0]
    Ay = W[1,:] / sd[1]
    b  = W[2,:] - mu[0]*Ax - mu[1]*Ay
    A  = np.vstack([Ax, Ay]).T  # 2x2

    svals = np.linalg.svd(Xn, full_matrices=False)[1]
    cond = (svals[0]/svals[-1]) if svals[-1] != 0 else np.inf
    pred = (world_xy @ A.T) + b
    rms = np.sqrt(((pred - robot_xy)**2).mean(axis=0))
    return A, b, rms, cond

def fit_similarity_2d(world_xy: np.ndarray, robot_xy: np.ndarray):
    """
    Least-squares similarity transform: robot ≈ s*R*world + t
    Returns (s, R(2x2), t(2,), rms, cond)
    """
    X = np.asarray(world_xy, float)  # Nx2
    Y = np.asarray(robot_xy, float)  # Nx2
    assert X.shape[0] >= 3 and X.shape == Y.shape

    # Center
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    # Covariance
    C = Xc.T @ Yc / X.shape[0]

    # SVD
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    # Enforce det(R)=+1
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1.0
        R = Vt.T @ U.T

    # Scale
    varX = (Xc**2).sum() / X.shape[0]
    s = (S.sum() / varX) if varX > 0 else 1.0

    # Translation
    t = muY - s * (R @ muX)

    # Predict + RMS
    pred = (s * (X @ R.T)) + t
    rms = np.sqrt(((pred - Y) ** 2).mean(axis=0))

    # Conditioning (of covariance)
    cond = (S.max() / S.min()) if S.min() > 0 else np.inf
    return s, R, t, rms, cond

def fit_quad_ridge(world_xy: np.ndarray, robot_xy: np.ndarray, alpha: float = 1e-3):
    x, y = world_xy[:,0], world_xy[:,1]
    A = np.column_stack([x*x, y*y, x*y, x, y, np.ones_like(x)])
    mu = A.mean(axis=0); sg = A.std(axis=0) + 1e-12
    An = (A - mu) / sg
    I = np.eye(An.shape[1])
    Wx = np.linalg.solve(An.T @ An + alpha*I, An.T @ robot_xy[:,0])
    Wy = np.linalg.solve(An.T @ An + alpha*I, An.T @ robot_xy[:,1])
    s = np.linalg.svd(An, full_matrices=False)[1]
    cond = (s[0]/s[-1]) if s[-1] != 0 else np.inf

    def predict(world_xy_):
        x_, y_ = world_xy_[:,0], world_xy_[:,1]
        A_ = np.column_stack([x_*x_, y_*y_, x_*y_, x_, y_, np.ones_like(x_)])
        An_ = (A_ - mu) / sg
        xr = An_ @ Wx; yr = An_ @ Wy
        return np.column_stack([xr, yr])

    pred = predict(world_xy)
    rms = np.sqrt(((pred - robot_xy)**2).mean(axis=0))
    return (mu, sg, Wx, Wy), predict, rms, cond

def fit_quadratic_equation(world_xy: np.ndarray, robot_xy: np.ndarray, alpha: float = 1e-6):
    """
    Closed-form quadratic (ridge) fit:
      x_r ≈ a1*x_w^2 + a2*y_w^2 + a3*x_w*y_w + a4*x_w + a5*y_w + a6
      y_r ≈ b1*x_w^2 + b2*y_w^2 + b3*x_w*y_w + b4*x_w + b5*y_w + b6
    Returns (coeff_x[6], coeff_y[6], rms_xy[2])
    """
    X = np.asarray(world_xy, float)
    Y = np.asarray(robot_xy, float)
    xw, yw = X[:,0], X[:,1]
    A = np.column_stack([xw*xw, yw*yw, xw*yw, xw, yw, np.ones_like(xw)])

    # Ridge: (A^T A + αI)^{-1} A^T y
    I = np.eye(6)
    M = A.T @ A + alpha * I
    coeff_x = np.linalg.solve(M, A.T @ Y[:,0])
    coeff_y = np.linalg.solve(M, A.T @ Y[:,1])

    pred = np.column_stack([A @ coeff_x, A @ coeff_y])
    rms = np.sqrt(((pred - Y) ** 2).mean(axis=0))
    return coeff_x, coeff_y, rms


# ---------- Main ----------
def main():
    import argparse, os
    import numpy as np

    parser = argparse.ArgumentParser(description="Deterministic robot vision calibration (locked world frame)")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=TCP_PORT)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)
    parser.add_argument("--dry-run", action="store_true", help="Test cameras only, no robot motion")
    # --- add flags BEFORE parse_args ---
    parser.add_argument("--fit", action="store_true", help="Fit and save mapping (overwrites similarity_mapping.npz)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Load saved mapping and only report residuals (no refit)")
    parser.add_argument("--require-all-steps", action="store_true",
                        help="Abort if not all planned poses produced a datapoint")
    args = parser.parse_args()

    # Pose list (unchanged)
    steps = [
        ({"xyz": [0.554,  0.297, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.538, -0.366, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.200, -0.336, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.376, -0.087, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.437,  0.087, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.279, -0.253, 0.04], "rxyz_deg": [180, 0, 90]}, False),
        ({"xyz": [0.357,  0.190, 0.04], "rxyz_deg": [180, 0, 90]}, False),
    ]

    # ---- Cameras ----
    try:
        capA, capB, K_A, D_A, K_B, D_B = initialize_cameras()
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return

    results = []  # (world_x, world_y, robot_x, robot_y)
    transport = router = session_manager = base = base_cyclic = None

    # ---- Robot ----
    try:
        if not args.dry_run:
            transport, router, session_manager, base, base_cyclic = connect_to_robot(
                args.ip, args.port, args.username, args.password
            )
            joints = get_kinova_current_joints(base_cyclic)
            tcp = get_kinova_current_cartesian(base_cyclic)
            if joints and tcp:
                print("Initial joint positions (deg):", [f"{v:.2f}" for v in joints])
                print("Initial TCP pose (m,rad):",   [f"{v:.4f}" for v in tcp])

        for i, (pose_cmd, open_state) in enumerate(steps, 1):
            xyz = pose_cmd["xyz"]; rxyz_deg = pose_cmd["rxyz_deg"]
            print(f"\n--- Step {i}/{len(steps)} ---")
            print(f"Target: XYZ={xyz}, RXYZ(deg)={rxyz_deg}, Gripper: {'Open' if open_state else 'Closed'}")

            if not args.dry_run:
                motion_success = False
                for attempt in range(CONNECTION_RETRIES):
                    try:
                        if not check_connection(base):
                            print("Connection lost, attempting to reconnect...")
                            if session_manager:
                                session_manager.CloseSession()
                            if transport:
                                transport.disconnect()
                            time.sleep(1)
                            transport, router, session_manager, base, base_cyclic = connect_to_robot(
                                args.ip, args.port, args.username, args.password
                            )
                        move_to_cartesian_pose(base, base_cyclic, xyz, rxyz_deg)
                        control_gripper(base, open_gripper=open_state)
                        motion_success = True
                        break
                    except Exception as e:
                        print(f"Motion attempt {attempt + 1} failed: {e}")
                        if attempt < CONNECTION_RETRIES - 1:
                            time.sleep(RECONNECT_DELAY)
                if not motion_success:
                    print(f"[WARN] Skipping step {i} due to motion failures")
                    continue
                tcp_actual = get_kinova_current_cartesian(base_cyclic)
                if tcp_actual:
                    print(f"Actual TCP: {[f'{v:.4f}' for v in tcp_actual[:3]]}")

            print(f"Settling for {SETTLE_SEC}s before capture...")
            time.sleep(SETTLE_SEC)

            fused_pos = average_world_xy(capA, capB, K_A, D_A, K_B, D_B, i)
            if fused_pos is None:
                print(f"[WARN] Step {i}: No marker average, skipping data point")
                continue
            w_x, w_y = fused_pos
            results.append((w_x, w_y, xyz[0], xyz[1]))
            print(f"Step {i} SUCCESS: World=({w_x:.4f}, {w_y:.4f}) -> Robot=({xyz[0]:.4f}, {xyz[1]:.4f})")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if session_manager:
            try:
                session_manager.CloseSession(); print("Robot session closed")
            except Exception: pass
        if transport:
            try:
                transport.disconnect(); print("Robot disconnected")
            except Exception: pass
        try:
            capA.release(); capB.release(); cv2.destroyAllWindows()
            print("Cameras released")
        except Exception: pass

    # ---- Save collected data ----
    print(f"\nCalibration sequence complete. Collected {len(results)} valid data points.")
    if not results:
        print("No valid data points collected!")
        return

    if args.require_all_steps and len(results) != len(steps):
        print(f"[ABORT] Only {len(results)}/{len(steps)} points captured. "
              f"Not fitting to keep outputs identical across runs.")
        return

    arr = np.array(results, dtype=float)
    # np.savetxt("world_kinova_xy_id2.csv", arr, delimiter=",",
    #            header="world_x,world_y,kinova_x,kinova_y", comments="")
    # np.save("world_kinova_xy_id2.npy", arr)
    # print("Saved: world_kinova_xy_id2.csv, world_kinova_xy_id2.npy")

    print("\nCollected data points:")
    for i, (wx, wy, rx, ry) in enumerate(results, 1):
        print(f"  {i}: World({wx:.4f}, {wy:.4f}) -> Robot({rx:.4f}, {ry:.4f})")

    # ---- Fit or Verify-only (do this ONCE) ----
    world = arr[:, :2]
    robot = arr[:, 2:]

    if args.verify_only:
        if not os.path.exists("similarity_mapping.npz"):
            print("[ERROR] verify-only requested but similarity_mapping.npz not found. Run with --fit first.")
            return
        data = np.load("similarity_mapping.npz")
        s, R, t = float(data["s"]), data["R"], data["t"]
        pred = (s * (world @ R.T)) + t
        rms = np.sqrt(((pred - robot) ** 2).mean(axis=0))
        print("\n=== VERIFY-ONLY (Similarity) ===")
        print(f"RMS vs saved mapping: X={rms[0]:.6f}m, Y={rms[1]:.6f}m")
        return

    # Fit (or refit if --fit) and save (Similarity used as a baseline diagnostic)
    if args.fit or not os.path.exists("similarity_mapping.npz"):
        s, R, t, srms, scond = fit_similarity_2d(world, robot)
        print("\n=== SIMILARITY (Fit) ===")
        print("s =", s)
        print("R =\n", R)
        print("t =", t)
        print(f"RMS: X={srms[0]:.6f}m, Y={srms[1]:.6f}m | cond≈{scond:.2e}")
        # np.savez("similarity_mapping.npz", s=s, R=R, t=t)
    else:
        data = np.load("similarity_mapping.npz")
        s, R, t = float(data["s"]), data["R"], data["t"]
        pred = (s * (world @ R.T)) + t
        srms = np.sqrt(((pred - robot) ** 2).mean(axis=0))
        print("\n=== USING SAVED SIMILARITY (No refit) ===")
        print(f"RMS vs saved mapping: X={srms[0]:.6f}m, Y={srms[1]:.6f}m")

    # ---- Quadratic (ridge) mapping: PRINT + SAVE + RETURN ----
    # Design: [x^2, y^2, x*y, x, y, 1]
    xw, yw = world[:, 0], world[:, 1]
    A = np.column_stack([xw*xw, yw*yw, xw*yw, xw, yw, np.ones_like(xw)])

    alpha = 1e-6  # tiny ridge for stability
    I6 = np.eye(6)
    M = A.T @ A + alpha * I6
    coeff_x = np.linalg.solve(M, A.T @ robot[:, 0])
    coeff_y = np.linalg.solve(M, A.T @ robot[:, 1])

    # Diagnostics
    pred_x = A @ coeff_x
    pred_y = A @ coeff_y
    qrms = np.sqrt(np.mean((pred_x - robot[:, 0])**2)), np.sqrt(np.mean((pred_y - robot[:, 1])**2))

    # Round for display only
    cx = np.round(coeff_x, 6)
    cy = np.round(coeff_y, 6)

    eq_x = (f"x_r = {cx[0]:.6f}*x_w^2 + {cx[1]:.6f}*y_w^2 + {cx[2]:.6f}*x_w*y_w "
            f"+ {cx[3]:.6f}*x_w + {cx[4]:.6f}*y_w + {cx[5]:.6f}")
    eq_y = (f"y_r = {cy[0]:.6f}*x_w^2 + {cy[1]:.6f}*y_w^2 + {cy[2]:.6f}*x_w*y_w "
            f"+ {cy[3]:.6f}*x_w + {cy[4]:.6f}*y_w + {cy[5]:.6f}")

    print("\n=== QUADRATIC (ridge) MAPPING ===")
    print(eq_x)
    print(eq_y)
    print(f"RMS: X={qrms[0]:.6f}m, Y={qrms[1]:.6f}m")

    # np.savez("quadratic_mapping.npz",
    #          coeff_x=coeff_x, coeff_y=coeff_y, rms=np.array(qrms),
    #          design='[x^2, y^2, x*y, x, y, 1]')

    # Return equations so you can import this module and call main() to get them
    return eq_x, eq_y


#
if __name__ == "__main__":
    main()
