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

# ===== Niryo import shim (works with pyniryo2 or pyniryo) =====
NiryoRobot = None
PoseClass = None
try:
    # pyniryo2
    from pyniryo import NiryoRobot as _NR2, Pose as _Pose2
    NiryoRobot = _NR2
    PoseClass = _Pose2
except Exception:
    try:
        # pyniryo (legacy)
        from pyniryo import NiryoRobot as _NR1, PoseObject as _Pose1
        NiryoRobot = _NR1
        PoseClass = _Pose1
    except Exception:
        pass

# External deps
import aruco_core as ac

np.random.seed(42)
cv2.setNumThreads(1)

# ---------------- Configuration ----------------
SETTLE_SEC = 20
CONNECTION_RETRIES = 3
RECONNECT_DELAY = 2
DEFAULT_IP = "192.168.1.15"

# Camera Config
CAM_A_INDEX = ac.CAM_A_INDEX
CAM_B_INDEX = 2
CAP_W       = ac.CAP_W
CAP_H       = ac.CAP_H
HFOV_A_DEG  = ac.HFOV_A_DEG
HFOV_B_DEG  = ac.HFOV_B_DEG
MARKER_ID   = 1
MARKER_LEN_M = ac.MARKER_LEN_M
DICT_ID      = ac.DICT_ID
WORLD_ANCHOR_TRY_ORDER = ac.WORLD_ANCHOR_TRY_ORDER

# === ArUco & anchor determinism ===
EXPECTED_MARKER_LEN_M = MARKER_LEN_M
EXPECTED_DICT_ID      = DICT_ID
FORCE_ANCHOR_CAMERA   = "B"  # "A" or "B"
ANCHOR_ERR_MAX_PX     = 0.35
ANCHOR_TOL_MM         = 0.8
ANCHOR_TOL_DEG        = 0.4

# Capture parameters (deterministic)
TARGET_VALID_FRAMES = 40
REPROJ_THRESHOLD_PX = 0.45
PER_POSE_TIMEOUT_S  = 12.0

# ===== Common Niryo pose (meters, radians) =====
NIRYO_Z   = 0.12
NIRYO_RPY = (0.001, 1.500, 0.001)  # (roll, pitch, yaw) radians

# ===== Pose reach tolerances =====
POSE_POS_TOL = 0.008   # 8 mm
POSE_ANG_TOL = math.radians(3.0)  # 3 deg in radians

# ------------- Helpers -------------
def _hash(arr) -> str:
    return hashlib.sha1(np.asarray(arr, float).tobytes()).hexdigest()[:10]

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

# ---------- Strict intrinsics loader ----------
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
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    except Exception:
        pass

def initialize_cameras():
    assert abs(MARKER_LEN_M - EXPECTED_MARKER_LEN_M) < 1e-9, \
        f"MARKER_LEN_M mismatch: {MARKER_LEN_M} vs EXPECTED_MARKER_LEN_M={EXPECTED_MARKER_LEN_M}"
    assert DICT_ID == EXPECTED_DICT_ID, "DICT_ID mismatch—check your aruco dictionary"

    print(f"Opening cameras: A={CAM_A_INDEX}, B={CAM_B_INDEX}")
    capA = ac.open_cam(CAM_A_INDEX, CAP_W, CAP_H)
    capB = ac.open_cam(CAM_B_INDEX, CAP_W, CAP_H)
    if capA is None or capB is None:
        raise RuntimeError("Failed to open cameras")

    try_lock_camera_settings(capA)
    try_lock_camera_settings(capB)

    for cap, name in [(capA, 'A'), (capB, 'B')]:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Camera {name} not providing valid frames")
        print(f"Camera {name}: {frame.shape} OK")

    calibA_path = os.getenv("CALIB_A_PATH", "calib_cam0.npz")
    calibB_path = os.getenv("CALIB_B_PATH", "calib_cam2.npz")

    K_A, D_A = load_calibration_npz_strict(calibA_path)
    K_B, D_B = load_calibration_npz_strict(calibB_path)

    print("K_A hash:", _hash(K_A), "D_A hash:", _hash(D_A))
    print("K_B hash:", _hash(K_B), "D_B hash:", _hash(D_B))

    # quick visibility sanity (first frame only)
    okA, frA = capA.read(); okB, frB = capB.read()
    if okA and okB:
        sa = ac.detect_aruco(frA, K_A, D_A, MARKER_LEN_M, DICT_ID)
        sb = ac.detect_aruco(frB, K_B, D_B, MARKER_LEN_M, DICT_ID)
        print(f"CamA sees IDs: {sorted(map(int, sa['markers'].keys()))}")
        print(f"CamB sees IDs: {sorted(map(int, sb['markers'].keys()))}")

    return capA, capB, K_A, D_A, K_B, D_B

# ---------- Anchor pose helpers ----------
def rvec_to_rot(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3,1))
    return R

def rotation_angle_deg(R):
    tr = np.clip((np.trace(R)-1)/2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def anchor_pose_delta(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float]:
    Ra = rvec_to_rot(a['rvec']); Rb = rvec_to_rot(b['rvec'])
    dR = Ra.T @ Rb
    d_ang = rotation_angle_deg(dR)
    ta = np.asarray(a['tvec'], float).reshape(3); tb = np.asarray(b['tvec'], float).reshape(3)
    d_trans_mm = float(np.linalg.norm(tb - ta) * 1000.0)
    return d_trans_mm, d_ang

def _get_marker(d: Dict, key):
    # Try both int and str forms
    if key in d:
        return d[key]
    ks = str(key)
    if ks in d:
        return d[ks]
    try:
        ki = int(key)
        if ki in d:
            return d[ki]
    except Exception:
        pass
    return None

def _has_key(d: Dict, key) -> bool:
    return _get_marker(d, key) is not None

def _del_key(d: Dict, key):
    # Delete both if present
    for k in (key, str(key)):
        if k in d:
            del d[k]
    try:
        ki = int(key)
        if ki in d:
            del d[ki]
    except Exception:
        pass


def _anchor_info(res, K, D, anchor_id, marker_len):
    m = _get_marker(res['markers'], anchor_id)
    if m is None:
        return None
    err = m.get("reproj_err_px", None)
    if err is None or np.isinf(err):
        if all(k in m for k in ("corners","rvec","tvec")):
            err = compute_reprojection_error(m['corners'], m['rvec'], m['tvec'], K, D, marker_len)
        else:
            # If we can't compute, treat as large error
            err = float('inf')
    return {"err": float(err), "m": m}

def capture_once(frameA: np.ndarray, frameB: np.ndarray, K_A: np.ndarray, D_A: np.ndarray,
                 K_B: np.ndarray, D_B: np.ndarray, capture_idx: int) -> Dict[str, Any]:

    resA = ac.detect_aruco(frameA, K_A, D_A, MARKER_LEN_M, DICT_ID)
    resB = ac.detect_aruco(frameB, K_B, D_B, MARKER_LEN_M, DICT_ID)

    if capture_idx % 10 == 0:
        def _sorted_ids(res):
            try:
                return sorted([int(k) for k in res['markers'].keys()])
            except Exception:
                return sorted(list(res['markers'].keys()))
        print(f"[dbg] frame {capture_idx}: CamA IDs={_sorted_ids(resA)}, CamB IDs={_sorted_ids(resB)}")

    anchor_id = WORLD_ANCHOR_TRY_ORDER[0]
    infoA = _anchor_info(resA, K_A, D_A, anchor_id, MARKER_LEN_M)
    infoB = _anchor_info(resB, K_B, D_B, anchor_id, MARKER_LEN_M)

    anchor_valid = False
    anchor_pose   = None

    chosen = infoB if FORCE_ANCHOR_CAMERA == "B" else infoA
    if chosen is not None and chosen["err"] < ANCHOR_ERR_MAX_PX:
        anchor_valid = True
        anchor_pose  = {"rvec": chosen["m"].get("rvec"), "tvec": chosen["m"].get("tvec")}
        # remove anchor from the other camera so world is unambiguous
        if FORCE_ANCHOR_CAMERA == "B":
            _del_key(resA['markers'], anchor_id)
        else:
            _del_key(resB['markers'], anchor_id)
    else:
        if capture_idx % 10 == 1:
            errA = None if infoA is None else infoA["err"]
            errB = None if infoB is None else infoB["err"]
            print(f"[anchor] invalid at frame {capture_idx}: "
                  f"CamA_err={errA} CamB_err={errB} (max {ANCHOR_ERR_MAX_PX})")

    ac.ensure_world_anchor(resA, WORLD_ANCHOR_TRY_ORDER)
    ac.ensure_world_anchor(resB, WORLD_ANCHOR_TRY_ORDER)

    fused = ac.fuse_two_results(resA, resB)

    per_id, aruco_base = {}, {}
    for mid in sorted(fused.keys(), key=lambda x: int(x)):
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
            "pose": anchor_pose,
            "source": FORCE_ANCHOR_CAMERA if anchor_valid else None,
        },
        "per_id": per_id,
        "aruco_base": aruco_base,
    }


# ---------- Deterministic per-pose averaging with anchor freeze ----------
def median_of_means(arr: np.ndarray, chunks: int = 5) -> float:
    n = len(arr)
    k = (n // chunks) * chunks
    if k == 0: return float(np.median(arr))
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
        if not (okA and okB): continue

        result = capture_once(frameA, frameB, K_A, D_A, K_B, D_B, frame_count)
        frame_count += 1

        if not result["world_anchor"]["anchor_valid"]: continue

        # freeze anchor pose for this step
        if first_anchor_pose is None:
            first_anchor_pose = result["world_anchor"]["pose"]
        else:
            d_mm, d_deg = anchor_pose_delta(first_anchor_pose, result["world_anchor"]["pose"])
            if (d_mm > ANCHOR_TOL_MM) or (d_deg > ANCHOR_TOL_DEG):
                continue

        if MARKER_ID not in result["per_id"]:
            if frame_count % 10 == 1:
                print(f"[skip] target ID {MARKER_ID} not found; fused IDs: {sorted(result['per_id'].keys())}")
            continue

        w_x, w_y, w_z = result["per_id"][MARKER_ID]["world_xyz"]
        reproj_err = result["per_id"][MARKER_ID]["error_px"]
        if reproj_err < REPROJ_THRESHOLD_PX:
            xs.append(w_x); ys.append(w_y); zs.append(w_z)
            raw.append((w_x, w_y, w_z, reproj_err))

    if len(xs) < TARGET_VALID_FRAMES:
        print(f"[WARN] Only {len(xs)}/{TARGET_VALID_FRAMES} valid frames captured at step {step_idx}")
        if len(xs) < 5:
            return None

    xs, ys, zs = np.array(xs, float), np.array(ys, float), np.array(zs, float)
    mean_x = median_of_means(xs, chunks=5)
    mean_y = median_of_means(ys, chunks=5)
    mean_z = median_of_means(zs, chunks=5)


    print(f"Step {step_idx} avg: ({mean_x:.4f}, {mean_y:.4f}, {mean_z:.4f}) from {len(xs)} frames")
    return mean_x, mean_y

# ---------- Angle helpers ----------
def _wrap_pi(a):
    """wrap to [-pi, pi]"""
    return (a + math.pi) % (2*math.pi) - math.pi

def _maybe_deg_to_rad(rpy_tuple):
    """Auto-detect deg vs rad. If any abs angle > pi, assume degrees."""
    r, p, y = rpy_tuple
    if max(abs(r), abs(p), abs(y)) > math.pi*1.1:
        return (math.radians(r), math.radians(p), math.radians(y)), True
    return (r, p, y), False

# ---------- Niryo helpers ----------
def connect_to_niryo(ip=DEFAULT_IP):
    if NiryoRobot is None or PoseClass is None:
        raise ImportError("pyniryo2 or pyniryo not installed (or Pose class unavailable)")
    last_err = None
    for attempt in range(CONNECTION_RETRIES):
        try:
            print(f"Connecting to Niryo at {ip} (attempt {attempt+1}/{CONNECTION_RETRIES})...")
            n = NiryoRobot(ip)
            try: n.set_learning_mode(False)
            except Exception: pass
            try: n.update_tool()
            except Exception: pass
            print("Niryo connection established")
            return n
        except Exception as e:
            last_err = e
            print(f"Connection attempt {attempt+1} failed: {e}")
            time.sleep(RECONNECT_DELAY)
    raise ConnectionError(f"Failed to connect to Niryo after {CONNECTION_RETRIES} attempts: {last_err}")

def niryo_move_pose(n: NiryoRobot, x, y, z, roll, pitch, yaw):
    print(f"Moving to: XYZ=({x:.4f}, {y:.4f}, {z:.4f}), RPY(rad)=({roll:.3f}, {pitch:.3f}, {yaw:.3f})")
    # Build pose class compatible with whichever SDK we loaded
    try:
        p = PoseClass(x, y, z, roll, pitch, yaw)
        # Most SDKs accept move_pose(Pose)
        if hasattr(n, "move_pose"):
            n.move_pose(p)
        else:
            # Fallback to numeric signature
            n.move_pose(x, y, z, roll, pitch, yaw)
    except TypeError:
        # If PoseClass signature differs, use numeric signature
        n.move_pose(x, y, z, roll, pitch, yaw)

def niryo_get_pose(n: NiryoRobot):
    try:
        p = n.get_pose()
        # Return as tuple; handle both SDKs fields
        rpy = (getattr(p, "roll"), getattr(p, "pitch"), getattr(p, "yaw"))
        (r, pch, yw), was_deg = _maybe_deg_to_rad(rpy)
        return [float(p.x), float(p.y), float(p.z), float(r), float(pch), float(yw)]
    except Exception:
        return None

def _pose_errors(meas, target_xyz, target_rpy):
    px, py, pz, rr, pp, yy = meas
    tx, ty, tz = target_xyz
    tr, tp, tyaw = target_rpy
    pos_err_vec = np.array([px-tx, py-ty, pz-tz], float)
    pos_err = float(np.linalg.norm(pos_err_vec))
    ang_err_vec = np.array([_wrap_pi(rr-tr), _wrap_pi(pp-tp), _wrap_pi(yy-tyaw)], float)
    ang_err = float(np.linalg.norm(ang_err_vec))
    return pos_err, ang_err, pos_err_vec, ang_err_vec

def wait_until_pose_reached(n: NiryoRobot, target_xyz, target_rpy,
                            pos_tol=POSE_POS_TOL, ang_tol=POSE_ANG_TOL,
                            timeout=20.0, poll=0.1):
    t0 = time.time()
    stable = 0
    while time.time() - t0 < timeout:
        meas = niryo_get_pose(n)
        if meas is not None:
            pos_err, ang_err, pv, av = _pose_errors(meas, target_xyz, target_rpy)
            print(f"  pose chk: |dp|={pos_err:.4f} m  |dθ|={math.degrees(ang_err):.2f}°   "
                  f"dp={pv}  dθ(rad)={av}")
            if (pos_err <= pos_tol) and (ang_err <= ang_tol):
                stable += 1
                if stable >= 3:
                    return True, meas
            else:
                stable = 0
        time.sleep(poll)
    return False, niryo_get_pose(n)

def niryo_control_gripper(n: NiryoRobot, open_gripper: bool):
    try:
        if open_gripper:
            if hasattr(n, "open_gripper"): n.open_gripper()
            elif hasattr(n, "open_gripper_with_tool"): n.open_gripper_with_tool()
            else: print("[WARN] No open_gripper method found")
        else:
            if hasattr(n, "close_gripper"): n.close_gripper()
            elif hasattr(n, "close_gripper_with_tool"): n.close_gripper_with_tool()
            else: print("[WARN] No close_gripper method found")
        time.sleep(1.0)
        print(f"Gripper {'opened' if open_gripper else 'closed'}")
    except Exception as e:
        print(f"[WARN] Gripper control error: {e}")

# ---------- Model fitting ----------
def fit_similarity_2d(world_xy: np.ndarray, robot_xy: np.ndarray):
    X = np.asarray(world_xy, float)  # Nx2
    Y = np.asarray(robot_xy, float)  # Nx2
    assert X.shape[0] >= 3 and X.shape == Y.shape

    muX = X.mean(axis=0); muY = Y.mean(axis=0)
    Xc  = X - muX;        Yc  = Y - muY

    C = Xc.T @ Yc / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1.0
        R = Vt.T @ U.T

    varX = (Xc**2).sum() / X.shape[0]
    s = (S.sum() / varX) if varX > 0 else 1.0
    t = muY - s * (R @ muX)

    pred = (s * (X @ R.T)) + t
    rms = np.sqrt(((pred - Y) ** 2).mean(axis=0))
    cond = (S.max() / S.min()) if S.min() > 0 else np.inf
    return s, R, t, rms, cond

# ---------- Main ----------
def main():
    import argparse, os as _os
    parser = argparse.ArgumentParser(description="Deterministic Niryo vision calibration (locked world frame)")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--dry-run", action="store_true", help="Test cameras only, no robot motion")
    parser.add_argument("--fit", action="store_true", help="Fit & save mapping (overwrites similarity_mapping_niryo.npz)")
    parser.add_argument("--verify-only", action="store_true", help="Load saved mapping and only report residuals")
    parser.add_argument("--require-all-steps", action="store_true", help="Abort fit if not all poses succeeded")
    args = parser.parse_args()

    # Pose list (XY vary; Z + RPY fixed)
    xy_steps = [
        (0.299,  0.099),
        (0.233,  0.181),
        (0.207, -0.201),
        (0.376, -0.087),
        (0.335, -0.044),
    ]
    steps = [({"xyz": [x, y, NIRYO_Z], "rpy": list(NIRYO_RPY)}, False) for (x, y) in xy_steps]

    # ---- Cameras ----
    try:
        capA, capB, K_A, D_A, K_B, D_B = initialize_cameras()
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return

    results = []  # (world_x, world_y, robot_x_meas, robot_y_meas)
    n = None

    # ---- Robot ----
    try:
        if not args.dry_run:
            n = connect_to_niryo(args.ip)

        for i, (pose_cmd, open_state) in enumerate(steps, 1):
            x_cmd, y_cmd, z_cmd = pose_cmd["xyz"]
            r_cmd, p_cmd, yv_cmd = pose_cmd["rpy"]
            print(f"\n--- Step {i}/{len(steps)} ---")
            print(f"Target: XY=({x_cmd:.4f},{y_cmd:.4f}), Z={z_cmd:.4f}, RPY(rad)=({r_cmd:.3f},{p_cmd:.3f},{yv_cmd:.3f}), "
                  f"Gripper: {'Open' if open_state else 'Closed'}")

            # Move & VERIFY pose reached
            p_meas = None
            if not args.dry_run:
                moved = False
                for attempt in range(CONNECTION_RETRIES):
                    try:
                        try:
                            if hasattr(n, "clear_collision_detected"):
                                n.clear_collision_detected()
                        except Exception:
                            pass
                        niryo_move_pose(n, x_cmd, y_cmd, z_cmd, r_cmd, p_cmd, yv_cmd)
                        ok, p_meas = wait_until_pose_reached(
                            n,
                            target_xyz=(x_cmd, y_cmd, z_cmd),
                            target_rpy=(r_cmd, p_cmd, yv_cmd),
                            timeout=25.0
                        )
                        if not ok:
                            raise RuntimeError("Pose not reached within tolerance/time")
                        niryo_control_gripper(n, open_gripper=open_state)
                        moved = True
                        break
                    except Exception as e:
                        print(f"Motion attempt {attempt + 1} failed: {e}")
                        time.sleep(RECONNECT_DELAY)
                if not moved:
                    print(f"[WARN] Skipping step {i} due to motion failures")
                    continue
                if p_meas:
                    print(f"Measured TCP: x={p_meas[0]:.4f}, y={p_meas[1]:.4f}, z={p_meas[2]:.4f}")

            print(f"Settling for {SETTLE_SEC}s before capture...")
            time.sleep(SETTLE_SEC)

            fused_pos = average_world_xy(capA, capB, K_A, D_A, K_B, D_B, i)
            if fused_pos is None:
                print(f"[WARN] Step {i}: No marker average, skipping data point")
                continue

            w_x, w_y = fused_pos
            rx, ry = (p_meas[0], p_meas[1]) if p_meas is not None else (x_cmd, y_cmd)
            results.append((w_x, w_y, rx, ry))
            print(f"Step {i} SUCCESS: World=({w_x:.4f}, {w_y:.4f}) -> Robot(meas)=({rx:.4f}, {ry:.4f})")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            capA.release(); capB.release(); cv2.destroyAllWindows()
            print("Cameras released")
        except Exception: pass
        if n is not None:
            try:
                n.set_learning_mode(True)
            except Exception:
                pass

    # ---- Save & Fit / Verify ----
    print(f"\nCalibration sequence complete. Collected {len(results)} valid data points.")
    if not results:
        print("No valid data points collected!")
        return

    if args.require_all_steps and len(results) != len(steps):
        print(f"[ABORT] Only {len(results)}/{len(steps)} points captured. Not fitting to keep runs comparable.")
        return

    arr = np.array(results, dtype=float)
    print("Saved: world_niryo_xy_id2.csv, world_niryo_xy_id2.npy")

    print("\nCollected data points:")
    for i, (wx, wy, rx, ry) in enumerate(results, 1):
        print(f"  {i}: World({wx:.4f}, {wy:.4f}) -> Robot(meas)({rx:.4f}, {ry:.4f})")

    world = arr[:, :2]
    robot = arr[:, 2:]

    import os as _os
    if args.verify_only:
        if not _os.path.exists("similarity_mapping_niryo.npz"):
            print("[ERROR] verify-only requested but similarity_mapping_niryo.npz not found. Run with --fit first.")
            return
        data = np.load("similarity_mapping_niryo.npz")
        s, R, t = float(data["s"]), data["R"], data["t"]
        pred = (s * (world @ R.T)) + t
        rms = np.sqrt(((pred - robot) ** 2).mean(axis=0))
        print("\n=== VERIFY-ONLY (Similarity, Niryo) ===")
        print(f"RMS vs saved mapping: X={rms[0]:.6f}m, Y={rms[1]:.6f}m")
        return

    # Fit (or refit if --fit) and save
    if args.fit or not _os.path.exists("similarity_mapping_niryo.npz"):
        s, R, t, srms, scond = fit_similarity_2d(world, robot)
        print("\n=== SIMILARITY (Fit, Niryo) ===")
        print("s =", s)
        print("R =\n", R)
        print("t =", t)
        print(f"RMS: X={srms[0]:.6f}m, Y={srms[1]:.6f}m | cond≈{scond:.2e}")
        # np.savez("similarity_mapping_niryo.npz", s=s, R=R, t=t)
    else:
        data = np.load("similarity_mapping_niryo.npz")
        s, R, t = float(data["s"]), data["R"], data["t"]
        pred = (s * (world @ R.T)) + t
        rms = np.sqrt(((pred - robot) ** 2).mean(axis=0))
        print("\n=== USING SAVED SIMILARITY (No refit, Niryo) ===")
        print(f"RMS vs saved mapping: X={rms[0]:.6f}m, Y={rms[1]:.6f}m")

        # ---- Quadratic (ridge) mapping: PRINT ONLY (no file save) ----
        # Design matrix columns: [x_w^2, y_w^2, x_w*y_w, x_w, y_w, 1]
        xw, yw = world[:, 0], world[:, 1]
        A = np.column_stack([xw * xw, yw * yw, xw * yw, xw, yw, np.ones_like(xw)])

        alpha = 1e-6  # tiny ridge L2 regularization for stability
        I6 = np.eye(6)
        M = A.T @ A + alpha * I6

        coeff_x = np.linalg.solve(M, A.T @ robot[:, 0])
        coeff_y = np.linalg.solve(M, A.T @ robot[:, 1])

        # Diagnostics (RMS in meters)
        pred_x = A @ coeff_x
        pred_y = A @ coeff_y
        qrms_x = float(np.sqrt(np.mean((pred_x - robot[:, 0]) ** 2)))
        qrms_y = float(np.sqrt(np.mean((pred_y - robot[:, 1]) ** 2)))

        # Pretty equations (rounded only for display)
        cx = np.round(coeff_x, 6)
        cy = np.round(coeff_y, 6)
        eq_x = (f"x_r = {cx[0]:.6f}*x_w^2 + {cx[1]:.6f}*y_w^2 + {cx[2]:.6f}*x_w*y_w "
                f"+ {cx[3]:.6f}*x_w + {cx[4]:.6f}*y_w + {cx[5]:.6f}")
        eq_y = (f"y_r = {cy[0]:.6f}*x_w^2 + {cy[1]:.6f}*y_w^2 + {cy[2]:.6f}*x_w*y_w "
                f"+ {cy[3]:.6f}*x_w + {cy[4]:.6f}*y_w + {cy[5]:.6f}")

        print("\n=== QUADRATIC (ridge) MAPPING — Niryo ===")
        print(eq_x)
        print(eq_y)
        print(f"RMS: X={qrms_x:.6f} m, Y={qrms_y:.6f} m")

        return eq_x, eq_y

if __name__ == "__main__":
    main()
