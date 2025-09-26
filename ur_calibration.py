from __future__ import annotations

import socket
import time
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import cv2

import aruco_core as ac  # expects: open_cam, K_A, D_A, K_B, D_B, MARKER_LEN_M, DICT_ID, CAP_W, CAP_H, WORLD_ANCHOR_TRY_ORDER

# ==================== UR Socket Config ====================
UR5_IP = "192.168.1.13"
UR5_PORT = 30002
ACC = 0.05
VEL = 0.20
SETTLE_SEC = 10.0

# ==================== Camera / ArUco Config ====================
CAM_A_INDEX = 0
CAM_B_INDEX = 2
CAP_W, CAP_H = ac.CAP_W, ac.CAP_H

# the ArUco you're averaging (change if needed)
MARKER_ID = 3

MARKER_LEN_M = ac.MARKER_LEN_M
DICT_ID = ac.DICT_ID
WORLD_ANCHOR_TRY_ORDER = ac.WORLD_ANCHOR_TRY_ORDER
ANCHOR_ID = WORLD_ANCHOR_TRY_ORDER[0] if WORLD_ANCHOR_TRY_ORDER else 0

# deterministic collection & quality gates
FORCE_ANCHOR_CAMERA = "A"     # choose "A" or "B" and keep it fixed
ANCHOR_ERR_MAX_PX = 1.5       # anchor must be below this reprojection error
REPROJ_THRESHOLD_PX = 1.5     # sample quality for the target marker
TARGET_VALID_FRAMES = 20      # collect exactly this many good frames (or timeout)
MIN_VALID_FRAMES_TO_USE = 6   # need at least this many good frames for a pose
PER_POSE_TIMEOUT_S = 20.0     # timeout per pose

# ==================== UR helpers ====================
def send_ur5(cmd: str, retries: int = 1) -> None:
    """Send a URScript command to UR controller via socket."""
    msg = (cmd.strip() + "\n").encode("utf-8")
    for attempt in range(retries + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.5)
                s.connect((UR5_IP, UR5_PORT))
                s.sendall(msg)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(0.05)

def movep_xyzrpy(x: float, y: float, z: float, rx: float, ry: float, rz: float,
                 a: float = ACC, v: float = VEL) -> None:
    """Move UR in tool-space using movep."""
    cmd = f"movep(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}], a={a}, v={v})"
    send_ur5(cmd)

# ==================== ArUco helpers ====================
def _compute_reprojection_error(corners, rvec, tvec, K, D, marker_length: float) -> float:
    """Compute per-marker reprojection error in pixels."""
    try:
        obj_points = np.array(
            [
                [-marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0],
            ],
            dtype=np.float32,
        )
        img_points, _ = cv2.projectPoints(
            obj_points,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).ravel(),
        )
        img_points = img_points.reshape(-1, 2)
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        err = np.mean(np.sqrt(np.sum((img_points - corners) ** 2, axis=1)))
        return float(err)
    except Exception:
        return float("inf")

def _anchor_err(res: Dict[str, Any], K, D, anchor_id: int) -> Optional[float]:
    """Return anchor reprojection error if present, else None."""
    key = str(anchor_id)
    if "markers" not in res or key not in res["markers"]:
        return None
    m = res["markers"][key]
    e = m.get("reproj_err_px", float("inf"))
    if not np.isfinite(e):
        # compute if corners/rvec/tvec available
        if all(k in m for k in ("corners", "rvec", "tvec")):
            e = _compute_reprojection_error(m["corners"], m["rvec"], m["tvec"], K, D, MARKER_LEN_M)
        else:
            e = float("inf")
    return float(e)

def capture_once(frameA: np.ndarray, frameB: np.ndarray, capture_idx: int) -> Dict[str, Any]:
    """
    Detect ArUco in both frames, FORCE a single-camera world anchor (A or B),
    fuse, and return rich structured results.
    """
    resA = ac.detect_aruco(frameA, ac.K_A, ac.D_A, MARKER_LEN_M, DICT_ID)
    resB = ac.detect_aruco(frameB, ac.K_B, ac.D_B, MARKER_LEN_M, DICT_ID)

    # compute anchor quality
    errA = _anchor_err(resA, ac.K_A, ac.D_A, ANCHOR_ID)
    errB = _anchor_err(resB, ac.K_B, ac.D_B, ANCHOR_ID)

    # choose and enforce a single camera as the source of the world anchor
    anchor_valid = False
    if FORCE_ANCHOR_CAMERA.upper() == "B":
        if errB is not None and errB < ANCHOR_ERR_MAX_PX:
            anchor_valid = True
            if str(ANCHOR_ID) in resA.get("markers", {}):
                del resA["markers"][str(ANCHOR_ID)]
    else:  # "A"
        if errA is not None and errA < ANCHOR_ERR_MAX_PX:
            anchor_valid = True
            if str(ANCHOR_ID) in resB.get("markers", {}):
                del resB["markers"][str(ANCHOR_ID)]

    # flag world anchor for downstream logic
    ac.ensure_world_anchor(resA, WORLD_ANCHOR_TRY_ORDER)
    ac.ensure_world_anchor(resB, WORLD_ANCHOR_TRY_ORDER)

    fused = ac.fuse_two_results(resA, resB)
    dims = ac.compute_world_dimensions(fused)

    pairwise = {f"{i}-{j}": float(d) for (i, j, d) in dims.get("pairs", [])}

    e = dims.get("extents")
    world_extents = None
    if e is not None:
        mn, mx, sz = e["min"], e["max"], e["size"]
        world_extents = {
            "x": (float(mn[0]), float(mx[0]), float(sz[0])),
            "y": (float(mn[1]), float(mx[1]), float(sz[1])),
            "z": (float(mn[2]), float(mx[2]), float(sz[2])),
        }

    per_id: Dict[int, Any] = {}
    aruco_base: Dict[str, List[float]] = {}
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
        if len(m.get("cands", [])) == 2:
            a = np.asarray(m["cands"][0]["xyz_world"], dtype=float)
            b = np.asarray(m["cands"][1]["xyz_world"], dtype=float)
            dmm = float(np.linalg.norm(a - b) * 1000.0)
            per_id[int(mid)]["delta_AB_mm"] = dmm

    return {
        "capture_idx": capture_idx,
        "world_anchor": {
            "anchor_valid": anchor_valid,
            "forced": FORCE_ANCHOR_CAMERA.upper(),
            "errA": None if errA is None else float(errA),
            "errB": None if errB is None else float(errB),
        },
        "per_id": per_id,
        "pairwise_dist": pairwise,
        "world_extents": world_extents,
        "aruco_base": aruco_base,
    }

# ==================== Averaging ====================
def average_world_xy(capA: cv2.VideoCapture, capB: cv2.VideoCapture) -> Optional[Tuple[float, float]]:
    """
    Deterministically collect exactly TARGET_VALID_FRAMES good samples
    (anchor valid + marker seen + low reproj error), or stop at timeout.
    """
    xs: List[float] = []
    ys: List[float] = []
    t0 = time.time()
    tries = 0

    while len(xs) < TARGET_VALID_FRAMES and (time.time() - t0) < PER_POSE_TIMEOUT_S:
        okA, frameA = capA.read()
        okB, frameB = capB.read()
        if not (okA and okB):
            continue

        res = capture_once(frameA, frameB, tries)
        tries += 1

        if not res["world_anchor"]["anchor_valid"]:
            continue

        # get the target marker
        if MARKER_ID not in res["per_id"]:
            continue

        w_x, w_y, _ = res["per_id"][MARKER_ID]["world_xyz"]
        err_px = res["per_id"][MARKER_ID]["error_px"]
        if err_px < REPROJ_THRESHOLD_PX:
            xs.append(float(w_x))
            ys.append(float(w_y))
        # small nap to prevent CPU burn
        time.sleep(0.01)

    if len(xs) < MIN_VALID_FRAMES_TO_USE:
        # too few good frames for a stable average
        return None

    return float(np.mean(xs)), float(np.mean(ys))

# ==================== Mapping helpers ====================
def fit_quadratic(world_xy: np.ndarray, robot_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit x_r = a1 x^2 + a2 y^2 + a3 xy + a4 x + a5 y + a6
        y_r = b1 x^2 + b2 y^2 + b3 xy + b4 x + b5 y + b6
    Returns (coeff_x[6], coeff_y[6])
    """
    x_w, y_w = world_xy[:, 0], world_xy[:, 1]
    A = np.column_stack((x_w**2, y_w**2, x_w*y_w, x_w, y_w, np.ones_like(x_w)))
    coeff_x = np.linalg.lstsq(A, robot_xy[:, 0], rcond=None)[0]
    coeff_y = np.linalg.lstsq(A, robot_xy[:, 1], rcond=None)[0]
    return coeff_x, coeff_y

def quadratic_equation_strings(coeff_x: np.ndarray, coeff_y: np.ndarray) -> Tuple[str, str]:
    cx, cy = coeff_x, coeff_y
    eq_x = (
        f"x_r = {cx[0]:.3f}x_w^2 + {cx[1]:.3f}y_w^2 + {cx[2]:.3f}x_w y_w + "
        f"{cx[3]:.3f}x_w + {cx[4]:.3f}y_w + {cx[5]:.3f}"
    )
    eq_y = (
        f"y_r = {cy[0]:.3f}x_w^2 + {cy[1]:.3f}y_w^2 + {cy[2]:.3f}x_w y_w + "
        f"{cy[3]:.3f}x_w + {cy[4]:.3f}y_w + {cy[5]:.3f}"
    )
    return eq_x, eq_y

# ==================== Main ====================
def main() -> Tuple[List[Tuple[float, float, float, float]], Optional[str], Optional[str]]:
    """
    Returns:
        results: list of (world_x, world_y, ur_x, ur_y)
        eq_x, eq_y: quadratic mapping strings if fitted, else None, None
    """
    # your 5 waypoint plan (X,Y); fixed Z & orientation below
    FIXED_Z = 0.250
    FIXED_RX, FIXED_RY, FIXED_RZ = 0.0, 3.1416, 0.0
    XY_TARGETS = [
        (0.35, -0.76),
        (0.01, -0.76),
        (0.272, -0.60),
        (0.35, -0.35),
        (0.219, -0.491),
    ]

    capA = ac.open_cam(CAM_A_INDEX, CAP_W, CAP_H)
    capB = ac.open_cam(CAM_B_INDEX, CAP_W, CAP_H)

    for c in (capA, capB):
        c.set(cv2.CAP_PROP_BRIGHTNESS, 10)
        c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disable auto, backend-dependent


    if capA is None or capB is None:
        raise RuntimeError("Failed to open cameras")

    results: List[Tuple[float, float, float, float]] = []
    eq_x: Optional[str] = None
    eq_y: Optional[str] = None

    try:
        for i, (ux, uy) in enumerate(XY_TARGETS, start=1):
            # Move UR
            movep_xyzrpy(ux, uy, FIXED_Z, FIXED_RX, FIXED_RY, FIXED_RZ)
            time.sleep(SETTLE_SEC)

            # Average world position of the chosen marker
            fused = average_world_xy(capA, capB)
            if fused is None:
                print(f"[Pose {i}] No stable detection within {PER_POSE_TIMEOUT_S:.1f}s. Skipping.")
                continue

            w_x, w_y = fused
            results.append((w_x, w_y, ux, uy))
            print(f"[Pose {i}] WORLD=({w_x:.3f}, {w_y:.3f})  UR=({ux:.3f}, {uy:.3f})")

    finally:
        try:
            capA.release()
            capB.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
    #
    # if results:
    #     arr = np.asarray(results, dtype=float)
    #     np.savetxt(
    #         "world_ur_xy_id3.csv",
    #         arr,
    #         delimiter=",",
    #         header="world_x,world_y,ur_x,ur_y",
    #         comments="",
    #     )
    #     np.save("world_ur_xy_id3.npy", arr)
    #     print("Saved: world_ur_xy_id3.csv, world_ur_xy_id3.npy")

    # Fit mapping if we have enough points
    if len(results) >= 4:
        arr = np.asarray(results, dtype=float)
        world_xy = arr[:, :2]
        robot_xy = arr[:, 2:]
        cx, cy = fit_quadratic(world_xy, robot_xy)
        eq_x, eq_y = quadratic_equation_strings(cx, cy)
        print("\nQuadratic Mapping Equations:")
        print(eq_x)
        print(eq_y)
    else:
        print(f"\nOnly {len(results)} point(s) collected — need ≥ 4 to fit a quadratic mapping.")

    # return results, eq_x, eq_y
    return eq_x, eq_y

if __name__ == "__main__":
    main()