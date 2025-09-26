# aruco_core.py
"""
Core for dual-camera ArUco fusion (no calibration files).
- All config lives here.
- Intrinsics are approximated from horizontal FOV.
- Live detection, WORLD anchoring, fusion, dimension reporting, overlay drawing.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# ===================== CONFIG =====================

# Cameras (set CAM_A_INDEX/CAM_B_INDEX after calling quick_probe() if needed)
CAM_A_INDEX = 0
CAM_B_INDEX = 1
CAP_W, CAP_H = 640, 480

# Intrinsics (FOV-based; no .npz)
HFOV_A_DEG = 65.0         # assumed horizontal FOV for Cam A
HFOV_B_DEG = 65.0         # assumed horizontal FOV for Cam B

# ArUco
DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_LEN_M = 0.05       # edge of the BLACK square in meters (measure your print!)
# WORLD_ANCHOR_TRY_ORDER = [0, 2, 1, 3]  # 0(table), 2(Kinova), 1(UR), 3(Niryo)
WORLD_ANCHOR_TRY_ORDER = [0]  # 0(table), 2(Kinova), 1(UR), 3(Niryo)


# UI / printing
PRINT_INTERVAL_S = 1.0    # how often main prints the summary
AXIS_LEN_FACTOR = 0.5     # axes length = factor * marker edge
WINDOW_NAME = "CamA | CamB (ArUco)"

CALIB_A = "calib_cam0.npz"
CALIB_B = "calib_cam2.npz"
# ==================================================


# ----------------- Camera utilities -----------------

def quick_probe(max_index: int = 8) -> List[int]:
    """
    Probe available camera indices quickly. Returns a list of indices that can grab a frame.
    """
    good: List[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        if cap:
            cap.release()
        if ok:
            good.append(i)
    print("[INFO] Working camera indices:", good)
    return good


def _try_open_with_backends(index: int):
    """
    Try multiple Windows backends then default. Works on non-Windows too (default).
    """
    candidates = []
    # Prefer MSMF then DSHOW on Windows; default otherwise
    if hasattr(cv2, "CAP_MSMF"):
        candidates.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_DSHOW"):
        candidates.append(cv2.CAP_DSHOW)
    candidates.append(0)  # default

    for be in candidates:
        cap = cv2.VideoCapture(index, be) if be != 0 else cv2.VideoCapture(index)
        if cap and cap.isOpened():
            return cap
        if cap:
            cap.release()
    return None


def open_cam(index: int, w: int = CAP_W, h: int = CAP_H) -> cv2.VideoCapture:
    """
    Open camera quickly with resolution + MJPG setting.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open camera {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # Force MJPG for faster grab at higher resolutions
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Just one quick warm-up read
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"[ERROR] Camera {index} opened but no frames available")
    return cap



# ----------------- Intrinsics -----------------

CALIB_A = "calib_cam0.npz"
CALIB_B = "calib_cam2.npz"

calA = np.load(CALIB_A)
K_A, D_A = calA["K"], calA["D"]

calB = np.load(CALIB_B)
K_B, D_B = calB["K"], calB["D"]



# ----------------- Math helpers -----------------

def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quad_area_px(c4: np.ndarray) -> float:
    x, y = c4[:, 0], c4[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def reproj_err(K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
               img_corners_px: np.ndarray, edge_len_m: float) -> float:
    obj = np.array([[-.5, .5, 0],
                    [ .5, .5, 0],
                    [ .5,-.5, 0],
                    [-.5,-.5, 0]], dtype=np.float32) * float(edge_len_m)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    return float(np.linalg.norm(proj - img_corners_px.astype(float), axis=1).mean())


def choose_best(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prefer lower reprojection error; tie-breaker: larger pixel area."""
    return min(cands, key=lambda c: (c["reproj_err_px"], -c["area_px"]))


# ----------------- Detection & World mapping -----------------

def _detect_markers(gray: np.ndarray, dict_id: int):
    """
    Compatibility wrapper for OpenCV's ArUco detector across versions.
    Returns (corners, ids, rejected).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    # New API (OpenCV 4.7+): ArucoDetector
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector.detectMarkers(gray)
    except Exception:
        # Legacy API
        try:
            params = cv2.aruco.DetectorParameters_create()
        except Exception:
            params = cv2.aruco.DetectorParameters()  # some builds still expose this
        return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


def detect_aruco(frame_bgr: np.ndarray, K: np.ndarray, D: np.ndarray,
                 marker_len_m: float, dict_id: int) -> Dict[str, Any]:
    """
    Return {'world_anchor_seen', 'world_T_cam', 'markers': {id: {...}}} for one frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect_markers(gray, dict_id)

    out: Dict[str, Any] = {"world_anchor_seen": False, "world_T_cam": None, "markers": {}}
    if ids is None or len(ids) == 0:
        return out

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, float(marker_len_m), K, D
    )

    for i, mid in enumerate(ids.flatten()):
        rvec = rvecs[i].reshape(3)
        tvec = tvecs[i].reshape(3)
        T_cam_tag = rvec_tvec_to_T(rvec, tvec)
        entry = {
            "id": int(mid),
            "corners_px": corners[i].reshape(4, 2),
            "area_px": quad_area_px(corners[i].reshape(4, 2)),
            "rvec": rvec, "tvec": tvec,
            "T_cam_tag": T_cam_tag,
            "xyz_cam_m": tvec,
            "reproj_err_px": reproj_err(K, D, rvec, tvec, corners[i].reshape(4, 2), marker_len_m),
            "T_world_tag": None, "xyz_world_m": None
        }
        out["markers"][str(int(mid))] = entry

    return out


def reanchor_in_place(res: Dict[str, Any], anchor_id: int) -> bool:
    """
    Make 'anchor_id' the WORLD anchor inside a detection result.
    """
    m = res["markers"].get(str(anchor_id))
    if not m:
        return False
    T_cam_world = m["T_cam_tag"]      # cam -> anchor
    T_world_cam = T_inv(T_cam_world)  # anchor -> cam
    res["world_anchor_seen"] = True
    res["world_T_cam"] = T_world_cam
    for e in res["markers"].values():
        T_world_tag = T_world_cam @ e["T_cam_tag"]
        e["T_world_tag"] = T_world_tag
        e["xyz_world_m"] = T_world_tag[:3, 3]
    return True


def ensure_world_anchor(res: Dict[str, Any], try_order: List[int]) -> bool:
    """
    Try multiple possible anchor IDs until one works.
    """
    if res["world_anchor_seen"]:
        return True
    for aid in try_order:
        if reanchor_in_place(res, aid):
            return True
    return False


# ----------------- Fusion & Reporting -----------------

def fuse_two_results(resA: Dict[str, Any], resB: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    For each seen ID, pick the best WORLD xyz among cameras.
    Returns: {
      id: {
        "xyz_world": np.array([x,y,z]),
        "src": "CamA"/"CamB",
        "reproj_err_px": float,
        "cands": [
            {"src": "...", "xyz_world": np.array(...), "area_px": float, "reproj_err_px": float, "cam_xyz": np.array(...)},
            ...
        ]
      }
    }
    """
    idsA = set(int(k) for k in resA["markers"].keys())
    idsB = set(int(k) for k in resB["markers"].keys())
    all_ids = sorted(list(idsA | idsB))

    fused: Dict[int, Dict[str, Any]] = {}
    for mid in all_ids:
        cands: List[Dict[str, Any]] = []

        mA = resA["markers"].get(str(mid))
        if mA is not None and resA["world_anchor_seen"] and (mA["xyz_world_m"] is not None):
            cands.append({
                "src": "CamA",
                "xyz_world": np.array(mA["xyz_world_m"], float),
                "area_px": float(mA["area_px"]),
                "reproj_err_px": float(mA["reproj_err_px"]),
                "cam_xyz": np.array(mA["xyz_cam_m"], float)
            })

        mB = resB["markers"].get(str(mid))
        if mB is not None and resB["world_anchor_seen"] and (mB["xyz_world_m"] is not None):
            cands.append({
                "src": "CamB",
                "xyz_world": np.array(mB["xyz_world_m"], float),
                "area_px": float(mB["area_px"]),
                "reproj_err_px": float(mB["reproj_err_px"]),
                "cam_xyz": np.array(mB["xyz_cam_m"], float)
            })

        if cands:
            best = choose_best(cands)
            fused[mid] = {
                "xyz_world": best["xyz_world"],
                "src": best["src"],
                "reproj_err_px": best["reproj_err_px"],
                "cands": cands
            }

    return fused


def compute_world_dimensions(fused: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pairwise distances + axis extents over fused WORLD points.
    Returns: {"pairs":[(i,j,d)], "extents":{"min":mn, "max":mx, "size":sz}} (arrays are np.ndarray)
    """
    ids = sorted(fused.keys())
    out: Dict[str, Any] = {"pairs": [], "extents": None}
    if len(ids) < 2:
        return out

    P = np.stack([fused[i]["xyz_world"] for i in ids], axis=0)

    # pairwise distances
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dij = float(np.linalg.norm(P[i] - P[j]))
            out["pairs"].append((ids[i], ids[j], dij))

    # extents
    mn = P.min(axis=0)
    mx = P.max(axis=0)
    out["extents"] = {"min": mn, "max": mx, "size": mx - mn}
    return out


def get_summary_str(resA: Dict[str, Any], resB: Dict[str, Any], fused: Dict[int, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"World anchor established?  CamA={resA.get('world_anchor_seen', False)}  CamB={resB.get('world_anchor_seen', False)}")
    lines.append("----- Per-ID coordinates -----")
    if not fused:
        lines.append("(none)")
    else:
        for mid in sorted(fused.keys()):
            x, y, z = fused[mid]["xyz_world"]
            err = fused[mid]["reproj_err_px"]
            src = fused[mid]["src"]
            lines.append(f"ID {mid}: WORLD XYZ=({x:.3f}, {y:.3f}, {z:.3f}) [{src}] (err~{err:.2f}px)")
            for c in fused[mid]["cands"]:
                cx, cy, cz = c["cam_xyz"]
                lines.append(f"   {c['src']} CAM XYZ=({cx:.3f}, {cy:.3f}, {cz:.3f})")
            if len(fused[mid]["cands"]) == 2:
                a, b = fused[mid]["cands"]
                dmm = np.linalg.norm(a["xyz_world"] - b["xyz_world"]) * 1000.0
                lines.append(f"   Δ CamA-CamB in WORLD: {dmm:.1f} mm")
    return "\n".join(lines)


def get_dimensions_str(dim: Dict[str, Any]) -> str:
    lines: List[str] = []
    if not dim["pairs"]:
        lines.append("[INFO] Fewer than 2 fused tags → cannot compute world dimensions.")
        return "\n".join(lines)

    lines.append("----- Pairwise distances in WORLD (meters) -----")
    for i, j, d in dim["pairs"]:
        lines.append(f"d(ID {i} ↔ ID {j}) = {d:.3f} m")

    e = dim["extents"]
    if e is not None:
        mn, mx, sz = e["min"], e["max"], e["size"]
        lines.append("----- WORLD extents over all visible tags -----")
        lines.append(f"X-range: {mn[0]:.3f} → {mx[0]:.3f}  (width ≈ {sz[0]:.3f} m)")
        lines.append(f"Y-range: {mn[1]:.3f} → {mx[1]:.3f}  (depth ≈ {sz[1]:.3f} m)")
        lines.append(f"Z-range: {mn[2]:.3f} → {mx[2]:.3f}  (height ≈ {sz[2]:.3f} m)")
    return "\n".join(lines)

# ----------------- Drawing -----------------

def draw_overlays(frame: np.ndarray, res: Dict[str, Any],
                  K: np.ndarray, D: np.ndarray,
                  axis_len_factor: float = AXIS_LEN_FACTOR,
                  marker_len_m: float = MARKER_LEN_M) -> None:
    """
    Draw IDs and axes on a frame (camera frame).
    """
    if not res["markers"]:
        return
    axis_len = marker_len_m * float(axis_len_factor)
    for m in res["markers"].values():
        mid = m["id"]
        corners = m["corners_px"].astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 255), 2, cv2.LINE_AA)
        c = corners.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID {mid}", (c[0] - 20, c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
        try:
            cv2.drawFrameAxes(frame, K, D,
                              m["rvec"].reshape(3, 1),
                              m["tvec"].reshape(3, 1),
                              axis_len)
        except Exception:
            # Legacy fallback
            cv2.aruco.drawAxis(frame, K, D,
                               m["rvec"],
                               m["tvec"],
                               axis_len)
