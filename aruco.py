# aruco.py
import cv2
import numpy as np
from typing import Dict, Any, Optional

# -------------------- helpers --------------------

def load_calibration_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load intrinsics saved as {K, dist} from np.savez."""
    d = np.load(path)
    K = d["K"].astype(float)
    dist = d["dist"].astype(float).reshape(1, -1)
    return K, dist

def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Rodrigues rvec + tvec -> 4x4 transform (tag in camera frame)."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = tvec.reshape(3)
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti

def euler_zyx_from_R(R: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw (deg) using ZYX convention for readability."""
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])

def marker_object_corners_3d(edge_len_m: float) -> np.ndarray:
    """3D marker square corners (order matches OpenCV aruco corners)."""
    L = float(edge_len_m)
    return np.array([
        [-L/2,  L/2, 0.0],
        [ L/2,  L/2, 0.0],
        [ L/2, -L/2, 0.0],
        [-L/2, -L/2, 0.0],
    ], dtype=np.float32)

def quad_area_px(c4: np.ndarray) -> float:
    """Signed polygon area (px^2) for a 4x2 corners array."""
    x, y = c4[:, 0], c4[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def reprojection_error_px(K: np.ndarray, dist: np.ndarray,
                          rvec: np.ndarray, tvec: np.ndarray,
                          img_corners_px: np.ndarray,
                          edge_len_m: float) -> float:
    """Mean pixel reprojection error of the 4 corners."""
    obj = marker_object_corners_3d(edge_len_m)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_corners_px.astype(float), axis=1).mean()
    return float(err)

# -------------------- main API --------------------

def detect_aruco(
    frame_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_m: float,
    world_anchor_id: Optional[int] = 0,
    dictionary: int = cv2.aruco.DICT_4X4_50,
) -> Dict[str, Any]:
    """
    Detect ArUco markers in an image and return per-ID coordinates.

    Returns a dict:
    {
      "world_anchor_seen": bool,
      "world_T_cam": 4x4 list | None,     # camera pose in world if anchor seen
      "rpy_world_cam_deg": [r, p, y] | None,
      "markers": {
        "<id>": {
          "id": int,
          "corners_px": [[u,v],... x4],
          "area_px": float,
          "rvec": [3], "tvec": [3],                   # camera frame
          "T_cam_tag": [[...4],... x4],
          "xyz_cam_m": [x,y,z],                       # meters (camera frame)
          "rpy_cam_deg": [r,p,y],
          "T_world_tag": [[...4],... x4] | None,      # only if anchor seen
          "xyz_world_m": [x,y,z] | None,
          "rpy_world_deg": [r,p,y] | None,
          "reproj_err_px": float
        }, ...
      }
    }
    """
    # Detector setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Convert to gray and detect
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    out: Dict[str, Any] = {
        "world_anchor_seen": False,
        "world_T_cam": None,
        "rpy_world_cam_deg": None,
        "markers": {}
    }
    if ids is None or len(ids) == 0:
        return out

    # Pose for each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, float(marker_length_m), camera_matrix, dist_coeffs
    )

    # Build per-marker entries
    for i, mid in enumerate(ids.flatten()):
        rvec = rvecs[i].reshape(3)
        tvec = tvecs[i].reshape(3)
        T_cam_tag = rvec_tvec_to_T(rvec, tvec)
        R_cam_tag = T_cam_tag[:3, :3]

        entry = {
            "id": int(mid),
            "corners_px": corners[i].reshape(4, 2).tolist(),
            "area_px": quad_area_px(corners[i].reshape(4, 2)),
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            "T_cam_tag": T_cam_tag.tolist(),
            "xyz_cam_m": tvec.tolist(),
            "rpy_cam_deg": euler_zyx_from_R(R_cam_tag).tolist(),
            "T_world_tag": None,
            "xyz_world_m": None,
            "rpy_world_deg": None,
            "reproj_err_px": reprojection_error_px(
                camera_matrix, dist_coeffs, rvec, tvec, corners[i].reshape(4, 2), marker_length_m
            ),
        }
        out["markers"][str(int(mid))] = entry

    # If we have a world anchor (e.g., tag 0), compute world->cam and world coords
    if world_anchor_id is not None:
        sid = str(int(world_anchor_id))
        if sid in out["markers"]:
            T_cam_world = np.array(out["markers"][sid]["T_cam_tag"], dtype=float)
            T_world_cam = T_inv(T_cam_world)        # world == anchor tag frame
            out["world_anchor_seen"] = True
            out["world_T_cam"] = T_world_cam.tolist()
            out["rpy_world_cam_deg"] = euler_zyx_from_R(T_world_cam[:3, :3]).tolist()

            # Map every tag to world
            for entry in out["markers"].values():
                T_cam_tag = np.array(entry["T_cam_tag"], dtype=float)
                T_world_tag = T_world_cam @ T_cam_tag
                entry["T_world_tag"] = T_world_tag.tolist()
                entry["xyz_world_m"] = T_world_tag[:3, 3].tolist()
                entry["rpy_world_deg"] = euler_zyx_from_R(T_world_tag[:3, :3]).tolist()

    return out
