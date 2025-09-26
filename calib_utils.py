import cv2, numpy as np
from pathlib import Path

# ---------- Camera ----------
def open_camera(index, width=1280, height=720, fps=30):
    """Open a camera at 1280x720 with safe defaults."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Enable autofocus if supported
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass

    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open camera {index}")
    print(f"✅ Camera {index} opened at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap

# ---------- Preprocess ----------
def _prep_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize and smooth for more stable corner detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

# ---------- Robust corner finder ----------
def _find_corners_robust(gray, pattern):
    cols, rows = pattern
    cflags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_FILTER_QUADS)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

    # Try both orientations, classic and SB
    attempts = [
        ("classic", (cols, rows)),
        ("sb", (cols, rows)),
        ("classic_swapped", (rows, cols)),
        ("sb_swapped", (rows, cols)),
    ]

    for mode, ps in attempts:
        if "classic" in mode:
            ok, corners = cv2.findChessboardCorners(gray, ps, flags=cflags)
            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
                return True, corners, ps
        else:
            if hasattr(cv2, "findChessboardCornersSB"):
                corners = cv2.findChessboardCornersSB(gray, ps)
                if corners is not None and len(corners) == ps[0]*ps[1]:
                    return True, corners.astype(np.float32), ps
    return False, None, (cols, rows)

# ---------- Calibrate one camera ----------
def calibrate_single_camera_npz(cam_index:int,
                                rows:int=6, cols:int=9, square:float=0.03,
                                samples:int=15, save_path:str="calib_cam.npz"):
    cap = open_camera(cam_index, 1280, 720, 30)
    pattern = (cols, rows)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square)

    objpoints, imgpoints = [], []
    win = f"Calibrate cam{cam_index}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("👉 Show the chessboard and press SPACE to capture, C to calibrate, ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Failed to read frame.")
            break
        gray = _prep_gray(frame)
        found, corners, used_ps = _find_corners_robust(gray, pattern)
        vis = frame.copy()

        if found:
            cv2.drawChessboardCorners(vis, used_ps, corners, found)
            cv2.putText(vis, f"FOUND {used_ps[1]}x{used_ps[0]} | {len(objpoints)}/{samples}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(vis, f"Show {rows}x{cols} chessboard",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord(' '):  # SPACE to capture
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                print(f"✅ Captured {len(objpoints)}")
            else:
                print("❌ No corners found, try again.")
        elif key in (ord('c'), ord('C')):
            if len(objpoints) >= 3:
                break

        if len(objpoints) >= samples:
            break

    cap.release()
    cv2.destroyWindow(win)

    if len(objpoints) < 3:
        print("Not enough samples — nothing saved.")
        return None, None

    h, w = gray.shape[:2]
    print("📐 Calibrating…")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        errs.append(float(err))

    out = {
        "K": K, "D": D, "rms": float(rms),
        "image_size": np.array([w,h], dtype=np.int32),
        "per_view_error": np.array(errs, dtype=np.float32),
        "rows": rows, "cols": cols, "square": float(square),
        "samples": len(objpoints)
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    print(f"💾 Saved {save_path} (RMS={rms:.4f}, size={w}x{h})")
    return K, D

# ---------- Loader ----------
def load_calibration_npz(path:str):
    data = np.load(path, allow_pickle=False)
    K, D = data["K"].astype(float), data["D"].astype(float)
    if D.ndim == 1:
        D = D.reshape(1, -1)
    return K, D
