from __future__ import annotations
import os
import time
from typing import Tuple, Dict, Any

import cv2
import numpy as np

import aruco_core as ac  # your module with detection/fusion utilities
import kinova_calibration
import niryo_calibration
import prompt
import ur_calibration
from home_cartesian import fetch_all_cartesian
from home_joints import get_all_robot_joints
import utilities
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from prompt import build_joint_planning_prompt, generate_robot_program_prompt
import voice
from llm import run_llm1, run_llm2, run_llm3
import subprocess
import sys


# ===================== Config helpers =====================

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

CAM_A_INDEX = 0  # default from aruco_core (0)
CAM_B_INDEX = 2  # default from aruco_core (1)
CAP_W       = _get_env_int("CAP_W", ac.CAP_W)
CAP_H       = _get_env_int("CAP_H", ac.CAP_H)
HFOV_A_DEG  = _get_env_float("HFOV_A", ac.HFOV_A_DEG)
HFOV_B_DEG  = _get_env_float("HFOV_B", ac.HFOV_B_DEG)

WINDOW_NAME = ac.WINDOW_NAME if hasattr(ac, "WINDOW_NAME") else "CamA | CamB (ArUco)"


# ===================== Intrinsics =====================

def first_K_from_cap(cap: cv2.VideoCapture, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grab one frame to infer intrinsics from resolution + assumed HFOV.
    Returns (K, D).
    """
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("[ERROR] Could not read initial frame from camera.")
    h, w = frame.shape[:2]
    K, D = ac.intrinsics_from_fov(w, h, hfov_deg)
    return K, D


# ===================== Capture & fusion =====================

def capture_once(frameA: np.ndarray, frameB: np.ndarray,
                 K_A: np.ndarray, D_A: np.ndarray,
                 K_B: np.ndarray, D_B: np.ndarray,
                 capture_idx: int) -> Dict[str, Any]:
    """
    Run detection on two frames, anchor to WORLD, fuse, compute dims, return structured dict.
    """
    resA = ac.detect_aruco(frameA, K_A, D_A, ac.MARKER_LEN_M, ac.DICT_ID)
    resB = ac.detect_aruco(frameB, K_B, D_B, ac.MARKER_LEN_M, ac.DICT_ID)

    ac.ensure_world_anchor(resA, ac.WORLD_ANCHOR_TRY_ORDER)
    ac.ensure_world_anchor(resB, ac.WORLD_ANCHOR_TRY_ORDER)

    fused = ac.fuse_two_results(resA, resB)
    dims  = ac.compute_world_dimensions(fused)

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

    per_id, aruco_base = {}, {}
    for mid in sorted(fused.keys()):
        m = fused[mid]
        x, y, z = map(float, m["xyz_world"])
        per_id[mid] = {
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
            a, b = m["cands"]
            dmm = float(np.linalg.norm(a["xyz_world"] - b["xyz_world"]) * 1000.0)
            per_id[mid]["delta_AB_mm"] = dmm

    return {
        "capture_idx": capture_idx,
        "world_anchor": {
            "camA": bool(resA.get("world_anchor_seen", False)),
            "camB": bool(resB.get("world_anchor_seen", False)),
        },
        "per_id": per_id,
        "pairwise_dist": pairwise,
        "world_extents": world_extents,
        "aruco_base": aruco_base,
    }


def format_capture_result(result: Dict[str, Any]) -> str:
    """
    Pretty-print the capture result dictionary.
    """
    lines = [f"=== Capture #{result['capture_idx']} ==="]
    wa = result["world_anchor"]
    lines.append(f"World anchor established?  CamA={wa['camA']}  CamB={wa['camB']}")
    lines.append("")

    lines.append("----- Per-ID coordinates -----")
    for mid, info in result["per_id"].items():
        x, y, z = info["world_xyz"]
        lines.append(f"ID {mid}: WORLD XYZ=({x:.3f}, {y:.3f}, {z:.3f}) "
                     f"[{info['src']}] (err~{info['error_px']:.2f}px)")
        for c in info["cands"]:
            cx, cy, cz = c["cam_xyz"]
            lines.append(f"   {c['src']} CAM XYZ=({cx:.3f}, {cy:.3f}, {cz:.3f})")
        if "delta_AB_mm" in info:
            lines.append(f"   Δ CamA-CamB in WORLD: {info['delta_AB_mm']:.1f} mm")
    lines.append("")

    lines.append("----- Pairwise distances in WORLD (meters) -----")
    for k, d in result["pairwise_dist"].items():
        i, j = k.split("-")
        lines.append(f"d(ID {i} ↔ ID {j}) = {d:.3f} m")
    lines.append("")

    e = result["world_extents"]
    if e:
        lines.append("----- WORLD extents over all visible tags -----")
        lines.append(f"X-range: {e['x'][0]:.3f} → {e['x'][1]:.3f}  (width ≈ {e['x'][2]:.3f} m)")
        lines.append(f"Y-range: {e['y'][0]:.3f} → {e['y'][1]:.3f}  (depth ≈ {e['y'][2]:.3f} m)")
        lines.append(f"Z-range: {e['z'][0]:.3f} → {e['z'][1]:.3f}  (height ≈ {e['z'][2]:.3f} m)")
    lines.append("")

    lines.append("----- aruco_base (WORLD XYZ) -----")
    for k, v in result["aruco_base"].items():
        lines.append(f"ID {k}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")

    return "\n".join(lines)


def save_and_run_code(code_str: str, filename: str = "testin.py") -> str:
    """
    Save code string to file and run it, capturing stdout + stderr.
    """
    # Save code to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code_str)

    try:
        # Run file with current Python interpreter
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=30   # prevent infinite loops
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()

        if errors:
            return f"[Error]\n{errors}"
        return output if output else "[No output]"
    except Exception as e:
        return f"[Exception] {e}"


def main() -> None:
    # --- Open Kinova session once ---
    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base_cyclic = BaseCyclicClient(router)

        # --- One-time readouts before opening cameras (optional) ---
        poses = fetch_all_cartesian("192.168.1.15", "192.168.1.13", base_cyclic)
        print("Raw dict:", poses, "\n")

        # map once for the prompt and reuse inside loop
        robots_current_cartesians = {
            "UR": poses["ur"],
            "Kinova": poses["kinova"],
            "Niryo": poses["niryo"],
        }

        joints = get_all_robot_joints(
            base_cyclic=base_cyclic,
            niryo_ip="192.168.1.15",
            ur_ip="192.168.1.13"
        )

        robots_current_joints = {
            "UR": joints["ur"],
            "Kinova": joints["kinova"],
            "Niryo": joints["niryo"],
        }

        # pass all the joints to the prompt
        print("=== JOINTS (rad, formatted .2f) ===")
        for name, vals in joints.items():
            print(f"{name.capitalize()} joints: {vals}")
        print()

        # --- Cameras ---
        capA = ac.open_cam(CAM_B_INDEX, CAP_W, CAP_H)
        capB = ac.open_cam(CAM_A_INDEX, CAP_W, CAP_H)
        for c in (capA, capB):
            c.set(cv2.CAP_PROP_BRIGHTNESS, 5)
            c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disable auto, backend-dependent

        try:
            K_A, D_A = ac.K_A, ac.D_A
            K_B, D_B = ac.K_B, ac.D_B
            print("Press 'C' to capture (multiple times). Press ESC to quit.\n")

            capture_idx = 1
            debounce_until = 0.0

            while True:
                okA, frameA = capA.read()
                okB, frameB = capB.read()
                if not (okA and okB):
                    print("[WARN] Frame grab failed; continuing...")
                    continue

                # overlays
                resA = ac.detect_aruco(frameA, K_A, D_A, ac.MARKER_LEN_M, ac.DICT_ID)
                resB = ac.detect_aruco(frameB, K_B, D_B, ac.MARKER_LEN_M, ac.DICT_ID)
                try:
                    ac.draw_overlays(frameA, resA, K_A, D_A)
                    ac.draw_overlays(frameB, resB, K_B, D_B)
                except Exception:
                    pass

                # side-by-side view
                h = max(frameA.shape[0], frameB.shape[0])
                w = frameA.shape[1] + frameB.shape[1]
                canvas = np.zeros((h, w, 3), dtype=frameA.dtype)
                canvas[:frameA.shape[0], :frameA.shape[1]] = frameA
                canvas[:frameB.shape[0], frameA.shape[1]:frameA.shape[1] + frameB.shape[1]] = frameB

                cv2.imshow(WINDOW_NAME, canvas)
                k = cv2.waitKey(1) & 0xFF
                now = time.time()

                if k in (ord('c'), ord('C')) and now >= debounce_until:
                    debounce_until = now + 0.25

                    # fresh snaps for fusion
                    okA2, snapA = capA.read()
                    okB2, snapB = capB.read()
                    if not (okA2 and okB2):
                        print("[ERROR] Capture failed; try again.")
                        continue

                    result = capture_once(snapA.copy(), snapB.copy(), K_A, D_A, K_B, D_B, capture_idx)
                    print(result)

                    calib = 0
                    if calib == 1:
                        niryo_eqx, niryo_eqy = niryo_calibration.main()
                    elif calib == 2:
                        kinova_eqx, kinova_eqy = kinova_calibration.main()
                    elif calib == 3:
                        ur_eqx, ur_eqy = ur_calibration.main()

                    choice = input("Enter your choice")

                    if choice == 0:
                        user_task = input("Enter your task: ")
                    elif choice == 1:
                        user_task = voice.listen_and_transcribe()

                    robot_class = "kinova"  # or "kinova" / "niryo"

                    llm1prompt = prompt.llm1_safety_prompt(user_input=user_task)
                    response1 = run_llm1(llm1prompt, model="gpt-oss", stream=False)
                    if response1 ==  True:
                        if robot_class == "kinova":
                            llm2prompt = build_joint_planning_prompt(
                                robot_class=robot_class,
                                user_task=user_task,
                                aruco_base=format_capture_result(result),
                                robots_current_joints=robots_current_joints,
                                robots_current_cartesians=robots_current_cartesians,
                                eqx = kinova_eqx,
                                eqy = kinova_eqy
                            )
                        elif robot_class == "niryo":
                            llm2prompt = build_joint_planning_prompt(
                                robot_class=robot_class,
                                user_task=user_task,
                                aruco_base=format_capture_result(result),
                                robots_current_joints=robots_current_joints,
                                robots_current_cartesians=robots_current_cartesians,
                                eqx = niryo_eqx,
                                eqy = niryo_eqy
                            )
                        elif robot_class == "ur":
                            llm2prompt = build_joint_planning_prompt(
                                robot_class=robot_class,
                                user_task=user_task,
                                aruco_base=format_capture_result(result),
                                robots_current_joints=robots_current_joints,
                                robots_current_cartesians=robots_current_cartesians,
                                eqx=ur_eqx,
                                eqy=ur_eqy
                            )



                        response2 = run_llm2(llm2prompt, model="gpt-oss", stream=False)
                        print(response2)
                        llm3prompt = generate_robot_program_prompt(waypoints=response2, robot_type=robot_class) #build a prompt
                        response3 = run_llm3(llm3prompt, model="codegemma", stream=False)
                        print(response3)

                        run = input("Do you want to run the code? (y/n): ")
                        if run.lower() == 'y':
                            print("\n=== Running testin.py ===")
                            result_output = save_and_run_code(response3, "testin.py")
                            print(result_output)
                        else:
                            print("Code execution skipped.")


                    else:
                        print("Unsafe task, please try again")
                        continue



                    capture_idx += 1

                if k == 27:  # ESC
                    break

        finally:
            capA.release()
            capB.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

