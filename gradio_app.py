import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import gradio as gr
import cv2
import numpy as np
import json
import time
import threading
import subprocess
import sys
import os
from typing import Dict, Any, Tuple, Optional
import tempfile
from queue import Queue, Empty
import uuid

# Import existing modules
try:
    import aruco_core as ac
    import kinova_calibration
    import niryo_calibration
    import ur_calibration
    from home_cartesian import fetch_all_cartesian
    from home_joints import get_all_robot_joints
    import utilities
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    import prompt
    import voice
    from llm import run_llm1, run_llm2, run_llm3
    from main import format_capture_result

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


# Global state variables
class AppState:
    def __init__(self):
        self.capA = None
        self.capB = None
        self.camera_running = False
        self.show_aruco = True
        self.latest_capture_result = None
        self.robots_current_cartesians = {}
        self.robots_current_joints = {}
        self.kinova_session = None
        self.calibration_values = {
            'ur_ex': 0.0, 'ur_ey': 0.0,
            'kinova_ex': 0.0, 'kinova_ey': 0.0,
            'niryo_ex': 0.0, 'niryo_ey': 0.0
        }
        self.llm_responses = {
            'llm1_prompt': '', 'llm1_response': '',
            'llm2_prompt': '', 'llm2_response': '',
            'llm3_response': '', 'execution_output': ''
        }
        self.robot_class = "kinova"
        self.user_task = ""

        # NEW: Real-time streaming components
        self.current_frameA = None
        self.current_frameB = None
        self.camera_thread = None
        self.stop_thread = False
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Event()

    def cleanup(self):
        self.stop_thread = True
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if self.capA:
            self.capA.release()
        if self.capB:
            self.capB.release()
        cv2.destroyAllWindows()


app_state = AppState()

# Configuration
CAM_A_INDEX = 0
CAM_B_INDEX = 2
CAP_W = 640
CAP_H = 480


def setup_camera_fast(cap_index: int) -> cv2.VideoCapture:
    """Fast camera setup with minimal delay"""
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        print(f"Warning: Camera {cap_index} not available")
        return None

    # Fast setup - only essential settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Critical for real-time
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def camera_streaming_thread():
    """Background thread for continuous camera streaming"""
    print("Starting camera streaming thread...")

    while not app_state.stop_thread:
        if not app_state.camera_running:
            time.sleep(0.1)
            continue

        # Capture from both cameras
        frameA = frameB = None

        if app_state.capA is not None:
            ret_a, frameA = app_state.capA.read()
            if ret_a and app_state.show_aruco and MODULES_AVAILABLE:
                try:
                    res = ac.detect_aruco(frameA, ac.K_A, ac.D_A, ac.MARKER_LEN_M, ac.DICT_ID)
                    ac.draw_overlays(frameA, res, ac.K_A, ac.D_A)
                except:
                    pass
            if ret_a:
                frameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2RGB)

        if app_state.capB is not None:
            ret_b, frameB = app_state.capB.read()
            if ret_b and app_state.show_aruco and MODULES_AVAILABLE:
                try:
                    res = ac.detect_aruco(frameB, ac.K_B, ac.D_B, ac.MARKER_LEN_M, ac.DICT_ID)
                    ac.draw_overlays(frameB, res, ac.K_B, ac.D_B)
                except:
                    pass
            if ret_b:
                frameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2RGB)

        # Update frames thread-safely
        with app_state.frame_lock:
            if frameA is not None:
                app_state.current_frameA = frameA
            if frameB is not None:
                app_state.current_frameB = frameB
            app_state.frame_ready.set()

        time.sleep(1 / 30)  # 30 FPS

    print("Camera streaming thread stopped")


def get_camera_frame(cap, K, D):
    """Get a frame from camera with ArUco detection overlay"""
    if cap is None:
        return None

    ret, frame = cap.read()
    if not ret:
        return None

    if app_state.show_aruco and MODULES_AVAILABLE:
        try:
            res = ac.detect_aruco(frame, K, D, ac.MARKER_LEN_M, ac.DICT_ID)
            ac.draw_overlays(frame, res, K, D)
        except:
            pass

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio


def get_camera_feed_a():
    """Real-time feed for Camera A"""
    with app_state.frame_lock:
        return app_state.current_frameA


def get_camera_feed_b():
    """Real-time feed for Camera B"""
    with app_state.frame_lock:
        return app_state.current_frameB


def start_cameras():
    """Initialize and start both cameras - FAST VERSION"""
    print("Starting cameras (fast mode)...")
    start_time = time.time()

    try:
        # Quick camera setup
        app_state.capA = setup_camera_fast(CAM_A_INDEX)
        app_state.capB = setup_camera_fast(CAM_B_INDEX)

        if not app_state.capA and not app_state.capB:
            return "Error: No cameras available"

        # Start streaming immediately
        app_state.camera_running = True
        app_state.stop_thread = False

        # Start background streaming thread
        if app_state.camera_thread is None or not app_state.camera_thread.is_alive():
            app_state.camera_thread = threading.Thread(target=camera_streaming_thread, daemon=True)
            app_state.camera_thread.start()

        elapsed = time.time() - start_time
        available_cameras = []
        if app_state.capA:
            available_cameras.append("Camera A")
        if app_state.capB:
            available_cameras.append("Camera B")

        return f"Cameras started in {elapsed:.1f}s: {', '.join(available_cameras)}"

    except Exception as e:
        return f"Error starting cameras: {str(e)}"


def stop_cameras():
    """Stop both cameras"""
    app_state.camera_running = False
    app_state.stop_thread = True

    if app_state.capA:
        app_state.capA.release()
        app_state.capA = None
    if app_state.capB:
        app_state.capB.release()
        app_state.capB = None
    return "Cameras stopped"


def toggle_aruco_overlay(show_overlay):
    """Toggle ArUco detection overlay"""
    app_state.show_aruco = show_overlay
    return f"ArUco overlay {'enabled' if show_overlay else 'disabled'}"


def capture_aruco_data():
    """Capture current frame and perform ArUco detection"""
    if not app_state.camera_running or not MODULES_AVAILABLE:
        return "Error: Cameras not running or modules unavailable"

    try:
        ret_a, frameA = app_state.capA.read()
        ret_b, frameB = app_state.capB.read()

        if not (ret_a and ret_b):
            return "Error: Failed to capture frames"

        from main import capture_once, format_capture_result
        result = capture_once(frameA, frameB, ac.K_A, ac.D_A, ac.K_B, ac.D_B, 1)
        app_state.latest_capture_result = result

        return format_capture_result(result)
    except Exception as e:
        return f"Error in ArUco capture: {str(e)}"


def fetch_robot_telemetry():
    """Fetch current robot positions and joints"""
    if not MODULES_AVAILABLE:
        return "Modules not available", "Modules not available"

    try:
        args = utilities.parseConnectionArguments()
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base_cyclic = BaseCyclicClient(router)

            poses = fetch_all_cartesian("192.168.1.15", "192.168.1.13", base_cyclic)
            app_state.robots_current_cartesians = {
                "UR": poses["ur"],
                "Kinova": poses["kinova"],
                "Niryo": poses["niryo"],
            }

            joints = get_all_robot_joints(
                base_cyclic=base_cyclic,
                niryo_ip="192.168.1.15",
                ur_ip="192.168.1.13"
            )
            app_state.robots_current_joints = {
                "UR": joints["ur"],
                "Kinova": joints["kinova"],
                "Niryo": joints["niryo"],
            }

            cartesian_json = json.dumps(app_state.robots_current_cartesians, indent=2)
            joints_json = json.dumps(app_state.robots_current_joints, indent=2)

            return cartesian_json, joints_json
    except Exception as e:
        error_msg = f"Error fetching telemetry: {str(e)}"
        return error_msg, error_msg


def calibrate_robot(robot_type):
    """Run calibration for specified robot based on calib value"""
    if not MODULES_AVAILABLE:
        return f"Modules not available for {robot_type} calibration", 0.0, 0.0

    try:
        if robot_type == 'niryo':
            ex, ey = niryo_calibration.main()
        elif robot_type == 'kinova':
            ex, ey = kinova_calibration.main()
        elif robot_type == 'ur':
            ex, ey = ur_calibration.main()
        else:
            return f"Unknown robot type: {robot_type}", 0.0, 0.0

        app_state.calibration_values[f'{robot_type}_ex'] = ex
        app_state.calibration_values[f'{robot_type}_ey'] = ey

        return f"{robot_type.upper()} calibration completed: EX={ex}, EY={ey}", ex, ey
    except Exception as e:
        return f"Error in {robot_type} calibration: {str(e)}", 0.0, 0.0


def listen_with_voice():
    """Use voice input to get user command"""
    if not MODULES_AVAILABLE:
        return "Voice module not available"

    try:
        transcription = voice.listen_and_transcribe()
        return transcription
    except Exception as e:
        return f"Error with voice input: {str(e)}"


def generate_llm1_prompt(robot_class, user_task, instruction_mode):
    """Generate and execute LLM1 safety check"""
    if not MODULES_AVAILABLE:
        return "Modules not available", "N/A"

    try:
        app_state.robot_class = robot_class
        app_state.user_task = user_task
        llm1_prompt = prompt.llm1_safety_prompt(user_input=user_task)
        app_state.llm_responses['llm1_prompt'] = llm1_prompt

        response = run_llm1(llm1_prompt, model="gpt-oss", stream=False)
        response_text  = json.loads(response.text)['response']
        app_state.llm_responses['llm1_response'] = response_text

        if response_text == "True":
            response_text1 = "Task approved as SAFE."
        else:
            response_text1 = "Task flagged as UNSAFE."
        # verdict = "SAFE" if response_text else "UNSAFE"
        return llm1_prompt, f"Response: {response_text}"
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"


def run_llm2_planning():
    """Execute LLM2 for motion planning"""
    if not MODULES_AVAILABLE or not app_state.llm_responses['llm1_response']:
        return "Error: Prerequisites not met"

    try:
        if not app_state.llm_responses['llm1_response']:
            return "Error: Task not approved by safety check"

        eqx = app_state.calibration_values[f"{app_state.robot_class}_ex"]
        eqy = app_state.calibration_values[f"{app_state.robot_class}_ey"]
        aruco_base = format_capture_result(app_state.latest_capture_result) if app_state.latest_capture_result else "{}"

        llm2_prompt = prompt.build_joint_planning_prompt(
            robot_class=app_state.robot_class,
            user_task=app_state.user_task,
            aruco_base=aruco_base,
            robots_current_joints=app_state.robots_current_joints,
            robots_current_cartesians=app_state.robots_current_cartesians,
            eqx=eqx,
            eqy=eqy
        )
        app_state.llm_responses['llm2_prompt'] = llm2_prompt

        response2 = run_llm2(llm2_prompt, model="gemma3:12b", stream=False)
        app_state.llm_responses['llm2_response'] = response2

        return app_state.llm_responses['llm2_prompt'], f"LLM2 Motion Planning:\n{response2}"
    except Exception as e:
        return "Error in LLM2: {str(e)}", "Error in LLM2: {str(e)}"

def run_llm3_codegen():
    """Execute LLM3 for code generation"""
    if not MODULES_AVAILABLE or not app_state.llm_responses['llm2_response']:
        return "Error: LLM2 must be run first"

    try:
        llm3_prompt = prompt.generate_robot_program_prompt(waypoints=app_state.llm_responses['llm2_response'], robot_type=app_state.robot_class)
        response = run_llm3(llm3_prompt, model="codegemma", stream=False)
        response_full = response.text
        parsed = json.loads(response_full)
        code_raw = parsed['response']
        if code_raw.startswith('```python\n'):
            code_raw = code_raw[10:]
        if code_raw.endswith('\n```'):
            code_raw = code_raw[:-4]
        response3 = code_raw.strip()
        app_state.llm_responses['llm3_response'] = response3
        return response3
    except Exception as e:
        return f"Error in LLM3: {str(e)}"


def execute_generated_code():
    """Execute the LLM3 generated code"""
    if not app_state.llm_responses['llm3_response']:
        return "Error: No generated code available"

    try:
        with open(r"E:\A_Dissertation\Final_Python_ws\execute.py", "w") as f:
            f.write(app_state.llm_responses['llm3_response'])

        result = subprocess.run(
            [sys.executable, r"E:\A_Dissertation\Final_Python_ws\execute.py"],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()
        errors = result.stderr.strip()

        if errors:
            execution_result = f"[Error]\n{errors}"
        else:
            execution_result = output if output else "[No output]"

        app_state.llm_responses['execution_output'] = execution_result
        return execution_result

    except Exception as e:
        return f"Error executing code: {str(e)}"

def create_interface():
    with gr.Blocks(title="Multi-Robot ArUco Control System", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# Multi-Robot ArUco Detection & LLM Control System")
        with gr.Tabs():
            with gr.TabItem("Calibration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### UR Robot")
                        ur_calib_btn = gr.Button("Calibrate UR")
                        ur_status = gr.Textbox(label="UR Status", interactive=False)
                        ur_ex = gr.Textbox(label="EX", interactive=False, value="0.0")
                        ur_ey = gr.Textbox(label="EY", interactive=False, value="0.0")

                    with gr.Column():
                        gr.Markdown("### Kinova Robot")
                        kinova_calib_btn = gr.Button("Calibrate Kinova")
                        kinova_status = gr.Textbox(label="Kinova Status", interactive=False)
                        kinova_ex = gr.Textbox(label="EX", interactive=False, value="0.0")
                        kinova_ey = gr.Textbox(label="EY", interactive=False, value="0.0")

                    with gr.Column():
                        gr.Markdown("### Niryo Robot")
                        niryo_calib_btn = gr.Button("Calibrate Niryo")
                        niryo_status = gr.Textbox(label="Niryo Status", interactive=False)
                        niryo_ex = gr.Textbox(label="EX", interactive=False, value="0.0")
                        niryo_ey = gr.Textbox(label="EY", interactive=False, value="0.0")


            with gr.TabItem("Robot Telemetry"):
                    fetch_telemetry_btn = gr.Button("Fetch Robot Data", variant="primary")

                    with gr.Row():
                        cartesian_output = gr.Textbox(label="Cartesian Positions (JSON)", lines=10)
                        joints_output = gr.Textbox(label="Joint Positions (JSON)", lines=10)

            with gr.TabItem("Camera & Detection"):
                with gr.Row():
                    start_cam_btn = gr.Button("Start Cameras", variant="primary")
                    stop_cam_btn = gr.Button("Stop Cameras", variant="secondary")
                    aruco_checkbox = gr.Checkbox(label="ArUco Overlay", value=True)

                cam_status = gr.Textbox(label="Camera Status", interactive=False)

                with gr.Row():
                    cam_a_feed = gr.Image(
                        label="Camera A",
                        height=300
                    )
                    cam_b_feed = gr.Image(
                        label="Camera B",
                        height=300
                    )

                capture_btn = gr.Button("Capture ArUco Data", variant="primary")
                aruco_output = gr.Textbox(label="ArUco Detection Results", lines=15, max_lines=20)




            with gr.TabItem("Home"):
                with gr.Row():
                    robot_class = gr.Dropdown(
                        choices=["ur", "kinova", "niryo"],
                        value="kinova",
                        label="Robot Class"
                    )
                    instruction_mode = gr.Radio(
                        choices=["Text", "Voice"],
                        value="Text",
                        label="Instruction Mode"
                    )

                with gr.Row():
                    task_input = gr.Textbox(label="Task Description", lines=3, placeholder="Enter your task...")
                    with gr.Column():
                        voice_btn = gr.Button("🎤 Voice Input", visible=False)

                llm1_btn = gr.Button("Safety check & Call LLM1", variant="primary")

                with gr.Row():
                    llm1_prompt_output = gr.Textbox(label="LLM1 Prompt", lines=5)
                    llm1_response_output = gr.Textbox(label="LLM1 Response", lines=5)

            with gr.TabItem("LLM"):
                with gr.Row():
                    llm2_btn = gr.Button("Run LLM2 (Planning)", variant="primary")
                    llm3_btn = gr.Button("Run LLM3 (Code Gen)", variant="primary")
                    execute_btn = gr.Button("Execute Generated Code", variant="secondary")
                llm2_prompt_output = gr.Textbox(label="LLM2 Prompt", lines=8)

                llm2_output = gr.Textbox(label="LLM2 Output", lines=8)
                llm3_output = gr.Code(label="LLM3 Generated Code", language="python", lines=10)
                execution_output = gr.Textbox(label="Execution Output", lines=8)

        # Event handlers
        start_cam_btn.click(
            fn=start_cameras,
            outputs=[cam_status]
        )

        stop_cam_btn.click(
            fn=stop_cameras,
            outputs=[cam_status]
        )

        aruco_checkbox.change(
            fn=toggle_aruco_overlay,
            inputs=[aruco_checkbox],
            outputs=[cam_status]
        )

        capture_btn.click(
            fn=capture_aruco_data,
            outputs=[aruco_output]
        )

        fetch_telemetry_btn.click(
            fn=fetch_robot_telemetry,
            outputs=[cartesian_output, joints_output]
        )

        ur_calib_btn.click(
            fn=lambda: calibrate_robot("ur"),
            outputs=[ur_status, ur_ex, ur_ey]
        )

        kinova_calib_btn.click(
            fn=lambda: calibrate_robot("kinova"),
            outputs=[kinova_status, kinova_ex, kinova_ey]
        )

        niryo_calib_btn.click(
            fn=lambda: calibrate_robot("niryo"),
            outputs=[niryo_status, niryo_ex, niryo_ey]
        )

        def update_voice_button(mode):
            return gr.update(visible=(mode == "Voice"))

        instruction_mode.change(
            fn=update_voice_button,
            inputs=[instruction_mode],
            outputs=[voice_btn]
        )

        voice_btn.click(
            fn=listen_with_voice,
            outputs=[task_input]
        )

        llm1_btn.click(
            fn=generate_llm1_prompt,
            inputs=[robot_class, task_input, instruction_mode],
            outputs=[llm1_prompt_output, llm1_response_output]
        )

        llm2_btn.click(
            fn=run_llm2_planning,
            outputs=[llm2_prompt_output, llm2_output]
        )

        llm3_btn.click(
            fn=run_llm3_codegen,
            outputs=[llm3_output]
        )

        execute_btn.click(
            fn=execute_generated_code   ,
            outputs=[execution_output]
        )

        # NEW: Real-time camera feed updates
        def update_feeds():
            return get_camera_feed_a(), get_camera_feed_b()

        # Set up automatic feed updates every 100ms
        feed_timer = gr.Timer(0.1)  # 10 FPS UI updates
        feed_timer.tick(
            fn=update_feeds,
            outputs=[cam_a_feed, cam_b_feed]
        )

    return iface


if __name__ == "__main__":
    try:
        iface = create_interface()
        iface.launch(
            server_name="0.0.0.0",
            server_port=None,
            share=False,
            inbrowser=True
        )
    finally:
        app_state.cleanup()