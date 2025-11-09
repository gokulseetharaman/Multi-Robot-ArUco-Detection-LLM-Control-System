<h1 align="center">🤖 Multi-Robot ArUco Detection & LLM Control System</h1>

<p align="center">
  <strong>Multi-Robot Manipulation: Natural Language–to–Motion Planning with Large Language Models</strong><br>
  <em>By Gokul Seetharaman — Aston University, UK</em>
</p>

<p align="center">
  🎥 <a href="https://www.youtube.com/playlist?list=PL9tuJVtOLPdCYtUmvfCv5NJPdTQTr07uD" target="_blank"><strong>YouTube Playlist</strong></a> |
  📄 <a href="" target="_blank"><strong>Research Paper (PDF)</strong></a> |
  💻 <a href="https://github.com/gokulseetharaman/Multi-Robot-ArUco-Detection-LLM-Control-System" target="_blank"><strong>GitHub Repository</strong></a>
</p>

---

## 📘 Overview

This project implements a **vision-to-action multi-robot system** that translates **natural language or voice commands** into safe, executable programs for three different robot arms:

- 🦾 **Universal Robots UR5** (via socket and URScript)  
- 🦿 **Kinova Kortex Gen3** (via Kinova-Py SDK)  
- 🧠 **Niryo Ned2** (via PyNiryo API)

It integrates **ArUco-based computer vision**, **speech recognition**, and a **three-stage Large Language Model (LLM) pipeline** for perception-driven robotic control — **without using ROS2**.  
Safety is governed by **Asimov’s Laws of Robotics**, ensuring human-safe and workspace-bounded execution.

---

## 🎯 Research Aim

To develop a **voice-controlled multi-robot manipulation system** that allows non-expert users to control heterogeneous robots using natural language, while maintaining safety, transparency, and modular extensibility.

---

## 🧩 System Architecture

The system follows a **sense → think → act** model:

1. **Perception Layer (Vision)**  
   - Dual-camera **ArUco marker tracking**  
   - Real-time 6-DoF pose estimation  
   - Dual-view fusion to minimize occlusions  

2. **Reasoning Layer (AI + Safety)**  
   - **LLM1** – Validates user commands against Asimov’s Laws  
   - **LLM2** – Generates safe, robot-agnostic waypoint plans (JSON)  
   - **LLM3** – Translates waypoints into executable robot code (URScript / Py SDK / PyNiryo)  

3. **Execution Layer (Adapters)**  
   - Unified API across UR, Kinova, and Niryo  
   - Modular design — add new robots by implementing an adapter  
   - Real-time telemetry for joint and Cartesian feedback  

4. **User Interface (UI)**  
   - Built with **Gradio**  
   - Live camera feeds with ArUco overlays  
   - Displays LLM reasoning chain, waypoints, and execution logs  

---

## 🧠 LLM Reasoning Pipeline

| Stage | Model | Description |
|-------|--------|-------------|
| **LLM1** | Safety Gate | Checks if the command is safe using Asimov-inspired policies |
| **LLM2** | Waypoint Planner | Generates canonical JSON waypoint plan using robot-specific calibration |
| **LLM3** | Program Generator | Converts waypoints to platform-specific code (URScript / Kortex / PyNiryo) |

All three models are run locally using **Ollama** for privacy and low latency.

---

## 🦾 Supported Robots

| Robot | Framework | Control Method | IP / Port |
|--------|-------------|----------------|------------|
| **Universal Robots UR5** | URScript | TCP socket | 192.168.1.13:30003 |
| **Kinova Kortex Gen3** | Kinova-Py SDK | API Client | 192.168.1.10 |
| **Niryo Ned2** | PyNiryo | High-Level Python API | 192.168.1.15 |

---

## 🧮 Hardware & Software Requirements

### Hardware
- 3 Robot Arms (UR5, Kinova, Niryo)
- 2× RGB Cameras (≥1080p, 30 FPS)
- Calibration Chessboard
- PC with GPU recommended (for LLM inference)

### Software
- Python ≥ 3.10  
- OpenCV, NumPy, Gradio, PyNiryo, Kinova SDK, Socket library  
- Ollama (for local LLM execution)  
- Optional: Whisper (speech-to-text)

---

## 🧰 Folder Structure

```
📂 Multi-Robot-ArUco-Detection-LLM-Control-System
 ├── aruco.py                  # Single-camera ArUco detection
 ├── aruco_core.py             # Dual-camera ArUco fusion
 ├── calib.py / calib_utils.py # Camera calibration utilities
 ├── main.py                   # Core orchestration
 ├── gradio_app.py             # Web interface
 ├── llm.py / prompt.py        # LLM coordination and prompts
 ├── kinova.py / ur.py / niryo.py   # Robot-specific control
 ├── *_calibration.py          # Per-robot world mapping
 ├── voice.py                  # Speech transcription
 ├── execute.py                # Generated program executor
 └── utilities.py              # General helper functions
```

---

## 🚀 How It Works

1. **Start Cameras & Calibrate**
   ```bash
   python calib.py
   ```

2. **Run Main Application**
   ```bash
   python gradio_app.py
   ```

3. **Access Gradio UI**
   - Open the local URL (e.g. `http://127.0.0.1:7860`)
   - Select robot → input voice/text command → run pipeline  

4. **Observe**
   - LLM1 → safety check  
   - LLM2 → waypoint generation  
   - LLM3 → program creation/execution  

---

## 🎥 Demonstration Videos

- [Camera Calibration](https://youtu.be/2VNVjzndNNY)  
- [3-Robot Synchronization](https://youtu.be/VRhSfO9us60)  
- [Safety Gate Demo](https://youtu.be/d-GZQA13wuk)  
- [Waypoint Generation](https://youtu.be/N7wOHBZU8-w)  
- [UR5 Execution](https://youtu.be/iKN_JurQ1kw)  
- [Kinova Execution](https://youtu.be/j4kEW-6yqwo)  
- [Niryo Execution](https://youtu.be/DUWKm3L2NcE)

---

## 📊 Results Summary

| Metric | Description | Performance |
|--------|--------------|--------------|
| **Task Success Rate** | Successful pick/place executions | ~60% (lab conditions) |
| **Safety Compliance** | LLM1 rejections prevented unsafe actions | 100% |
| **UI Latency** | Real-time refresh with minimal lag | ~10 FPS |
| **Multi-Robot Portability** | Same logic across UR, Kinova, Niryo | ✅ Successful |

---

## 🧱 Limitations

- 2D cameras introduce minor pose jitter under glare or occlusion.  
- Occasional malformed JSON from LLM2/3 requires safety-layer correction.  
- Cartesian plans can approach joint-space singularities.  
- Performance bounded by LLM inference time and robot speed.

---

## 🔭 Future Work

- Integrate **RGB-D / stereo vision** for depth-aware perception  
- Add **multi-robot coordination (MRTA)** and dynamic scheduling  
- Extend **human-robot collaboration (HRC)** with real-time proximity sensing  
- Enforce **JSON-schema-constrained decoding** for LLM safety  
- Containerize (Docker + profiles) for secure deployment  

---


## 📎 Links

- 🎓 [Dissertation PDF](…)  
- 🧠 [YouTube Playlist](https://www.youtube.com/playlist?list=PL9tuJVtOLPdCYtUmvfCv5NJPdTQTr07uD)  
- 💻 [Project Repository](https://github.com/gokulseetharaman/Multi-Robot-ArUco-Detection-LLM-Control-System)  


---

### ⭐ If you find this project helpful, please consider giving it a star on GitHub!
