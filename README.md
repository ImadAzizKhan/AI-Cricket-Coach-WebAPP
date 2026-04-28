# 🏏 OmniComm: Smart Bat & AI Cricket Coach

**Author:** Muhammad Imad Aziz Khan  
**Context:** Developed as a Human-Computer Interaction (HCI) project at Sapienza University of Rome.

This project merges Computer Vision, IoT hardware, and an interactive web application to create a real-time, closed-loop AI cricket coach. It uses MediaPipe for pose estimation, a Flask web server for mobile interaction, and communicates via UDP to an ESP32 micro-controller embedded in a physical smart cricket bat to deliver haptic feedback.

## ✨ Key Features
* **Targeted Lesson Mode:** Select a specific shot (e.g., Cover Drive) on the mobile web app.
* **Real-Time Grading:** The system waits for your swing's peak extension and grades your specific joint angles.
* **Haptic Feedback:** If your form is incorrect (e.g., dropped front elbow), a UDP packet is sent to an ESP32 to vibrate the bat handle mid-swing.
* **Ghost Skeleton UI:** The mobile app displays the ideal joint alignment for the selected shot before you practice.

## 📁 Project Structure
```text
/
├── app.py                 # Main Flask server and OpenCV/MediaPipe loop
└── templates/
    └── index.html         # Mobile web interface (UI, WebSocket logic, Canvas skeleton)
## 🚀 How to Run the App (Mobile + Laptop)
For seamless, zero-latency communication between your Python backend and mobile interface, both devices must be on the exact same local network. Using a mobile hotspot is the most reliable method for testing.

**1. Network Setup**
* Enable your mobile phone's Wi-Fi Hotspot and connect your laptop to this network.

**2. Locate Your IP Address**
* Open PowerShell or Command Prompt on your laptop.
* Execute the `ipconfig` command.
* Locate your **IPv4 Address** under the active connection (typically formatted as `192.168.x.x` or `10.x.x.x`).

**3. Initialize the Server**
* Open a terminal within your project directory.
* Launch the backend by running: `python app.py`

**4. Connect via Mobile**
* Open a web browser (Chrome or Safari) on your mobile phone.
* Enter your laptop's IPv4 address followed by `:5000` in the URL bar (e.g., `http://10.240.141.2:5000`).
* The AI Coach interface will instantly load on your screen.

## ⚙️ Hardware Integration & Future VR Scope
This computer vision module is engineered to operate in tandem with a custom physical smart bat, which is powered by an ESP32 microcontroller and a BNO085 IMU sensor. Ultimately, these combined kinematic data streams will be transmitted via UDP to the Unity game engine, establishing a fully immersive, physics-accurate VR training environment.

## 🤝 Credits & Acknowledgements
The foundational MediaPipe pose extraction logic and landmark mapping were adapted from the repository by [Baymax07-ig](https://github.com/Baymax07-ig/cricket-pose-detection-analysis). Building upon that baseline, this project introduces a targeted action-review loop, dynamic web UI integration, and hardware-level haptic feedback communication.
