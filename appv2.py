"""
appv2.py  –  Cricket AI Coach (Flask + SocketIO + MediaPipe + Random Forest)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import threading
import socket
import math

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

# ─────────────────────────────────────────────
#  CONFIGURE THESE PATHS
# ─────────────────────────────────────────────
MODEL_FILE = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\4_Model_Training\cricket_ai_model.pkl"
META_FILE  = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\4_Model_Training\model_meta.json"
ESP32_IP   = "192.168.1.100"   
UDP_PORT   = 4210
# ─────────────────────────────────────────────

print("Loading model …")
rf_model   = joblib.load(MODEL_FILE)
with open(META_FILE) as f:
    meta = json.load(f)
CLASS_NAMES = meta['classes']
print(f"  Loaded. Classes: {CLASS_NAMES}")

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

_lock            = threading.Lock()
_target_shot     = "Cover Drive"
_evaluate_flag   = False  

def set_target(shot):
    global _target_shot
    with _lock:
        _target_shot = shot

def get_target():
    with _lock:
        return _target_shot

def arm_evaluation():
    global _evaluate_flag
    with _lock:
        _evaluate_flag = True

def consume_evaluation():
    global _evaluate_flag
    with _lock:
        if _evaluate_flag:
            _evaluate_flag = False
            return True
    return False

# ── Geometry helpers (Matches extract_data.py exactly) ──────────────────────
def joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def dist(p1, p2):
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def tilt_deg(left_pt, right_pt):
    dy = right_pt[1] - left_pt[1]
    dx = right_pt[0] - left_pt[0]
    return float(np.degrees(np.arctan2(dy, dx)))

# ── Feature Vector Extraction ───────────────────────────────────────────────
def extract_feature_vector(lms):
    PL = mp.solutions.pose.PoseLandmark
    def pt(lm): return [lms[lm.value].x, lms[lm.value].y]

    r_sh   = pt(PL.RIGHT_SHOULDER);   l_sh   = pt(PL.LEFT_SHOULDER)
    r_el   = pt(PL.RIGHT_ELBOW);      l_el   = pt(PL.LEFT_ELBOW)
    r_wr   = pt(PL.RIGHT_WRIST);      l_wr   = pt(PL.LEFT_WRIST)
    r_hip  = pt(PL.RIGHT_HIP);        l_hip  = pt(PL.LEFT_HIP)
    r_knee = pt(PL.RIGHT_KNEE);       l_knee = pt(PL.LEFT_KNEE)
    r_ank  = pt(PL.RIGHT_ANKLE);      l_ank  = pt(PL.LEFT_ANKLE)
    nose   = pt(PL.NOSE)

    mid_sh  = [(r_sh[0]+l_sh[0])/2,   (r_sh[1]+l_sh[1])/2]
    mid_hip = [(r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2]

    # The 12 core features
    frame_features = [
        joint_angle(r_sh, r_el, r_wr),
        joint_angle(l_sh, l_el, l_wr),
        joint_angle(r_hip, r_knee, r_ank),
        joint_angle(l_hip, l_knee, l_ank),
        dist(r_wr, nose)   * 100,
        dist(r_ank, l_ank) * 100,
        dist(r_wr, r_hip)  * 100,
        dist(l_wr, l_hip)  * 100,
        tilt_deg(l_sh, r_sh),
        tilt_deg(l_hip, r_hip),
        tilt_deg(mid_hip, mid_sh),
        dist(mid_sh, mid_hip) * 100,
    ]
    
    # Replicate across 3 phases (setup/impact/follow) for 36 total features
    return np.array(frame_features * 3).reshape(1, -1)

def build_angle_dict(lms):
    PL = mp.solutions.pose.PoseLandmark
    def pt(lm): return [lms[lm.value].x, lms[lm.value].y]

    r_sh,  r_el,  r_wr  = pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_ELBOW),  pt(PL.RIGHT_WRIST)
    r_hip, r_knee, r_ank = pt(PL.RIGHT_HIP),      pt(PL.RIGHT_KNEE),   pt(PL.RIGHT_ANKLE)
    l_sh,  l_el,  l_wr  = pt(PL.LEFT_SHOULDER),  pt(PL.LEFT_ELBOW),   pt(PL.LEFT_WRIST)
    l_hip, l_knee, l_ank = pt(PL.LEFT_HIP),       pt(PL.LEFT_KNEE),    pt(PL.LEFT_ANKLE)
    nose = pt(PL.NOSE)

    return {
        'r_el':       joint_angle(r_sh,  r_el,   r_wr),
        'l_el':       joint_angle(l_sh,  l_el,   l_wr),
        'r_knee':     joint_angle(r_hip, r_knee, r_ank),
        'l_knee':     joint_angle(l_hip, l_knee, l_ank),
        'wrist_nose': dist(r_wr,  nose) * 100,
        'leg_dist':   dist(r_ank, l_ank) * 100,
    }

# ── Rule-based grading rubric ─────────────────────────────────────────────────
def grade_shot(shot_name, angles):
    errors = []

    if shot_name == "Cover Drive":
        if angles['l_el'] < 120:
            errors.append(f"Keep front elbow higher (was {int(angles['l_el'])}°).")
        if angles['l_knee'] > 160:
            errors.append("Bend your front knee to lean into the drive.")
            
    elif shot_name == "Defensive":
        if angles['wrist_nose'] > 20:
            errors.append("Play closer to your body. Keep bat and pad together.")
        if angles['r_el'] > 120:
            errors.append("Play with soft hands! Don't push at the ball.")

    elif shot_name == "Sweep":
        if angles['r_knee'] > 110:
            errors.append("Get lower! Drop your back knee to the ground.")
        if angles['l_el'] < 90:
            errors.append("Extend your arms through the sweep.")

    elif shot_name == "Down The Wicket":
        if angles['leg_dist'] < 30:
            errors.append("Take a bigger stride to get to the pitch of the ball.")
        if angles['wrist_nose'] > 25:
            errors.append("Keep your head still and over the ball as you advance.")

    elif shot_name == "Lofted Legside":
        if angles['l_knee'] > 150:
            errors.append("Clear your front leg to make room for the swing.")
        if angles['r_el'] < 70:
            errors.append("Follow through completely with your bottom hand.")

    elif shot_name == "Lofted Offside":
        if angles['l_el'] < 100:
            errors.append("Fully extend your arms to get under the ball.")
        if angles['l_knee'] > 160:
            errors.append("Keep a strong base. Bend the front knee slightly.")

    elif shot_name == "Upper Cut":
        # Using tilt/distance features could be great here later!
        if angles['wrist_nose'] < 15:
            errors.append("Wait for the ball! Play it late over your shoulder.")
        if angles['r_el'] > 150:
            errors.append("Keep elbows slightly bent to guide the ball, don't slash.")

    return errors

def run_camera():
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose       = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Camera running. Press 'q' in the OpenCV window to quit.")

    while True:
        ret, img = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result  = pose.process(img_rgb)
        target = get_target()

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark

            if consume_evaluation():
                fv = extract_feature_vector(lms)
                pred_idx   = rf_model.predict(fv)[0]
                pred_proba = rf_model.predict_proba(fv)[0]
                pred_name  = CLASS_NAMES[pred_idx]
                confidence = float(pred_proba[pred_idx]) * 100

                angles = build_angle_dict(lms)
                errors = grade_shot(target, angles)

                if pred_name != target:
                    payload = {
                        'status': 'wrong_shot',
                        'errors': [f"Expected '{target}' but you played '{pred_name}' ({confidence:.0f}% confident)."],
                        'points': 0,
                        'prediction': pred_name,
                        'confidence': confidence,
                    }
                    udp_sock.sendto(b"VIBRATE", (ESP32_IP, UDP_PORT))
                elif errors:
                    payload = {
                        'status': 'error',
                        'errors': errors,
                        'points': 0,
                        'prediction': pred_name,
                        'confidence': confidence,
                    }
                    udp_sock.sendto(b"VIBRATE", (ESP32_IP, UDP_PORT))
                else:
                    payload = {
                        'status': 'success',
                        'errors': ["Perfect form!"],
                        'points': 10,
                        'prediction': pred_name,
                        'confidence': confidence,
                    }
                socketio.emit('shot_review', payload)

            mp_drawing.draw_landmarks(
                img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3),
            )

        cv2.putText(img, f"EXPECTING: {target}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
        cv2.imshow("AI Coach - press Q to quit", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Camera stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('set_target_shot')
def handle_target_shot(data):
    set_target(data['shot'])

@socketio.on('trigger_evaluation')
def handle_eval():
    arm_evaluation()

if __name__ == '__main__':
    flask_thread = threading.Thread(
        target=lambda: socketio.run(
            app, host='0.0.0.0', port=5000,
            allow_unsafe_werkzeug=True, use_reloader=False,
        ),
        daemon=True,
    )
    flask_thread.start()
    print("Flask server started on http://localhost:5000")
    run_camera()