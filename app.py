import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import socket
from flask import Flask, render_template
from flask_socketio import SocketIO

# --- FLASK & WEB SERVER SETUP ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- ESP32 SETUP ---
ESP32_IP = "000.000.0.100"  # Replace with your ESP32 IP
UDP_PORT = 4210
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- GLOBAL VARIABLES ---
target_shot = "Cover Drive"
evaluate_next_frame = False  # Flag triggered by the web app countdown
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
data = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- NEW: TARGETED GRADING RUBRIC ---
def grade_shot(shot_name, angles):
    errors = []
    
    if shot_name == "Cover Drive":
        if angles['l_el'] < 120: errors.append(f"Keep front elbow higher (was {int(angles['l_el'])}°).")
        if angles['l_knee'] > 160: errors.append(f"Bend front knee more to lean in.")
        if angles['wrist_nose'] > 13: errors.append(f"Head is too far from the bat.")
            
    elif shot_name == "Front Foot Defensive":
        if angles['wrist_nose'] > 20: errors.append("Keep bat and pad closer together.")
        if angles['r_el'] > 105: errors.append("Soft hands! Don't push hard at the ball.")
            
    elif shot_name == "Sweep Shot":
        if angles['r_knee'] > 110: errors.append("Drop your back knee lower to the ground.")
        if angles['l_el'] < 90: errors.append("Extend arms more through the sweep.")
            
    elif shot_name == "Pull Shot":
        if angles['r_el'] < 65: errors.append("Extend arms fully on impact.")
        if angles['leg_dist'] < 12: errors.append("Widen your stance for better balance.")
    
    # --- NEWLY ADDED SHOTS ---
    elif shot_name == "Back Foot Defensive":
        if angles['l_knee'] < 150: errors.append("Stand taller! Keep front leg straight.")
        if angles['leg_dist'] > 10: errors.append("Bring feet closer for back foot defense.")
        if angles['wrist_nose'] > 10: errors.append("Play it late, right under your eyes.")
            
    elif shot_name == "Back Foot Punch":
        if angles['l_knee'] < 150: errors.append("Stay tall on your crease.")
        if angles['l_el'] < 70: errors.append("Punch through with high elbows.")
            
    elif shot_name == "Flick Shot":
        if angles['l_el'] < 110: errors.append("Use your wrists, keep arms extended.")
        if angles['l_knee'] < 150: errors.append("Keep front leg mostly straight to pivot.")
        
    return errors

# --- BACKGROUND CAMERA LOOP ---
def generate_camera_feed():
    global target_shot, evaluate_next_frame
    video = cv2.VideoCapture(0)

    while True:
        suc, img = video.read()
        if not suc: break
        
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = data.process(img1)
        
        if result.pose_landmarks:
            # THIS IS YOUR EXACT WORKING CODE
            landmarks = result.pose_landmarks.landmark
            
            r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ank = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ank = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            
            # Bundle angles for the grader
            current_angles = {
                'r_el': calculate_angle(r_sh, r_el, r_wr),
                'r_knee': calculate_angle(r_hip, r_knee, r_ank),
                'l_el': calculate_angle(l_sh, l_el, l_wr),
                'l_knee': calculate_angle(l_hip, l_knee, l_ank),
                'wrist_nose': calculate_distance(r_wr, nose) * 100,
                'leg_dist': calculate_distance(r_ank, l_ank) * 100
            }

            # --- TARGETED EVALUATION ---
            if evaluate_next_frame:
                evaluate_next_frame = False # Reset flag immediately
                
                errors = grade_shot(target_shot, current_angles)
                
                if errors:
                    socketio.emit('shot_review', {'status': 'error', 'errors': errors, 'points': 0})
                    udp_sock.sendto(b"VIBRATE", (ESP32_IP, UDP_PORT))
                else:
                    socketio.emit('shot_review', {'status': 'success', 'errors': ["Perfect form!"], 'points': 10})

            # Draw Markers
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3))

        # Visual Overlay
        cv2.putText(img, f"EXPECTING: {target_shot}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)

        cv2.imshow("Laptop View: AI Coach Running", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()

# --- WEB ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('set_target_shot')
def handle_target_shot(data):
    global target_shot
    target_shot = data['shot']

@socketio.on('trigger_evaluation')
def handle_eval():
    global evaluate_next_frame
    evaluate_next_frame = True # Tells the camera loop to grade the very next frame

if __name__ == '__main__':
    socketio.start_background_task(generate_camera_feed)
    print("Server running on port 5000.")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
