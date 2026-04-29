"""
extract_data.py  –  Cricket pose extraction (CPU-only, rich feature set)
                    with optional DEBUG window for crop verification.

Feature set per phase (setup / impact / follow = 3 phases):
  - 4 joint angles      (elbow L/R, knee L/R)
  - 4 distances         (wrist-nose, leg-dist, wrist-hip L/R)
  - 2 tilt angles       (shoulder tilt, hip tilt)
  - 2 spine metrics     (trunk lean, torso extension/crouch depth)
  = 12 features × 3 phases = 36 features total  +  label

DEBUG MODE
──────────
Set DEBUG_MODE = True to open a live OpenCV window while processing.
The window shows:
  • LEFT  panel  – cropped region being fed to MediaPipe
  • RIGHT panel  – full original frame with crop rectangle drawn on it
  • Skeleton drawn in CYAN/RED on whichever view was actually used
  • Banner at top: CROP USED (green) or FALLBACK – FULL FRAME (orange)
  • Impact marker: flashes yellow when the impact frame is detected
  • Wrist-Y graph scrolling across the bottom so you can watch the
    velocity peak and verify impact detection is landing in the right place

Controls while the window is open:
  Q  – quit the entire script immediately
  N  – skip to the next video (don't save this one)
  P  – pause / resume
  S  – save a screenshot of the current frame to your Desktop
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import csv
import sys
import time

# ─────────────────────────────────────────────
#  CONFIGURE THESE PATHS
# ─────────────────────────────────────────────
DATASET_PATH = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\1_Raw_Videos"
CSV_FILE     = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\3_Extracted_Data\training_data.csv"
SCREENSHOT_DIR = os.path.expanduser("~/Desktop")   # where S-key screenshots go
# ─────────────────────────────────────────────

# ── Debug toggle ──────────────────────────────────────────────────────────────
DEBUG_MODE = True     # ← set False to run silently (much faster, no window)
DEBUG_SCALE = 0.6     # scale the debug window down if it's too big on your screen
GRAPH_HEIGHT = 80     # height in pixels of the wrist-Y scrolling graph

MP_POSE    = mp.solutions.pose
MP_DRAWING = mp.solutions.drawing_utils

# ── Column names ──────────────────────────────────────────────────────────────
PHASES = ['setup', 'impact', 'follow']
FEAT_NAMES = [
    'r_el', 'l_el', 'r_knee', 'l_knee',
    'wn_dist', 'leg_dist',
    'r_wr_hip_dist', 'l_wr_hip_dist',
    'sh_tilt', 'hip_tilt',
    'trunk_lean', 'torso_ext',
]
COLUMNS    = [f"{phase}_{feat}" for phase in PHASES for feat in FEAT_NAMES] + ['label']
N_FEATURES = len(FEAT_NAMES)   # 12


# ── Geometry helpers ──────────────────────────────────────────────────────────
def joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2  = a - b, c - b
    cos_a   = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def dist(p1, p2):
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def tilt_deg(left_pt, right_pt):
    dy = right_pt[1] - left_pt[1]
    dx = right_pt[0] - left_pt[0]
    return float(np.degrees(np.arctan2(dy, dx)))


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(lms):
    PL = MP_POSE.PoseLandmark

    def pt(lm):
        d = lms[lm.value]
        if d.visibility < 0.30:
            raise ValueError(f"Low visibility: {lm.name} ({d.visibility:.2f})")
        return [d.x, d.y]

    try:
        r_sh,  l_sh  = pt(PL.RIGHT_SHOULDER),  pt(PL.LEFT_SHOULDER)
        r_el,  l_el  = pt(PL.RIGHT_ELBOW),     pt(PL.LEFT_ELBOW)
        r_wr,  l_wr  = pt(PL.RIGHT_WRIST),     pt(PL.LEFT_WRIST)
        r_hip, l_hip = pt(PL.RIGHT_HIP),       pt(PL.LEFT_HIP)
        r_knee,l_knee= pt(PL.RIGHT_KNEE),      pt(PL.LEFT_KNEE)
        r_ank, l_ank = pt(PL.RIGHT_ANKLE),     pt(PL.LEFT_ANKLE)
        nose         = pt(PL.NOSE)

        mid_sh  = [(r_sh[0]+l_sh[0])/2,   (r_sh[1]+l_sh[1])/2]
        mid_hip = [(r_hip[0]+l_hip[0])/2, (r_hip[1]+l_hip[1])/2]

        return [
            joint_angle(r_sh, r_el, r_wr),
            joint_angle(l_sh, l_el, l_wr),
            joint_angle(r_hip, r_knee, r_ank),
            joint_angle(l_hip, l_knee, l_ank),
            dist(r_wr, nose)    * 100,
            dist(r_ank, l_ank)  * 100,
            dist(r_wr, r_hip)   * 100,
            dist(l_wr, l_hip)   * 100,
            tilt_deg(l_sh, r_sh),
            tilt_deg(l_hip, r_hip),
            tilt_deg(mid_hip, mid_sh),
            dist(mid_sh, mid_hip) * 100,
        ]
    except Exception:
        return None


# ── Impact detection ──────────────────────────────────────────────────────────
def find_impact_index(frame_buffer):
    if len(frame_buffer) < 5:
        return len(frame_buffer) // 2

    wrist_ys = np.array([f['wrist_y'] for f in frame_buffer])
    smoothed = np.convolve(wrist_ys, np.ones(5)/5, mode='same')
    velocity = np.diff(smoothed)
    peak_idx = int(np.argmax(velocity))

    lo = int(len(frame_buffer) * 0.10)
    hi = int(len(frame_buffer) * 0.90)
    return peak_idx if lo <= peak_idx <= hi else int(np.argmax(wrist_ys))


# ── Debug rendering helpers ───────────────────────────────────────────────────
def draw_skeleton(frame, landmarks, h, w):
    """Draw pose skeleton scaled to frame dimensions."""
    MP_DRAWING.draw_landmarks(
        frame, landmarks, MP_POSE.POSE_CONNECTIONS,
        landmark_drawing_spec=MP_DRAWING.DrawingSpec(
            color=(0, 255, 255), thickness=2, circle_radius=3),
        connection_drawing_spec=MP_DRAWING.DrawingSpec(
            color=(0, 0, 255), thickness=2),
    )


def build_wrist_graph(wrist_ys, current_idx, width, height):
    """
    Renders a scrolling wrist-Y graph as a numpy image.
    Green vertical line = current frame position.
    Yellow vertical line = detected impact.
    """
    graph = np.zeros((height, width, 3), dtype=np.uint8)

    if len(wrist_ys) < 2:
        return graph

    n = len(wrist_ys)
    ys = np.array(wrist_ys)
    ys_norm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    # invert because wrist_y increases downward in image coordinates
    ys_px = ((1 - ys_norm) * (height - 6) + 3).astype(int)

    # Draw the wrist-Y curve
    for i in range(1, n):
        x1 = int((i-1) / n * width)
        x2 = int(i     / n * width)
        cv2.line(graph, (x1, ys_px[i-1]), (x2, ys_px[i]), (100, 200, 100), 1)

    # Current frame marker (green)
    cx = int(current_idx / n * width)
    cv2.line(graph, (cx, 0), (cx, height), (0, 255, 0), 2)

    # Try to overlay impact position (yellow)
    try:
        imp = find_impact_index(
            [{'wrist_y': y} for y in wrist_ys])
        ix = int(imp / n * width)
        cv2.line(graph, (ix, 0), (ix, height), (0, 220, 220), 2)
        cv2.putText(graph, "impact", (max(0, ix-28), height-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)
    except Exception:
        pass

    cv2.putText(graph, "Wrist-Y  (green=now  yellow=impact)",
                (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return graph


def build_debug_frame(full_frame, crop, crop_box, landmarks_used,
                       used_crop, class_name, video_name,
                       wrist_ys, frame_idx, is_impact_frame):
    """
    Assembles the side-by-side debug view.
    LEFT  = crop fed to MediaPipe (padded to same height as full frame)
    RIGHT = full frame with crop rectangle
    BOTTOM = wrist-Y graph
    """
    h_full, w_full = full_frame.shape[:2]
    h_crop, w_crop = crop.shape[:2]

    # ── Draw skeleton on whichever frame was actually used ─────────────────
    full_disp = full_frame.copy()
    crop_disp = crop.copy()

    if landmarks_used and used_crop:
        draw_skeleton(crop_disp, landmarks_used, h_crop, w_crop)
    elif landmarks_used and not used_crop:
        draw_skeleton(full_disp, landmarks_used, h_full, w_full)

    # ── Draw crop rectangle on full frame ──────────────────────────────────
    y1, y2, x1, x2 = crop_box
    color = (0, 255, 0) if used_crop else (0, 165, 255)   # green=crop, orange=fallback
    cv2.rectangle(full_disp, (x1, y1), (x2, y2), color, 2)

    # ── Resize crop panel to match full frame height ───────────────────────
    crop_resized = cv2.resize(crop_disp, (int(w_crop * h_full / h_crop), h_full))

    # ── Banner strip at the top ────────────────────────────────────────────
    banner_h = 36
    side_by_side = np.hstack([crop_resized, full_disp])
    total_w = side_by_side.shape[1]

    banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    if used_crop:
        banner[:] = (0, 80, 0)
        status_text = "CROP USED"
        status_color = (0, 255, 80)
    else:
        banner[:] = (0, 60, 100)
        status_text = "FALLBACK – FULL FRAME"
        status_color = (0, 200, 255)

    if is_impact_frame:
        banner[:] = (0, 80, 80)
        status_text = "★  IMPACT FRAME  ★"
        status_color = (0, 255, 255)

    cv2.putText(banner, status_text,
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(banner,
                f"{class_name}  |  {video_name}  |  frame {frame_idx}",
                (300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Divider label between crop and full ────────────────────────────────
    crop_w = crop_resized.shape[1]
    cv2.putText(side_by_side, "CROP (fed to AI)",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(side_by_side, "FULL FRAME (crop box shown)",
                (crop_w + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.line(side_by_side, (crop_w, 0), (crop_w, side_by_side.shape[0]), (80, 80, 80), 1)

    # ── Wrist-Y graph strip ────────────────────────────────────────────────
    graph = build_wrist_graph(wrist_ys, frame_idx, total_w, GRAPH_HEIGHT)

    combined = np.vstack([banner, side_by_side, graph])

    # ── Scale down if too large ────────────────────────────────────────────
    if DEBUG_SCALE != 1.0:
        new_w = int(combined.shape[1] * DEBUG_SCALE)
        new_h = int(combined.shape[0] * DEBUG_SCALE)
        combined = cv2.resize(combined, (new_w, new_h))

    return combined


# ── Per-video processing (with optional debug) ────────────────────────────────
def process_video(video_path, pose, class_name=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "OPEN_FAILED"

    video_name  = os.path.basename(video_path)
    frame_buffer = []
    frame_idx   = 0
    paused      = False
    skip_video  = False
    wrist_ys_so_far = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # When paused just re-show the last debug frame without reading
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                paused = False
            elif key == ord('q'):
                cap.release()
                return None, "QUIT"
            elif key == ord('n'):
                skip_video = True
                break
            continue

        h, w = frame.shape[:2]
        y1 = int(h * 0.05);  y2 = int(h * 0.98)
        x1 = int(w * 0.25);  x2 = int(w * 0.75)
        crop      = frame[y1:y2, x1:x2]
        crop_box  = (y1, y2, x1, x2)

        result    = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        used_crop = bool(result.pose_landmarks)

        if not result.pose_landmarks:
            result    = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        landmarks_used = result.pose_landmarks

        if result.pose_landmarks:
            lms   = result.pose_landmarks.landmark
            feats = extract_features(lms)
            if feats:
                wy = lms[MP_POSE.PoseLandmark.RIGHT_WRIST.value].y
                wrist_ys_so_far.append(wy)
                frame_buffer.append({'wrist_y': wy, 'features': feats})

        # ── Debug window ──────────────────────────────────────────────────
        if DEBUG_MODE:
            imp_frame = False
            if len(frame_buffer) >= 10:
                imp_idx_so_far = find_impact_index(frame_buffer)
                imp_frame = (len(frame_buffer) - 1 == imp_idx_so_far)

            dbg = build_debug_frame(
                frame, crop, crop_box,
                landmarks_used, used_crop,
                class_name, video_name,
                wrist_ys_so_far, frame_idx, imp_frame,
            )

            cv2.imshow("Cricket Extraction Debug", dbg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                return None, "QUIT"
            elif key == ord('n'):
                skip_video = True
                break
            elif key == ord('p'):
                paused = True
            elif key == ord('s'):
                fname = os.path.join(
                    SCREENSHOT_DIR,
                    f"cricket_debug_{int(time.time())}.png")
                cv2.imwrite(fname, dbg)
                print(f"\n  📸  Screenshot saved → {fname}")

        frame_idx += 1

    cap.release()

    if skip_video:
        return None, "SKIPPED"

    if len(frame_buffer) < 10:
        return None, "TOO_FEW_FRAMES"

    imp_idx    = find_impact_index(frame_buffer)
    setup_idx  = max(0, imp_idx - 5)
    follow_idx = min(len(frame_buffer) - 1, imp_idx + 5)

    row = (frame_buffer[setup_idx]['features']
         + frame_buffer[imp_idx]['features']
         + frame_buffer[follow_idx]['features'])

    return (row if len(row) == N_FEATURES * 3 else None), "OK"


# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current, total, prefix='', width=30):
    filled = int(width * current / total)
    bar    = '█' * filled + '─' * (width - filled)
    sys.stdout.write(f'\r  [{bar}] {current/total*100:5.1f}%  {prefix[:35]:<35}')
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    written = skipped = 0

    if DEBUG_MODE:
        print("🔍  DEBUG MODE ON")
        print("    Q = quit  |  N = next video  |  P = pause  |  S = screenshot\n")
    else:
        print("🚀  Silent mode  (set DEBUG_MODE=True to open visual window)\n")

    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)

        with MP_POSE.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            classes = sorted([d for d in os.listdir(DATASET_PATH)
                              if os.path.isdir(os.path.join(DATASET_PATH, d))])

            quit_all = False
            for class_name in classes:
                if quit_all:
                    break

                class_dir = os.path.join(DATASET_PATH, class_name)
                videos    = [v for v in os.listdir(class_dir)
                             if v.lower().endswith(('.mp4', '.avi', '.mkv'))]
                if not videos:
                    continue

                print(f"\n── {class_name}  ({len(videos)} videos) ──")

                for idx, vname in enumerate(videos, 1):
                    if not DEBUG_MODE:
                        progress_bar(idx, len(videos), prefix=vname)

                    try:
                        feats, status = process_video(
                            os.path.join(class_dir, vname), pose, class_name)
                    except Exception as e:
                        feats, status = None, f"ERROR: {e}"
                        print(f"\n  ⚠  {vname}: {e}")

                    if status == "QUIT":
                        quit_all = True
                        break

                    if DEBUG_MODE:
                        tag = "✓" if feats else f"✗ ({status})"
                        print(f"  {idx:>3}/{len(videos)}  {vname[:45]:<45}  {tag}")

                    if feats:
                        writer.writerow(feats + [class_name])
                        file.flush()
                        written += 1
                    else:
                        skipped += 1

                if not DEBUG_MODE:
                    print()

    if DEBUG_MODE:
        cv2.destroyAllWindows()

    print(f"\n✅  Done!  {written} rows written, {skipped} videos skipped.")
    print(f"   Features per row: {N_FEATURES * 3}  (12 × 3 phases)")
    print(f"   Saved → {CSV_FILE}")


if __name__ == '__main__':
    main()