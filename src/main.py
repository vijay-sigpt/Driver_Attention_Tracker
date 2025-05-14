import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import numpy as np
import csv
from collections import deque
from utils.face_utils import process_frame, calculate_ear, calculate_mar, draw_landmarks, calculate_head_pose

# Configuration
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.5
DROWSINESS_FRAMES = 50
YAW_THRESHOLD = 35  # Increased for normal glances
PITCH_THRESHOLD = 25  # Increased for normal tilts
DISTRACTION_FRAMES = 50  # ~2 seconds at 25 FPS
WINDOW_SECONDS = 60
FPS = 25
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# Camera matrix and distortion coefficients
CAMERA_MATRIX = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH / 2],
                          [0, FRAME_WIDTH, FRAME_HEIGHT / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

def compute_attention_score(blink_count, yawn_count, drowsiness_duration, distraction_duration, num_frames):
    """Compute driver attention score (0-100) based on metrics."""
    if num_frames == 0:
        return 100

    # Normalize metrics
    blink_rate = (sum(blink_count) / num_frames) * FPS * 60
    yawn_rate = (sum(yawn_count) / num_frames) * FPS * 60
    drowsiness_ratio = sum(drowsiness_duration) / num_frames
    distraction_ratio = sum(distraction_duration) / num_frames

    # Score components (weights sum to 100)
    blink_score = max(0, 25 - abs(blink_rate - 17.5) * 1.5)  # Softer penalty
    yawn_score = max(0, 25 - yawn_rate * 5)  # Softer penalty
    drowsiness_score = max(0, 30 - drowsiness_ratio * 50)  # Reduced penalty
    distraction_score = max(0, 20 - distraction_ratio * 50)  # Penalize distractions

    return int(blink_score + yawn_score + drowsiness_score + distraction_score)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Ensure logs directory exists
    os.makedirs("docs/logs", exist_ok=True)
    log_file = "docs/logs/attention_log.csv"
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if os.path.getsize(log_file) == 0:
            writer.writerow(["Timestamp", "Event", "Details"])

    # Tracking variables
    prev_time = time.time()
    fps_list = []
    drowsiness_counter = 0
    distraction_counter = 0
    blink_detected = False
    yawn_detected = False
    window_frames = int(WINDOW_SECONDS * FPS)
    blink_count = deque(maxlen=window_frames)
    yawn_count = deque(maxlen=window_frames)
    drowsiness_duration = deque(maxlen=window_frames)
    distraction_duration = deque(maxlen=window_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 10:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
        prev_time = current_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Process frame
        landmarks, frame_rgb = process_frame(frame)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detection and tracking
        is_drowsy = False
        is_distracted = False
        ear_avg = None
        mar = None
        yaw = None
        if landmarks:
            # EAR calculations
            ear_left = calculate_ear(LEFT_EYE, landmarks, frame.shape)
            ear_right = calculate_ear(RIGHT_EYE, landmarks, frame.shape)
            ear_avg = (ear_left + ear_right) / 2.0

            # MAR calculation
            mar = calculate_mar(MOUTH, landmarks, frame.shape)

            # Head pose estimation
            yaw, pitch, roll = calculate_head_pose(landmarks, frame.shape, CAMERA_MATRIX, DIST_COEFFS)
            head_pose_ok = (yaw is not None and abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD)
            
            # Distraction detection
            if not head_pose_ok:
                distraction_counter += 1
                if distraction_counter >= DISTRACTION_FRAMES:
                    is_distracted = True
                    distraction_duration.append(1)
                    with open(log_file, mode="a", newline="") as f:
                        csv.writer(f).writerow([timestamp, "Distraction", f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}"])
            else:
                distraction_counter = 0
                distraction_duration.append(0)

            # Blink detection (log only)
            if ear_avg < EAR_THRESHOLD and not blink_detected:
                with open(log_file, mode="a", newline="") as f:
                    csv.writer(f).writerow([timestamp, "Blink", f"EAR: {ear_avg:.3f}"])
                blink_count.append(1)
                blink_detected = True
            elif ear_avg >= EAR_THRESHOLD:
                blink_detected = False
            else:
                blink_count.append(0)

            # Drowsiness detection
            if ear_avg < EAR_THRESHOLD:
                drowsiness_counter += 1
                drowsiness_duration.append(1)
                if drowsiness_counter >= DROWSINESS_FRAMES:
                    is_drowsy = True
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    with open(log_file, mode="a", newline="") as f:
                        csv.writer(f).writerow([timestamp, "Drowsiness", f"EAR: {ear_avg:.3f}, Frames: {drowsiness_counter}"])
            else:
                drowsiness_counter = 0
                drowsiness_duration.append(0)

            # Yawn detection
            if mar > MAR_THRESHOLD and not yawn_detected:
                print("Yawn detected!")
                with open(log_file, mode="a", newline="") as f:
                    csv.writer(f).writerow([timestamp, "Yawn", f"MAR: {mar:.3f}"])
                yawn_count.append(1)
                yawn_detected = True
            elif mar <= MAR_THRESHOLD:
                yawn_detected = False
            else:
                yawn_count.append(0)

            # Visualize landmarks
            frame = draw_landmarks(frame, landmarks, LEFT_EYE + RIGHT_EYE + MOUTH)

        else:
            blink_count.append(0)
            yawn_count.append(0)
            drowsiness_duration.append(0)
            distraction_duration.append(0)
            distraction_counter = 0

        # Compute attention score
        attention_score = compute_attention_score(
            blink_count, yawn_count, drowsiness_duration, distraction_duration, len(blink_count)
        )

        # Display metrics
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Attention: {attention_score}/100", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_distracted:
            cv2.putText(frame, "DISTRACTION DETECTED!", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if yaw is not None:
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Driver Monitoring", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()