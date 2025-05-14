import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame):
    """Process a frame to extract facial landmarks using Mediapipe."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark, frame_rgb
    return None, frame_rgb

def calculate_ear(eye_points, landmarks, frame_shape):
    """Calculate Eye Aspect Ratio (EAR) for blink/drowsiness detection."""
    points = [
        (int(landmarks[p].x * frame_shape[1]), int(landmarks[p].y * frame_shape[0]))
        for p in eye_points
    ]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

def calculate_mar(mouth_points, landmarks, frame_shape):
    """Calculate Mouth Aspect Ratio (MAR) for yawning detection."""
    points = [
        (int(landmarks[p].x * frame_shape[1]), int(landmarks[p].y * frame_shape[0]))
        for p in mouth_points
    ]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[7]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
    C = np.linalg.norm(np.array(points[3]) - np.array(points[5]))
    D = np.linalg.norm(np.array(points[0]) - np.array(points[4]))
    mar = (A + B + C) / (2.0 * D) if D != 0 else 0
    return mar

def draw_landmarks(frame, landmarks, points, color=(0, 255, 0)):
    """Draw specified facial landmarks on the frame."""
    if landmarks:
        for idx in points:
            x = int(landmarks[idx].x * frame.shape[1])
            y = int(landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, color, -1)
    return frame

def calculate_head_pose(landmarks, frame_shape, camera_matrix, dist_coeffs):
    """Estimate head pose (yaw, pitch, roll) using Mediapipe landmarks."""
    # 3D model points (reference points on a generic face, in arbitrary units)
    model_points = np.array([
        [0.0, 0.0, 0.0],        # Nose tip (landmark 1)
        [0.0, -330.0, -65.0],   # Chin (landmark 152)
        [-225.0, 170.0, -135.0], # Left eye left corner (landmark 33)
        [225.0, 170.0, -135.0],  # Right eye right corner (landmark 263)
        [-150.0, -150.0, -125.0],# Left mouth corner (landmark 61)
        [150.0, -150.0, -125.0]  # Right mouth corner (landmark 291)
    ], dtype=np.float32)

    # 2D image points (corresponding Mediapipe landmarks)
    image_points = np.array([
        (landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0])
        for i in [1, 152, 33, 263, 61, 291]
    ], dtype=np.float32)

    # Solve PnP to estimate rotation and translation
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if success:
        # Convert rotation vector to Euler angles (yaw, pitch, roll)
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, np.zeros((3, 1))))
        euler_angles = cv2.decomposeProjectionMatrix(P)[6]
        pitch, yaw, roll = euler_angles.flatten()

        # Normalize angles to degrees
        pitch = pitch % 360
        yaw = yaw % 360
        roll = roll % 360
        if pitch > 180:
            pitch -= 360
        if yaw > 180:
            yaw -= 360
        if roll > 180:
            roll -= 360

        return yaw, pitch, roll
    return None, None, None