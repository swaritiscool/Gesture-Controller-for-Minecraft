import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import numpy as np
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as ctrlmou
from pynput.mouse import Button

keyboard = Controller()
mouse = ctrlmou()

# --- Model Definition (from train_and_test.py) ---
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 128
        self.lin1 = nn.Linear(42, self.hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, 3) # Assuming 3 classes based on the original data

    def forward(self, x):
        return self.lin3(self.relu(self.lin2(self.relu(self.lin1(x)))))

# --- Load the trained model ---
model = Model()
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

# --- MediaPipe HandLandmarker setup (from cv_mp_detection.py) ---
MODEL_PATH = "hand_landmarker.task"

baseOptions = mp.tasks.BaseOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOptions = vision.HandLandmarkerOptions
visionRunningMode = vision.RunningMode

options = handLandmarkerOptions(
    base_options=baseOptions(model_asset_path=MODEL_PATH),
    running_mode=visionRunningMode.VIDEO,
    num_hands=1
)

landmarker = handLandmarker.create_from_options(options)

# --- Gesture label mapping ---
# Assuming these labels correspond to the integer outputs of the model
# User might need to define this based on their training data
# For now, using generic labels
gesture_labels = {
    0: "forward",
    1: "backward",
    2: "fight"
}

w_enb = False
s_enb = False

# --- Video Capture and Inference Loop ---
cap = cv2.VideoCapture(0)
timestamp = 0

def draw_hand_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw connections manually
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),       # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),# Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                               # Palm base
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        # Assuming only one hand is detected as num_hands=1
        hand_landmarks = result.hand_landmarks[0]
        
        # Prepare input for the model
        landmarks_flat = []
        for lm in hand_landmarks:
            landmarks_flat.append(lm.x)
            landmarks_flat.append(lm.y)
        
        input_tensor = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0) # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_gesture = gesture_labels.get(predicted_idx.item(), "Unknown Gesture")
        
        # Draw landmarks and predicted gesture
        draw_hand_landmarks(frame, result.hand_landmarks)
        cv2.putText(frame, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if predicted_gesture == "forward":
            keyboard.tap(Key.ctrl)
            keyboard.press("w")
            w_enb = True
        else:
            if w_enb:
                keyboard.release("w")
                w_enb = False
        if predicted_gesture == "backward":
            keyboard.tap(Key.ctrl)
            keyboard.press("s")
            s_enb = True
        else:
            if s_enb:
                keyboard.release("s")
                s_enb = False
        if predicted_gesture == "fight":
            mouse.click(Button.left, 2)

    else:
        if w_enb:
            keyboard.release("w")
            w_enb = False

        if s_enb:
            keyboard.release("s")
            s_enb = False
    cv2.imshow("Hand Gesture Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
