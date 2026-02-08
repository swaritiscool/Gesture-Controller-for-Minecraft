"""
Data collection + visualization of hand gestures
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas

MODEL_PATH = "hand_landmarker.task"

num_samples = int(input("Enter num samples: "))
internal_clock=1

baseOptions = mp.tasks.BaseOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOptions = vision.HandLandmarkerOptions
visionRunningMode = vision.RunningMode

run = True

cols = ['sample_id', 'label']
for i in range(1, 22):
    cols.append(f'x{i}')
    cols.append(f'y{i}')

data = pandas.DataFrame(columns=cols)
print(data)

options = handLandmarkerOptions(
    base_options=baseOptions(model_asset_path=MODEL_PATH),
    running_mode=visionRunningMode.VIDEO,
    num_hands=1
)

label = input("Enter label name: ")
rec = input("Ready? If recording Y/n: ")
if rec.lower() == "y": record = True
else: record= False

landmarker = handLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

def save_landmarks(hand_landmarks, data):
    buffer_poses = []
    for lm in hand_landmarks:
        if buffer_poses.count((lm.x, lm.y)) < 10:
            buffer_poses.append((lm.x, lm.y))
    # buffer_poses = [(x1,y1), (x2,y2), (x3,y3), ... , (x21, y21)]
    row = [len(data), label]
    for a in buffer_poses:
        row.append(a[0])
        row.append(a[1])
    row_df = pandas.DataFrame([row], columns=data.columns)
    data = pandas.concat([data, row_df])
    return data

def draw_hand_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

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
    
    # Converting from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )
    timestamp += 1
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)
            if record and timestamp % 5 == 0: data = save_landmarks(hand_landmarks, data)
            print(data)
            len(data.label) < num_samples

    print(type(list(data.label)))
    print(data)
    print(type(num_samples))
    print(num_samples)
    if len(data.label) >= int(num_samples):
        print(f"Sampling Complete For {label}")
        cont = input("Would you wish to continue? [Y/n]: ")
        if cont.lower() == "n": break
        label = input("Enter label name: ")
        rec = input("Ready? If recording Y/n: ")
        if rec.lower() == "y": record = True
        else: record= False
        internal_clock +=1
        num_samples = num_samples/(internal_clock-1) * internal_clock
        print(f"Samples left: {num_samples - num_samples/internal_clock}")
        print(f"End Sample: {num_samples-1}")
    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print(data)
data.to_csv("data/hand_gestures.csv", index=False)

cap.release()
cv2.destroyAllWindows()

