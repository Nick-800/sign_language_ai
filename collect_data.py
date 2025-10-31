import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

gesture_name = input("Enter gesture name: ")

os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{gesture_name}.csv"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        print("Press 's' to start recording and 'q' to quit.")
        recording = False

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    if recording:
                        writer.writerow(row)

            cv2.imshow("Collecting Gesture Data", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("Recording started!")
                recording = True
            elif key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
