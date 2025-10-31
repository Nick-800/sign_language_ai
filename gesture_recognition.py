import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load model and label names
model = tf.keras.models.load_model("gesture_model_tf.h5")
labels = np.load("gesture_labels.npy", allow_pickle=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
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

                row = np.array(row).reshape(1, -1)
                pred = model.predict(row)
                gesture_name = labels[np.argmax(pred)]

                cv2.putText(img, gesture_name, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
