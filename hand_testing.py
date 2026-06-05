import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
from tensorflow.keras.models import load_model
from collections import deque, Counter

# =============================
# Initialize Text-to-Speech
# =============================

engine = pyttsx3.init()

engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0) 

# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# =============================
# Load model and encoder
# =============================

model = load_model("gesture_model.h5")
encoder = joblib.load("label_encoder.pkl")

# =============================
# Gesture to sentence mapping
# IMPORTANT: use SAME labels as CSV
# =============================

gesture_sentences = {

    "one": "Hello, how are you?",
    "two": "I am fine.",
    "three": "Thank you very much.",
    "four": "Please help me.",
    "five": "Good morning.",
    "six": "Good afternoon.",
    "seven": "Good evening.",
    "8": "Nice to meet you.",
    "9": "See you later.",
    "10": "Yes.",
    "11": "No.",
    "12": "Stop.",
    "13": "Call emergency."
}

# =============================
# MediaPipe setup
# =============================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =============================
# Webcam
# =============================

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=10)

CONFIDENCE_THRESHOLD = 0.8

current_sentence = ""
last_spoken_label = ""

# =============================
# Main loop
# =============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    label = "No Hand"
    confidence = 0

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        row = []

        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        for lm in hand_landmarks.landmark:

            row.append(lm.x - base_x)
            row.append(lm.y - base_y)
            row.append(lm.z - base_z)

        row = np.array(row).reshape(1, -1)

        pred = model.predict(row, verbose=0)

        confidence = np.max(pred)

        if confidence > CONFIDENCE_THRESHOLD:

            class_id = np.argmax(pred)

            predicted_label = str(
                encoder.inverse_transform([class_id])[0]
            )

            prediction_buffer.append(predicted_label)

            label = Counter(prediction_buffer).most_common(1)[0][0]

            # =============================
            # Map sentence
            # =============================

            if label in gesture_sentences:

                current_sentence = gesture_sentences[label]

                # =============================
                # Speak ONLY when gesture changes
                # =============================

                if label != last_spoken_label:

                    speak(current_sentence)

                    last_spoken_label = label

    # =============================
    # Display gesture
    # =============================

    cv2.putText(frame,
                f"Gesture: {label} ({confidence:.2f})",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    # =============================
    # Display sentence
    # =============================

    cv2.putText(frame,
                f"Sentence: {current_sentence}",
                (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2)

    cv2.imshow("Gesture to Speech", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
