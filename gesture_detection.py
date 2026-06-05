import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
import os
from gtts import gTTS
import pygame
from tensorflow.keras.models import load_model
from collections import deque, Counter

# =============================
# LOAD MODEL
# =============================

print("Loading model...")
model = load_model("gesture_model.h5")

print("Loading label encoder...")
encoder = joblib.load("label_encoder.pkl")

# =============================
# INIT AUDIO
# =============================

print("Initializing audio...")
pygame.mixer.init()

speech_lock = threading.Lock()

print("Initialization completed")


# =============================
# SPEAK FUNCTION
# =============================

def speak(text):

    def run():

        try:

            with speech_lock:

                print("Speaking:", text)

                filename = "temp_voice.mp3"

                tts = gTTS(text=text, lang='en')
                tts.save(filename)

                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                pygame.mixer.music.unload()

                if os.path.exists(filename):
                    os.remove(filename)

        except Exception as e:
            print("Speech Error:", e)

    threading.Thread(target=run, daemon=True).start()


# =============================
# GESTURE SENTENCES
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
# MAIN DETECTION FUNCTION
# =============================

def run_gesture_detection():

    print("Gesture detection started...")

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    mp_draw = mp.solutions.drawing_utils

    print("Opening camera...")

    cap = cv2.VideoCapture(0)

    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("Camera not opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prediction_buffer = deque(maxlen=5)

    CONFIDENCE_THRESHOLD = 0.80
    COOLDOWN = 2

    current_label = "No Hand"
    current_sentence = ""

    last_spoken_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

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

            try:

                pred = model.predict(row, verbose=0)

                confidence = np.max(pred)

                if confidence > CONFIDENCE_THRESHOLD:

                    detected_label = str(
                        encoder.inverse_transform(
                            [np.argmax(pred)]
                        )[0]
                    )

                    prediction_buffer.append(detected_label)

                    detected_label = Counter(
                        prediction_buffer
                    ).most_common(1)[0][0]

                    if detected_label in gesture_sentences:

                        current_label = detected_label
                        current_sentence = gesture_sentences[detected_label]

                        current_time = time.time()

                        if current_time - last_spoken_time > COOLDOWN:

                            last_spoken_time = current_time

                            speak(current_sentence)

            except Exception as e:
                print("Prediction Error:", e)

        else:
            current_label = "No Hand"

        cv2.putText(
            frame,
            f"Gesture: {current_label}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Sentence: {current_sentence}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("Gesture Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("ESC pressed")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Gesture detection stopped.")


# =============================
# START PROGRAM
# =============================

if __name__ == "__main__":
    run_gesture_detection()
