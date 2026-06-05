# Hand Gesture Recognition with Audio Feedback

## Overview

An AI-powered real-time hand gesture recognition system that detects hand gestures using a webcam and converts them into meaningful speech. The project combines Computer Vision, Deep Learning, and Speech Synthesis to enable intuitive and accessible human-computer interaction.

## Features

* Real-time hand gesture detection
* MediaPipe-based hand landmark extraction
* Deep learning gesture classification
* Gesture-to-speech conversion
* Confidence-based prediction filtering
* Prediction smoothing for stable outputs
* Real-time visual feedback
* Audio feedback generation

## Tech Stack

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* gTTS
* Pygame
* Matplotlib
* Seaborn

## System Architecture

1. Webcam captures hand movements.
2. MediaPipe extracts 21 hand landmarks.
3. Landmark coordinates are normalized and converted into feature vectors.
4. Deep Neural Network classifies the gesture.
5. Predicted gesture is mapped to a predefined sentence.
6. gTTS converts the sentence into speech output.
7. Visual and audio feedback are provided in real time.

## Model Architecture

* Input Layer: 63 Features
* Dense Layer: 256 Neurons (ReLU)
* Dropout: 30%
* Dense Layer: 128 Neurons (ReLU)
* Dropout: 30%
* Output Layer: Softmax Classification

## Project Structure

```text
Hand-Gesture-Recognition-with-Audio-Feedback/
│
├── dataset/
│   └── hand_landmarks.csv
│
├── models/
│   ├── gesture_model.h5
│   └── label_encoder.pkl
│
├── training.py
├── gesture_detection.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── demo_1.png
    ├── demo_2.png
```

## Installation

```bash
git clone https://github.com/chandru-python/Hand-Gesture-Recognition-with-Audio-Feedback.git

cd Hand-Gesture-Recognition-with-Audio-Feedback

pip install -r requirements.txt
```

## Training

```bash
python training.py
```

## Run Application

```bash
python gesture_detection.py
```

## Applications

* Assistive Communication Systems
* Gesture-Based Human Computer Interaction
* Accessibility Solutions
* Smart Interfaces
* AI-Powered Communication Tools

## Future Enhancements

* Support for complete sign language recognition
* Multi-hand gesture detection
* Offline text-to-speech support
* Mobile and web deployment
* Transformer-based gesture classification

## Results

* High-accuracy gesture classification
* Real-time prediction and speech generation
* Robust landmark-based feature extraction
* Smooth user interaction experience

## Author

Chandru M

AI/ML Engineer | Generative AI | Computer Vision | Deep Learning

LinkedIn: https://www.linkedin.com/in/chandrum071202/

GitHub: https://github.com/chandru-python
