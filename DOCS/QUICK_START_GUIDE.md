# Aragog - Quick Start Guide

## The project in one paragraph

Aragog is a six-legged robot you steer with your hand. A webcam watches you, a Python program figures out which hand sign you are showing, and it sends a single letter (`f`, `b`, `l`, `r`, `e`) over Wi-Fi to the robot. The robot's ESP32 microcontroller reads that letter and moves its legs to walk forward, backward, left, right, or stop.

The name comes from Aragog, the giant spider in Harry Potter — the robot walks on six legs like a spider.

---

## What actually happens, step by step

1. You hold your hand up in front of the webcam.
2. **MediaPipe** finds your hand and returns 21 points on it — fingertips, knuckles, wrist.
3. Those 21 points are cleaned up so that the position and size of your hand on screen do not matter.
4. A **small neural network** looks at the cleaned-up points and says which of the trained gestures it is: Front, Left, Right, Back, Stop, and so on.
5. The gesture is converted to a one-letter command.
6. The command is **published to an MQTT broker** over Wi-Fi.
7. The **ESP32 on the robot** is subscribed to that same topic, receives the letter, and runs the matching walking routine in its C firmware.
8. The robot walks. Total delay: a fraction of a second.

---

## The gestures

The trained gesture classes live in `model/keypoint_classifier/keypoint_classifier_label.csv`:

| Gesture label | Command letter | Robot does |
|---|---|---|
| Front | `f` | Walk forward |
| Back | `b` | Walk backward |
| Left | `l` | Turn left |
| Right | `r` | Turn right |
| Stop | `e` | Stop moving |
| Banned From Spider | `f` | Ignored / mapped to a safe default |
| Spider | — | Extra class collected during training |

**Important safety detail:** if no hand is visible at all, the program defaults to `e` (stop). The robot should never keep walking just because you moved out of frame.

---

## Main technologies

- **Robot firmware:** C, ESP32, Arduino IDE
- **Computer vision:** Python, OpenCV, MediaPipe
- **Machine learning:** TensorFlow (training), TensorFlow Lite (running)
- **Communication:** MQTT via `paho-mqtt`
- **Supporting:** NumPy, scikit-learn

---

## Repository layout (the vision side)

```
Aragog/
├── Gesture.py                       ← main program, run this
├── app.py, app_new.py, Untitled-2.py ← earlier working versions
├── communication.py                 ← Arduino/Bluetooth experiment with a Tkinter button
├── requirements.txt
│
├── model/
│   ├── keypoint_classifier/         ← the "which hand sign is this?" model
│   │   ├── keypoint.csv                 (training data you recorded)
│   │   ├── keypoint_classifier.tflite   (the trained model that runs live)
│   │   ├── keypoint_classifier_label.csv (gesture names)
│   │   └── keypoint_classifier.py       (loads and runs the model)
│   │
│   └── point_history_classifier/    ← the "which motion is this?" model
│       └── (same structure)
│
├── keypoint_classification_EN.ipynb     ← notebook that trains the hand-sign model
├── point_history_classification.ipynb   ← notebook that trains the motion model
│
├── utils/cvfpscalc.py               ← frames-per-second counter
└── Ignore/                          ← scratch experiments (LED tests, early drafts)
```

The ESP32 firmware (C, written in the Arduino IDE) lives outside this repository, on the robot side of the project.

---

## How to run it

```bash
pip install -r requirements.txt
python Gesture.py
```

Useful flags:

```bash
python Gesture.py --device 0 --width 960 --height 540 --min_detection_confidence 0.7
```

Press `ESC` to quit.

**To record new training data:** press `k` to enter key-point logging mode, then press a number key `0`–`9` while holding a gesture. Each press appends one row to `keypoint.csv` with that number as the class label. Then re-run `keypoint_classification_EN.ipynb` to retrain.

---

## What to say in an interview

- I built a six-legged walking robot and controlled it with hand gestures instead of a remote.
- The robot runs C firmware on an ESP32; the gesture recognition runs in Python on a laptop.
- The two communicate over MQTT, which keeps them decoupled — the robot only ever sees a single-character command.
- The gesture recognition uses MediaPipe for hand landmarks and a small neural network I trained on my own recorded data, exported to TensorFlow Lite so it runs fast on CPU.
