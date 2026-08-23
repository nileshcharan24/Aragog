# Aragog - Documentation Index

> Beginner-friendly documentation for understanding your Aragog project and explaining it clearly in interviews.

---

## What this project is

Aragog is a **six-legged walking robot that you control with hand gestures**.

There is no remote, no joystick, and no app. You show your hand to a webcam, the computer works out which gesture you are making, and the robot walks in that direction.

The system has two halves that talk to each other over Wi-Fi:

1. **The control station** (a laptop running Python) — watches the webcam, recognises the gesture, and sends a one-letter command.
2. **The robot** (an ESP32 microcontroller running C firmware) — receives that command and moves its twelve leg joints to walk forward, backward, left, right, or stop.

The two halves are connected by **MQTT**, a lightweight messaging protocol commonly used in IoT.

---

## The one-line version

I built a six-legged walking robot controlled by hand gestures: a Python computer-vision module recognises hand signs with MediaPipe and a small neural network, publishes movement commands over MQTT, and an ESP32 running C firmware translates those commands into leg movements.

---

## How to read this documentation

If you are short on time, read them in this order:

1. **QUICK_START_GUIDE.md** — the fastest possible understanding of the project.
2. **PROJECT_CONTEXT.md** — the full story: what the project is, why you built it, what each part does.
3. **TECHNICAL_EXPLANATION.md** — the deep beginner-friendly explanation of how gesture recognition actually works.
4. **INTERVIEW_GUIDE.md** — ready-made answers for the questions you will actually be asked.

The remaining files are reference material.

---

## File overview

| File | What it covers |
|---|---|
| [QUICK_START_GUIDE.md](./QUICK_START_GUIDE.md) | The project in five minutes, plus how to run it |
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | The complete story, goals, and what each component does |
| [TECHNICAL_EXPLANATION.md](./TECHNICAL_EXPLANATION.md) | How hand tracking, the classifier, and MQTT work, explained from zero |
| [ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md) | The whole system in compact diagram form |
| [CODE_WALKTHROUGH.md](./CODE_WALKTHROUGH.md) | File-by-file tour of the repository |
| [INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md) | 30-second pitch, likely questions, honest answers |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Things that broke and how they were fixed |

---

## Technologies at a glance

**Robot side:** C, ESP32, Arduino IDE, servo motor control
**Vision side:** Python, OpenCV, MediaPipe, TensorFlow / TensorFlow Lite, scikit-learn
**Communication:** MQTT (paho-mqtt on the laptop, an MQTT client library on the ESP32)
**Also explored:** Arduino + pyFirmata over USB, and HC-05 Bluetooth, before settling on MQTT over Wi-Fi

---

## Timeline

March 2025 – April 2025.
