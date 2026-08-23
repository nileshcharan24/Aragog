# Aragog - Architecture Summary

## The whole system in one picture

```
┌─────────────────────────── CONTROL STATION (laptop, Python) ───────────────────────────┐
│                                                                                         │
│   Webcam                                                                                │
│     │  BGR frame, 960×540                                                               │
│     ▼                                                                                   │
│   OpenCV          flip horizontally (mirror), convert BGR → RGB                          │
│     │                                                                                   │
│     ▼                                                                                   │
│   MediaPipe Hands ──────────────► 21 hand landmarks (x, y per point)                     │
│     │                                                                                   │
│     ▼                                                                                   │
│   pre_process_landmark()                                                                │
│     • make relative to wrist  (kills position dependence)                               │
│     • flatten to 42 values                                                              │
│     • divide by max absolute  (kills scale dependence)                                  │
│     │                                                                                   │
│     ├──────────────────────────────┬──────────────────────────────┐                     │
│     ▼                              ▼                              │                     │
│   KeyPointClassifier         fingertip buffer (16 frames)          │                     │
│   (TFLite MLP)                     │                              │                     │
│   "what shape?"                    ▼                              │                     │
│     │                        PointHistoryClassifier               │                     │
│     │                        (TFLite MLP) "what motion?"          │                     │
│     │                              │                              │                     │
│     │                              ▼                              │                     │
│     │                        majority vote over last 16 ──────────┘                     │
│     ▼                                                                                   │
│   give_names()  →  'f' / 'b' / 'l' / 'r' / 'e'      (default 'e' if no hand)             │
│     │                                                                                   │
│     ▼                                                                                   │
│   publish only if changed  ──►  paho-mqtt client                                        │
│                                                                                         │
└──────────────────────────────────────┬──────────────────────────────────────────────────┘
                                       │  Wi-Fi, 1-byte payload
                                       ▼
                              ┌────────────────┐
                              │  MQTT  BROKER  │
                              └────────┬───────┘
                                       │  forwards to subscribers
                                       ▼
┌──────────────────────────────── ROBOT (ESP32, C) ───────────────────────────────────────┐
│                                                                                         │
│   MQTT client callback  ──►  command letter                                             │
│     │                                                                                   │
│     ▼                                                                                   │
│   Movement logic:  select gait  (forward / backward / turn left / turn right / hold)     │
│     │                                                                                   │
│     ▼                                                                                   │
│   Servo control:  timed PWM sequences across the six legs' joints                        │
│     │                                                                                   │
│     ▼                                                                                   │
│   Robot walks                                                                           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer by layer

### Perception layer — MediaPipe

Pre-trained, off-the-shelf. Turns a 1.5-million-number image into 42 meaningful numbers. This is the expensive stage and dominates the frame budget.

### Feature layer — normalisation

Pure arithmetic, no learning. Makes the features translation- and scale-invariant. This is what allows the model above it to be tiny.

### Decision layer — two TFLite classifiers

| | Keypoint classifier | Point history classifier |
|---|---|---|
| Question answered | What shape is the hand? | What motion is the hand making? |
| Input | 42 values (one frame) | 32 values (16 frames of fingertip position) |
| Architecture | Dense 20 → Dense 10 → Softmax | Dense 24 → Dense 10 → Softmax |
| Classes | Front, Left, Right, Back, Stop, Banned From Spider, Spider | Stop, Clockwise, Counter Clockwise, Move |
| Training data | ~1,800 self-recorded samples | ~5,300 self-recorded samples |
| Runtime | TensorFlow Lite, quantised | TensorFlow Lite, quantised |

### Smoothing layer

A 16-frame majority vote over recent predictions. Removes single-frame misclassifications before they ever reach the robot.

### Protocol layer — MQTT

One character per command. Published only when the command changes. Default is `e` (stop) whenever no hand is detected.

### Actuation layer — ESP32 firmware in C

Owns all movement logic. Receives intent, produces gait. The network never carries joint angles.

---

## Command protocol

| Character | Meaning |
|---|---|
| `f` | forward |
| `b` | backward |
| `l` | turn left |
| `r` | turn right |
| `e` | stop (also the default when no hand is visible) |

---

## Why the responsibilities are split this way

| Concern | Lives where | Reason |
|---|---|---|
| Perception | Laptop | Needs CPU and a camera; the ESP32 could never run MediaPipe |
| Classification | Laptop | Sits next to the perception it depends on |
| Smoothing | Laptop | Cheaper to filter noise at the source than to transmit it |
| Gait generation | Robot | Must be timing-precise and must survive network loss |
| Servo PWM | Robot | Hard real-time; only the microcontroller can guarantee it |

---

## In one sentence

Aragog splits a gesture-controlled robot into a perception-heavy Python control station and a timing-critical C firmware on an ESP32, joined by a deliberately minimal one-byte MQTT protocol that carries intent rather than low-level control.
