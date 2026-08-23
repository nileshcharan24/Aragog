# Aragog - Project Context

## Project goal

Build a **six-legged walking robot that a person can drive using hand gestures alone**, with no controller, no phone app, and no physical contact.

Stated as a set of questions the project had to answer:

- Can a robot walk reliably on six legs using a cheap microcontroller?
- Can a laptop recognise hand gestures accurately enough, and fast enough, to be used as a live control input?
- Can the two be connected wirelessly with low enough delay that the robot feels responsive?

Aragog answers yes to all three, and the interesting engineering is in how the three pieces fit together.

---

## Why the project matters

Most hobby robots are controlled by a phone app or a Bluetooth remote. That is a solved and slightly boring problem. Gesture control is more interesting because it forces you to deal with three different domains at once:

1. **Embedded systems** — writing C that drives a lot of servo motors in a precise, repeating pattern, on a chip with very little memory.
2. **Computer vision and machine learning** — turning a noisy camera image into a reliable, single, discrete decision many times per second.
3. **Networking and system design** — getting two very different machines to agree on a protocol without tightly coupling them.

That combination is what makes it a good project to talk about: it is not just a model, and it is not just a robot.

---

## The three parts of the system

### 1. The robot (hardware + C firmware)

A six-legged (hexapod) walking robot built around an **ESP32** microcontroller, programmed in **C** using the **Arduino IDE**.

The firmware is responsible for:

- connecting to Wi-Fi and subscribing to the MQTT command topic,
- holding the **movement logic** — the sequences of joint angles that make up a walking gait,
- **onboard motor control** — driving the leg servos to the right angles at the right times,
- reacting to an incoming command letter by switching to the matching gait.

The important idea here is that **the robot owns the walking**. The laptop never says "move joint 4 to 37 degrees." It says `f`, and the robot already knows what forward means. This is what makes the system robust: if the network hiccups, the robot is still executing a coherent motion, and the worst case is that it keeps doing the last thing it was told until the next message arrives.

### 2. The control station (Python computer vision)

A Python program (`Gesture.py`) running on a laptop with a webcam. Every frame, it:

- captures the image and mirrors it so the view feels natural,
- uses **MediaPipe Hands** to locate the hand and extract 21 landmark points,
- normalises those points so hand position and distance from the camera stop mattering,
- feeds them to a small trained neural network that outputs a gesture class,
- converts the gesture to a one-letter command,
- publishes that command over MQTT,
- draws the landmarks, bounding box, and detected gesture on screen so you can see what the system thinks it is seeing.

### 3. The link (MQTT)

**MQTT** is a publish/subscribe messaging protocol designed for exactly this kind of situation: small devices, unreliable networks, tiny messages.

- The laptop is the **publisher**. It publishes to a topic.
- The ESP32 is the **subscriber**. It listens to the same topic.
- A **broker** sits in the middle and routes messages.

Neither side knows anything about the other beyond the topic name and the meaning of the letters. You could replace the entire vision system with a keyboard script and the robot would not notice.

---

## The journey: how the communication was chosen

This is one of the most interesting things to talk about, because the final design was not the first one.

**Attempt 1 — Arduino over USB with pyFirmata.**
`Ignore/led_trial.py` is the very first test: connect to a board on `COM4`, blink an LED on pin 12. Then `Ignore/trial.py` and `Ignore/Untitled-1.py` wire the gesture recogniser directly to that LED — show one gesture, LED on; show another, LED off. This proved the vision-to-hardware pipeline worked end to end.

The fatal problem: **pyFirmata needs a USB cable**. A robot on a leash is not a robot.

**Attempt 2 — HC-05 Bluetooth.**
`communication.py` is the next step: a small Tkinter window with an On button, talking to a board over a Bluetooth serial link. This removes the cable, but Bluetooth has short range, is fiddly to pair, and pushes you toward a one-to-one connection.

**Attempt 3 — MQTT over Wi-Fi.** This is what shipped.

- range is whatever your Wi-Fi covers,
- the ESP32 has Wi-Fi built in, so no extra module is needed,
- publish/subscribe means you could add a second robot, or a logging dashboard, without changing the sender,
- messages are one byte, so latency is negligible.

Being able to explain *why you moved from A to B to C* is far more valuable in an interview than only being able to describe C.

---

## Design decisions worth defending

### One-letter commands

Commands are single characters: `f`, `b`, `l`, `r`, `e`. Not JSON, not a struct.

Why: the ESP32 has limited RAM and no spare cycles for parsing. A single byte is unambiguous, cannot be half-received, and costs nothing to handle. The protocol is small enough to hold in your head, which makes debugging trivial — you can test the robot by publishing `f` from any MQTT client on your phone.

### Default to stop

If MediaPipe finds no hand in the frame, the command defaults to `e` (stop). This is a deliberate **fail-safe**. The alternative — hold the last command — means a robot that walks off a table because you scratched your nose.

### Only track one hand

The main program sets `max_num_hands=1`. Two hands would mean two competing commands with no rule for resolving them. Restricting to one hand removes the ambiguity entirely and makes each frame cheaper to process.

### Send only on change

The MQTT publishing logic tracks the last gesture sent and only publishes when the gesture actually changes. At 20–30 frames per second, publishing every frame would flood the broker with hundreds of identical messages per second for no benefit.

### Run the model in TensorFlow Lite

The model is trained in regular TensorFlow but exported to a quantised `.tflite` file for live use. It loads faster, uses less memory, and runs inference in well under a millisecond on CPU — which matters when you have a 30 ms frame budget to share with camera capture, MediaPipe, and drawing.

### Force CPU only

`Gesture.py` sets `CUDA_VISIBLE_DEVICES = '-1'`, disabling the GPU. The classifier is tiny — a handful of small dense layers. Moving data to a GPU and back costs more than just doing the maths on CPU, and it avoids a whole category of driver problems.

---

## What you learned from building this

- How to split a system across two machines and design a protocol between them.
- How to collect your own dataset, label it, train a model, and deploy it — the whole ML loop, not just the training step.
- Why normalisation matters: without it, the model learns *where your hand is* instead of *what shape it is making*.
- How to write timing-sensitive C for a microcontroller with tight memory.
- That iterating on the communication layer (USB → Bluetooth → MQTT) is a legitimate and important part of engineering, not a sign of failure.
- The value of fail-safe defaults in any system that moves in the physical world.

---

## Interview story

> "I built Aragog, a six-legged walking robot controlled entirely by hand gestures. The robot runs C firmware on an ESP32 that handles the servo control and the walking gaits. Separately, a Python program on a laptop uses MediaPipe to extract 21 hand landmarks from a webcam feed, normalises them, and classifies them with a small neural network I trained on data I recorded myself. The recognised gesture becomes a one-character command published over MQTT, which the robot subscribes to. I deliberately kept the protocol minimal so the robot owns all the movement logic and the network only carries intent — that made the system resilient and easy to debug."
