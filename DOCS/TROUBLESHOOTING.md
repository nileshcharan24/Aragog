# Aragog - Troubleshooting

Real problems that come up with this project, why they happen, and how to fix them. Useful both for getting it running again and for answering "tell me about a bug you had to debug".

---

## Camera and OpenCV

### The camera window never opens, or the frame is black

The camera index is wrong. `--device 0` is the built-in webcam on most laptops; an external USB camera is usually `1`. Try `python Gesture.py --device 1`.

Also check nothing else is holding the camera — video call apps keep an exclusive lock on it.

### The window opens but detection is poor

Almost always lighting. MediaPipe needs a reasonably lit hand with contrast against the background. Backlighting — a bright window behind you — is the worst case, because your hand becomes a silhouette.

If lighting is fine, try lowering the detection threshold: `--min_detection_confidence 0.5`.

### Detection quality is subtly bad for no obvious reason

Check that the BGR → RGB conversion is still in place. OpenCV reads frames as BGR; MediaPipe expects RGB. Skip the conversion and nothing crashes — the model just receives colour-swapped images and quietly performs worse. This is the classic silent bug in any OpenCV + MediaPipe pipeline.

---

## The model and predictions

### The prediction flickers between two gestures

Expected behaviour without smoothing — camera noise causes occasional bad frames. The fix already in the code is the 16-frame majority vote over `finger_gesture_history`. If it still flickers, increase the history length, at the cost of feeling slightly less responsive.

### One gesture is never predicted

Two likely causes:

1. **Not enough training data for that class.** Check the class distribution in `keypoint.csv` — count how many rows start with each class ID. If one class has far fewer samples than the others, the model will under-predict it. Record more and retrain.
2. **The gesture is too similar to another one.** If two hand shapes produce nearly identical normalised landmarks, no amount of data will separate them. Redesign one of the gestures to be visually distinct.

The confusion matrix in the training notebook tells you which is happening: similar gestures show up as off-diagonal confusion between a specific pair.

### The first gesture label doesn't match

Byte-order-mark problem. If the label CSV was saved from Excel, it starts with an invisible BOM character that gets glued to the first label, so `"Front"` becomes `"﻿Front"` and never matches. That is why the code opens label files with `encoding='utf-8-sig'` — it strips the BOM. If you re-create the CSV and this breaks, that is the cause.

### Accuracy looks great in the notebook but the live system is unusable

Train/test accuracy measures the model. Live behaviour measures the *system*. The gap is usually one of:

- **Missing normalisation at inference time** — the live path must apply the exact same preprocessing as training. Any mismatch here is fatal and produces near-random predictions.
- **No temporal smoothing** — a 97% accurate model still gets one frame in 33 wrong, which at 30 FPS is roughly once a second.
- **Training data that doesn't match reality** — if all your samples were recorded at one distance and angle, the model has never seen the conditions it now faces.

### The model file won't load

Check the path. `KeyPointClassifier` defaults to `model/keypoint_classifier/keypoint_classifier.tflite`, a **relative** path — so the program must be run from the repository root, not from inside a subfolder.

---

## Performance

### Frame rate is too low

In rough order of impact:

1. Lower the capture resolution: `--width 640 --height 480`. MediaPipe cost scales with image size.
2. Confirm `max_num_hands=1`. Detecting two hands roughly doubles the work.
3. Confirm the GPU is disabled (`CUDA_VISIBLE_DEVICES = '-1'`). For a model this small, GPU transfer overhead exceeds the compute saved, and a misconfigured CUDA setup can be dramatically slower than plain CPU.
4. Comment out the debug drawing. `draw_landmarks` draws dozens of circles and lines per frame.

### TensorFlow prints a wall of CUDA warnings on startup

Cosmetic, caused by TensorFlow looking for a GPU it has been told not to use. Setting `CUDA_VISIBLE_DEVICES` before importing TensorFlow (which `Gesture.py` does) suppresses most of it. The rest can be silenced with `TF_CPP_MIN_LOG_LEVEL=2`.

---

## MQTT and networking

### Connection refused / the client never connects

Work through this order:

1. **Is the broker actually running and reachable?** Ping the broker IP from the laptop.
2. **Is the IP still correct?** This is the most common failure by far. The broker address is hardcoded, and on a DHCP network the machine hosting it gets a new IP whenever it reconnects. Different files in this repo have different hardcoded IPs precisely because of this.
3. **Are both devices on the same network?** Many campus and guest Wi-Fi networks use client isolation, which blocks device-to-device traffic entirely even though both devices have internet.
4. **Is the port right?** MQTT defaults to 1883 unencrypted.
5. **Are the credentials right?** Wrong username or password gives a connection-refused return code, not a timeout.

### The laptop publishes but the robot never reacts

Confirm the topic strings match **exactly** on both sides — they are case-sensitive, and a trailing space is invisible and fatal. Different iteration files in this repo use different topic names, which is a good reason to check rather than assume.

To isolate which half is broken, subscribe to the topic from a third client (MQTT Explorer, or a phone app). If you see the messages arriving there, the laptop is fine and the problem is on the robot. If you don't, the problem is on the laptop.

### The robot reacts sluggishly or lags behind

Check that publish-on-change is active. Publishing every frame at 25–30 FPS floods the broker with identical messages and the robot ends up working through a backlog. Only publish when the command actually changes.

### The robot keeps walking after you lower your hand

The fail-safe isn't firing. Verify that `current_sign` is being reset to `'e'` in the no-hand-detected branch, and that the publish call is outside the `if results.multi_hand_landmarks` block — if publishing only happens when a hand is found, the stop command can never be sent.

---

## Hardware

### `SerialException` on COM4 (the pyFirmata scripts in `Ignore/`)

The port number changes between machines and USB ports. Check Device Manager on Windows for the actual COM port. Also make sure the Arduino IDE's serial monitor isn't open — it holds the port exclusively.

Note that pyFirmata also requires the **StandardFirmata** sketch to be flashed to the board first. Without it, the board simply won't respond.

### Servos jitter or the robot's movement is erratic

Usually power, not code. Many servos drawing current at once will brown out a supply that is fine when they're idle, and a browning-out microcontroller behaves unpredictably. Give the servos their own power supply with adequate current, and share a common ground with the ESP32.

### The robot falls over instead of walking

Gait sequencing. A hexapod must always keep enough legs planted to stay stable while the others swing forward. If too many legs lift at once, or the timing between lift and plant is wrong, it topples. Slow the gait down first — if it walks correctly at half speed, the problem is timing rather than the leg ordering.

---

## Security note

Earlier iteration files (`app.py`, `app_new.py`) contain a hardcoded MQTT broker address, username, and password. Before this repository is shared or made public:

1. Move all of it to environment variables or a git-ignored `.env` file.
2. Rotate the credentials, since they are already in the git history.
3. Consider `git filter-repo` if the history itself needs cleaning.

This is worth fixing, and it is also worth mentioning proactively in an interview — noticing it yourself reads far better than having someone else point it out.
