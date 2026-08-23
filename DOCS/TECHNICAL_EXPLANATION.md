# Aragog - Technical Explanation

This document explains how the gesture recognition actually works, starting from zero knowledge. If you can follow this, you can answer almost any technical question about the project.

---

## The big picture

A webcam gives you a picture. A robot needs a decision. Everything in between is the interesting part.

The naive approach would be to feed the raw camera image into a big neural network and ask "which gesture is this?" That works, but it is slow, needs a huge amount of training data, and breaks the moment you change your shirt or the lighting.

Aragog uses a much smarter two-stage approach:

1. **Stage 1 — find the hand skeleton.** Use a ready-made, heavily-optimised model (MediaPipe) to turn the image into 21 numbered points on the hand.
2. **Stage 2 — classify the skeleton.** Use a tiny custom neural network to turn those 21 points into a gesture name.

This split is the single most important design idea in the project, so it is worth understanding why it is so much better.

---

## Step 1: Why not just use the raw image?

A 960×540 colour image is about **1.5 million numbers**. Most of them are irrelevant — the wall behind you, your sleeve, the lighting.

A neural network fed raw pixels has to learn, from scratch, all of the following before it can even start on the actual problem:

- what a hand looks like,
- how to ignore the background,
- how to cope with different skin tones and lighting,
- how to cope with the hand being near or far, left or right.

That needs tens of thousands of labelled images and a big model.

By contrast, 21 landmark points is **42 numbers** (an x and a y for each point). All the irrelevant information is gone. Your background, your lighting, and your skin tone have already been stripped away. What is left is pure hand geometry.

**This is the key insight to say out loud in an interview:** "I used MediaPipe as a feature extractor. It reduced 1.5 million pixels to 42 meaningful numbers, which meant my classifier could be tiny and train on a few hundred samples per gesture instead of thousands of images."

---

## Step 2: What MediaPipe gives you

MediaPipe Hands is a Google library that runs a pre-trained hand-tracking pipeline. Given an image, it returns for each detected hand:

- **21 landmarks**, each with an x, y (and z) coordinate. Landmark 0 is the wrist. Landmarks 1–4 are the thumb, 5–8 the index finger, and so on to 17–20 for the little finger. Landmark 8 is the index fingertip.
- **Handedness** — whether it thinks this is a left or right hand.

In the code this is `hands.process(image)`, and the landmarks come back in `results.multi_hand_landmarks`.

Two small but real details in `Gesture.py`:

- The image is converted from OpenCV's **BGR** colour order to **RGB**, because MediaPipe expects RGB. Getting this wrong is a classic silent bug — detection quality quietly drops.
- `image.flags.writeable = False` is set before processing. This lets MediaPipe skip making a defensive copy of the array, which is a free speed-up.

MediaPipe returns coordinates as fractions between 0 and 1. `calc_landmark_list()` multiplies them by the image width and height to get actual pixel positions.

---

## Step 3: Normalisation — the part everyone underestimates

Raw pixel coordinates are useless for classification. Here is why.

Suppose you make a "Front" gesture on the left side of the screen. Landmark 0 might be at (200, 300). Now make the exact same gesture on the right side. Landmark 0 is now at (700, 300). To a network, those are completely different inputs — but they are the same gesture.

The same problem happens with distance: a hand near the camera produces large coordinate spreads, a hand far away produces small ones.

`pre_process_landmark()` fixes both problems in three moves:

**Move 1 — make coordinates relative to the wrist.**

Subtract landmark 0's position from every landmark. Now landmark 0 is always (0, 0), and every other point is expressed as "how far from the wrist". *Where the hand is on screen no longer matters.*

**Move 2 — flatten to a 1D list.**

`[[x0,y0], [x1,y1], ...]` becomes `[x0, y0, x1, y1, ...]` — 42 numbers in a row, which is what a dense neural network layer expects.

**Move 3 — scale by the largest absolute value.**

Find the biggest absolute number in the list and divide everything by it. All 42 values now sit between −1 and 1. *How big the hand appears no longer matters.*

After these three steps, the same gesture produces nearly the same 42 numbers whether it is near or far, left or right. **That is what makes a tiny model possible.**

> **Likely interview question:** "Why did you normalise the landmarks?"
> **Answer:** "To make the features invariant to translation and scale. Without it the model would learn hand position rather than hand shape, and it would fail as soon as you moved."

---

## Step 4: The classifier

The model that decides which gesture you are making is defined in `keypoint_classification_EN.ipynb` and is deliberately small:

```
Input        : 42 values  (21 landmarks × 2 coordinates)
Dropout 0.2
Dense 20     : ReLU
Dropout 0.4
Dense 10     : ReLU
Dense N      : Softmax  → one probability per gesture class
```

Reading that in plain language:

- **Input (42)** — the normalised landmark list.
- **Dense 20, ReLU** — 20 neurons, each looking at all 42 inputs and learning some combination of them. ReLU means "if the result is negative, output zero" — a cheap non-linearity that lets the network learn curved decision boundaries instead of just straight lines.
- **Dense 10, ReLU** — squeezes to 10 numbers, forcing the network to keep only what matters.
- **Dense N, Softmax** — one output per gesture. Softmax turns raw scores into probabilities that add up to 1, so you can read the output as "82% confident this is Front".
- **Dropout** — during training, randomly switch off some fraction of neurons on each pass. This stops the network leaning too heavily on any single neuron and is the main defence against overfitting on a small dataset.

Training setup:

- **Loss:** `sparse_categorical_crossentropy` — the standard choice for multi-class classification with integer labels.
- **Optimiser:** Adam — a good default that adapts its own learning rate.
- **Split:** 75% train, 25% test, with a fixed random seed so results are reproducible.
- **Early stopping:** patience 20 — if the validation score stops improving for 20 epochs, stop, and do not waste time overfitting.
- **Checkpointing:** save the model every time it improves.

The whole model is a few thousand parameters. It trains in seconds on a laptop.

> **Likely interview question:** "Why such a small model?"
> **Answer:** "Because MediaPipe had already done the hard perceptual work. My input was 42 clean, normalised geometric features, not raw pixels. A larger model would have overfitted my few-hundred-sample-per-class dataset without being any more accurate."

---

## Step 5: Collecting your own dataset

There was no existing dataset for gestures like "Front", "Back", or "Spider", so you built one.

`Gesture.py` has a built-in logging mode:

- Press `k` → the program enters key-point logging mode.
- Hold a gesture and press a number key `0`–`9`.
- One row is appended to `model/keypoint_classifier/keypoint.csv`: the number you pressed as the class label, followed by the 42 normalised landmark values.

Repeat a couple of hundred times per gesture, varying hand position, distance, and angle so the model sees natural variety.

The result is roughly 1,800 labelled samples across the gesture classes, with 200-ish samples for most of them.

**One thing to be honest about:** the class counts are not perfectly balanced — one class has noticeably more samples than the others. That is a real, nameable weakness, and being able to say "yes, my dataset was somewhat imbalanced, and I would fix that by collecting more samples for the smaller classes or applying class weights" is a much stronger answer than pretending everything was perfect.

---

## Step 6: TensorFlow Lite and why quantisation matters

After training, the notebook converts the Keras model to **TensorFlow Lite** with default optimisations enabled:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

That optimisation flag turns on **quantisation** — storing weights as 8-bit integers instead of 32-bit floats.

What you get:

- roughly **4× smaller** model file,
- **faster** inference, because integer arithmetic is cheaper than floating point,
- almost no accuracy loss for a model this simple.

At runtime, `KeyPointClassifier` loads the `.tflite` file with a `tf.lite.Interpreter`, pushes in the 42 values, calls `invoke()`, and takes `argmax` of the output — the index of the highest-probability class.

> **Likely interview question:** "What is quantisation?"
> **Answer:** "Storing model weights at lower precision — 8-bit integers instead of 32-bit floats. It shrinks the model and speeds up inference, at the cost of a small amount of precision that a model this simple doesn't miss."

---

## Step 7: The second model — recognising motion, not shape

There is a second, separate classifier: the **point history classifier**.

The difference matters:

- The **keypoint classifier** answers *"what shape is the hand making right now?"* — a single frame.
- The **point history classifier** answers *"what motion has the hand been making?"* — across time.

It works by storing the position of the index fingertip (landmark 8) for the last 16 frames in a `deque(maxlen=16)`. That gives 16 points × 2 coordinates = 32 numbers, which are normalised the same way and fed to a similar small network. Its classes are Stop, Clockwise, Counter Clockwise, and Move.

The training notebook contains **both an LSTM version and a plain dense version**, selectable with a `use_lstm` flag. An LSTM is a recurrent network designed for sequences; the dense version just treats the 32 numbers as a flat vector. The dense version was used — it is far cheaper and, for a fixed-length 16-frame window, works well enough.

> **Likely interview question:** "Why did you have two models?"
> **Answer:** "One is static and one is temporal. A single frame tells you the hand's shape but nothing about movement. Buffering the fingertip position over 16 frames lets a second model recognise motion patterns like a circular sweep."

---

## Step 8: Temporal smoothing — stopping the flicker

Camera frames are noisy. A single bad frame can produce a wrong classification, and if you sent every classification straight to the robot it would twitch.

The fix in `Gesture.py` is a **majority vote over recent history**:

```python
finger_gesture_history.append(finger_gesture_id)
most_common_fg_id = Counter(finger_gesture_history).most_common()
```

The last 16 predictions are kept, and the most frequent one wins. A single outlier frame is outvoted and never reaches the robot.

This is a simple, cheap, very effective technique, and it is a good thing to bring up unprompted — it shows you thought about the difference between a model that works on a test set and a system that works in the real world.

---

## Step 9: From gesture to command

`give_names()` maps a gesture label to a single character:

| Gesture | Character |
|---|---|
| Front | `f` |
| Back | `b` |
| Left | `l` |
| Right | `r` |
| Stop | `e` |

And critically, in the main loop:

```python
current_sign = 'e'   # default before any detection
...
else:
    current_sign = 'e'   # no hand detected → stop
```

The command variable is **initialised to stop** and **reset to stop** whenever no hand is found. Safety is the default state, not an afterthought.

---

## Step 10: MQTT, explained simply

MQTT is a **publish/subscribe** messaging protocol.

Instead of the laptop opening a direct connection to the robot, both connect to a **broker**:

- The laptop **publishes** the character `f` to a topic.
- The robot **subscribes** to that same topic, so the broker forwards `f` to it.

Why this is a good fit here:

- **Designed for constrained devices.** MQTT's whole reason for existing is small, low-power, unreliable-network devices. That is exactly an ESP32.
- **Decoupling.** The laptop does not know or care that a robot exists. It publishes intent. Anything subscribed acts on it.
- **Extensibility.** Want a phone dashboard showing the current command? Subscribe to the same topic. Nothing else changes.
- **Tiny messages.** A one-byte payload means the network delay is essentially the round trip through the broker.

The code also avoids spamming the broker: it keeps a `last_gesture` variable and only publishes when the gesture actually changes. Without this, at 25 FPS you would publish 25 identical messages a second.

> **Likely interview question:** "Why MQTT instead of HTTP or a raw socket?"
> **Answer:** "HTTP has a heavy per-request overhead and is request/response, which is a bad fit for a continuous stream of small commands. A raw TCP socket would work but I would have to write my own reconnection and framing logic. MQTT gives me pub/sub, automatic reconnection, and quality-of-service levels for free, and it is the standard in IoT — the ESP32 has well-supported client libraries."

---

## Step 11: The robot side

The ESP32 firmware, written in C in the Arduino IDE, has two jobs:

**Job 1 — networking.** Connect to Wi-Fi, connect to the MQTT broker, subscribe to the command topic, and handle incoming messages in a callback.

**Job 2 — movement logic and motor control.** A hexapod has six legs with multiple joints each — that is a lot of servos, each driven by a PWM signal whose pulse width sets the angle.

Making a hexapod walk is not "turn on the motors". It is a **gait**: a repeating, carefully ordered sequence of joint angles where some legs lift and swing forward while the others stay planted and push the body along. Get the ordering wrong and the robot falls over. Each command letter selects a different gait — forward, backward, turn left, turn right, or hold still.

**Why this split is the right architecture:** the laptop sends *intent*, not *joint angles*. If the laptop instead streamed servo positions, then every network hiccup would become a physical stumble, and the robot would be useless on its own. By keeping the gait logic on the robot, a dropped message just means the robot continues its current motion until the next command arrives.

---

## Step 12: Performance details

**FPS measurement.** `utils/cvfpscalc.py` measures frame rate using OpenCV's high-resolution tick counter, averaging over a rolling buffer. The buffer length was tuned (a commit is literally titled "updated buffer length in cvfpscalc to increase framerate") — a longer buffer gives a smoother, more stable reading.

**GPU disabled.** `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` forces CPU execution. For a model this small, the cost of moving data to the GPU exceeds the cost of the computation itself, and it sidesteps CUDA/driver issues entirely.

**One hand only.** `max_num_hands=1` halves the per-frame detection work and removes the ambiguity of two hands giving conflicting commands.

**Frame budget.** At 30 FPS you have about 33 ms per frame, shared between camera capture, MediaPipe inference, classification, drawing, and publishing. MediaPipe dominates; the classifier is negligible.

---

## The core message for interviews

> "The interesting part of Aragog is not any one component — it is how the responsibilities are split. MediaPipe does the heavy perception and turns pixels into 42 clean geometric features. My own small neural network turns those features into a discrete decision. MQTT carries that decision as a single byte. And the robot's C firmware owns all the movement logic, so the network only ever carries intent, never low-level control. That separation is what makes the system fast, debuggable, and resilient to network problems."
