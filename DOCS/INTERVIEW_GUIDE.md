# Aragog - Interview Guide

## The 30-second pitch

> "Aragog is a six-legged walking robot controlled entirely by hand gestures. I built the robot around an ESP32 running C firmware that handles the servo control and walking gaits. Separately, I wrote a Python computer-vision module that uses OpenCV and MediaPipe to extract hand landmarks from a webcam, normalises them, and classifies them with a small neural network I trained on data I recorded myself. The recognised gesture is published as a single character over MQTT, and the robot subscribes to it. The interesting design decision was keeping the movement logic on the robot, so the network only carries intent — that made the whole system fast and resilient to dropped messages."

---

## The 10-second version

> "A six-legged robot you drive with hand gestures. OpenCV and MediaPipe on a laptop recognise the gesture, MQTT carries it over Wi-Fi, and an ESP32 running C translates it into a walking gait."

---

## Questions you will almost certainly get

### "Walk me through how a gesture becomes robot movement."

Webcam frame → MediaPipe extracts 21 hand landmarks → normalise them so position and scale don't matter → a small neural network classifies them into a gesture → majority vote over the last 16 frames to remove flicker → map to a single character → publish over MQTT → the ESP32 receives it and runs the matching walking gait. End to end it is a fraction of a second.

### "Why MediaPipe instead of training on raw images?"

MediaPipe reduces 1.5 million pixels to 42 meaningful numbers, and it does it with a model that Google has already trained on far more data than I could ever collect. That meant my classifier only needed to learn hand geometry, not "what is a hand" — so it could be tiny and train on a few hundred samples per class instead of tens of thousands of images. It also made the system robust to lighting and background changes for free.

### "Why did you normalise the landmarks?"

To make the features invariant to where the hand is and how big it appears. I subtract the wrist position from every landmark, which removes translation, then divide by the largest absolute value, which removes scale. Without that, the model would learn hand *position* rather than hand *shape*, and it would fail the moment you moved across the frame or leaned back.

### "Why MQTT and not HTTP or a socket?"

HTTP is request/response with heavy per-request overhead — a bad fit for a continuous stream of tiny commands. A raw TCP socket would work but I'd have to write my own reconnection and message framing. MQTT is publish/subscribe, designed specifically for constrained IoT devices, gives me reconnection and quality-of-service for free, and has well-supported ESP32 client libraries. It also decouples the two sides: the laptop publishes intent and doesn't know a robot exists.

### "Why send just one character?"

The ESP32 has limited RAM and tight timing constraints, so I didn't want it parsing JSON. One byte is unambiguous, can't be partially received, and costs nothing to handle. It also made debugging trivial — I could test the robot by publishing a letter from an MQTT client on my phone, with the whole vision system switched off.

### "What happens if the network drops?"

The robot keeps executing its last commanded gait until a new message arrives, because the gait logic lives on the robot rather than being streamed from the laptop. That's a deliberate trade-off: a dropped message causes continuation, not a stumble. On the vision side, if no hand is detected the command defaults to stop, so walking out of frame stops the robot rather than leaving it running.

### "How did you get your training data?"

There was no dataset for gestures like "Front" or "Back", so I built a logging mode into the program: press a key to enter logging mode, hold a gesture, press a number, and one row of normalised landmarks gets appended to a CSV with that number as the label. I recorded a couple of hundred samples per gesture, varying position, distance, and angle. About 1,800 samples total.

### "How accurate was it?"

The training notebook produces a confusion matrix and classification report on a held-out 25% test split. In practice the more useful measure was live behaviour — whether it held a stable prediction while I moved my hand around naturally — which is why I added the 16-frame majority vote. A model can look great on a test set and still flicker unusably in real time.

### "What was the hardest part?"

Two things. On the firmware side, getting a stable walking gait — a hexapod needs its legs lifted and planted in a specific order, and getting the sequencing or timing wrong makes it fall over rather than walk. On the vision side, the jump from "the model classifies correctly" to "the system feels responsive and doesn't twitch". Solving that meant normalisation, temporal smoothing, and publish-on-change rather than anything to do with the model itself.

### "What would you do differently?"

Four things, in order of importance:

1. **Move the credentials out of the source.** Earlier versions have a hardcoded broker address, username, and password. Those belong in environment variables or a git-ignored config file.
2. **Balance the dataset.** My class counts weren't even — one class had noticeably more samples than the others. I'd collect more for the smaller classes or apply class weights.
3. **Add a heartbeat and a watchdog.** Right now the robot can't tell "no new command" from "the laptop died". A periodic heartbeat with a timeout that falls back to stop would close that gap.
4. **Clean up the repository.** Several near-duplicate iteration files (`app.py`, `app_new.py`, `Untitled-2.py`) should be branches or deleted, not files.

### "You used a pre-trained model — what did you actually build?"

I built the robot itself and its firmware, the gesture dataset, the classifier, the smoothing logic, and the communication design. MediaPipe replaced one component — hand landmark detection — which is a solved problem that Google has spent enormous resources on. Choosing to use it was itself an engineering decision: it let me spend my time on the parts that were actually novel to this project. Reimplementing hand tracking would have been a worse use of the time and produced a worse result.

---

## Concepts to be able to define in one line

| Term | One-line definition |
|---|---|
| **Landmark** | A specific labelled point on the hand, like a fingertip or knuckle — 21 of them per hand |
| **Normalisation** | Rescaling features so irrelevant variation (position, size) is removed |
| **ReLU** | An activation that outputs zero for negatives — the cheap non-linearity that lets networks learn curves |
| **Softmax** | Converts raw output scores into probabilities that sum to 1 |
| **Dropout** | Randomly disabling neurons during training to prevent over-reliance on any one of them |
| **Overfitting** | Memorising the training data instead of learning the pattern; shows up as good train and bad test scores |
| **Early stopping** | Halting training once validation performance stops improving |
| **Quantisation** | Storing weights at lower precision (int8 instead of float32) to shrink and speed up a model |
| **TensorFlow Lite** | A runtime for running small optimised models efficiently on CPU and edge devices |
| **MQTT** | A lightweight publish/subscribe messaging protocol built for IoT devices |
| **Broker** | The MQTT server that routes messages from publishers to subscribers |
| **PWM** | Pulse-width modulation — the timed pulse signal that tells a servo what angle to hold |
| **Gait** | The repeating, ordered pattern of leg movements that produces walking |

---

## The one strong closing answer

> "The thing I'm most pleased with in Aragog isn't the model — it's the interface between the two halves. I had a laptop doing heavy perception and a microcontroller doing hard real-time motor control, and they had completely different constraints. Deciding that the network would carry a single character of *intent*, and that all the movement logic would live on the robot, is what made the system work. It meant the robot stayed functional and coherent even when the network wasn't, and it meant I could debug either half completely independently of the other."
