# Aragog - Code Walkthrough

A file-by-file tour of the repository, so you can answer "walk me through your code" without opening it.

---

## `Gesture.py` — the main program

This is the file to run and the file to talk about. Roughly 630 lines. Its structure:

### `get_args()`
Command-line arguments: camera device number, capture width and height, static-image mode, and the two MediaPipe confidence thresholds. Defaults are 960×540 and a detection confidence of 0.7.

### `main()` — setup phase
- Opens the camera with OpenCV and sets the resolution.
- Creates the MediaPipe `Hands` object with `max_num_hands=1`.
- Loads `KeyPointClassifier` and `PointHistoryClassifier` (both TFLite).
- Reads the two label CSV files so numeric class IDs can be turned back into names. Note `encoding='utf-8-sig'` — this strips the byte-order mark that Excel adds when saving CSVs, a small detail that otherwise makes the first label mysteriously fail to match.
- Creates `CvFpsCalc` and two `deque(maxlen=16)` buffers: one for fingertip positions, one for recent gesture predictions.

### `main()` — the per-frame loop
1. Read a key press. `ESC` exits; `k` / `h` / `n` switch logging modes; `0`–`9` set the class label when logging.
2. Capture a frame; flip it horizontally so it behaves like a mirror.
3. Deep-copy the frame into `debug_image` — the copy is what gets drawn on, so the original stays clean for processing.
4. Convert BGR → RGB, set `writeable = False`, run `hands.process()`, set `writeable = True`.
5. **Initialise `current_sign = 'e'`.** The fail-safe default.
6. If landmarks were found: compute the bounding box, extract landmarks, normalise, optionally log to CSV, classify, update history buffers, majority-vote, map to a command letter, and draw the overlay.
7. If no landmarks: append `[0, 0]` to the point history and set `current_sign = 'e'`.
8. Publish over MQTT (only when the command has changed).
9. `cv.imshow()` the debug image.

### Geometry helpers
- `calc_bounding_rect()` — builds an array of landmark pixel positions and calls `cv.boundingRect()` for the box drawn around the hand.
- `calc_landmark_list()` — converts MediaPipe's 0-to-1 fractions into pixel coordinates, clamped to stay inside the image.
- `pre_process_landmark()` — the three-step normalisation described in TECHNICAL_EXPLANATION.md: relative to wrist, flatten, divide by max absolute value.
- `pre_process_point_history()` — the same idea for the fingertip trail, but divided by image width and height rather than by a max value.

### `logging_csv()`
When in logging mode, appends one row to the appropriate CSV: the pressed number as the class label, then the normalised feature values. This is the entire dataset-collection mechanism.

### `give_names()`
Maps a gesture label string to its command character.

### Drawing helpers
`draw_landmarks()`, `draw_bounding_rect()`, `draw_info_text()`, `draw_point_history()`, `draw_info()`. These are purely for the on-screen debug view. `draw_info()` uses the classic trick of drawing text twice — once thick in one colour, once thin in another — to create an outline that stays readable over any background.

---

## `model/keypoint_classifier/`

| File | Purpose |
|---|---|
| `keypoint.csv` | The dataset you recorded. ~1,800 rows; column 1 is the class ID, columns 2–43 are the 42 normalised values. |
| `keypoint_classifier_label.csv` | Human-readable gesture names, one per line, in class-ID order. |
| `keypoint_classifier.keras` / `.hdf5` | The trained Keras model. |
| `keypoint_classifier.tflite` | The quantised model that actually runs live. |
| `keypoint_classifier.py` | A thin wrapper class: loads the TFLite interpreter, allocates tensors, and exposes `__call__` so the classifier can be used like a function — `hand_sign_id = keypoint_classifier(landmarks)`. Returns `argmax` of the output. |

---

## `model/point_history_classifier/`

Same structure, for the motion classifier. One meaningful difference: its wrapper has a **confidence threshold** (`score_th=0.5`). If the top class scores below 0.5, it returns class 0 instead. This is a deliberate "I am not sure, so do nothing" behaviour — safer than committing to a low-confidence guess about motion.

---

## `utils/cvfpscalc.py`

A small FPS counter. Uses `cv.getTickCount()` and `cv.getTickFrequency()` for high-resolution timing, stores frame durations in a rolling `deque`, and reports the average. The buffer length was tuned during development to stabilise the reading.

---

## The notebooks

**`keypoint_classification_EN.ipynb`** — loads `keypoint.csv`, splits 75/25 with a fixed seed, defines the Dense 20 → Dense 10 → Softmax model, trains with Adam and early stopping, prints a confusion matrix and classification report, then converts to a quantised `.tflite` file and verifies the converted model gives the same prediction.

**`point_history_classification.ipynb`** — the same flow for the motion model, with a `use_lstm` flag that switches between an LSTM version and a plain dense version. The dense version was used.

---

## `communication.py`

The Bluetooth experiment. Connects to a board on `COM4` via pyFirmata, opens a fullscreen Tkinter window, and gives you a button that turns pin 12 on. The window title is "HC-05 Connection" — the HC-05 is a Bluetooth serial module.

This is a stepping stone, not part of the final system, but it is worth keeping because it documents the wireless-communication step between USB and MQTT.

---

## `Ignore/`

Scratch work from the earliest phase:

- **`led_trial.py`** — the hello-world: blink an LED on pin 12 over USB. Proves the board connection works.
- **`trial.py`** — first integration: MediaPipe gesture recognition drives that same LED. Gesture "Spider" turns it on, "Stop" turns it off. This is the moment the vision-to-hardware pipeline first worked end to end.
- **`Untitled-1.py`** — a cleaner rewrite of the same idea with its own minimal preprocessing.

These files are why you can tell a *story* about the project rather than just describe its final state.

---

## `app.py`, `app_new.py`, `Untitled-2.py`

Earlier iterations of `Gesture.py`, kept around during development. They differ mainly in MQTT configuration and in whether the publishing code is active or commented out. `Gesture.py` is the current version.

**Worth knowing:** these older files contain a hardcoded broker address, username, and password. Before this repository goes anywhere public, those should be moved to environment variables or a `.env` file that is git-ignored, and the old credentials rotated. If an interviewer asks "what would you improve?", this is a strong, honest answer — it shows you know what secret management is and that you can audit your own code.

---

## Things worth pointing out unprompted

These are the details that separate "I followed a tutorial" from "I understand this system":

- **BGR → RGB conversion.** OpenCV and MediaPipe disagree on channel order. Get it wrong and nothing crashes — accuracy just quietly degrades.
- **`writeable = False`.** Lets MediaPipe skip a defensive array copy. Free performance.
- **The deep copy for `debug_image`.** Keeps the processing image pristine while drawing on a separate one.
- **`utf-8-sig` when reading labels.** Handles the BOM that Excel writes into CSVs.
- **Fail-safe default of `e`.** The command variable is initialised to stop and reset to stop, not merely left unchanged.
- **Publish-on-change.** Avoids flooding the broker with dozens of identical messages per second.
- **The confidence threshold on the motion classifier.** An explicit "not sure, do nothing" path.
