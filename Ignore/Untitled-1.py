import time
import csv
import cv2 as cv
import numpy as np
import mediapipe as mp
from pyfirmata import Arduino  # ✅ Use PyFirmata instead of MQTT
from model import KeyPointClassifier

# ✅ Connect to Arduino Due on COM4
board = Arduino('COM4')
pin_12 = board.get_pin('d:12:o')  # Set pin 12 as output

def draw_info_text(image, text):
    """ Draws the detected gesture text on the screen """
    cv.putText(image, f"Gesture: {text}", (50, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    return image

# ✅ Function to preprocess landmarks (same as original `app.py`)
def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    temp_landmark_list = []

    for x, y in landmark_list:
        temp_landmark_list.append(x - base_x)
        temp_landmark_list.append(y - base_y)

    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    keypoint_classifier = KeyPointClassifier()

    # Load gesture labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = image.copy()

        # Process Hand Landmarks
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        detected_gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = [(l.x, l.y) for l in hand_landmarks.landmark]  # Normalized (0-1)

                # ✅ Apply pre-processing (same as original `app.py`)
                processed_landmark_list = pre_process_landmark(landmark_list)

                # ✅ Ensure correct shape and type
                processed_landmark_list = np.array(processed_landmark_list, dtype=np.float32).reshape(1, -1)

                print("🔍 Input Shape Before Passing to Model:", processed_landmark_list.shape)
                print("📌 Expected Model Input Shape:", keypoint_classifier.interpreter.get_input_details()[0]['shape'])
                print("🧪 Data Type of Input:", processed_landmark_list.dtype)

                # ✅ Pass correctly formatted input
                hand_sign_id = keypoint_classifier(processed_landmark_list[0])  # Fixes shape mismatch

                if hand_sign_id < len(keypoint_classifier_labels):
                    detected_gesture = keypoint_classifier_labels[hand_sign_id]

                    # ✅ Directly Control LED
                    if detected_gesture.lower() == "left":
                        print("🟢 Gesture Detected: LEFT → Turning LED ON")
                        pin_12.write(1)  # ✅ Directly turn LED ON
                    elif detected_gesture.lower() == "right":
                        print("🔴 Gesture Detected: RIGHT → Turning LED OFF")
                        pin_12.write(0)  # ✅ Directly turn LED OFF

        # Draw Gesture Info on Screen
        debug_image = draw_info_text(debug_image, detected_gesture)

        cv.imshow("Hand Gesture Recognition", debug_image)
        if cv.waitKey(10) == 27:  # Press ESC to exit
            break

    cap.release()
    cv.destroyAllWindows()
    board.exit()  # ✅ Close connection to Arduino

if __name__ == "__main__":
    main()
