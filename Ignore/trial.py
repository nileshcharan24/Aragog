from pyfirmata import Arduino
import time
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Connect to Arduino Due (Check the COM port in Arduino IDE)
board = Arduino('COM4')  # Change 'COM4' to the correct port
led_pin = board.get_pin('d:12:o')  # Using pin 12 for LED

# Define gestures for turning LED ON and OFF
GESTURE_ON = "Spider"  # Modify this based on your trained gestures
GESTURE_OFF = "Stop"   # Modify this based on your trained gestures

def control_led(action):
    """Function to turn LED ON/OFF based on gesture recognition."""
    if action == "ON":
        led_pin.write(1)  # Turn LED ON
        print("LED Turned ON")
    elif action == "OFF":
        led_pin.write(0)  # Turn LED OFF
        print("LED Turned OFF")

def main():
    cap = cv.VideoCapture(0)  # Open camera

    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    keypoint_classifier = KeyPointClassifier()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        # Convert frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get keypoints and classify gesture
                landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                gesture_id = keypoint_classifier(landmark_list)

                if gesture_id is not None:
                    detected_gesture = keypoint_classifier_labels[gesture_id]
                    print(f"Detected Gesture: {detected_gesture}")

                    # Control LED based on detected gesture
                    if detected_gesture == GESTURE_ON:
                        control_led("ON")
                    elif detected_gesture == GESTURE_OFF:
                        control_led("OFF")

        # Display the frame
        cv.imshow("Hand Gesture Recognition", debug_image)
        if cv.waitKey(10) == 27:  # Exit on ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
