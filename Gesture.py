#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

# Option 2: Absolute import (recommended)
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from paho.mqtt import client as mqtt_client

import random
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Explicitly disable GPU


# broker = '192.168.203.7'
# port = 1883
# topic = 'nilesh'
# # Generate a Client ID with the subscribe prefix.
# client_id = f'subscribe-{random.randint(0, 100)}'
# # client_id = f'mqttx_7dd103e7'
# username = 's'
# password = '12345678'


# def connect_mqtt():
#     def on_connect(client, userdata, flags, rc):
#         if rc == 0:
#             print("Connected to MQTT Broker!")
#         else:
#             print("Failed to connect, return code %d\n", rc)

#     client = mqtt_client.Client(client_id)
#     client.username_pw_set(username, password)
#     client.on_connect = on_connect
#     client.connect(broker, port)
#     return client


# def publish(client, control):
#     msg_count = 1
#     while True:
#         time.sleep(1)
#         msg = control
#         result = client.publish(topic, msg)
#         # result: [0, 1]
#         status = result[0]
#         if status == 0:
#             print(f"Send `{msg}` to topic `{topic}`")
#         else:
#             print(f"Failed to send message to topic {topic}")
#         msg_count += 1
#         if msg_count > 1:
#             break


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) and exit window
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation - for detection only
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #cant change the frame
        image.flags.writeable = False
        #detect hands
        results = hands.process(image)
        #allow changing the frame
        image.flags.writeable = True

        current_sign = 'e'  # Default to stop ('e') when no hand is detected

        #if hand has landmarks
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                        pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not Applicable":  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Get the current gesture sign
                _, current_sign = give_names(
                    handedness.classification[0].label[0:],
                    keypoint_classifier_labels[hand_sign_id]
                )

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])
            current_sign = 'e'  # No hand detected - send stop command

        # MQTT Publishing (only initialize client once outside the loop)
        # if 'client' not in locals():
        #     client = connect_mqtt()
        #     client.loop_start()
        
        # publish(client, current_sign)  # Pass the current_sign as control

        # debug_image = draw_point_history(debug_image, point_history)
        # debug_image = draw_info(debug_image, fps, mode, number)
        
        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Cleanup
    if 'client' in locals():
        client.loop_stop()
    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0,255,0), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0,255,0), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0,255,0), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0,255,0), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0,255,0), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255,0,0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0,255,0), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 1: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255,0,0), 1)
        if index == 5: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 6: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 7: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255,0,0), 1)
        if index == 9: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 11: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255,0,0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 14: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255,0,0), 1)
        if index == 17: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255,0,0), 1)
        if index == 20: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,255,0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255,0,0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255,0,0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (255,0,0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        
        #info_text = info_text + ':' + hand_sign_text
        info_text, sign = give_names(info_text, hand_sign_text)
        print(sign)
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2,
                   cv.LINE_AA)

    return image

def give_names(info_text, hand_sign_text):
    info_text = hand_sign_text

    if(hand_sign_text == 'Left'):
        sign = 'l'
    elif(hand_sign_text == 'Right'):
        sign = 'r'
    elif(hand_sign_text == 'Front'):
        sign = 'f'
    elif(hand_sign_text == 'Back'):
        sign = 'b'
    elif(hand_sign_text == 'Stop'):
        sign = 'e'  
    elif(hand_sign_text == 'Banned From Spider'):
        sign = 'f'

    return info_text, sign


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255,0,0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0,255,0), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
