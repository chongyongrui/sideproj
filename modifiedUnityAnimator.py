import cv2
import mediapipe as mp
import numpy as np
import socket
import struct
import os
import psycopg2
from psycopg2 import Error


mppose = mp.solutions.pose
mpdraw = mp.solutions.drawing_utils
pose = mppose.Pose()

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
cap = cv2.VideoCapture(parent_directory + "\\video1.mp4")
alpha = 0.5

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.resize(img, (400, 800))
    overlay = np.zeros_like(img, dtype=np.uint8)
    cv2.addWeighted(img, alpha, overlay, 1 - alpha, 2, img)

    result = pose.process(img)
    landmarks = result.pose_landmarks.landmark

    # Extract relevant landmarks (modify based on your requirements)
    shoulder = [landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].y]
    
    # Calculate the angle
    elbowBend = calculate_angle(shoulder, elbow, wrist)

    # Display the angle on the image
    cv2.putText(img, str(elbowBend), tuple(np.multiply(elbow, [800, 400]).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Send the landmarks data to Unity over the socket connection

    cv2.imshow("Pose Estimation", img)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close the socket connection

cv2.destroyAllWindows()
