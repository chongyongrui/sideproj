import cv2
import mediapipe as mp
import numpy as np

mppose = mp.solutions.pose
mpdraw = mp.solutions.drawing_utils
pose = mppose.Pose()
cap = cv2.VideoCapture("C:\\Users\\userAdmin\\Videos\\video1.mp4")
## for webcame
#cap = cv2.VideoCapture(0)
alpha = 0.5

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    
    return angle



while cap.isOpened():
    ret, img = cap.read()
    img = cv2.resize(img, (400, 800))
    overlay = np.zeros_like(img, dtype=np.uint8)

    cv2.addWeighted(img, alpha, overlay, 1 - alpha, 2, img)

    result = pose.process(img)
    mpdraw.draw_landmarks(img, result.pose_landmarks, mppose.POSE_CONNECTIONS)
    
    h,w,c = img.shape
    opImg = np.zeros([h,w,c])
    opImg.fill(255)
    mpdraw.draw_landmarks(opImg, result.pose_landmarks, mppose.POSE_CONNECTIONS)
    cv2.imshow("Extracted Pose", opImg)
    
    
    
    landmarks = result.pose_landmarks.landmark
    shoulder = np.array([landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].y])
    elbow = np.array([landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].y])
    wrist = np.array([landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].y])
    elbowBend = calculate_angle(shoulder,elbow,wrist)

    cv2.putText(img, str(elbowBend), tuple(np.multiply(elbow, [w, h]).astype(int)),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
    
    
    print(result.pose_landmarks)
    cv2.imshow("Pose Estimation" , img)
    cv2.waitKey(1)