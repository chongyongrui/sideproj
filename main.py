import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import psycopg2
from psycopg2 import Error
from matplotlib import pyplot as plt


create_table_query = """
   
    CREATE TABLE IF NOT EXISTS pose_data (
        Rhip_x FLOAT,
        Rhip_y FLOAT,
        Rshoulder_x FLOAT,
        Rshoulder_y FLOAT,
        Relbow_x FLOAT,
        Relbow_y FLOAT,
        Rwrist_x FLOAT,
        Rwrist_y FLOAT,
        Rknee_x FLOAT,
        Rknee_y FLOAT,
        Rankle_x FLOAT,
        Rankle_y FLOAT,
        Lhip_x FLOAT,
        Lhip_y FLOAT,
        Lshoulder_x FLOAT,
        Lshoulder_y FLOAT,
        Lelbow_x FLOAT,
        Lelbow_y FLOAT,
        Lwrist_x FLOAT,
        Lwrist_y FLOAT,
        Lknee_x FLOAT,
        Lknee_y FLOAT,
        Lankle_x FLOAT,
        Lankle_y FLOAT,
        RshoulderBend FLOAT,
        RelbowBend FLOAT,
        RhipBend FLOAT,
        Rkneebend FLOAT,
        LshoulderBend FLOAT,
        LelbowBend FLOAT,
        LhipBend FLOAT,
        Lkneebend FLOAT, 
        sceneNumber INT
    );
"""


try:
    # Connect to an existing database
    connection = psycopg2.connect(user="sysadmin",
                                  password="D5taCard",
                                  host="localhost",
                                  port="5433",
                                  database="postgres")

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    print("PostgreSQL server information")
    print(connection.get_dsn_parameters(), "\n")
    # Executing a SQL query
    cursor = connection.cursor()

    # Execute the SQL command to create the table
    cursor.execute(create_table_query)


except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)
    
finally:
    if (connection):
        connection.commit()

        mppose = mp.solutions.pose
        mpdraw = mp.solutions.drawing_utils
        pose = mppose.Pose()
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        cap = cv2.VideoCapture(parent_directory + "\\video2.mp4")
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


        frameNum = 1
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
            
            Rhip = np.array([landmarks[mppose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mppose.PoseLandmark.RIGHT_HIP.value].y])
            Rshoulder = np.array([landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mppose.PoseLandmark.RIGHT_SHOULDER.value].y])
            Relbow = np.array([landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mppose.PoseLandmark.RIGHT_ELBOW.value].y])
            Rwrist = np.array([landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mppose.PoseLandmark.RIGHT_WRIST.value].y])
            Rknee = np.array([landmarks[mppose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mppose.PoseLandmark.RIGHT_KNEE.value].y])
            Rankle = np.array([landmarks[mppose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mppose.PoseLandmark.RIGHT_ANKLE.value].y])
            
            Lhip = np.array([landmarks[mppose.PoseLandmark.LEFT_HIP.value].x, landmarks[mppose.PoseLandmark.LEFT_HIP.value].y])
            Lshoulder = np.array([landmarks[mppose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mppose.PoseLandmark.LEFT_SHOULDER.value].y])
            Lelbow = np.array([landmarks[mppose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mppose.PoseLandmark.LEFT_ELBOW.value].y])
            Lwrist = np.array([landmarks[mppose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mppose.PoseLandmark.LEFT_WRIST.value].y])
            Lknee = np.array([landmarks[mppose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mppose.PoseLandmark.LEFT_KNEE.value].y])
            Lankle = np.array([landmarks[mppose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mppose.PoseLandmark.LEFT_ANKLE.value].y])
            
            RshoulderBend = calculate_angle(Rhip,Rshoulder,Relbow)
            RelbowBend = calculate_angle(Rshoulder,Relbow,Rwrist)
            RhipBend = calculate_angle(Rshoulder,Rhip,Rknee)
            Rkneebend= calculate_angle(Rhip,Rknee,Rankle)
            
            LshoulderBend = calculate_angle(Lhip,Lshoulder,Lelbow)
            LelbowBend = calculate_angle(Lshoulder,Lelbow,Lwrist)
            LhipBend = calculate_angle(Lshoulder,Lhip,Lknee)
            Lkneebend= calculate_angle(Lhip,Lknee,Lankle)

            cv2.putText(img, str(RshoulderBend), tuple(np.multiply(Rshoulder, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img, str(RelbowBend), tuple(np.multiply(Relbow, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img, str(RhipBend), tuple(np.multiply(Rhip, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img, str(Rkneebend), tuple(np.multiply(Rknee, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            pose_data = {
                "Rhip_x": Rhip[0],
                "Rhip_y": Rhip[1],
                "Rshoulder_x": Rshoulder[0],
                "Rshoulder_y": Rshoulder[1],
                "Relbow_x": Relbow[0],
                "Relbow_y": Relbow[1],
                "Rwrist_x": Rwrist[0],
                "Rwrist_y": Rwrist[1],
                "Rknee_x": Rknee[0],
                "Rknee_y": Rknee[1],
                "Rankle_x": Rankle[0],
                "Rankle_y": Rankle[1],
                "Lhip_x": Lhip[0],
                "Lhip_y": Lhip[1],
                "Lshoulder_x": Lshoulder[0],
                "Lshoulder_y": Lshoulder[1],
                "Lelbow_x": Lelbow[0],
                "Lelbow_y": Lelbow[1],
                "Lwrist_x": Lwrist[0],
                "Lwrist_y": Lwrist[1],
                "Lknee_x": Lknee[0],
                "Lknee_y": Lknee[1],
                "Lankle_x": Lankle[0],
                "Lankle_y": Lankle[1],
                "RshoulderBend": RshoulderBend,
                "RelbowBend": RelbowBend,
                "RhipBend": RhipBend,
                "Rkneebend": Rkneebend,
                "LshoulderBend": LshoulderBend,
                "LelbowBend": LelbowBend,
                "LhipBend": LhipBend,
                "Lkneebend": Lkneebend, 
                "sceneNumber": frameNum
            }


            # SQL query for inserting data into the dancemoves table
            insert_query = """
                INSERT INTO pose_data (
                    Rhip_x, Rhip_y, Rshoulder_x, Rshoulder_y, Relbow_x, Relbow_y, Rwrist_x, Rwrist_y,
                    Rknee_x, Rknee_y, Rankle_x, Rankle_y, Lhip_x, Lhip_y, Lshoulder_x, Lshoulder_y,
                    Lelbow_x, Lelbow_y, Lwrist_x, Lwrist_y, Lknee_x, Lknee_y, Lankle_x, Lankle_y,
                    RshoulderBend, RelbowBend, RhipBend, Rkneebend,
                    LshoulderBend, LelbowBend, LhipBend, Lkneebend, sceneNumber
                ) VALUES (
                    %(Rhip_x)s, %(Rhip_y)s, %(Rshoulder_x)s, %(Rshoulder_y)s, %(Relbow_x)s, %(Relbow_y)s, %(Rwrist_x)s, %(Rwrist_y)s,
                    %(Rknee_x)s, %(Rknee_y)s, %(Rankle_x)s, %(Rankle_y)s, %(Lhip_x)s, %(Lhip_y)s, %(Lshoulder_x)s, %(Lshoulder_y)s,
                    %(Lelbow_x)s, %(Lelbow_y)s, %(Lwrist_x)s, %(Lwrist_y)s, %(Lknee_x)s, %(Lknee_y)s, %(Lankle_x)s, %(Lankle_y)s,
                    %(RshoulderBend)s, %(RelbowBend)s, %(RhipBend)s, %(Rkneebend)s,
                    %(LshoulderBend)s, %(LelbowBend)s, %(LhipBend)s, %(Lkneebend)s, %(sceneNumber)s
                );
            """

# Execute the SQL query and pass the data_to_insert list as parameters
            cursor.execute(insert_query, pose_data)

            # Commit the changes
            connection.commit()
            #print(result.pose_landmarks)
            cv2.imshow("Pose Estimation" , img)
            cv2.waitKey(1)
            frameNum = frameNum +1
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")