import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# calculate Angle
def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    radius = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radius * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# video
cap = cv2.VideoCapture('5.mp4')

# curl counter variables
counter = 0
stage = None

# setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # recolor image to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # make detection
        results = pose.process(image)

        # recolor back to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates
            shoulderLeft = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbowLeft = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wristLeft = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]
            shoulderRight = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            elbowRight = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            wristRight = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]
            # calculate angle
            angleLeft = calculate_angle(shoulderLeft, elbowLeft, wristLeft)
            angleRight = calculate_angle(shoulderRight, elbowRight, wristRight)

            #cul counter logic
            if angleLeft>160:
                stage='down'
            if angleLeft<30 and stage=='down':
                stage='up'
                counter+=1
                print(counter)
        except:
            pass

        #render curl counter
        #setup status box
        cv2.rectangle(image,(0,0),(250,73),(245,117,16),-1)

        #rep data
        cv2.putText(image,'REPS',(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,'Stage',(75,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(90,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

        #render detections
        mp_drawing.draw_landmarks(
            image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2),
        )

        cv2.imshow('Mediapipe feed',image)
        if cv2.waitKey(10)&0xff==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
