import pickle
import math
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
from dollarpy import Recognizer, Template, Point

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles

mp_holistic = mp.solutions.holistic # Mediapipe Solutions


def getPoints(videoURL,label):
    cap = cv2.VideoCapture(videoURL)#web cam =0 , else enter filename
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        points = []
        left_shoulder=[]#
        right_shoulder=[] #
        left_elbos=[]#
        right_elbos=[]#
        nose=[]#
        left_wirst=[]#
        right_wrist=[]#
        left_hip=[]#
        right_hip=[]#     
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            if ret==True:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)


                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Drawing on Frame (You can remove it)
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Export coordinates

                try:

                    # add points of wrist , elbow and shoulder
                    left_shoulder.append(Point(results.pose_landmarks.landmark[11].x,results.pose_landmarks.landmark[11].y,1))
                    right_shoulder.append(Point(results.pose_landmarks.landmark[12].x,results.pose_landmarks.landmark[12].y,2))
                    left_elbos.append(Point(results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,3))
                    right_elbos.append(Point(results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,4))
                    left_wirst.append(Point(results.pose_landmarks.landmark[15].x,results.pose_landmarks.landmark[15].y,5))
                    right_wrist.append(Point(results.pose_landmarks.landmark[16].x,results.pose_landmarks.landmark[16].y,6))
                    left_hip.append(Point(results.pose_landmarks.landmark[23].x,results.pose_landmarks.landmark[23].y,7))
                    right_hip.append(Point(results.pose_landmarks.landmark[24].x,results.pose_landmarks.landmark[24].y,8))
                    nose.append(Point(results.pose_landmarks.landmark[0].x,results.pose_landmarks.landmark[0].y,9))

                except:
                    pass

                cv2.imshow(label, image)
            else :
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

    cap.release()
    cv2.destroyAllWindows()
    points = left_shoulder+right_shoulder+left_elbos+right_elbos+left_wirst+right_wrist+left_hip+right_hip+nose
    print(label)
    return points

#  TRAIN  #

templates=[] #list of templates for $1 training

#CatCow Correct
vid = "../dataset/CatCow Correct 1.mp4"
points = getPoints(vid,"CatCow Correct")
tmpl = Template('CatCow Correct', points)
templates.append(tmpl)

#CatCow Wrong
vid = "../dataset/CatCow Wrong 1.mp4"
points = getPoints(vid,"CatCow Wrong")
tmpl = Template('CatCow Wrong', points)
templates.append(tmpl)

#FlagPole
vid = "../dataset/Kyphosis-All Seated ex1-v1.mp4"
points = getPoints(vid,"FlagPole Correct")
tmpl = Template('FlagPole Correct', points)
templates.append(tmpl)

#CrissCross
vid = "../dataset/Kyphosis-All Seated ex2-v1.mp4"
points = getPoints(vid,"CrissCross Correct")
tmpl = Template('CrissCross Correct', points)
templates.append(tmpl)

#  TEST  #
vid = "../dataset/CatCow-Wrong-4.mp4" # use "/videos/"" + username
points = getPoints(vid,"Unknown") # keep as unknown

# Load the recognizer model from the file
#with open('recognizer_model.pkl', 'rb') as file:
#    recognizer = pickle.load(file)

import time
start = time.time()
recognizer = Recognizer(templates)
result = recognizer.recognize(points)
end = time.time()
print(result[0])
print("time taken to classify:"+ str(end-start))

# Save the recognizer model to a file
#with open('recognizer_model.pkl', 'wb') as f:
#    pickle.dump(recognizer, f)

score = math.floor(result[1] * 1000)

print(score)