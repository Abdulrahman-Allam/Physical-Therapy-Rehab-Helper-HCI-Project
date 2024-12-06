import os
import cv2
import mediapipe as mp
# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
framecnt=0
from dollarpy import Recognizer, Template, Point
cir = Template('cir', [
Point(219, 264, 1),
Point(206, 269, 1),
Point(218, 263, 1),
Point(210, 270, 1),
Point(217, 263, 1),
Point(218, 269, 1),
Point(217, 263, 1),
Point(222, 268, 1),
Point(216, 262, 1),
Point(246, 260, 1),
Point(216, 254, 1),
Point(204, 296, 1),
Point(218, 233, 1),
Point(158, 322, 1),
Point(211, 183, 1),
Point(152, 327, 1),
Point(145, 201, 1),
Point(152, 326, 1),
Point(128, 190, 1),
Point(161, 323, 1),
Point(147, 161, 1),
Point(156, 324, 1),
Point(134, 167, 1),
Point(158, 323, 1),
Point(105, 172, 1),
Point(152, 324, 1),
Point(93, 215, 1),
Point(144, 326, 1),
Point(118, 232, 1),
Point(145, 329, 1),
Point(148, 236, 1),
Point(150, 326, 1),
Point(180, 258, 1),
Point(150, 318, 1),
Point(191, 266, 1),
Point(151, 314, 1),
Point(209, 267, 1),
Point(150, 305, 1),
Point(223, 258, 1),
Point(175, 293, 1),
Point(228, 232, 1),
Point(177, 306, 1),
Point(228, 181, 1),
Point(152, 318, 1),
Point(235, 160, 1),
Point(199, 327, 1),
Point(143, 185, 1),
Point(158, 326, 1),
Point(153, 154, 1),
Point(154, 326, 1),
Point(164, 156, 1),
Point(153, 325, 1),
Point(163, 168, 1),
Point(151, 325, 1),
Point(149, 189, 1),
Point(144, 325, 1),
Point(151, 195, 1),
Point(142, 326, 1),
Point(154, 196, 1),
Point(141, 326, 1),
Point(155, 196, 1),
Point(142, 326, 1),
Point(153, 198, 1),
Point(143, 326, 1),
Point(150, 199, 1),
Point(143, 327, 1),
Point(147, 199, 1),
Point(143, 327, 1),
Point(142, 196, 1),
Point(143, 326, 1),
Point(139, 193, 1),
Point(144, 326, 1),
Point(140, 191, 1),
Point(144, 326, 1),
Point(139, 190, 1),
Point(144, 326, 1),
Point(139, 190, 1),
Point(143, 326, 1),
Point(139, 189, 1),
Point(143, 326, 1),
])
down = Template('down', [
Point(29, 323, 1),
Point(310, 380, 1),
Point(60, 309, 1),
Point(313, 384, 1),
Point(85, 282, 1),
Point(311, 389, 1),
Point(91, 265, 1),
Point(310, 390, 1),
Point(106, 261, 1),
Point(310, 391, 1),
Point(120, 286, 1),
Point(311, 389, 1),
Point(118, 269, 1),
Point(317, 394, 1),
Point(118, 266, 1),
Point(316, 391, 1),
Point(116, 273, 1),
Point(316, 393, 1),
Point(115, 278, 1),
Point(313, 392, 1),
Point(114, 277, 1),
Point(311, 394, 1),
Point(112, 281, 1),
Point(311, 395, 1),
Point(110, 279, 1),
Point(312, 397, 1),
Point(111, 278, 1),
Point(311, 398, 1),
Point(115, 258, 1),
Point(311, 397, 1),
Point(132, 270, 1),
Point(310, 396, 1),
Point(147, 283, 1),
Point(310, 388, 1),
Point(173, 293, 1),
Point(310, 375, 1),
Point(210, 291, 1),
Point(309, 374, 1),
Point(249, 289, 1),
Point(285, 355, 1),
Point(266, 288, 1),
Point(198, 305, 1),
Point(282, 288, 1),
Point(230, 298, 1),
Point(301, 287, 1),
Point(239, 258, 1),
Point(314, 287, 1),
Point(300, 273, 1),
Point(323, 289, 1),
Point(306, 271, 1),
Point(325, 289, 1),
Point(324, 237, 1),
Point(325, 288, 1),
Point(322, 269, 1),
Point(322, 289, 1),
Point(314, 280, 1),
Point(319, 285, 1),
Point(329, 273, 1),
Point(309, 284, 1),
Point(340, 229, 1),
Point(280, 288, 1),
Point(224, 283, 1),
Point(244, 301, 1),
Point(233, 295, 1),
Point(224, 298, 1),
Point(281, 350, 1),
Point(54, 318, 1),
Point(313, 372, 1),
Point(11, 328, 1),
Point(311, 373, 1),
Point(2, 339, 1),
Point(318, 385, 1),
Point(5, 342, 1),
Point(318, 386, 1),
Point(5, 346, 1),
Point(321, 388, 1),
Point(6, 346, 1),
Point(322, 396, 1),
])
up = Template('up', [
Point(-2, 352, 1),
Point(144, 107, 1),
Point(-3, 360, 1),
Point(135, 99, 1),
Point(-3, 366, 1),
Point(138, 100, 1),
Point(-5, 379, 1),
Point(136, 98, 1),
Point(-2, 378, 1),
Point(135, 99, 1),
Point(-5, 383, 1),
Point(136, 98, 1),
Point(-2, 380, 1),
Point(141, 94, 1),
Point(-5, 381, 1),
Point(155, 95, 1),
Point(-7, 384, 1),
Point(174, 97, 1),
Point(0, 383, 1),
Point(232, 101, 1),
Point(3, 386, 1),
Point(256, 96, 1),
Point(5, 385, 1),
Point(287, 102, 1),
Point(3, 384, 1),
Point(312, 96, 1),
Point(1, 384, 1),
Point(336, 99, 1),
Point(0, 379, 1),
Point(347, 98, 1),
Point(1, 379, 1),
Point(362, 101, 1),
Point(1, 377, 1),
Point(367, 104, 1),
Point(2, 377, 1),
Point(366, 105, 1),
Point(2, 375, 1),
Point(364, 104, 1),
Point(3, 374, 1),
Point(359, 103, 1),
Point(1, 374, 1),
Point(349, 103, 1),
Point(-1, 372, 1),
Point(320, 102, 1),
Point(-3, 372, 1),
Point(281, 100, 1),
Point(-3, 373, 1),
Point(246, 100, 1),
Point(-5, 378, 1),
Point(210, 99, 1),
Point(-4, 380, 1),
Point(171, 102, 1),
Point(-5, 384, 1),
Point(147, 110, 1),
Point(-6, 384, 1),
Point(132, 117, 1),
Point(-5, 385, 1),
Point(123, 119, 1),
Point(-7, 384, 1),
Point(116, 121, 1),
Point(-7, 386, 1),
Point(118, 122, 1),
Point(-7, 388, 1),
Point(121, 123, 1),
Point(-5, 388, 1),
Point(127, 122, 1),
Point(-3, 388, 1),
Point(130, 122, 1),
Point(0, 388, 1),
Point(133, 121, 1),
])
recognizer = Recognizer([cir, down, up])

Allpoints=[]


while cap.isOpened():
    # read frame from capture object
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.resize(frame, (480, 320))
    framecnt+=1
    try:
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print (framecnt)
        # process the RGB frame to get the result
        results = pose.process(RGB)
            # Loop through the detected poses to visualize.
        #for idx, landmark in enumerate(results.pose_landmarks.landmark):
            #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        
            # Print nose landmark.
        image_hight, image_width, _ = frame.shape
        x=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width))
        y=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_hight))
        
        Allpoints.append(Point(x,y,1))
        x=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width))
        y=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight))
        
        Allpoints.append(Point(x,y,1))

        if framecnt%30==0:
              framecnt=0
              #print (Allpoints)
              result = recognizer.recognize(Allpoints)
              print (result)
              Allpoints.clear()  
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # show the final output
        cv2.imshow('Output', frame)
        
    except:
            #break
            print ('Camera Error')
    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()