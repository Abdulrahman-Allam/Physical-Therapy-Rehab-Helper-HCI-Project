import cv2
import mediapipe as mp

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic=mp_holistic.Holistic(static_image_mode=True, 
                              min_detection_confidence=0.5, 
                              model_complexity=2)

cap = cv2.VideoCapture(0)# you can use video path(D:/video.mp4) if you want

while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        f_frame = cv2.flip(frame, 1)
         # convert to RGB
        frame_rgb = cv2.cvtColor(f_frame, cv2.COLOR_BGR2RGB)
         # Be carefull as some cameras flip the image. MAC cameras don't
         # process the frame for pose detection
        results = holistic.process(frame_rgb)        
         # We make a copy of the frame as we don't want to override the image
        annotated_image = f_frame.copy()
        
        #Drawing Landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)
            
            left_middle_top = results.left_hand_landmarks.landmark[12]
            left_middle_bot = results.left_hand_landmarks.landmark[9]

            left_middle_top_y = str(int(left_middle_top.y * image_height))
            left_middle_bot_y = str(int(left_middle_bot.y * image_height))

            if(left_middle_top_y > left_middle_bot_y):
                print(f"left top: {left_middle_top_y}, left bot: {left_middle_bot_y}")
                print(f"left hand closed")
            
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)
            
            right_middle_top = results.right_hand_landmarks.landmark[12]
            right_middle_bot = results.right_hand_landmarks.landmark[9]

            right_middle_top = results.right_hand_landmarks.landmark[12]
            right_middle_bot = results.right_hand_landmarks.landmark[9]

            right_middle_top_y = str(int(right_middle_top.y * image_height))
            right_middle_bot_y = str(int(right_middle_bot.y * image_height))

            if(right_middle_top_y > right_middle_bot_y):
                print(f"right top: {right_middle_top_y}, right bot: {right_middle_bot_y}")
                print(f"right hand closed")
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())
        

            image_height, image_width, _ = f_frame.shape  # Get frame dimensions
            left_wrist = results.pose_landmarks.landmark[15]

            xl = str(int(left_wrist.x * image_width))
            yl = str(int(left_wrist.y * image_height))
            print(f"Left wrist coordinates: x={xl}, y={yl}")

            right_wrist = results.pose_landmarks.landmark[16]
        
            xr = str(int(right_wrist.x * image_width))
            yr = str(int(right_wrist.y * image_height))
            print(f"Right wrist coordinates: x={xr}, y={yr}")
        
        cv2.imshow('Output', annotated_image)
    except:
        break
    # Close Window on pressing "Q"
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(100)
        break
          
cap.release()
cv2.destroyAllWindows()