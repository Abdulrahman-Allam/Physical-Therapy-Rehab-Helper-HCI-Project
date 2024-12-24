import threading
import tkinter as tk
import cv2
import numpy as np
from deepface import DeepFace
import face_recognition
from ultralytics import YOLO
import mediapipe as mp
import math
from collections import Counter
import asyncio
import bluetooth
import socket
import time
import json
from dollarpy import Point
import pickle
from PIL import Image, ImageTk
import dlib
from math import hypot
import matplotlib.pyplot as plt



with open('Users.txt', 'r') as file:
    users = json.load(file)



with open('recognizer_model.pkl', 'rb') as file:
                recognizer = pickle.load(file)






cx = 0
cy = 0
obj_flag = 0
obj_user = None

tuio = 1
flag = 0
camera_flag = 0
handX = 0
handY = 0
canvas = None

currentUser = None
client_data = None




# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Font for displaying text
font = cv2.FONT_HERSHEY_PLAIN

# List to store pupil coordinates
pupil_coords = []

# Screen dimensions
screen_width, screen_height = 700,550



# Define server IP and port
host_ip = '0.0.0.0'  # Listen on all available interfaces
port = 8000

# Initialize the server socket and set it up to accept connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(1)  # Listen for 1 client connection at a time

client_socket = None
is_client_connected = False
server_thread = None  # To keep track of the server thread


# Define mediapipe holistics
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles

mp_holistic = mp.solutions.holistic # Mediapipe Solutions


# Helper function to calculate the midpoint
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to map coordinates to screen resolution
def map_to_screen(pupil_x, pupil_y, frame_width, frame_height):
    screen_x = int(pupil_x * screen_width / frame_width)
    screen_y = int(pupil_y * screen_height / frame_height)
    return screen_x, screen_y

# Function to detect pupil center
def get_pupil_center(eye_points, facial_landmarks, gray, scale=0.6):
    # Get the bounding box of the eye region
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    x, y, w, h = cv2.boundingRect(eye_region)

    # Shrink the bounding box around the eye region
    center_x, center_y = x + w // 2, y + h // 2
    w, h = int(w * scale), int(h * scale)
    x, y = center_x - w // 2, center_y - h // 2

    # Crop the region of interest (ROI)
    eye_roi = gray[y:y + h, x:x + w]
    eye_roi = cv2.equalizeHist(eye_roi)

    # Thresholding to detect dark areas (pupils)
    _, threshold_eye = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        return int(cx + x), int(cy + y)  # Convert to original frame coordinates
    return None




def socket_server():
    global client_socket, is_client_connected, client_data
    print(f"Server listening on {host_ip}:{port}...")

    try:
        # Wait for a client to connect
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client at {client_address}")
        is_client_connected = True  # Mark client as connected
        send_to_client(client_data)

        while is_client_connected:
            time.sleep(5)  # Keep the connection open and sleep
    except Exception as e:
        print(f"An error occurred in socket server: {e}")
    finally:
        # Clean up the connection
        if client_socket:
            client_socket.close()
        server_socket.close()
        is_client_connected = False
        print("Socket server closed")


def send_to_client(message):
    try:
        if is_client_connected and client_socket:
            client_socket.send(message.encode('utf-8'))
            print(f"Sent to client: {message}")
        else:
            print("Client not connected, cannot send data.")
    except Exception as e:
        print(f"Error sending to client: {e}")



def start_server():
    global server_thread
    if server_thread is None or not server_thread.is_alive():
        server_thread = threading.Thread(target=socket_server, daemon=True)
        server_thread.start()
        print("Server started in a separate thread.")
    else:
        print("Server is already running.")

async def scan_bluetooth_devices():
    print("Scanning for nearby Bluetooth devices...")
    devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=False)
    
    if len(devices) == 0:
        print("No Bluetooth devices found.")
    else:
        print(f"Found {len(devices)} device(s):")
        for addr, name in devices:
            try:
                print(f"Device ID (MAC Address): {addr}, Device Name: {name}")
            except UnicodeEncodeError:
                print(f"Device ID (MAC Address): {addr}, Device Name: (Could not decode name)")

async def bt():
    await scan_bluetooth_devices()



def getPoints(videoURL,label):
    global currentUser
    cap2 = cv2.VideoCapture(videoURL)#web cam =0 , else enter filename
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
        while cap2.isOpened():
            ret, frame = cap2.read()

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
                cap2.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap2.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

    cap2.release()
    cv2.destroyAllWindows()
    points = left_shoulder+right_shoulder+left_elbos+right_elbos+left_wirst+right_wrist+left_hip+right_hip+nose
    print(label)
    return points

def eyes_tracking(frame):
    global pupil_coords
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get pupil centers
        left_pupil = get_pupil_center([36, 37, 38, 39, 40, 41], landmarks, gray, scale=0.6)
        right_pupil = get_pupil_center([42, 43, 44, 45, 46, 47], landmarks, gray, scale=0.6)

        # Display pupil coordinates
        if left_pupil:
            cv2.circle(frame, left_pupil, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Left: {left_pupil}", (10, 30), font, 1, (255, 255, 255), 1)
            pupil_coords.append(left_pupil)

            # Map to screen coordinates
            screen_left_pupil = map_to_screen(left_pupil[0], left_pupil[1], frame.shape[1], frame.shape[0])
            cv2.putText(frame, f"Screen Left: {screen_left_pupil}", (10, 50), font, 1, (255, 255, 0), 1)

        if right_pupil:
            cv2.circle(frame, right_pupil, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Right: {right_pupil}", (10, 60), font, 1, (255, 255, 255), 1)
            pupil_coords.append(right_pupil)

            # Map to screen coordinates
            screen_right_pupil = map_to_screen(right_pupil[0], right_pupil[1], frame.shape[1], frame.shape[0])
            cv2.putText(frame, f"Screen Right: {screen_right_pupil}", (10, 80), font, 1, (255, 255, 0), 1)

    # Show frame
    cv2.putText(frame, "Press ESC to exit", (10, frame.shape[0] - 10), font, 1, (255, 255, 255), 1)


def face_detection(frame):
    global users
    global flag, camera_flag, currentUser, client_data
    # Load reference images for face recognition
    try:
        obama_image = face_recognition.load_image_file("people/obama.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    
        Abdulrahman_image = face_recognition.load_image_file("people/Abdulrahman Allam.jpg")
        Abdulrahman_face_encoding = face_recognition.face_encodings(Abdulrahman_image)[0]
        
        kena_image = face_recognition.load_image_file("people/Abdelrahman Moustafa.jpg")
        kena_face_encoding = face_recognition.face_encodings(kena_image)[0]

        hamza_image = face_recognition.load_image_file("people/Hamza Moustafa.jpeg")
        hamza_face_encoding = face_recognition.face_encodings(hamza_image)[0]

        assem_image = face_recognition.load_image_file("people/Assem Omar.jpg")
        assem_face_encoding = face_recognition.face_encodings(assem_image)[0]
    except IndexError:
        print("Error: No face detected in one of the reference images.")
        exit()
    
    # Known face encodings and names
    known_face_encodings = [obama_face_encoding, Abdulrahman_face_encoding, kena_face_encoding, hamza_face_encoding, assem_face_encoding]
    known_face_names = ["Barack Obama", "Abdulrahman Allam", "Abdelrahman Moustafa", "Hamza Moustafa", "Assem Omar"]

    # Flip and preprocess frame
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Face Recognition
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]


        for user in users:
            if name == user['name']:
                currentUser = user
                client_data = user['name']
                camera_flag = 0
                if user['role'] == "doctor":
                    flag = 1
                else:
                    flag = 3
                break


    
        # Draw bounding box for face
        #if name == "Abdulrahman Allam":
           
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    

def Emotion_detection(frame):

    # Emotion counter
    emotion_counter = Counter()

    # Flip and preprocess frame
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Face Recognition
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


    # Emotion Detection
    try:
        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            emotion_result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion_result[0]['dominant_emotion']
            emotion_counter[dominant_emotion] += 1
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print("Emotion detection error:", e)

def Object_detection(frame):
    global cx, cy, obj_flag, obj_user, users, flag

    # Initialize models
    yolo_model = YOLO("yolo-Weights/yolov8n.pt")
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Object detection classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    results = yolo_model(frame, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])

            # center of cellphone
            if(classNames[cls] == "cell phone"):
                width = int(x2 - x1)
                height = int(y2 - y1)

                cx = int(x1 + width / 2) # center x
                cy = int(y1 + height / 2) # center y
                print(cx)
                print(cy)

                #700x550

                if cx < 200 and cy < 200:
                    obj_flag = 1
                elif cx > 200 and cy < 200:
                    obj_flag = 2
                elif cx < 200 and cy > 200:
                    obj_flag = 3
                elif cx > 200 and cy > 200:
                    obj_flag = 4


                r = int(3) # radius
                cv2.circle(frame, (cx, cy), r, (0, 0, 255), 3) # center point drawn on camera

                if(width > height): #rotated on if horizontal
                    print("Rotated: ON")
                    if obj_flag == 1:
                        obj_user = users[1]
                    elif obj_flag == 2:
                        obj_user = users[2]
                    elif obj_flag == 3:
                        obj_user = users[3]
                    elif obj_flag == 4:
                        obj_user = users[4]
                    flag = 5
                elif(height >= width): #rotated off if vertical
                    print("Rotated: OFF")

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


def hand_TUIO(frame):
    global flag, camera_flag, tuio, currentUser, recognizer

    # Initialize models
    yolo_model = YOLO("yolo-Weights/yolov8n.pt")
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize MediaPipe Holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        model_complexity=2
    )


    # Flip and preprocess frame
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape
    
           
    
            
    
           
    
    # MediaPipe Pose and Hand Detection
    results = holistic.process(rgb_frame)
    
    # Draw Pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw hand landmarks and detect hand movements
    for hand_landmarks, hand_label in zip(
            [results.left_hand_landmarks, results.right_hand_landmarks],
            ["Left", "Right"]):
        if hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            middle_top = hand_landmarks.landmark[12].y * image_height
            middle_bot = hand_landmarks.landmark[9].y * image_height
            print(hand_landmarks.landmark[9].x * image_width)
            print(hand_landmarks.landmark[9].y * image_height)
            global handX, handY
            handX = hand_landmarks.landmark[9].x * image_width
            handY = hand_landmarks.landmark[9].y * image_height
            if middle_top > middle_bot:
                if flag == 0:
                    if handX > 300 and handX < 400 and handY > 280 and handY < 330:
                        camera_flag = 1
                elif flag == 1:
                    if handX > 220 and handX < 450 and handY > 90 and handY < 150:
                        camera_flag = 2
                        flag = 2
                        tuio = 0
                    elif handX > 220 and handX < 450 and handY > 190 and handY < 250:
                        start_server()
                        camera_flag = -1
                elif flag == 3:
                    if handX > 220 and handX < 450 and handY > 290 and handY < 350:  
                        points = getPoints(currentUser['video'], "Unknown") # keep as unknown
                        score = recognizer.recognize(points)
                        score = math.floor(score[1] * 1000)
                        currentUser['score'] = score
                        # Write the modified data back to the file
                        with open('Users.txt', 'w') as file:
                            json.dump(users, file, indent=4)
                        flag = 4    


                print(f"{hand_label} hand closed")


# Detection Functionality
def detection_thread():
    global canvas, flag, tuio, camera_flag, handX, handY, cx, cy, pupil_coords
    
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 700)
    cap.set(4, 550)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        try:
            Emotion_detection(frame)
            eyes_tracking(frame)

            if tuio == 1:
                hand_TUIO(frame)
            
            if camera_flag == 1:
                face_detection(frame)
            elif camera_flag == 2:
                Object_detection(frame)
            elif camera_flag == -1:
                break
            
            
            print(flag)

            if flag == 0:
                show_login_page()
            elif flag == 1:
                Doctor_page()
            elif flag == 3:
                Patiant_page()
            elif flag == 2:
                open_tuio_page()
            elif flag == 4:
                Score_page()
            elif flag == 5:
                objpat_page()


            if camera_flag != 2:
                root.after(0, update_red_dot, handX, handY)
            else:
                root.after(0, update_red_dot, cx, cy)
            

        except Exception as e:
            print("Error processing frame:", e)
            break

        cv2.imshow("Multifunction Detection (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate heatmap
    print("before entering")
    if pupil_coords:
        print("after entering")
        heatmap, xedges, yedges = np.histogram2d(
            [coord[0] for coord in pupil_coords],
            [coord[1] for coord in pupil_coords],
            bins=(frame.shape[1] // 10, frame.shape[0] // 10),
            range=[[0, frame.shape[1]], [0, frame.shape[0]]]
        )
    
        plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, frame.shape[1], 0, frame.shape[0]])
        plt.colorbar(label="Focus Intensity")
        plt.title("Eye Focus Heatmap")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("eye_focus_heatmap.png")
        plt.show()
    
        # Save pupil coordinates to file
        np.savetxt("pupil_coords.csv", pupil_coords, delimiter=",", fmt="%d", header="x,y")


# Update the existing canvas with the new dot position
# Do not recreate the canvas here!
def update_red_dot(x, y):
    global canvas
    canvas.delete("red_dot")
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", outline="red", tags="red_dot")


# Function to display profile picture
def display_profile_picture(canvas, image_path):
    # Load the image
    img = Image.open(image_path)
    # Resize the image to fit the canvas (optional)
    # img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
    img = img.resize((40, 40), Image.Resampling.LANCZOS)
    # Convert the image to PhotoImage
    img_tk = ImageTk.PhotoImage(img)
    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    # Keep a reference to avoid garbage collection
    canvas.image = img_tk


# Function to display profile picture
def display_profile_picture2(canvas, image_path):
    # Load the image
    img = Image.open(image_path)
    # Resize the image to fit the canvas (optional)
    img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
    # Convert the image to PhotoImage
    img_tk = ImageTk.PhotoImage(img)
    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    # Keep a reference to avoid garbage collection
    canvas.image = img_tk


def sign_out_all():
    global tuio, flag, camera_flag


    tuio = 1
    flag = 0
    camera_flag = 0

def back_to_doctor():
    global tuio, flag, camera_flag


    tuio = 1
    flag = 1
    camera_flag = 0


def back_to_obj():
    global tuio, flag, camera_flag


    flag = 2
    camera_flag = 2


def back_to_patient():
    global tuio, flag, camera_flag


    tuio = 1
    flag = 3
    camera_flag = 0

# Function to open the TUIO page
def open_tuio_page():
    global canvas, obj_flag, users
    # Destroy the current page
    for widget in root.winfo_children():
        widget.destroy()

    # Configure the TUIO page
    root.title("TUIO Interface")
    root.geometry("700x550")  # Updated size
    root.configure(bg="#ADD8E6")  # Light blue background

    # Create a canvas to draw the circle
    canvas = tk.Canvas(root, width=400, height=400, bg="#ADD8E6", highlightthickness=0)
    canvas.pack(pady=20)



    # Define colors for sections
    default_colors = ["#FF9999", "#99FF99", "#9999FF", "#FFFF99"]  # Default section colors
    hover_colors = ["#FF4D4D", "#4DFF4D", "#4D4DFF", "#FFFF4D"]  # Hover section colors

    # Functions to change colors on hover
    def on_enter(event, section):
        canvas.itemconfig(section, fill=hover_colors[sections.index(section)])

    def on_leave(event, section):
        canvas.itemconfig(section, fill=default_colors[sections.index(section)])

    # Create sections of the circle
    sections = [
        canvas.create_arc(50, 50, 350, 350, start=0, extent=90, fill=default_colors[0], outline="black"),
        canvas.create_arc(50, 50, 350, 350, start=90, extent=90, fill=default_colors[1], outline="black"),
        canvas.create_arc(50, 50, 350, 350, start=180, extent=90, fill=default_colors[2], outline="black"),
        canvas.create_arc(50, 50, 350, 350, start=270, extent=90, fill=default_colors[3], outline="black"),
    ]

    if obj_flag == 1:
        welcoming = tk.Label(root, text=users[1]['name'], bg="#ADD8E6", font=("Arial", 12, "bold"))
        canvas.create_window(180, 40, window=welcoming, anchor="nw")
    elif obj_flag == 2:
        welcoming = tk.Label(root, text=users[2]['name'], bg="#ADD8E6", font=("Arial", 12, "bold"))
        canvas.create_window(180, 40, window=welcoming, anchor="nw")
    elif obj_flag == 3:
        welcoming = tk.Label(root, text=users[3]['name'], bg="#ADD8E6", font=("Arial", 12, "bold"))
        canvas.create_window(180, 40, window=welcoming, anchor="nw")
    elif obj_flag == 4:
        welcoming = tk.Label(root, text=users[4]['name'], bg="#ADD8E6", font=("Arial", 12, "bold"))
        canvas.create_window(180, 40, window=welcoming, anchor="nw")

    # Bind hover events to each section
    for section in sections:
        canvas.tag_bind(section, "<Enter>", lambda event, s=section: on_enter(event, s))
        canvas.tag_bind(section, "<Leave>", lambda event, s=section: on_leave(event, s))

    # Back button to return to the second page
    back_button = tk.Button(
        root,
        text="Back",
        bg="#4682B4",
        fg="white",
        font=("Arial", 12, "bold"),
        command=back_to_doctor
    )
    back_button.place(x=2, y=2)




# Function to open the second page
def objpat_page():
    global canvas, obj_user

    # Destroy current page
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Physical Therapy Dashboard")
    root.geometry("700x550")
    root.configure(bg="#ADD8E6")

    # Create and pack the main canvas
    canvas = tk.Canvas(root, bg="#ADD8E6", highlightthickness=0, width=700, height=550)
    canvas.pack(fill=tk.BOTH, expand=True)


    # Title bar frame
    title_bar = tk.Frame(root, bg="#0047AB", width=700, height=50)
    # Place the title bar at the top center (350 is half of 700, 25 is half of 50)
    canvas.create_window(350, 25, window=title_bar, anchor="center")



    # Back button to return to the second page
    back_button = tk.Button(
        root,
        text="Back",
        bg="#4682B4",
        fg="white",
        font=("Arial", 12, "bold"),
        command=back_to_obj
    )
    back_button.place(x=2, y=2)







    # Details labels
    your_name = tk.Label(root, text=' Name : '+obj_user['name'], bg="#ADD8E6", font=("Arial", 12, "bold"))
    age_label = tk.Label(root, text='Age: '+str(obj_user['age']), bg="#ADD8E6", font=("Arial", 12, "bold"))
    score_labe = tk.Label(root, text='Score: '+str(obj_user['score']), bg="#ADD8E6", font=("Arial", 12, "bold"))
    
    
       
   
    # PATIENT label
    canvas.create_window(150, 100, window=your_name, anchor="nw")

    # AGE label
    canvas.create_window(150, 130, window=age_label, anchor="nw")

    # ROLE label
    canvas.create_window(150, 160, window=score_labe, anchor="nw")



    

 


# Function to open the second page
def Doctor_page():
    global canvas, handX, handY, currentUser

    # Destroy current page
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Physical Therapy Dashboard")
    root.geometry("700x550")
    root.configure(bg="#ADD8E6")

    # Create and pack the main canvas
    canvas = tk.Canvas(root, bg="#ADD8E6", highlightthickness=0, width=700, height=550)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Title bar frame
    title_bar = tk.Frame(root, bg="#0047AB", width=700, height=50)
    # Place the title bar at the top center (350 is half of 700, 25 is half of 50)
    canvas.create_window(350, 25, window=title_bar, anchor="center")

    # Profile frame inside the title bar
    profile_frame = tk.Frame(title_bar, bg="#0047AB")
    profile_frame.pack(side=tk.LEFT, padx=10)

    # Profile picture placeholder
    profile_pic = tk.Canvas(profile_frame, width=40, height=40, bg="#D3D3D3", highlightthickness=0)
    profile_pic.pack(side=tk.LEFT, padx=5)

    display_profile_picture(profile_pic, currentUser['image'])



    welcoming = tk.Label(root, text="Welcome Doctor", bg="#ADD8E6", font=("Arial", 12, "bold"))

    canvas.create_window(270, 65, window=welcoming, anchor="nw")

    # Name label
    # Name label
    title_label = tk.Label(
        profile_frame, text='Name: '+currentUser['name'], bg="#0047AB", fg="white", font=("Arial", 14, "bold")
    )
    title_label.pack(side=tk.LEFT)

    # Signout button on the title bar (to the right)
    signout_button = tk.Button(
        title_bar, text="SIGNOUT", bg="#FF0000", fg="white", font=("Arial", 12, "bold"), command=sign_out_all
    )
    signout_button.pack(side=tk.RIGHT, padx=10)




    # Continue using TUIO button
    tuio_button = tk.Button(
        root, text="using Object Detection",  bg="#4682B4", fg="white", font=("Arial", 10, "bold"), width=22, command=open_tuio_page
    )
    tuio_button2 = tk.Button(
        root, text="using Object Detection",  bg="red", fg="white", font=("Arial", 10, "bold"), width=22, command=open_tuio_page
    )
    if handX > 250 and handX < 500 and handY > 90 and handY < 150:
        canvas.create_window(270, 120, window=tuio_button2, anchor="nw")
    else:
        canvas.create_window(270, 120, window=tuio_button, anchor="nw")

    # Continue using Object Detection button
    obj_det_button = tk.Button(
        root, text="Continue using TUIO", bg="#4682B4", fg="white", font=("Arial", 10, "bold"), width=22
    )
    obj_det_button2 = tk.Button(
        root, text="Continue using TUIO", bg="red", fg="white", font=("Arial", 10, "bold"), width=22
    )
    if handX > 250 and handX < 500 and handY > 190 and handY < 250:
        canvas.create_window(270, 220, window=obj_det_button2, anchor="nw")
    else:
        canvas.create_window(270, 220, window=obj_det_button, anchor="nw")



# Function to open the second page
def Patiant_page():
    global canvas, handX, handY, currentUser

    # Destroy current page
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Physical Therapy Dashboard")
    root.geometry("700x550")
    root.configure(bg="#ADD8E6")

    # Create and pack the main canvas
    canvas = tk.Canvas(root, bg="#ADD8E6", highlightthickness=0, width=700, height=550)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Title bar frame
    title_bar = tk.Frame(root, bg="#0047AB", width=700, height=50)
    # Place the title bar at the top center (350 is half of 700, 25 is half of 50)
    canvas.create_window(350, 25, window=title_bar, anchor="center")

    # Profile frame inside the title bar
    profile_frame = tk.Frame(title_bar, bg="#0047AB")
    profile_frame.pack(side=tk.LEFT, padx=10)

    # Profile picture placeholder
    profile_pic = tk.Canvas(profile_frame, width=40, height=40, bg="#D3D3D3", highlightthickness=0)
    profile_pic.pack(side=tk.LEFT, padx=5)

    display_profile_picture(profile_pic, currentUser['image'])



    # Name label
    # Name label
    title_label = tk.Label(
        profile_frame, text='Name: '+currentUser['name'], bg="#0047AB", fg="white", font=("Arial", 14, "bold")
    )
    title_label.pack(side=tk.LEFT)

    # Signout button on the title bar (to the right)
    signout_button = tk.Button(
        title_bar, text="SIGNOUT", bg="#FF0000", fg="white", font=("Arial", 12, "bold"), command=sign_out_all
    )
    signout_button.pack(side=tk.RIGHT, padx=10)


    # Details labels
    age_label = tk.Label(root, text='Age: '+str(currentUser['age']), bg="#ADD8E6", font=("Arial", 12, "bold"))
    role_label = tk.Label(root, text='Role: '+currentUser['role'], bg="#ADD8E6", font=("Arial", 12, "bold"))
    your_label = None
    if currentUser['role'] == 'patient':
        your_label = tk.Label(root, text='Your doctor is: '+currentUser['doctor'], bg="#ADD8E6", font=("Arial", 12, "bold"))
    elif currentUser['role'] == 'doctor':
        your_label = tk.Label(root, text='Your patients: '+', '.join(currentUser['patients']), bg="#ADD8E6", font=("Arial", 12, "bold"))
    #tk.Label(root, text="Patient", bg="#ADD8E6", font=("Arial", 12, "bold")).place(x=50, y=160)

    # AGE label
    canvas.create_window(50, 100, window=age_label, anchor="nw")

    # ROLE label
    canvas.create_window(50, 130, window=role_label, anchor="nw")

    # PATIENT label
    canvas.create_window(50, 160, window=your_label, anchor="nw")

    # Continue using TUIO button
    tuio_button = tk.Button(
        root, text="Rate Your Exercise",  bg="#4682B4", fg="white", font=("Arial", 10, "bold"), width=22, command=open_tuio_page
    )
    tuio_button2 = tk.Button(
        root, text="Rate Your Exercise",  bg="red", fg="white", font=("Arial", 10, "bold"), width=22, command=open_tuio_page
    )
    if handX > 220 and handX < 450 and handY > 290 and handY < 350:
        canvas.create_window(270, 320, window=tuio_button2, anchor="nw")
    else:
        canvas.create_window(270, 320, window=tuio_button, anchor="nw")

    # Continue using Object Detection button
    # obj_det_button = tk.Button(
    #     root, text="show score", bg="#4682B4", fg="white", font=("Arial", 10, "bold"), width=22
    # )
    # obj_det_button2 = tk.Button(
    #     root, text="Show Score", bg="red", fg="white", font=("Arial", 10, "bold"), width=22
    # )
    # if handX > 250 and handX < 500 and handY > 190 and handY < 250:
    #     canvas.create_window(270, 420, window=obj_det_button2, anchor="nw")
    # else:
    #     canvas.create_window(270, 420, window=obj_det_button, anchor="nw")



# Function to open the second page
def Score_page():
    global canvas, handX, handY, currentUser

    # Destroy current page
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Physical Therapy Dashboard")
    root.geometry("700x550")
    root.configure(bg="#ADD8E6")

    # Create and pack the main canvas
    canvas = tk.Canvas(root, bg="#ADD8E6", highlightthickness=0, width=700, height=550)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Title bar frame
    title_bar = tk.Frame(root, bg="#0047AB", width=700, height=50)
    # Place the title bar at the top center (350 is half of 700, 25 is half of 50)
    canvas.create_window(350, 25, window=title_bar, anchor="center")

    # Profile frame inside the title bar
    profile_frame = tk.Frame(title_bar, bg="#0047AB")
    profile_frame.pack(side=tk.LEFT, padx=10)

    # Profile picture placeholder
    profile_pic = tk.Canvas(profile_frame, width=40, height=40, bg="#D3D3D3", highlightthickness=0)
    profile_pic.pack(side=tk.LEFT, padx=5)

    display_profile_picture(profile_pic, currentUser['image'])

    # Name label
    # Name label
    title_label = tk.Label(
        profile_frame, text='Name: '+currentUser['name'], bg="#0047AB", fg="white", font=("Arial", 14, "bold")
    )
    title_label.pack(side=tk.LEFT)

    # Signout button on the title bar (to the right)
    signout_button = tk.Button(
        title_bar, text="SIGNOUT", bg="#FF0000", fg="white", font=("Arial", 12, "bold"), command=sign_out_all
    )
    signout_button.pack(side=tk.RIGHT, padx=10)


    welcoming = tk.Label(root, text="your score is: " + str(currentUser['score']), bg="#ADD8E6", font=("Arial", 24, "bold"))

    canvas.create_window(200, 250, window=welcoming, anchor="nw")



    # Profile picture placeholder
    profile_pic2 = tk.Canvas(root, width=300, height=250, bg="#D3D3D3", highlightthickness=0)
    profile_pic2.place(x=200, y=300) 

    display_profile_picture2(profile_pic2, currentUser['image'])




    # Back button to return to the second page
    back_button = tk.Button(
        root,
        text="Back",
        bg="#4682B4",
        fg="white",
        font=("Arial", 12, "bold"),
        command=back_to_patient
    )
    back_button.place(x=2,y=2)

    


# Function to show the login page
def show_login_page():
    global canvas, handX, handY
    
    # Destroy current page
    for widget in root.winfo_children():
        widget.destroy()

    root.title("Physical Therapy For Kyphosis Application")
    root.geometry("700x550")
    root.configure(bg="#ADD8E6")

    # Create the canvas to hold all elements
    canvas = tk.Canvas(root, bg="#ADD8E6", highlightthickness=0, width=700, height=550)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Title label
    title_label = tk.Label(
        root,
        text="Physical Therapy For Kyphosis Application",
        bg="#0047AB",  # Dark blue background for title
        fg="white",    # White text
        font=("Arial", 14, "bold"),
        pady=10
    )
    # Place the label on the canvas (centered at x=350)
    canvas.create_window(350, 20, window=title_label, anchor="n")

    # Placeholder frame (e.g., could hold an image or form)
    placeholder_frame = tk.Frame(root, bg="#D3D3D3", width=300, height=150)
    # Place the frame on the canvas at roughly (350, 220)
    # Adjust the anchor so that the frame is centered around that point
    #canvas.create_window(350, 220, window=placeholder_frame, anchor="center")

    # Login button
    login_button = tk.Button(
        root,
        text="LOGIN",
        bg="#4682B4",  # Blue button background
        fg="white",    # White text
        font=("Arial", 12, "bold"),
        width=10,
        height=1,
    )

    login_button2 = tk.Button(
        root,
        text="LOGIN",
        bg="red",  # Blue button background
        fg="white",    # White text
        font=("Arial", 12, "bold"),
        width=10,
        height=1,
    )
    # Place the button on the canvas near the bottom (350, 440)
    if handX > 300 and handX < 400 and handY > 280 and handY < 330:
         canvas.create_window(350, 300, window=login_button2, anchor="center")
    else:
        canvas.create_window(350, 300, window=login_button, anchor="center")
   



    

    


 

    
# Run the async function for scanning Bluetooth devices
asyncio.run(bt())


# Start the threads
detection = threading.Thread(target=detection_thread, daemon=True)
detection.start()

root = tk.Tk()
show_login_page()  # Display the login page initially
root.mainloop()

