import socket
import time
import asyncio
from bleak import BleakScanner
from datetime import datetime
import threading
import pickle
import math
import mediapipe as mp 
import cv2 
from dollarpy import Recognizer, Template, Point

# Define mediapipe holistics
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles

mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# Define server IP and port
host_ip = '0.0.0.0'  # Listen on all available interfaces
port = 8000

# Initialize the server socket and set it up to accept connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(1)  # Listen for 1 client connection at a time

client_socket = None
is_client_connected = False

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

def socket_server():
    global client_socket, is_client_connected
    print(f"Server listening on {host_ip}:{port}...")

    try:
        # Wait for a client to connect
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client at {client_address}")
        is_client_connected = True  # Mark client as connected

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

async def scan_bluetooth_devices():
    # Wait until a client is connected before scanning
    while not is_client_connected:
        print("Waiting for client connection to start Bluetooth scan...")
        await asyncio.sleep(1)

    print("Scanning for Bluetooth devices...")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        devices = await BleakScanner.discover()
        
        print(f"\nScan completed at {scan_time}")
        print(f"Found {len(devices)} devices:\n")
        
        for device in devices:
            device_info = f"Device Name: {device.name or 'Unknown'}, MAC Address: {device.address}"
            print(device_info)
            # Send the device info to the client
            #send_to_client(device_info)

            # Load the recognizer model from the file
            with open('recognizer_model.pkl', 'rb') as file:
                recognizer = pickle.load(file)

            #TEST
            vid = "../dataset/CatCow-Wrong-4.mp4" # use "/videos/"" + username
            points = getPoints(vid,"Unknown") # keep as unknown

            start = time.time()
            result = recognizer.recognize(points)
            end = time.time()
            print(result[0])
            print("time taken to classify:"+ str(end-start))

            score = math.floor(result[1] * 1000)

            if device.address == 'A0:D0:5B:27:31:17':
                send_to_client(f"{device.address, score}")
            
    except Exception as e:
        print(f"An error occurred in Bluetooth scan: {str(e)}")

def send_to_client(message):
    try:
        if is_client_connected and client_socket:
            client_socket.send(message.encode('utf-8'))
            print(f"Sent to client: {message}")
        else:
            print("Client not connected, cannot send data.")
    except Exception as e:
        print(f"Error sending to client: {e}")

async def main():
    print("Starting Bluetooth scan (after client connection)...")
    await scan_bluetooth_devices()

if __name__ == "__main__":
    # Start the socket server in a separate thread
    server_thread = threading.Thread(target=socket_server)
    server_thread.start()

    # Run the async function for scanning Bluetooth devices
    asyncio.run(main())