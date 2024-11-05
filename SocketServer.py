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
import platform

macdic = [
    {"mac": "6F:F2:1E:3F:B2:BB", "video": "videos\Kyphosis-All Seated ex2-v2.mp4"},
    {"mac": "B0:E4:5C:37:97:1F", "video": "videos\CatCow-Wrong-5.mp4"},
    {"mac": "71:3C:9B:85:A4:95", "video": "videos\Kyphosis-All Seated ex1-v4.mp4"},
    {"mac": "D4:0E:76:81:C2:8D", "video": "videos\CatCow-Correct-5.mp4"},
    {"mac": "5B:FC:EA:FA:D1:62", "video": "videos\Kyphosis-All Seated ex1-v4.mp4"},
    {"mac": "4E:D3:A3:58:27:CA", "video": "videos\Kyphosis-All Seated ex2-v2.mp4"},
]

# Define mediapipe holistics
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

# Define server IP and port
host_ip = '0.0.0.0'  # Listen on all available interfaces
port = 8000

# Initialize the server socket and set it up to accept connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(1)  # Listen for 1 client connection at a time

client_socket = None
is_client_connected = False

def getPoints(videoURL, label):
    cap = cv2.VideoCapture(videoURL)  # webcam = 0, else enter filename
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        points = []
        left_shoulder, right_shoulder = [], []
        left_elbos, right_elbos = [], []
        nose, left_wirst, right_wrist = [], [], []
        left_hip, right_hip = [], []     

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                try:
                    left_shoulder.append(Point(results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, 1))
                    right_shoulder.append(Point(results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, 2))
                    left_elbos.append(Point(results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, 3))
                    right_elbos.append(Point(results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, 4))
                    left_wirst.append(Point(results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y, 5))
                    right_wrist.append(Point(results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y, 6))
                    left_hip.append(Point(results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[23].y, 7))
                    right_hip.append(Point(results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y, 8))
                    nose.append(Point(results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, 9))
                except:
                    pass

                cv2.imshow(label, image)
            else:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                break

    points = left_shoulder + right_shoulder + left_elbos + right_elbos + left_wirst + right_wrist + left_hip + right_hip + nose
    print(label)
    return points

def socket_server():
    global client_socket, is_client_connected
    print(f"Server listening on {host_ip}:{port}...")

    try:
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client at {client_address}")
        is_client_connected = True

        while is_client_connected:
            time.sleep(5)
    except Exception as e:
        print(f"An error occurred in socket server: {e}")
    finally:
        if client_socket:
            client_socket.close()
        server_socket.close()
        is_client_connected = False
        print("Socket server closed")

async def scan_bluetooth_devices():
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
            if device.address == 'A0:D0:5B:27:31:17':
                send_to_client(f"{device.address}, ")
                break

            else:
                for x in macdic:
                    if x.get("mac") == device.address:
                         with open('recognizer_model.pkl', 'rb') as file:
                            recognizer = pickle.load(file)

                         vid = x.get("video")
                         points = getPoints(vid, "Unknown")
                         start = time.time()
                         result = recognizer.recognize(points)
                         end = time.time()
                         print(result[0])
                         print("Time taken to classify: " + str(end - start))

                         score = math.floor(result[1] * 1000)
                         print(f"this is your score: {score}")
                         send_to_client(f"{device.address},{score}")
                         break
                break

               

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

def run_scan_in_loop():
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scan_bluetooth_devices())
    loop.close()

if __name__ == "__main__":
    server_thread = threading.Thread(target=socket_server)
    server_thread.start()

    scan_thread = threading.Thread(target=run_scan_in_loop)
    scan_thread.start()
