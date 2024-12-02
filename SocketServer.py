import socket
import threading
import cv2
import face_recognition
import numpy as np
import time


# Define server IP and port
host_ip = '0.0.0.0'  # Listen on all available interfaces
port = 8000

# Initialize the server socket and set it up to accept connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(1)  # Listen for 1 client connection at a time

client_socket = None
is_client_connected = False

# Load sample pictures and learn how to recognize them
try:
    obama_image = face_recognition.load_image_file("people/obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    allam_image = face_recognition.load_image_file("people/Abdulrahman Allam.jpg")
    allam_face_encoding = face_recognition.face_encodings(allam_image)[0]
except IndexError:
    print("Error: No face detected in one of the reference images. Ensure the image contains a clear face.")
    exit()

known_face_encodings = [
    obama_face_encoding,
    allam_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Abdulrahman Allam"
]


def socket_server():
    global client_socket, is_client_connected
    print(f"Server listening on {host_ip}:{port}...")

    try:
        # Wait for a client to connect
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client at {client_address}")
        is_client_connected = True  # Mark client as connected

        # Start face detection
        start_face_detection()

    except Exception as e:
        print(f"An error occurred in socket server: {e}")
    finally:
        # Clean up the connection
        if client_socket:
            client_socket.close()
        server_socket.close()
        is_client_connected = False
        print("Socket server closed")


def start_face_detection():
    # Access the webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return

    process_this_frame = True

    while is_client_connected:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Unable to capture video. Check your webcam.")
            break

        # Process every other frame for efficiency
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces and encode them
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Find the best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches and matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Send detected faces to the client
            if client_socket:
                detected = False
                for name in face_names:
                    try:
                        if name != "Unknown":
                            client_socket.send(f"{name}".encode('utf-8'))
                            print(f"{name} is sent to client")
                            detected = True
                            time.sleep(5)
                            break

                    except Exception as e:
                        print(f"Error sending to client: {e}")
                if detected:
                    break
                        

        process_this_frame = not process_this_frame

        # Display the results on screen
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    server_thread = threading.Thread(target=socket_server)
    server_thread.start()
