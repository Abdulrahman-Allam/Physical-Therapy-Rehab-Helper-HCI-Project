import cv2
from deepface import DeepFace
from collections import Counter

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# To store emotions detected during the session
emotion_counter = Counter()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to grayscale (optional, can improve face detection speed)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade for face detection (included with OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the frame
        face_region = frame[y:y + h, x:x + w]

        try:
            # Use DeepFace to analyze the emotion in the face region
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion from the result
            dominant_emotion = result[0]['dominant_emotion']

            # Increment the count of the dominant emotion
            emotion_counter[dominant_emotion] += 1

            # Put the emotion label on the frame
            cv2.putText(frame, f'{dominant_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("Error in emotion detection:", e)

    # Display the resulting frame with the face and emotion label
    cv2.imshow('Emotion Recognition (Press q to exit)', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# After closing the camera, print the most dominant emotion
if emotion_counter:
    most_common_emotion, count = emotion_counter.most_common(1)[0]
    print(f"The most dominant emotion during the session was: {most_common_emotion} (detected {count} times)")
else:
    print("No emotions detected.")