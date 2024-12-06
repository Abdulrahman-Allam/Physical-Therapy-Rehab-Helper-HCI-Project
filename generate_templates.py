import os
import cv2
import mediapipe as mp

# Initialize Pose Estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def loop_files(directory):
    # Create or overwrite TestDollar.Py
    output_file = os.path.join(directory, "TestDollar.Py")
    with open(output_file, "w") as f:
        f.write("from dollarpy import Recognizer, Template, Point\n")

        templates = []  # List to collect all template names

        for file_name in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file_name)) and file_name.endswith(".mp4"):
                print(f"Processing file: {file_name}")
                template_name = file_name[:-4]  # Remove ".mp4" from the file name
                templates.append(template_name)

                # Start template definition
                f.write(f"{template_name} = Template('{template_name}', [\n")

                # Open video
                cap = cv2.VideoCapture(os.path.join(directory, file_name))
                framecnt = 0
                points_written = False  # Track if any points are written for this template

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video: {file_name}")
                        break

                    frame = cv2.resize(frame, (480, 320))
                    framecnt += 1

                    # Convert the frame to RGB format
                    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame to get pose landmarks
                    results = pose.process(RGB)
                    if results.pose_landmarks:
                        print(f"Pose landmarks detected in frame {framecnt} of {file_name}")
                        # Get image dimensions
                        image_height, image_width, _ = frame.shape

                        try:
                            # Extract right wrist coordinates
                            rw_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
                            rw_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)
                            f.write(f"Point({rw_x}, {rw_y}, 1),\n")
                            print(f"Writing Point({rw_x}, {rw_y}, 1) to template {template_name}")
                            points_written = True

                            # Extract left wrist coordinates
                            lw_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
                            lw_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)
                            f.write(f"Point({lw_x}, {lw_y}, 1),\n")
                            print(f"Writing Point({lw_x}, {lw_y}, 1) to template {template_name}")
                            points_written = True
                        except Exception as e:
                            print(f"Error extracting points at frame {framecnt}: {e}")
                    else:
                        print(f"No pose landmarks detected in frame {framecnt} of {file_name}")

                    # Draw landmarks for debugging
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Display the frame for debugging (press 'q' to quit early)
                    cv2.imshow('Output', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

                # End the template definition
                f.write("])\n")

                if not points_written:
                    print(f"Warning: No points written for {file_name}. Check video content or pose detection.")

                cap.release()

        # Add the recognizer definition with all templates
        recognizer_templates = ", ".join(templates)
        f.write(f"recognizer = Recognizer([{recognizer_templates}])\n")

    # Clean up OpenCV windows
    cv2.destroyAllWindows()


# Example usage
directory_path = "videos"  # Replace with the path to your video directory
loop_files(directory_path)
