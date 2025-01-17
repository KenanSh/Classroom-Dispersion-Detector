import os
from ultralytics import YOLO
import cv2
import time

VIDEOS_DIR = os.path.join('D:\\human_detection_yolo\\train_yolov8\\local_env\\', 'videos')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the default device index for the webcam
threshold = 0.3

# Dictionary to keep track of unique IDs for detected people
people_ids = {}
current_id = 1

# Get the frame rate of the webcam
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0:  # If the webcam FPS cannot be determined, set a default value
    fps = 30.0
frames_per_log = int(fps * 2)  # Log every 2 seconds
frame_count = 0

# Open a log file to write detection information
log_file_path = 'detection_log.txt'

# Initialize the YOLO model
model = YOLO("D:\\human_detection_yolo\\train_yolov8\\local_env\\runs\\detect\\train3\\weights\\best.pt")

def is_face_in_person(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    return px1 <= fx1 <= px2 and py1 <= fy1 <= py2

with open(log_file_path, 'a') as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on the current frame
        results = model(frame)[0]

        people_in_frame = []
        faces_in_frame = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            bbox = (int(x1), int(y1), int(x2), int(y2))
            if score > threshold:
                if int(class_id) == 0:  # Assuming class_id 0 is 'Person'
                    if bbox not in people_ids:
                        people_ids[bbox] = current_id
                        current_id += 1
                    person_id = people_ids[bbox]
                    people_in_frame.append((person_id, bbox))
                elif int(class_id) == 1:  # Assuming class_id 1 is 'Face'
                    faces_in_frame.append(bbox)

        # List to keep track of detected persons with faces
        persons_with_faces = []
        face_only_detected = []

        for person_id, person_bbox in people_in_frame:
            distracted = True
            for face_bbox in faces_in_frame:
                if is_face_in_person(face_bbox, person_bbox):
                    distracted = False
                    persons_with_faces.append((person_id, person_bbox))
                    break

            label = "Distracted" if distracted else "Not distracted"
            color = (0, 0, 255) if distracted else (0, 255, 0)
            x1, y1, x2, y2 = person_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Handle faces detected without a corresponding person
        for face_bbox in faces_in_frame:
            face_in_person = False
            for _, person_bbox in people_in_frame:
                if is_face_in_person(face_bbox, person_bbox):
                    face_in_person = True
                    break
            if not face_in_person:
                face_only_detected.append(face_bbox)
                fx1, fy1, fx2, fy2 = face_bbox
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                cv2.putText(frame, "Face only", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Update the count of people (persons with faces and faces only)
        person_count = len(persons_with_faces) + len(face_only_detected)

        # Display the frame with the detections
        cv2.imshow('YOLO Real-Time Detection', frame)

        if frame_count % frames_per_log == 0:
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            log_file.write(f'Time: {current_time}\n')
            log_file.write(f'Number of people: {person_count}\n')
            for person_id, person_bbox in persons_with_faces:
                log_file.write(f'Person ID: {person_id}, Bounding Box: {person_bbox}, Distracted: False\n')
            for person_id, person_bbox in people_in_frame:
                if person_id not in [p_id for p_id, _ in persons_with_faces]:
                    log_file.write(f'Person ID: {person_id}, Bounding Box: {person_bbox}, Distracted: True\n')
            for face_bbox in face_only_detected:
                log_file.write(f'Face only detected, Bounding Box: {face_bbox}\n')
            log_file.write('\n')

        frame_count += 1

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
