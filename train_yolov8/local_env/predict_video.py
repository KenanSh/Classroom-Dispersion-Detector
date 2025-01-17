import os
from ultralytics import YOLO
import cv2
import time

VIDEOS_DIR = '.\train_yolov8\local_env\videos\footage1.mp4'
video_path = os.path.join(VIDEOS_DIR, 'footage1.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
log_file_path = os.path.join(VIDEOS_DIR, 'detection_log.txt')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model = YOLO("D:\\human_detection_yolo\\train_yolov8\\local_env\\runs\\detect\\train3\\weights\\best.pt")

threshold = 0.5
fps = int(cap.get(cv2.CAP_PROP_FPS))
frames_per_log = fps * 30  # Log every 30 seconds
frame_count = 0

# Dictionary to keep track of unique IDs for detected people
people_ids = {}
current_id = 1

def is_face_in_person(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    return px1 <= fx1 <= px2 and py1 <= fy1 <= py2

with open(log_file_path, 'w') as log_file:
    while ret:
        results = model(frame)[0]
        class_names = model.names

        people_in_frame = []
        faces_in_frame = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                bbox = (int(x1), int(y1), int(x2), int(y2))
                if int(class_id) == 0:  # Assuming class_id 0 is 'Person'
                    people_in_frame.append((current_id, bbox))
                    people_ids[bbox] = current_id
                    current_id += 1
                elif int(class_id) == 1:  # Assuming class_id 1 is 'Human face'
                    faces_in_frame.append(bbox)

        for person_id, person_bbox in people_in_frame:
            distracted = True
            for face_bbox in faces_in_frame:
                if is_face_in_person(face_bbox, person_bbox):
                    distracted = False
                    break

            label = "Distracted" if distracted else "Not distracted"
            color = (0, 0, 255) if distracted else (0, 255, 0)
            x1, y1, x2, y2 = person_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        out.write(frame)

        if frame_count % frames_per_log == 0:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            log_file.write(f'Time: {timestamp}\n')
            log_file.write(f'Number of people: {len(people_in_frame)}\n')
            for person_id, person_bbox in people_in_frame:
                distracted = True
                for face_bbox in faces_in_frame:
                    if is_face_in_person(face_bbox, person_bbox):
                        distracted = False
                        break
                log_file.write(f'Person ID: {person_id}, Bounding Box: {person_bbox}, Distracted: {distracted}\n')
            log_file.write('\n')

        frame_count += 1
        ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
