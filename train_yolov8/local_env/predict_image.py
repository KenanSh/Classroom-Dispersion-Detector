from ultralytics import YOLO
model = YOLO("D:\\human_detection_yolo\\train_yolov8\\local_env\\runs\\detect\\train3\\weights\\best.pt")
results = model("D:\\human_detection_yolo\\train_yolov8\\local_env\\photos\\classroom.jpg", save=True)
