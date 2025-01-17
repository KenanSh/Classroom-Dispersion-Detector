from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("D:\\human_detection_yolo\\train_yolov8\\local_env\\runs\\detect\\train\\weights\\best.pt")  # load a pretrained model

# train the model
model.train(data="config.yaml", epochs=100, val=True)
# evaluate model performance on the validation set
metrics = model.val()
print(metrics)