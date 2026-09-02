from ultralytics import YOLO

# Load pre-trained nano YOLOv8 model
model = YOLO('yolov8n.pt')

# Train on your custom dataset (data.yaml must be in E:\CrimeCatcher\)
model.train(data='E:/CrimeCatcher/data/data.yaml', epochs=10)
