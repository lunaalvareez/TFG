from ultralytics import YOLO

# Load a model
model = YOLO('best_1.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)