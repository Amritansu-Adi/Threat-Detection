from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')

# Train the model
if __name__ == '__main__':
    results = model.train(
        data='data/final dataset(2)/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='person_finetune_lr_adjusted',
        lr0=0.005,  # Lower initial learning rate
        patience=10 # Early stopping patience
    )
