from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='trash_dataset_final/data.yaml',
    epochs=40,           # Keep full epochs with smaller dataset
    imgsz=320,
    batch=8,             # Balanced batch size
    workers=2,
    device='cpu',
    patience=10,
    save=True,
    project='runs/detect',
    name='waste_50percent',
    verbose=True
)
