from ultralytics import solutions

inf = solutions.Inference(
    model="runs/segment/train14/weights/best.pt",  # You can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
)

inf.inference()