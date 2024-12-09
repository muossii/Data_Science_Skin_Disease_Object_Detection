from ultralytics import YOLO

# Load pre-trained model
model = YOLO('best.pt')

# run inferrence on the source
results = model(source=0, show=True, conf=0.4, save=True) #generator of results objects

