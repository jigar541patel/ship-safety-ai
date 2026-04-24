from ultralytics import YOLO
from report import generate_report

model = YOLO("models/best.pt")

def run(image):
    results = model(image)
    names = model.names
    detected = [names[int(b.cls[0])] for b in results[0].boxes]
    report = generate_report(detected)
    print(report)
    results[0].save(filename="outputs/result.jpg")

if __name__ == "__main__":
    run("test.jpg")
