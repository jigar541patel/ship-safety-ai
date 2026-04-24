import gradio as gr
from ultralytics import YOLO
from src.report import generate_report

model = YOLO("models/best.pt")

def detect(image):
    results = model(image)
    names = model.names
    detected = [names[int(b.cls[0])] for b in results[0].boxes]
    report = generate_report(detected)
    return results[0].plot(), report

gr.Interface(fn=detect, inputs="image", outputs=["image","text"]).launch()
