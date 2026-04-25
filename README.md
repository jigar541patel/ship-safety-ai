# 🚢 Ship Safety AI — Hazard & Compliance Detection System

An AI-powered computer vision system designed to detect hazardous materials and safety compliance items in shipyard and industrial environments.

This project combines **multi-dataset training**, **custom class remapping**, and **mobile-ready deployment (TFLite)** to build a real-world safety monitoring solution.

---

## 🎯 Problem Statement

Shipyard environments involve high-risk operations where the presence of hazardous materials (e.g., cigarettes, solvents, spray tools) can lead to fire or chemical accidents.

Manual monitoring is:
- ❌ inefficient  
- ❌ error-prone  
- ❌ not scalable  

---

## 💡 Solution

This system automatically:

- Detects **hazardous items** (fire/chemical risks)
- Detects **safety compliance items**
- Generates structured outputs for further safety analysis
- Supports deployment in **mobile and edge environments**

---

## 🔍 Features

### 🟡 Hazard Detection
- Cigarettes
- Lighters
- Matches
- Solvents
- Spray guns
- Two-pack paints
- Rust removers

### 🟢 Safety Compliance Detection
- Masks
- Gloves
- Goggles
- Gauges
- Compressors
- MSDS documents

---

## 🧠 Key Engineering Highlights

- 🔀 **Multi-dataset training pipeline**
- 🏷️ **Custom class remapping across datasets**
- 🧩 **Automated dataset merging**
- ⚡ **YOLOv8 optimized for edge deployment**
- 📦 **Export to TFLite for Flutter/mobile apps**

---

## 🏗️ Project Structure
ship-safety-ai/
│
├── src/
│ ├── train.py # Training pipeline
│ ├── infer.py # Inference script
│ ├── report.py # Safety analysis logic
│
├── demo/
│ └── app.py # Gradio demo (Hugging Face ready)
│
├── models/ # Trained models (not included)
├── outputs/ # Sample results
├── configs/
├── README.md


---

## ⚙️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- Roboflow datasets
- OpenCV
- TFLite (for mobile deployment)

---

## 📊 Dataset

- Aggregated from multiple sources via Roboflow  
- Custom class remapping applied across datasets  
- Dataset merging pipeline ensures consistency  

> ⚠️ Dataset is not included due to size and licensing constraints.

---

## 🚀 Getting Started

### 1. Clone repository

```bash
git clone https://github.com/yourusername/ship-safety-ai.git
cd ship-safety-ai