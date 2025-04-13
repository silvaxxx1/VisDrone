# 🚁 VisDrone Object Detection – Technofest Project

Welcome to the **VisDrone YOLOv11n Inference Demo**, part of our **Technofest Competition Submission**. This project focuses on detecting objects from drone-based video footage using a custom-trained **YOLOv11n** model.

> 🎯 This repository will eventually include the **full pipeline** — from training to inference — for drone-based object detection.

---

## 📂 Current Contents

- `inf.py` – Inference script that processes a drone video, displays live detections, and saves the annotated output.
- `.gitignore` – Excludes large model/video files from version control.

---

## 🚀 How to Use

1. Place your custom-trained model (`Vis.pt`) and input drone video (`demo.MP4`) in the root directory.
2. Run the inference script:
   ```bash
   python inf.py
   ```
3. Output video (`output_demo.mp4`) will be saved in the same directory.

---

## 🛠 Requirements

- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV
- PyTorch (CUDA support recommended)

Install them with:
```bash
pip install ultralytics opencv-python torch
```

---

## 🧠 Model Info

We're using the lightweight and fast **YOLOv11n** model, trained on a drone-perspective dataset (`Vis.pt`), tailored for real-time detection from aerial views.

---

## 🛤 Future Plans

This repository will grow to include:

- 📦 Dataset preprocessing
- 🧑‍🏫 Model training & evaluation
- 🧪 Experiment logging
- 💾 Model export (ONNX, TorchScript)
- 🌐 Deployment options (API, web app, edge device)

---

## 🤝 Credits

This work is part of the **SILVA.AI Lab** project submission for **Technofest 2025**.

---

## 📢 Note

Model weights (`*.pt`) and videos (`*.mp4`) are **excluded** from the repository via `.gitignore` to keep the repo clean.

```

