# Technofest 2025 - Drone Object Detection

# 🚀 YOLOv11 Object Detection Pipeline

A complete framework for drone-based object detection featuring dataset management, model validation, and video inference using YOLOv11 models.

![YOLO Inference Demo](assets/demo.gif) *Example inference output*

## 📂 Project Structure

### Core Scripts
- `data.py` - Automated dataset download from Roboflow
- `val.py` - Model validation with metrics & visualizations
- `inf.py` - Video inference with real-time detection
- `config.yaml` - Central configuration file

### Supporting Files
- `.gitignore` - File exclusion rules
- `requirements.txt` - Python dependencies

## 🛠 Features

- **End-to-End Pipeline**
  - Dataset download & preparation
  - Model validation with comprehensive metrics
  - Video inference with GPU acceleration
  - Automated report generation

- **Advanced Visualization**
  - Precision-Recall curves
  - Confusion matrices
  - Class distribution analysis
  - Detection examples

- **Flexible Configuration**
  - Central YAML configuration
  - CLI parameter overrides
  - Automatic device detection

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- Roboflow API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drone-object-detection.git
cd drone-object-detection
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

### 📦 Requirements

```txt
# Core Requirements
ultralytics>=8.0.0  # YOLO framework
torch>=2.0.0        # PyTorch base
opencv-python>=4.7.0  # Video processing

# Data Management
roboflow>=1.1.0     # Dataset handling
pyyaml>=6.0         # Config management

# Visualization
matplotlib>=3.7.0   # Metrics plotting
seaborn>=0.12.2     # Enhanced visuals

# Utilities
numpy>=1.24.0       # Array processing
tqdm>=4.65.0        # Progress tracking
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset Management (data.py)

### Usage
```bash
# Basic usage
python data.py --api-key YOUR_ROBOFLOW_KEY

# Custom config
python data.py --config custom_config.yaml
```

### Config.yaml Structure
```yaml
roboflow:
  api_key: "your_key"
  workspace: "enes-v8muh"
  project: "my-first-project-lk9ki"

dataset:
  name: "drone-dataset"
  version: 1
  overwrite: False

model:
  path: "Yolo11m.pt"

validation:
  batch_size: 16
  imgsz: 640
  conf_threshold: 0.001
```

## 🧪 Model Validation (val.py)

### Usage
```bash
# Basic validation
python val.py --model Yolo11m.pt --data dataset.yaml

# Advanced options
python val.py \
  --model Yolo11m.pt \
  --data dataset.yaml \
  --output validation_results \
  --batch 32 \
  --device cuda
```

### Outputs
- `metrics.json` - Quantitative results
- `combined_metrics.png` - Visual summary
- `class_performance.png` - Per-class AP

## 📹 Video Inference (inf.py)

### Usage
```bash
# Basic inference
python inf.py --model Yolo11n.pt --input input.mp4

# Advanced options
python inf.py \
  --model Yolo11n.pt \
  --input input.mp4 \
  --output results/output.mp4 \
  --conf 0.4 \
  --headless
```

### Features
- Real-time display (GUI/headless)
- Multiple codec support
- Progress tracking
- Batch processing

## ⚙️ Configuration Management

### Key Config Parameters
| Parameter          | Description                     | Default      |
|--------------------|---------------------------------|--------------|
| `validation.batch` | Batch size for validation       | 16           |
| `validation.imgsz` | Input image resolution          | 640          |
| `hardware.device`  | Processing device               | auto-detect  |
| `dataset.overwrite`| Overwrite existing datasets     | False        |

## 📈 Performance Metrics

| Metric            | Description                     |
|-------------------|---------------------------------|
| mAP@0.5           | Mean Average Precision @ IoU=0.5 |
| mAP@0.5-0.95      | mAP across different IoU thresholds |
| F1 Score          | Balance between precision/recall |

## 📝 Notes

- Model weights (`*.pt`) and media files are excluded via `.gitignore`
- Use `--headless` flag for server environments
- JSON metrics enable programmatic analysis
- Roboflow API key can be passed via CLI or config

## 🤝 Credits

Developed by **SILVA.AI Lab** for Technofest 2025  
Dataset powered by Roboflow  
Detection framework: Ultralytics YOLO

