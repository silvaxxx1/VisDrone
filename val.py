import argparse
import os
import logging
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Dict

def setup_logging():
    """Configure logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def validate_paths(model_path: str, data_config: str):
    """Validate input file paths"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found")
    if not Path(data_config).exists():
        raise FileNotFoundError(f"Data config file {data_config} not found")

def plot_metrics(results, output_dir: str) -> None:
    """Plot comprehensive validation metrics"""
    try:
        plt.figure(figsize=(15, 10))
        results_dir = results.save_dir

        # Precision-Recall Curve
        plt.subplot(2, 2, 1)
        pr_curve = plt.imread(Path(results_dir)/'PR_curve.png')
        plt.imshow(pr_curve)
        plt.title('Precision-Recall Curve')
        plt.axis('off')

        # Confusion Matrix
        plt.subplot(2, 2, 2)
        confusion_matrix = plt.imread(Path(results_dir)/'confusion_matrix.png')
        plt.imshow(confusion_matrix)
        plt.title('Confusion Matrix')
        plt.axis('off')

        # F1 Curve
        plt.subplot(2, 2, 3)
        f1_curve = plt.imread(Path(results_dir)/'F1_curve.png')
        plt.imshow(f1_curve)
        plt.title('F1 Score Curve')
        plt.axis('off')

        # Detection Examples
        plt.subplot(2, 2, 4)
        example_img = plt.imread(Path(results_dir)/'val_batch0_pred.jpg')
        plt.imshow(example_img)
        plt.title('Example Detections')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(Path(output_dir)/'combined_metrics.png')
        plt.close()
        logging.info("Saved combined metrics plot")
        
    except Exception as e:
        logging.error(f"Failed to generate metrics plots: {str(e)}")

def plot_class_distribution(results, output_dir: str) -> None:
    """Plot class distribution and performance"""
    try:
        plt.figure(figsize=(12, 6))
        results_dir = results.save_dir

        # Class distribution
        plt.subplot(1, 2, 1)
        labels_img = plt.imread(Path(results_dir)/'labels.jpg')
        plt.imshow(labels_img)
        plt.title('Class Distribution')
        plt.axis('off')

        # Class-wise AP
        plt.subplot(1, 2, 2)
        classes = list(results.box.ap_class_index)
        ap_values = [results.box.ap[i] for i in classes]
        plt.barh([str(c) for c in classes], ap_values)
        plt.title('Class-wise AP@0.5:0.95')
        plt.xlabel('Average Precision')
        plt.ylabel('Class ID')
        plt.tight_layout()
        plt.savefig(Path(output_dir)/'class_performance.png')
        plt.close()
        logging.info("Saved class performance plots")
        
    except Exception as e:
        logging.error(f"Failed to generate class plots: {str(e)}")

def save_metrics(results, output_dir: str) -> None:
    """Save validation metrics to file"""
    try:
        metrics = {
            'mAP50': results.box.map50,
            'mAP': results.box.map,
            'precision': results.box.p,
            'recall': results.box.r,
            'f1': 2 * ((results.box.p * results.box.r) / (results.box.p + results.box.r + 1e-16))
        }

        # Save to TXT
        with open(Path(output_dir)/'metrics.txt', 'w') as f:
            f.write("Validation Metrics:\n")
            f.write(f"mAP@0.5:       {metrics['mAP50']:.4f}\n")
            f.write(f"mAP@0.5-0.95: {metrics['mAP']:.4f}\n")
            f.write(f"Precision:     {metrics['precision']:.4f}\n")
            f.write(f"Recall:        {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:      {metrics['f1']:.4f}\n")

        # Save to JSON
        import json
        with open(Path(output_dir)/'metrics.json', 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        logging.info("Metrics saved in TXT and JSON formats")

    except Exception as e:
        logging.error(f"Failed to save metrics: {str(e)}")

def run_validation(
    model_path: str,
    data_config: str,
    output_dir: str,
    batch_size: int = 16,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    device: Optional[str] = None
) -> None:
    """Run complete validation pipeline"""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if not specified
        if not device:
            device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

        # Load model
        model = YOLO(model_path)

        # Validation configuration
        val_config = {
            'data': data_config,
            'batch': batch_size,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
            'device': device,
            'save_json': True,
            'save_hybrid': True,
            'plots': True,
            'rect': True,
            'name': output_dir  # Save results to specified directory
        }

        # Run validation
        results = model.val(**val_config)

        # Save metrics and generate plots
        save_metrics(results, output_dir)
        plot_metrics(results, output_dir)
        plot_class_distribution(results, output_dir)

        logging.info(f"\nValidation complete! Results saved to: {os.path.abspath(output_dir)}")

    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        raise

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Model Validation Pipeline')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data config YAML')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IOU threshold')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    
    try:
        validate_paths(args.model, args.data)
        run_validation(
            model_path=args.model,
            data_config=args.data,
            output_dir=args.output,
            batch_size=args.batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()


# EXAMPLE USAGE 

# python val.py --model Yolo11m.pt --data dataset.yaml