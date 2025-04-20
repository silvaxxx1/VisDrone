import argparse
import os
import logging
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO
import yaml

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class AutoYOLOValidator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.dataset_dir = Path("datasets") / self.config['dataset']['name']
        self.model = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def download_dataset(self):
        """Download dataset from Roboflow with error handling"""
        try:
            logging.info("Initializing Roboflow connection...")
            rf = Roboflow(api_key=self.config['roboflow']['api_key'])
            
            logging.info(f"Accessing project {self.config['roboflow']['project']}...")
            project = rf.workspace(self.config['roboflow']['workspace']).project(
                self.config['roboflow']['project']
            )
            
            logging.info(f"Downloading version {self.config['dataset']['version']}...")
            version = project.version(self.config['dataset']['version'])
            version.download(
                "yolov11", 
                location=str(self.dataset_dir),
                overwrite=self.config['dataset']['overwrite']
            )
            
            logging.info(f"Dataset downloaded to {self.dataset_dir}")
            return True
        except Exception as e:
            logging.error(f"Dataset download failed: {str(e)}")
            return False

    def validate_model(self):
        """Run full validation pipeline"""
        try:
            # Get dataset config path
            data_yaml = self.dataset_dir / "data.yaml"
            
            # Initialize model
            self.model = YOLO(self.config['model']['path'])
            
            # Run validation
            results = self.model.val(
                data=str(data_yaml),
                batch=self.config['validation']['batch_size'],
                imgsz=self.config['validation']['imgsz'],
                conf=self.config['validation']['conf_threshold'],
                iou=self.config['validation']['iou_threshold'],
                device=self.config['hardware']['device'],
                plots=True,
                save_json=True
            )
            
            logging.info("\nValidation Metrics:")
            logging.info(f"mAP@0.5: {results.box.map50:.4f}")
            logging.info(f"mAP@0.5-0.95: {results.box.map:.4f}")
            logging.info(f"Precision: {results.box.p:.4f}")
            logging.info(f"Recall: {results.box.r:.4f}")
            
            return True
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description="AutoYOLO Validator")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--api-key', help='Roboflow API key (overrides config)')
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    
    validator = AutoYOLOValidator(args.config)
    
    # Override API key if provided
    if args.api_key:
        validator.config['roboflow']['api_key'] = args.api_key
    
    if validator.download_dataset():
        if validator.validate_model():
            logging.info("✅ Full pipeline completed successfully!")
        else:
            logging.error("❌ Validation failed after successful download")
    else:
        logging.error("❌ Dataset download failed, aborting validation") 



# to run with default config 
# python data.py --api-key YOUR_ROBOFLOW_API_KEY

# to run with custom config
# python validate.py --config custom_config.yaml