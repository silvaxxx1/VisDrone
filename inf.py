import argparse
import cv2
import os
import sys
import logging
from typing import Optional
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def setup_logging():
    """Configure logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def process_video(
    model_path: str,
    input_video: str,
    output_path: Optional[str] = None,
    codec: str = 'mp4v',
    conf_threshold: float = 0.25,
    force_headless: bool = False,
    show_labels: bool = True
) -> None:
    """
    Process video with YOLO model and save annotated results
    
    Args:
        model_path: Path to YOLO model weights
        input_video: Path to input video file
        output_path: Path for output video (default: input_video + '_output.mp4')
        codec: Video codec for output
        conf_threshold: Confidence threshold for detections
        force_headless: Disable GUI even if available
        show_labels: Show detection labels on output
    """
    try:
        # Validate paths
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Input video {input_video} not found")
        
        # Automatic output path generation
        if output_path is None:
            input_path = Path(input_video)
            output_path = str(input_path.with_name(f"{input_path.stem}_output{input_path.suffix}"))
        
        # Initialize model
        model = YOLO(model_path)
        model.conf = conf_threshold

        # Video capture setup
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_video}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"Could not create video writer for: {output_path}")

        # GUI detection
        show_gui = False
        if not force_headless:
            try:
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow("Test", dummy)
                cv2.waitKey(1)
                cv2.destroyWindow("Test")
                show_gui = True
            except:
                pass

        # Processing loop
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot(labels=show_labels)

            # Write output
            out.write(annotated_frame)

            # Show progress
            if show_gui:
                cv2.imshow("YOLO Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User interrupted processing")
                    break

            # Progress reporting
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                logging.info(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames} frames)")

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        sys.exit(1)

    finally:
        # Cleanup resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        cv2.destroyAllWindows()
        logging.info(f"\n✅ Processing complete. Output saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser(description="YOLO Video Processing Automation")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights file")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path for output video file")
    parser.add_argument("--codec", type=str, default="mp4v", help="Video codec for output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--headless", action="store_true", help="Force headless mode (no GUI)")
    parser.add_argument("--hide-labels", action="store_true", help="Hide detection labels")

    args = parser.parse_args()

    process_video(
        model_path=args.model,
        input_video=args.input,
        output_path=args.output,
        codec=args.codec,
        conf_threshold=args.conf,
        force_headless=args.headless,
        show_labels=not args.hide_labels
    )



    # how to run this script
    # python inf.py --model Yolo11n.pt --input input.mp4 --output output.mp4 

    # optional arguments 

# --output custom_results  # Custom output directory
# --batch 64              # Larger batch size
# --imgsz 1280            # Higher resolution
# --conf 0.01             # Lower confidence threshold
# --device cuda:1         # Specific GPU selection


    