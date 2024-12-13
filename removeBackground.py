import argparse
from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='Device to use for computation: "cpu" or "gpu"')
    parser.add_argument('--conf', type=float, default=0.3, 
                        help='Confidence threshold for detections (default: 0.3).')
    parser.add_argument('--threads', type=int, default=8, 
                        help='Number of threads for CPU computations. Default is None, which uses all available threads.')
    return parser.parse_args()

# Load arguments
args = parse_arguments()
confidence_threshold = args.conf
num_threads = args.threads

if num_threads is not None:
    torch.set_num_threads(num_threads)

device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = YOLO('yolov8x-seg')
model.to(device)

def get_next_output_dir(base_path):
    i = 1
    while True:
        output_path = Path(base_path) / f'predict{i if i > 1 else ""}'
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
        i += 1


input_base_dir = Path('objectDetectionSource')
os.makedirs(input_base_dir, exist_ok=True)
output_path = get_next_output_dir('objectDetectionResults/')
stats_path = output_path / 'stats'
stats_path.mkdir(parents=True, exist_ok=True)

# Function to process images and save only masks if --presource is used
def process_images(input_dir, output_dir, confidence_threshold):
    # Iterate through subdirectories (Ok and Damaged) only if --presource is used
    for sub_dir_name in ['Ok', 'Damaged']:
        sub_dir = input_dir / sub_dir_name
        if not sub_dir.exists():
            print(f"Subdirectory does not exist: {sub_dir}")
            continue
        
        # Create corresponding output directory
        sub_output_dir = output_dir / sub_dir_name
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_name in os.listdir(sub_dir):
            img_path = sub_dir / img_name
            print(f"Processing image: {img_path}")
            
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                results = model(img_path, conf=confidence_threshold)
                print(f"Results for {img_name}: {results}")

                if results:
                    for result in results:
                        masks = result.masks
                        det = result.boxes

                        if masks is not None and len(masks.data) > 0:
                            print(f"Masks found for {img_name}: {len(masks.data)}")
                            largest_area = 0
                            largest_mask = None
                            largest_bbox = None

                            for j in range(len(det)):
                                bbox = det[j].xyxy.cpu().numpy().flatten()
                                mask = masks.data[j].cpu().numpy()

                                if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[0] == 1):
                                    if mask.ndim == 3:
                                        mask = mask.squeeze(0)
                                    area = np.sum(mask)
                                else:
                                    print(f"Unexpected mask dimensions for {img_name}: {mask.shape}")
                                    continue

                                if area > largest_area:
                                    largest_area = area
                                    largest_mask = mask
                                    largest_bbox = bbox

                            if largest_mask is not None and len(largest_bbox) == 4:
                                original_img = Image.open(img_path)

                                if largest_mask.shape[:2] != original_img.size[::-1]:
                                    largest_mask = np.array(
                                        Image.fromarray(largest_mask.astype(np.uint8)).resize(original_img.size, Image.NEAREST)
                                    )

                                largest_mask = (largest_mask > 0).astype(bool)

                                x_min, y_min, x_max, y_max = map(int, largest_bbox)

                                cropped_img = np.array(original_img)[y_min:y_max, x_min:x_max]
                                cropped_mask = largest_mask[y_min:y_max, x_min:x_max]

                                new_img = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
                                new_img[..., :3] = cropped_img
                                new_img[..., 3] = cropped_mask.astype(np.uint8) * 255

                                # Save only the mask
                                mask_file = sub_output_dir / f'{Path(img_name).stem}_C.png'  # Convert img_name to Path
                                Image.fromarray(new_img).save(mask_file)                
                        else:
                            print(f"No masks found for {img_name}.")

# Process images based on the input directory
process_images(input_base_dir, output_path, confidence_threshold)
