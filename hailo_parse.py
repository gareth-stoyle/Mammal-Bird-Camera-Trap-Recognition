import tensorflow as tf
import os
import numpy as np
import glob
from hailo_sdk_client import ClientRunner
import yaml

chosen_hw_arch = "hailo8l"

model_name = "EfficientNetV2"

tflite_model_path = "models/model2023/EfficientNetV2.tflite"

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_tf_model(tflite_model_path, model_name)

# IMAGES_TO_VISUALIZE = 5

def preproc(image, output_height=300, output_width=300, resize_side=300):
    """ Resize image while maintaining aspect ratio, then center crop to 300x300. """
    h, w = image.shape[:2]
    scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
    
    resized_image = tf.image.resize(image, [int(h * scale), int(w * scale)], method='bilinear')
    cropped_image = tf.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
    
    return cropped_image

def load_and_preprocess_image(metadata, output_height=300, output_width=300, resize_side=300):
    """ Load image, resize while keeping aspect ratio, then crop to 300x300. """
    img = tf.io.read_file(metadata['file'])
    img = tf.io.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)

    # First, apply aspect-preserving resize
    img = preproc(img, output_height=resize_side, output_width=resize_side)

    # If a bounding box is provided, crop using it
    if len(metadata['bbox']) == 0:
        bbox = [0.0, 0.0, 1.0, 1.0]  # Use the full image if no bbox
    else:
        bbox = metadata['bbox']
        bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
        img = tf.image.crop_and_resize([img], [bbox], [0], [output_height, output_width], method='bilinear')[0]

    return img

def get_all_images_with_bboxes(image_dir, yaml_dir, extensions=('.jpg', '.jpeg', '.png')):
    """ Recursively find all image files and match them with bounding boxes from YAML metadata. """
    
    # Step 1: Get all image paths (relative paths for matching)
    image_files = {}
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, image_dir)  # Example: "20210811/79/DSCF0010.JPG"
                relative_path = relative_path.lstrip("img/") if relative_path.startswith("img/") else relative_path
                image_files[relative_path] = full_path
    # print(image_files)
    # Step 2: Parse YAML files and match images
    image_data = []
    yaml_files = glob.glob(os.path.join(yaml_dir, '**', "*.yaml"), recursive=True)

    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'images' in data:
            for entry in data['images']:
                rel_path = entry.get('file', '')  # Example: "img/20210811/79/DSCF0010.JPG"
                
                # Remove 'img/' prefix if present
                # clean_rel_path = rel_path.lstrip("img/") if rel_path.startswith("img/") else rel_path

                # Get bounding boxes
                bboxes = [det['bbox'] for det in entry.get('detections', []) if 'bbox' in det]

                # Match YAML entry with actual image
                if rel_path in image_files:
                    image_data.append({
                        "file": image_files[rel_path],  # Full image path
                        "bbox": bboxes[0] if bboxes else [0, 0, 1, 1]  # Default bbox if missing
                    })

    return image_data

# Directories
image_dir = "data/MOF"
yaml_dir = "data/MOF/md"

# Get 100 images with bounding boxes
images_with_bboxes = get_all_images_with_bboxes(image_dir, yaml_dir)[:100]

calib_dataset = np.zeros((len(images_with_bboxes), 300, 300, 3))

for idx, metadata in enumerate(images_with_bboxes):
    # # Process Image
    img = load_and_preprocess_image(metadata)

    # # Convert to NumPy and Expand Dimensions for Model Inference
    img = np.expand_dims(img.numpy(), axis=0).astype(np.float32)

    calib_dataset[idx, :, :, :] = img


np.save("calib_set_100_float32.npy", calib_dataset)
runner.save_har("EfficientNetV2.har")