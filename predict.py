import numpy as np
import tensorflow as tf
import yaml
import time
import tflite_runtime.interpreter as tflite_runtime  # For Edge TPU

# Enable only one inference mode
hailo = True
tflite = False
tensorflow = False
edgetpu = False

if sum([hailo, tflite, tensorflow, edgetpu]) > 1:
    raise Exception("Only one of hailo, tflite_cpu, edgetpu, or tensorflow can be True")

# Load the correct model based on the chosen mode
if tensorflow:
    model_folder = "models/model2023/EfficientNetV2"
    model = tf.keras.models.load_model(model_folder, compile=False)
elif tflite:
    tflite_model_path = "models/EfficientNetV2.tflite"
    interpreter = tflite_runtime.Interpreter(model_path=tflite_model_path)
elif edgetpu:
    edgetpu_model_path = "models/edgetpu/EfficientNetV2_quantized_edgetpu.tflite"
    interpreter = tflite_runtime.Interpreter(model_path=edgetpu_model_path, 
                                     experimental_delegates=[tflite_runtime.load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1")])

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
elif hailo:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, 
                                FormatType)
    target = VDevice()
    hailo_model_path = "models/EfficientNetV2.hef"
    model = HEF(hailo_model_path)

    # Configure network groups
    configure_params = ConfigureParams.create_from_hef(hef=model, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(model, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # Create input and output virtual streams params
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

    # Get input and output stream information
    input_vstream_info = model.get_input_vstream_infos()[0]
    output_vstream_info = model.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info.shape

with open("models/labels.yaml") as f:
    labels = yaml.load(f, yaml.SafeLoader)

def load_and_crop_image(metadata, image_shape=[300, 300]):
    """
    Loads an image from metadata and crops it to its bounding box
    :param metadata: the image metadata
    :return: the cropped image
    """
    img = tf.io.read_file(metadata['file'])
    img = tf.io.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)

    if len(metadata['bbox']) == 0:
        bbox = [0.0, 0.0, 1.0, 1.0]
    else:
        bbox = metadata['bbox']
        bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]

    cropped_img = tf.image.crop_and_resize([img], [bbox], [0], image_shape, method='bilinear')

    return cropped_img

# Taken from the first entry of `md/species/Sus_scrofa.yaml`
metadata = {
    "file": "data/MOF/img/20211118/02/IMAG0080.JPG",
    "bbox": [0.0871, 0.4609, 0.3328, 0.4593],
}
img = load_and_crop_image(metadata)

# Preprocess image for inference
img = np.expand_dims(img.numpy()[0], axis=0).astype(np.float32)  # Ensure correct shape

# Ensure proper shape & data type for Edge TPU
if edgetpu:
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        img = (img / 255.0 * 255).astype(np.uint8)  # Normalize and convert to uint8
    else:
        img = img.astype(input_dtype)

print(f"[DEBUG]: Image shape before inference: {img.shape}, dtype: {img.dtype}")

start_time = time.time()
if tensorflow:
    print("[DEBUG]: Running inference on Tensorflow")
    preds = model.predict(img)
elif tflite or edgetpu:
    print(f"[DEBUG]: Running inference on {'Edge TPU' if edgetpu else 'TFLite CPU'}")
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
elif hailo:
    print("[DEBUG]: Running inference on Hailo")
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            preds = infer_pipeline.infer(img)['EfficientNetV2/softmax1'][0]

print(f"Prediction time: {time.time() - start_time:.4f} seconds")

print(preds)
index, pred = np.argmax(preds), np.max(preds)
print(f"Class {index}: {pred:.4f} confidence")
