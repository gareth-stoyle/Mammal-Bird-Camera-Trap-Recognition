import numpy as np
import tensorflow as tf
import yaml
import time

# Enable only one inference mode
hailo = True
tflite = False
tensorflow = False

if sum([hailo, tflite, tensorflow]) > 1:
    raise Exception("Only one of hailo, tflite, or tensorflow can be True")

# Load the correct model based on the chosen mode
if tensorflow:
    model_folder = "models/model2023/EfficientNetV2"
    model = tf.keras.models.load_model(model_folder, compile=False)
elif tflite:
    tflite_model_path = "models/model2023/EfficientNetV2.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
elif hailo:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, 
                                FormatType)
    target = VDevice()
    hailo_model_path = "models/model2023/EfficientNetV2_quantized.hef"
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


# Metadatas for MOF dataset
# metadatas = [{"file": "data/MOF/img/20211118/02/IMAG0080.JPG","bbox": [0.0871, 0.4609, 0.3328, 0.4593]}, {'file': 'data/MOF/img/20211118/02/IMAG0157.JPG', 'bbox': [0.5921, 0.6796, 0.264, 0.127]}, {'file': 'data/MOF/img/20211118/46/DSCF0270.JPG', 'bbox': [0.4521, 0.6934, 0.5478, 0.2644]}, {'file': 'data/MOF/img/20211202/36/IMAG0190.JPG', 'bbox': [0.0, 0.7604, 0.2058, 0.1369]}, {'file': 'data/MOF/img/20211202/36/IMAG0169.JPG', 'bbox': [0.02734, 0.4328, 0.1597, 0.1307]}, {'file': 'data/MOF/img/20211202/36/IMAG0027.JPG', 'bbox': [0.0, 0.5677, 0.08085, 0.07968]}, {'file': 'data/MOF/img/20220224/11/IMAG0236.JPG', 'bbox': [0.105, 0.7531, 0.1417, 0.2145]}, {'file': 'data/MOF/img/20211118/11/IMAG0107.JPG', 'bbox': [0.09375, 0.7463, 0.2421, 0.2218]}, {'file': 'data/MOF/img/20220413/11/IMAG0089.JPG', 'bbox': [0.2429, 0.5416, 0.08671, 0.09427]}, {'file': 'data/MOF/img/20220413/48/IMAG0102.JPG', 'bbox': [0.9484, 0.5395, 0.05156, 0.08229]}, {'file': 'data/MOF/img/20211202/22/IMAG0183.JPG', 'bbox': [0.5769, 0.3166, 0.148, 0.2937]}, {'file': 'data/MOF/img/20211202/36/IMAG0159.JPG', 'bbox': [0.4, 0.7776, 0.1281, 0.1213]}, {'file': 'data/MOF/img/20220413/48/IMAG0154.JPG', 'bbox': [0.641, 0.801, 0.1644, 0.1479]}, {'file': 'data/MOF/img/20211202/36/IMAG0170.JPG', 'bbox': [0.09882, 0.4104, 0.1066, 0.1557]}, {'file': 'data/MOF/img/20220120/11/IMAG0076.JPG', 'bbox': [0.1187, 0.5473, 0.2492, 0.1812]}, {'file': 'data/MOF/img/20220412/50/IMAG0035.JPG', 'bbox': [0.8945, 0.5958, 0.1054, 0.1177]}, {'file': 'data/MOF/img/20211202/22/IMAG0179.JPG', 'bbox': [0.4921, 0.552, 0.5078, 0.4156]}, {'file': 'data/MOF/img/20220413/48/IMAG0101.JPG', 'bbox': [0.9664, 0.5463, 0.03359, 0.07343]}, {'file': 'data/MOF/img/20211202/22/IMAG0182.JPG', 'bbox': [0.5664, 0.3145, 0.1542, 0.288]}, {'file': 'data/MOF/img/20211202/22/IMAG0100.JPG', 'bbox': [0.7371, 0.4697, 0.1066, 0.1791]}, {'file': 'data/MOF/img/20211220/02/IMAG0026.JPG', 'bbox': [0.08203, 0.6026, 0.3148, 0.1953]}]

# A few sample images I found online.
metadatas = [
    {"file": "test_images/badger.png","bbox": []},
    {"file": "test_images/fox1.png","bbox": []},
    {"file": "test_images/fox2.png","bbox": []},
    {"file": "test_images/deer.png","bbox": []},
    {"file": "test_images/marten.png","bbox": []},
    {"file": "test_images/song_thrush.png","bbox": []}
]

for i, metadata in enumerate(metadatas):
    img = load_and_crop_image(metadata)

    # Preprocess image for inference
    img = np.expand_dims(img.numpy()[0], axis=0).astype(np.float32)  # Ensure correct shape

    start_time = time.time()
    if tensorflow:
        print("[DEBUG]: Running inference on Tensorflow")
        preds = model.predict(img)
    elif tflite:
        print(f"[DEBUG]: Running inference on tflite")
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
    elif hailo:
        print("[DEBUG]: Running inference on Hailo")
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                preds = infer_pipeline.infer(img)['EfficientNetV2/softmax1'][0]

    print(f"Prediction time: {time.time() - start_time:.4f} seconds")

    index, pred = np.argmax(preds), np.max(preds)
    pred = (pred / 256) * 100
    print(f"{i} - Class {index}: {pred:.4f} confidence")
