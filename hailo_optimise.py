from hailo_sdk_client import ClientRunner
import numpy as np
import gc

# load files created from hailo_parse.py
chosen_hw_arch = "hailo8l"
model_name = "EfficientNetV2.har"
runner = ClientRunner(hw_arch=chosen_hw_arch, har=model_name)
calib_dataset = np.load("calib_set_100_float32.npy")

# Now we will create a model script
# Batch size is 8 by default
# Load the model script to ClientRunner so it will be considered on optimization
# these numbers can be played with...
runner.load_model_script("model_optimization_flavor(optimization_level=2, compression_level=0, batch_size=1)")
# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)

# desperate attempts to not run out of RAM...
del calib_dataset
gc.collect()

# Save the result state to a Quantized HAR file
quantized_model_har_path = f"EfficientNetV2_quantized.har"
runner.save_har(quantized_model_har_path)

# At this point you can run via cli: `hailo compile --hw-arch hailo8l EfficientNetV2_quantized.har`
# to get your .hef file for inferencing on hailo architecture.
