#  European Mammal and Bird Recognition in Camera Trap Images

## Fork Notes
N.B: This is a fork, which serves to provide you with a hailo compiled version of the EfficientNetV2 from the 2023 paper. 
It does a good job of detecting the 88 species in `models/labels.yaml`.

You will need a hailo accelerator to run the compiled .hef file. On a RPi 5 with hailo8l I was able to get 0.11s per inference, without much time spent playing with compression values etc. This is compoared to ~1s of the full tensorflow model (on RPi 5).

The tflite file is too large to include, ask your favourite LLM to do this step for you if you want access to the tflite.

## Original README continued...

This repository provides models, metric evaluations and the test data sets presented in the papers "Recognizing European mammals and birds in camera trap images using convolutional neural networks" (Schneider et al., 2023) and "Recognition of European mammals and birds in camera trap images using deep neural networks" (Schneider et al., 2024).

The repository is structured as follows:

- `models` contains download script for the best models from our 2023 and 2024 papers and a code snippet to perform predictions with these models.
- `evaluation` contains high-resolution images of the confusion matrices from our papers.
- `data` contains details about our training data sets as well as a download script for our Marburg Open Forest (MOF) and Białowieża National Park (BNP) test data sets.
