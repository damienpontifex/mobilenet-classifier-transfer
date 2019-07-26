#!/bin/bash

IMAGE_PATH="$1"

docker run \
    --rm -it \
    -v "${PWD}":/home/work \
    -w /home/work \
    tensorflow/tensorflow:2.0.0b1-py3 \
    python -m pip install -U tensorflow_hub --no-warn-conflicts && \
    python binary_classifier_predict.py --image-path "$IMAGE_PATH"