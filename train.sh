#!/bin/bash

docker run \
    --rm -it \
    -v "${PWD}":/home/work \
    -w /home/work \
    tensorflow/tensorflow:2.0.0b1-py3 \
    python -m pip install -U tensorflowjs tensorflow_hub --no-warn-conflicts && \
    python binary_classifier_train.py