ARG MODEL_VERSION=1

FROM tensorflow/serving

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME catdog

COPY export/mobilenet_finetuned /models/catdog/$MODEL_VERSION