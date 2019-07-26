ARG MODEL_VERSION=1

FROM tensorflow/serving

# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME catdog

COPY export/mobilenet_finetuned /models/catdog/$MODEL_VERSION