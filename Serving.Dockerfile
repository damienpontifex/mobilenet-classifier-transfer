FROM tensorflow/serving

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME catdog

COPY export/transformed_for_serving /models/catdog/1
