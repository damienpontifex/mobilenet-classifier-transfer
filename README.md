# Mobilenet Classifier by transfer learning

## Dependencies

```bash
python -m pip install -U tensorflow==2.0.0-beta1 tensorflow_hub
```

## Data
Setup data under a directory named 'images' with the two categories as folders themselves. i.e.

- images
    - cat
        - cat1.jpeg
        - cat2.jpeg
    - dog
        - dog1.jpeg
        - dog2.jpeg

## Train

To run inside the TF20 docker container
```bash
python binary_classifier_train.py --data-directory images
```

## Predict

```bash
python binary_classifier_predict.py --image-path <your-image>.jpeg
```

## Build TF Serving container

```bash
docker build -t <username>/myclassifier -f Serving.Dockerfile .
```

TODO: provide REST and gRPC examples calling server in container

## Use in TensorFlow.js

Run webserver in repo root to serve index.html
```bash
python3 -m http.server
```
TODO: Still to do predictions inside TF.js