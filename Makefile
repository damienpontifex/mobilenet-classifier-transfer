all:
	@echo "hello world"

train: 
	docker run \
		--rm -it \
		-v "$(CURDIR)":/home/work \
		-w /home/work \
		tensorflow/tensorflow:2.0.0-py3 \
		python -m pip install -U tensorflowjs tensorflow_hub --no-warn-conflicts && \
		python binary_classifier_train.py

predict: 
	docker run \
		--rm -it \
		-v "$(CURDIR)":/home/work \
		-w /home/work \
		tensorflow/tensorflow:2.0.0-py3 \
		python -m pip install -U tensorflow_hub --no-warn-conflicts && \
		python binary_classifier_predict.py --image-path "$(IMAGE_PATH)"

docker-serving: 
	docker build -t mobilenet-classifier:latest .

clean:
	rm -rf export