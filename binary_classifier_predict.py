import os
import math
import pickle
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import tensorflow_hub as hub

parser = ArgumentParser()
parser.add_argument('--image-path')

args = parser.parse_args()

model = tf.keras.models.load_model('export/mobilenet_finetuned.h5', custom_objects={'KerasLayer': hub.KerasLayer})

img_path = os.path.expanduser(args.image_path)
img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
image = tf.image.resize_with_pad(img, 224, 224)

image_batch = tf.expand_dims(image, axis=0)

prediction = model.predict(image_batch)
print('Prediction: ', prediction[0][0])

prediction_int_label = int(round(prediction[0][0]))

with open('export/idx2class.pkl', 'rb') as f:
    idx2class = pickle.load(f)

print('Prediction Class: ', idx2class[prediction_int_label])