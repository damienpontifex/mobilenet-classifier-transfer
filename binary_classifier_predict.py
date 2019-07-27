import os
import math
import pickle
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('--image-path', default='images/cat/iu.jpeg')

args = parser.parse_args()

model = tf.saved_model.load('export/mobilenet_finetuned')

img_path = os.path.expanduser(args.image_path)
img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
image = tf.image.resize_with_pad(img, 224, 224)

image_batch = tf.expand_dims(image, axis=0)

prediction = model(image_batch)
prediction = prediction.numpy()
print('Prediction: ', prediction[0][0])

prediction_int_label = int(round(prediction[0][0]))

with open('export/idx2class.pkl', 'rb') as f:
    idx2class = pickle.load(f)

print('Prediction Class: ', idx2class[prediction_int_label])