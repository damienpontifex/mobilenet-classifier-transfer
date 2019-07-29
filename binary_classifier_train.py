#%%
import os
import re
import glob
import pickle
from argparse import ArgumentParser
import logging

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs

l = tf.keras.layers

#%%
parser = ArgumentParser()
parser.add_argument('--data-directory', default='images')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=1, type=int)

args, _ = parser.parse_known_args()

args.data_directory = os.path.expanduser(args.data_directory)

#%%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

#%%
image_size = (224,224)

#%%
def make_dataset(path, batch_size=32):
    classes = [p for p in tf.io.gfile.listdir(path) if '.' not in p]
    class_values = list(range(len(classes)))

    log.info('Binary classifier with classes {}'.format(list(zip(classes, class_values))))

    os.makedirs('export', exist_ok=True)
    with open('export/idx2class.pkl', 'wb') as f:
        pickle.dump(classes, f)
    
    # Key-Value lookup to convert string labels to integer for label model uses
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(classes, class_values), 0)
    
    # List all the files recursively for our dataset
    ds = tf.data.Dataset.list_files(os.path.join(path, '**', '*.jpeg'))
    
    def _path_to_feature(p):
        # Get the example label name from it's parent folder
        label = tf.strings.split(p, os.sep)[-2]
        # Conver the string label to integer label
        label_idx = table.lookup(label)
        
        # Read, decode, resize image for network input
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, *image_size)
        
        return { 'input_image': img }, label_idx
    
    ds = (
        ds.map(_path_to_feature)
            .shuffle(1024)
            .batch(batch_size)
            .repeat(None)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds

#%%
mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

input = l.Input(shape=(*image_size,3), name='input_image')

mobilenet = hub.KerasLayer(mobilenet_url)(input)
logits = l.Dense(units=1, activation=tf.nn.sigmoid, name='prediction')(mobilenet)

model = tf.keras.Model(inputs=input, outputs=logits)

#%%
model.summary()

#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.binary_accuracy]
)

#%%
dataset = make_dataset(os.path.expanduser(args.data_directory), batch_size=args.batch_size)

#%%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./export/logs')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'export/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', 
    save_best_only=True, load_weights_on_restart=True)

callbacks = [tensorboard_callback, checkpoint_callback]

#%%
initial_epoch = 0
previous_checkpoint_path = glob.glob('export/weights.*.hdf5')
if any(previous_checkpoint_path):
    previous_checkpoint_path = sorted(previous_checkpoint_path)[-1]
    initial_epoch = int(re.search('export/weights.(\d*)-.*.hdf5', previous_checkpoint_path).group(1))
    log.info('Restoring model from {} and starting at initial epoch of {}'.format(previous_checkpoint_path, initial_epoch))

#%%
model.fit(
    dataset, 
    epochs=args.epochs, 
    steps_per_epoch=100,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

#%%
tf.saved_model.save(model, 'export/mobilenet_finetuned')

#%%
tfjs.converters.convert_tf_saved_model('export/mobilenet_finetuned', 'export/web_model')

#%%
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serving(input_image):

    # Convert bytes of jpeg input to float32 tensor for model
    def _input_to_feature(img):
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, 224, 224)
        return img
    img = tf.map_fn(_input_to_feature, input_image, dtype=tf.float32)

    # Predict
    predictions = model(img)

    with open('export/idx2class.pkl', 'rb') as f:
        class_names = pickle.load(f)
        class_names = tf.constant(class_names, dtype=tf.string)

    # Single output for model so collapse final axis for vector output
    predictions = tf.squeeze(predictions, axis=-1)

    # Predictions are output from sigmoid so float32 in range 0 -> 1
    # Round to integers for predicted class and string lookup for class name
    prediction_integers = tf.cast(tf.math.round(predictions), tf.int32)
    predicted_classes = tf.map_fn(lambda idx: class_names[idx], prediction_integers, dtype=tf.string)

    # Convert sigmoid output for probability
    # 1 (dog) will remain at logit output
    # 0 (cat) will be 1.0 - logit to give probability
    def to_probability(logit):
        if logit < 0.5:
            return 1.0 - logit
        else:
            return logit
    class_probability = tf.map_fn(to_probability, predictions, dtype=tf.float32)

    return {
        'classes': predicted_classes,
        'probabilities': class_probability
    }

tf.saved_model.save(model, export_dir='export/transformed_for_serving', signatures=serving)

#%%
