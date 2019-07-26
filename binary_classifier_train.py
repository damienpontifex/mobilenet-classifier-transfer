# %%
import os
import re
import glob
import pickle
from argparse import ArgumentParser
import logging

import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs

l = tf.keras.layers

# %%
parser = ArgumentParser()
parser.add_argument('--data-directory', default='images')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=1, type=int)

args, _ = parser.parse_known_args()

args.data_directory = os.path.expanduser(args.data_directory)

# %%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# %%
image_size = (224,224)

# %%
def make_dataset(path, batch_size=32):
    classes = [p for p in os.listdir(path) if '.' not in p]
    class_values = list(range(len(classes)))

    log.info('Binary classifier with classes {}'.format(list(zip(classes, class_values))))

    with open('export/idx2class.pkl', 'wb') as f:
        pickle.dump(classes, f)
    
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(classes, class_values), 0)
    
    ds = tf.data.Dataset.list_files(os.path.join(path, '**', '*.jpeg'))
    
    def _path_to_feature(p):
        label = tf.strings.split(p, os.sep)[-2]
        label_idx = table.lookup(label)
        
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, *image_size)
        
        return { 'input_image': img }, label_idx
    
    ds = ds.map(_path_to_feature).shuffle(1024).batch(batch_size).repeat(None).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# %%
mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

input = l.Input(shape=(*image_size,3), name='input_image')

mobilenet = hub.KerasLayer(mobilenet_url)(input)
logits = l.Dense(units=1, activation=tf.nn.sigmoid, name='prediction')(mobilenet)

model = tf.keras.Model(inputs=input, outputs=logits)

# %%
model.summary()

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.binary_accuracy]
)

# %%
dataset = make_dataset(os.path.expanduser(args.data_directory), batch_size=args.batch_size)

# %%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./export/logs')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'export/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', 
    save_best_only=True, load_weights_on_restart=True)

callbacks = [tensorboard_callback, checkpoint_callback]

# %%
initial_epoch = 0
previous_checkpoint_path = glob.glob('export/weights.*.hdf5')
if any(previous_checkpoint_path):
    previous_checkpoint_path = sorted(previous_checkpoint_path)[-1]
    initial_epoch = int(re.search('export/weights.(\d*)-.*.hdf5', previous_checkpoint_path).group(1))
    log.info('Restoring model from {} and starting at initial epoch of {}'.format(previous_checkpoint_path, initial_epoch))

# %%
model.fit(
    dataset, 
    epochs=args.epochs, 
    steps_per_epoch=100,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

# %%
os.makedirs('export', exist_ok=True)

model_filename = 'export/mobilenet_finetuned.h5'
if os.path.isfile(model_filename):
    os.remove(model_filename)
model.save(model_filename)

# %%
tfjs.converters.save_keras_model(model, 'export/tfjs_export')
