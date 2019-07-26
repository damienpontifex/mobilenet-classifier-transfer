# %%
import urllib
import json
from PIL import Image
import numpy as np
import tensorflow as tf
# %%
img = tf.io.read_file('images/cat/iu-1.jpeg')
img = tf.image.decode_jpeg(img)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize_with_pad(img, 224, 224)

# %%
body = {
    'instances': [
        { 'input_image': img.numpy().tolist() }
    ]
}
# %%
data = json.dumps(body)
# %%
req = urllib.request.Request(
    'http://localhost:8501/v1/models/catdog:predict',
    data=data.encode(),
    method='POST'
)
# %%
res = urllib.request.urlopen(req)
res = res.read().decode()
print(res)

#%%
