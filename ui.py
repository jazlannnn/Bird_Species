import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2

print("")
print("x")

import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential

import tempfile
from matplotlib import pyplot as plt 

batch_size = 32 
img_height = 300 
img_width = 300 
data_dir="Marine" 
train_ds = tf.keras.preprocessing.image_dataset_from_directory( 
    data_dir, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=(img_height, img_width),
    batch_size=batch_size)