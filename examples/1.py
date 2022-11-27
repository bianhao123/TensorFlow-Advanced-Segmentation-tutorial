import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K

dataset = tfds.load('caltech_birds2010', download=False, data_dir='/data111/bianhao/wyh/TensorFlow-Advanced-Segmentation-Models/data')
print(dataset)
pass