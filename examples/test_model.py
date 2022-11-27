import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append('..')
import tensorflow_advanced_segmentation_models as tasm
import tensorflow as tf

N_CLASSES = 2
HEIGHT = 320
WIDTH = 320
BACKBONE_NAME = "resnet50"
WEIGHTS = "imagenet"

# https://www.tensorflow.org/api_docs/python/tf/keras
base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)

BACKBONE_TRAINABLE = True
model = tasm.UNet(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=BACKBONE_TRAINABLE, include_top_conv=False, height=HEIGHT, width=WIDTH)

# 对类进行继承子类化
data = tf.random.normal((2, HEIGHT, WIDTH, 3))
output = model(data)
print(output.shape)
# print(model.summary())

# 函数式API
model2 = model.model()
pass

