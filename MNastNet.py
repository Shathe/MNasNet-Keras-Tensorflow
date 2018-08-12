from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing
from tensorflow.keras  import backend as K
import glob
import tensorflow as tf
import cv2
import numpy as np


def MNasNet(n_classes=1000, input_shape=(224, 224, 3), alpha=1):
	inputs = layers.Input(shape=input_shape)

	x = conv_bn(inputs, 32, 3,  padding='same', strides=2)
	x = sepConv_bn_noskip(x, 16, 3, padding='same', strides=1) 
	# MBConv3 3x3
	x = MBConv_idskip(x, 24, 3, padding='same', strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, 24, 3, padding='same', strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, 24, 3, padding='same', strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv3 5x5
	
	x = MBConv_idskip(x, 40, 5, padding='same', strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, 40, 5, padding='same', strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, 40, 5, padding='same', strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(x, 80, 5, padding='same', strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 80, 5, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 80, 5, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	x = MBConv_idskip(x, 96, 3, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 96, 3, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(x, 192, 5, padding='same', strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	x = MBConv_idskip(x, 320, 3, padding='same', strides=1, filters_multiplier=6, alpha=alpha)

	# FC + POOL
	x = conv_bn(x,_make_divisible(1152*alpha), 1,  padding='same', strides=1)
	x = layers.GlobalAveragePooling2D()(x)
	predictions = layers.Dense(n_classes, activation='softmax')(x)

	return models.Model(inputs=inputs, outputs=predictions)




# Convolution with batch normalization
def conv_bn(x, filters, kernel_size, padding='same', strides=1, alpha=1, activation=True):

	filters = _make_divisible(filters * alpha)
	x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x) # use_bias=False,
	x = layers.BatchNormalization()(x)  
	if activation:
		x = layers.ReLU(max_value=6)(x)
	return x

# Depth-wise Separable Convolution with batch normalization 
def depthwiseSepConv_bn(x, depth_multiplier, kernel_size, padding='same', strides=1):

	x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, depth_multiplier=depth_multiplier, use_bias=False)(x) # use_bias=False,
	x = layers.BatchNormalization()(x)  
	x = layers.ReLU(max_value=6)(x)
	return x

def sepConv_bn_noskip(x, filters, kernel_size, padding='same', strides=1):

	x = depthwiseSepConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, padding=padding, strides=strides)
	x = conv_bn(x, filters, 1, padding=padding, strides=1)

	return x

# Inverted bottleneck block with identity skip connection
def MBConv_idskip(x_input, filters, kernel_size, padding='same', strides=1, filters_multiplier=1, alpha=1):

	depthwise_conv_filters = _make_divisible(x_input.shape[3].value) 
	pointwise_conv_filters = _make_divisible(filters * alpha)

	x = conv_bn(x_input, depthwise_conv_filters * filters_multiplier, 1, padding=padding, strides=1)
	x = depthwiseSepConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, padding=padding, strides=strides)
	x = conv_bn(x, pointwise_conv_filters, 1, padding=padding, strides=1, activation=False)

	# Residual connection if possible
	if strides==1 and x.shape[3] == x_input.shape[3]:
		return  layers.add([x_input, x])
	else: 
		return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if __name__ == "__main__":

	model = MNasNet(alpha=1)
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	model.summary()
