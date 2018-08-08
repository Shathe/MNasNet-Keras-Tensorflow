from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing
from tensorflow.keras  import backend as K
import glob
import cv2
import numpy as np


def MNasNet(n_classes=1000, input_shape=(224, 224, 3)):
	inputs = layers.Input(shape=input_shape)

	x = conv_bn(inputs, 32, 3,  padding='same', strides=2)
	x = sepConv_bn_noskip(x, 16, 3, padding='same', strides=1) 
	# MBConv3 3x3
	x = MBConv_idskip(x, 24, 3, padding='same', strides=2, filters_multiplier=3)
	x = MBConv_idskip(x, 24, 3, padding='same', strides=1, filters_multiplier=3)
	x = MBConv_idskip(x, 24, 3, padding='same', strides=1, filters_multiplier=3)
	# MBConv3 5x5
	x = MBConv_idskip(x, 40, 5, padding='same', strides=2, filters_multiplier=3)
	x = MBConv_idskip(x, 40, 5, padding='same', strides=1, filters_multiplier=3)
	x = MBConv_idskip(x, 40, 5, padding='same', strides=1, filters_multiplier=3)
	# MBConv6 5x5
	x = MBConv_idskip(x, 80, 5, padding='same', strides=2, filters_multiplier=6)
	x = MBConv_idskip(x, 80, 5, padding='same', strides=1, filters_multiplier=6)
	x = MBConv_idskip(x, 80, 5, padding='same', strides=1, filters_multiplier=6)
	# MBConv6 3x3
	x = MBConv_idskip(x, 96, 3, padding='same', strides=1, filters_multiplier=6)
	x = MBConv_idskip(x, 96, 3, padding='same', strides=1, filters_multiplier=6)
	# MBConv6 5x5
	x = MBConv_idskip(x, 192, 5, padding='same', strides=2, filters_multiplier=6)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6)
	x = MBConv_idskip(x, 192, 5, padding='same', strides=1, filters_multiplier=6)
	# MBConv6 3x3
	x = MBConv_idskip(x, 320, 3, padding='same', strides=1, filters_multiplier=6)

	x = layers.GlobalAveragePooling2D()(x)
	predictions = layers.Dense(n_classes, activation='softmax')(x)
	return models.Model(inputs=inputs, outputs=predictions)




# Convolution with batch normalization
def conv_bn(x, filters, kernel_size, padding='same', strides=1):

	x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x) # use_bias=False,
	x = layers.BatchNormalization()(x)  
	x = layers.Activation('relu')(x)
	return x

# Depth-wise Separable Convolution with batch normalization 
def sepConv_bn(x, filters, kernel_size, padding='same', strides=1):

	x = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x) # use_bias=False,
	x = layers.BatchNormalization()(x)  
	x = layers.Activation('relu')(x)
	return x

def sepConv_bn_noskip(x, filters, kernel_size, padding='same', strides=1):

	x = sepConv_bn(x, filters, kernel_size, padding=padding, strides=strides)
	x = conv_bn(x, filters, 1, padding=padding, strides=1)

	return x

# Inverted bottleneck block with identity skip connection
def MBConv_idskip(x_input, filters, kernel_size, padding='same', strides=1, filters_multiplier=1):

	x = conv_bn(x_input, filters*filters_multiplier, 1, padding=padding, strides=1)
	x = sepConv_bn(x, filters*filters_multiplier, kernel_size, padding=padding, strides=strides)
	x = conv_bn(x, filters, 1, padding=padding, strides=1)

	# Residual connection if possible
	if x.shape[1] == x_input.shape[1] and x.shape[3] == x_input.shape[3]:
		return  layers.add([x_input, x])
	else: 
		return x


