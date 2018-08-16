import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing, regularizers, activations
from tensorflow.keras  import backend as K
import tensorflow.contrib.eager as tfe
# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)


# constants
image_size = 28
batch_size = 128
epochs = 20
num_classes = 10

# Get dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshape images
x_train = x_train.reshape(-1, image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(-1, image_size, image_size, 1).astype('float32')

# We are normalizing the images to the range of [-1, 1]
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()


print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

n_elements_train = x_train.shape[0]
n_elements_test = x_test.shape[0]
dset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe)).shuffle(n_elements_train).batch(batch_size)
dset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test_ohe)).shuffle(n_elements_test).batch(batch_size)



class Mnasnet(tf.keras.Model):
	def __init__(self, num_classes,  alpha=1, **kwargs):
		super(Mnasnet, self).__init__(**kwargs)
		self.blocks = []

		self.conv_initial = conv(filters=32*alpha, kernel_size=3, strides=2)
		self.bn_initial = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		# depthwise
		self.conv1_block1 = depthwiseConv(depth_multiplier=1, kernel_size=3, strides=1)
		self.bn1_block1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		# conv bottleneck

		self.conv2_block1 = conv(filters=16*alpha, kernel_size=1, strides=1)
		self.bn2_block1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		# MBConv3 3x3
		block1_part1 = MBConv_idskip(input_filters=16*alpha, filters=24, kernel_size=3, strides=2, filters_multiplier=3, alpha=alpha)
		self.register_block(block1_part1, 'block1_part1')
		block1_part2 = MBConv_idskip(input_filters=24*alpha, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
		self.register_block(block1_part2, 'block1_part2')
		block1_part3 = MBConv_idskip(input_filters=24*alpha, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
		self.register_block(block1_part3, 'block1_part3')

		# MBConv3 5x5
		block2_part1 = MBConv_idskip(input_filters=24*alpha, filters=40, kernel_size=5, strides=2, filters_multiplier=3, alpha=alpha)
		self.register_block(block2_part1, 'block2_part1')
		block2_part2 = MBConv_idskip(input_filters=40*alpha, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
		self.register_block(block2_part2, 'block2_part2')
		block2_part3 = MBConv_idskip(input_filters=40*alpha, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
		self.register_block(block2_part3, 'block2_part3')
		# MBConv6 5x5
		block3_part1 = MBConv_idskip(input_filters=40*alpha, filters=80, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
		self.register_block(block3_part1, 'block3_part1')
		block3_part2 = MBConv_idskip(input_filters=80*alpha, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block3_part2, 'block3_part2')
		block3_part3 = MBConv_idskip(input_filters=80*alpha, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block3_part3, 'block3_part3')

		# MBConv6 3x3
		block4_part1 = MBConv_idskip(input_filters=80*alpha, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block4_part1, 'block4_part1')
		block4_part2 = MBConv_idskip(input_filters=96*alpha, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block4_part2, 'block4_part2')
		# MBConv6 5x5
		block5_part1 = MBConv_idskip(input_filters=96*alpha, filters=192, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
		self.register_block(block5_part1, 'block5_part1')
		block5_part2 = MBConv_idskip(input_filters=192*alpha, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block5_part2, 'block5_part2')
		block5_part3 = MBConv_idskip(input_filters=192*alpha, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block5_part3, 'block5_part3')
		block5_part4 = MBConv_idskip(input_filters=192*alpha, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block5_part4, 'block5_part4')
		# MBConv6 3x3
		block6_part1 = MBConv_idskip(input_filters=192*alpha, filters=320, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
		self.register_block(block6_part1, 'block6_part1')

		self.conv_last = conv(filters=1152*alpha, kernel_size=1, strides=1)
		self.bn_last = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		self.avg_pool =  layers.GlobalAveragePooling2D()
		self.fc = layers.Dense(num_classes)


	def register_block(self, block, name):
		#setattr(self, name, block)
		self.blocks.append(block)



	def call(self, inputs, training=None, mask=None):
		out = self.conv_initial(inputs)
		out = self.bn_initial(out, training=training)
		out = tf.nn.relu(out)

		out = self.conv1_block1(out)
		out = self.bn1_block1(out, training=training)
		out = tf.nn.relu(out)

		out = self.conv2_block1(out)
		out = self.bn2_block1(out, training=training)
		out = tf.nn.relu(out)


		# forward pass through all the blocks
		# build all the blocks
		for block in self.blocks:
			out = block(out, training=training)

		out = self.conv_last(out)
		out = self.bn_last(out, training=training)
		out = tf.nn.relu(out)

		out = self.avg_pool(out)
		out = self.fc(out)


		
		# softmax op does not exist on the gpu, so always use cpu
		output = tf.nn.softmax(out)
		
		return output




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


# convolution
def conv(filters, kernel_size, strides=1):
	return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
								kernel_regularizer=regularizers.l2(l=0.0003))
# convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1):
	return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
								padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))

class MBConv_idskip(tf.keras.Model):

	def __init__(self, input_filters, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1):
		super(MBConv_idskip, self).__init__()

		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.filters_multiplier = filters_multiplier
		self.alpha = alpha

		self.depthwise_conv_filters = _make_divisible(input_filters) 
		self.pointwise_conv_filters = _make_divisible(filters * alpha)

		#conv1
		self.conv1 = conv(filters=self.depthwise_conv_filters*filters_multiplier, kernel_size=1, strides=1)
		self.bn1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		#depthwiseconv2
		self.conv2 = depthwiseConv(depth_multiplier=1, kernel_size=kernel_size, strides=strides)
		self.bn2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)

		#conv3
		self.conv3 = conv(filters=self.pointwise_conv_filters, kernel_size=1, strides=1)
		self.bn3 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)


	def call(self, inputs, training=None):

		x = self.conv1(inputs)
		x = self.bn1(x, training=training)
		x = tf.nn.relu(x)

		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x = tf.nn.relu(x)

		x = self.conv3(x)
		x = self.bn3(x, training=training)

		if self.strides==1 and x.shape[3] == inputs.shape[3]:
			return  layers.add([inputs, x])
		else: 
			return x



def loss_function(model, x, y, training=True):
	y_ = model(x, training=training)
	loss = tf.losses.softmax_cross_entropy(y, y_)
	print(loss)
	return loss


def get_params(model):
	total_parameters = 0
	for variable in model.variables:
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1

		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")


if __name__ == "__main__":


	# build model and optimizer
	model = Mnasnet(num_classes=10)

	# Initialize the metric
	accuracy = tfe.metrics.Accuracy()
 
 	# optimizer
	optimizer = tf.train.AdamOptimizer(0.001)

	
	# OPTION 1 TRAIN
	for epoch in xrange(epochs):
		print('epoch: '+ str(epoch))
		for x, y in dset_train: # for every batch
			with tf.GradientTape() as g:
				y_ = model(x, training=True)
				loss = tf.losses.softmax_cross_entropy(y, y_)
				print('training loss: ' + str(loss.numpy()))

			grads = g.gradient(loss, model.variables)
			optimizer.apply_gradients(zip(grads, model.variables))


	# EVALUATION
	for x, y in dset_train: # for every batch
		y_ = model(x, training=False)
		accuracy(tf.argmax(y, 1), tf.argmax(y_, 1))

	# Print the accuracy
	print('Accuracy: ' + str(accuracy.result().numpy()))
	get_params(model)


	

	# OPTION 2 TRAIN
	'''
	for epoch in xrange(epochs):
		print('epoch: '+ str(epoch))
		for x, y in dset_train: # for every batch
			optimizer.minimize(lambda: loss_function(model, x, y, training=True))
	'''


	
	# OTHER OPTION TRAIN AND TEST: COMMON KERAS CODE
	'''
	model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

	# TF Keras tries to use entire dataset to determine shape without this step when using .fit()
	# Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
	dummy_x = tf.zeros((1, image_size, image_size, 1))
	model._set_inputs(dummy_x)
	#model.summary()

	# train
	model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
			  validation_data=(x_test, y_test_ohe), verbose=1)

	# evaluate on test set
	scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
	print("Final test loss and accuracy :", scores)

	
	saver = tf.Saver(model.variables)
	#saver = tfe.Saver(model.variables)
	saver.save('weights.ckpt')
	'''
