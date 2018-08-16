import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import MnasnetEager

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

# Define the loss function
def loss_function(model, x, y, training=True):
	y_ = model(x, training=training)
	loss = tf.losses.softmax_cross_entropy(y, y_)
	print(loss)
	return loss

# Prints the number of parameters of a model
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


# Trains the model for certains epochs on a dataset
def train(dset_train, model, epochs=5):
	for epoch in xrange(epochs):
		print('epoch: '+ str(epoch))
		for x, y in dset_train: # for every batch
			with tf.GradientTape() as g:
				y_ = model(x, training=True)
				loss = tf.losses.softmax_cross_entropy(y, y_)
				print('Training loss: ' + str(loss.numpy()))

			# Gets gradients and applies them
			grads = g.gradient(loss, model.variables)
			optimizer.apply_gradients(zip(grads, model.variables))

# Tests the model on a dataset
def test(dset_test, model):
	for x, y in dset_test: # for every batch
		y_ = model(x, training=False)
		accuracy(tf.argmax(y, 1), tf.argmax(y_, 1))

	# Print the accuracy
	print('Accuracy: ' + str(accuracy.result().numpy()))
	get_params(model)

if __name__ == "__main__":


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

	# Onehot: from 28,28 to 28,28,n_classes
	y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
	y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()


	print('x train', x_train.shape)
	print('y train', y_train_ohe.shape)
	print('x test', x_test.shape)
	print('y test', y_test_ohe.shape)

	# Creates the tf.Dataset
	n_elements_train = x_train.shape[0]
	n_elements_test = x_test.shape[0]
	dset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe)).shuffle(n_elements_train).batch(batch_size)
	dset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test_ohe)).shuffle(n_elements_test).batch(batch_size)

	# build model and optimizer
	model = MnasnetEager.Mnasnet(num_classes=10)

	# Initialize the metric
	accuracy = tfe.metrics.Accuracy()
 
 	# optimizer
	optimizer = tf.train.AdamOptimizer(0.001)

	train(dset_train, model, epochs=epochs)
	test(dset_test, model)



	

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
