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

# Returns a pretrained model (can be used in eager execution)
def get_pretrained_model(num_classes, input_shape=(224, 224, 3)):
	model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
	logits = tf.keras.layers.Dense(num_classes, name='fc')(model.output)
	model = tf.keras.models.Model(model.inputs, logits)
	return model
		

# Trains the model for certains epochs on a dataset
def train(dset_train, dset_test, model, epochs=5, show_loss=False):

	for epoch in xrange(epochs):
		print('epoch: '+ str(epoch))
		for x, y in dset_train: # for every batch
			with tf.GradientTape() as g:
				y_ = model(x, training=True)
				loss = tf.losses.softmax_cross_entropy(y, y_)
				if show_loss: print('Training loss: ' + str(loss.numpy()))

			# Gets gradients and applies them
			grads = g.gradient(loss, model.variables)
			optimizer.apply_gradients(zip(grads, model.variables))

		train_acc = get_accuracy(dset_train, model, training=True)
		test_acc = get_accuracy(dset_test, model)
		print('Train accuracy: ' + str(train_acc))
		print('Test accuracy: ' + str(test_acc))


# Tests the model on a dataset
def get_accuracy(dset_test, model, training=False):
	accuracy = tfe.metrics.Accuracy()
	for x, y in dset_test: # for every batch
		y_ = model(x, training=training)
		accuracy(tf.argmax(y, 1), tf.argmax(y_, 1))

	return accuracy.result().numpy()


def restore_state(saver, checkpoint):
	try:
		saver.restore(checkpoint)
		print('Model loaded')
	except Exception:
		print('Model not loaded')


def init_model(model, input_shape):
	model._set_inputs(np.zeros(input_shape))

if __name__ == "__main__":


	# constants
	image_size = 28
	batch_size = 128
	epochs = 20
	num_classes = 10
	channels= 1

	# Get dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	# Reshape images
	x_train = x_train.reshape(-1, image_size, image_size, channels).astype('float32')
	x_test = x_test.reshape(-1, image_size, image_size, channels).astype('float32')

	# We are normalizing the images to the range of [-1, 1]
	x_train = x_train / 127.5 - 1
	x_test = x_test / 127.5 - 1

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

 	# optimizer
	optimizer = tf.train.AdamOptimizer(0.001)

	# Init model
	init_model(model, input_shape=(batch_size, image_size, image_size, channels))
	
	get_params(model)
	
	# Init saver 
	saver_model = tfe.Saver(var_list=model.variables) # can use also ckpt = tfe.Checkpoint(model=model) 

	restore_state(saver_model, 'weights/last_saver')

	train(dset_train=dset_train, dset_test=dset_test, model=model, epochs=epochs)
	
	saver_model.save('weights/last_saver')

	



	'''
	You can olso optimize with only:
	optimizer.minimize(lambda: loss_function(model, x, y, training=True))
	
	or you can build the Keras model:
	model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	'''

