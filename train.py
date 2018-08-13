import tensorflow as tf
mnist = tf.keras.datasets.mnist
import Mnasnet
import numpy as np

# Load Mnist data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# Preprocess data
x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Load model
model = Mnasnet.MNasNet(input_shape=(28, 28, 1))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Train it
model.fit(x_train, y_train, epochs=20)
# Evaluate it
loss, acc = model.evaluate(x_test, y_test)
print('Accuracy of: ' + str(acc*100.) + '%')