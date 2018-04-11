import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Sequential 
import numpy as np

# create a model
model = Sequential()

# stack layers
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# configure learning
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# create random np arrays
x_train = np.random.random((64, 100))
y_train = np.random.random((64, 10))
x_test = np.random.random((4, 100))
y_test = np.random.random((4, 10))

# train using numpy arrays
model.fit(x_train, y_train, epochs=5, batch_size=32)

# evaluate on existing data
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
# generate predictions on new data
classes = model.predict(x_test, batch_size=128)
print(classes)
