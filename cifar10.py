import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

import os



batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Split data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f'x_train shape: {x_train.shape}')
print(f'{x_train.shape[0]} train samples')
print(f'{x_test.shape[0]} test samples')


print(f"Old y shapes: {y_train.shape} && {y_test.shape}")
# convert class vectors into binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"New y shapes: {y_train.shape} && {y_test.shape}")


##################
# build NN model #
##################

model = Sequential()

# input shape ==> specifies all dimesion except the 'image count' one
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# train model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,\
              metrics=['accuracy'])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# check for data augmentation
if not data_augmentation:
    print('Not using data augmentation')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)




# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'Saved trained model at {model_path}')

# score trained model
score = model.evaluete(x_test, y_test, verbose=1)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])
















