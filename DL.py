from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend
from keras.utils import plot_model
import pydot


width = 300
height = 300

training_set = 'path/to/directory'
testing_set = 'path/to/directory'

training_samples = 2500
testing_samples = 2000

epochs = 10
batch_size = 16

if backend.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen_from_directory = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen_from_directory = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen_from_directory.flow_from_directory(training_set, target_size=(width, height), batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen_from_directory.flow_from_directory(testing_set, target_size=(width, height), batch_size=batch_size, class_mode='binary')

model.fit_generator( train_generator, steps_per_epoch=training_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=testing_samples // batch_size)

model.save_weights('first_try.h5')