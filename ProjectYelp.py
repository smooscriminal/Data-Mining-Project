# Libraries used
import os
import numpy as np
import cv2
import matplotlib as plt
import json

# Keras library is used on top of the TensorFlow library.
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# -----------------------Loading the photos-----------------------
# partially using tutorial :
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# loading the labels
os.chdir("dataset")
photo_labels = []
photo_labels_file = open("photos.json")
for i in photo_labels_file.readlines():
    photo_labels.append(json.loads(i))


# changing it back to the previous
os.chdir("..")

# Changing the directory to the photo folder
os.chdir("yelp_photos")
os.chdir("photos")

# testing
# img = cv2.imread("___3eKRa-uA97sdEZTJ9rQ.jpg", 0)
# cv2.imshow('image',img)
# cv2.waitKey(0)

# Now actually loading the photos
batch_size = 200

# this is the augmentation configuration for training the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation conifuration for testing the model
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory()


# ---------------------Extracting the photos---------------------


# Building a classification model with ___


# Building another classification model with ___
