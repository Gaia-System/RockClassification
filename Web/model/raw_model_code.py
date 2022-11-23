"""
I had been operated this codes on the Google colab.
This code is just reference.
"""


# # import libraries
# import tensorflow as tf
# import numpy as np

# import pandas as pd
# import cv2
# import os
# import math
# import scipy as sp
# import PIL

# # Tensorflow
# from tensorflow.keras import models, layers, Model
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
# from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# from tensorflow.keras import optimizers
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import EfficientNetB4, EfficientNetB6, ResNet50V2
# #from keras_tuner.tuners import RandomSearch

# import scikitplot as skplt
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt



# # setting directories
# data_dir = r'C:\\Users\\성우진\\Desktop\\YearDreamSchool2nd\\CV_mini_PJT\\Data\\'
# train_dir = data_dir + 'Train\\'
# test_dir = data_dir + 'Test\\'
# model_dir = data_dir + 'Model\\'


# # setting parameters
# # batch_size: 한번에 forward & Backword 하는 샘플의 수
# batch_size = 16

# # Training 수
# epochs = 15

# # Weight 조절 parameter
# LearningRate = 1e-3 # 0.001
# Decay = 1e-6

# img_width = 224
# img_height = 224


# # download the pre-trained model via tf.keras
# VGGmodel = tf.keras.applications.VGG16(include_top=False,
#     weights='imagenet', input_tensor=None, input_shape=(img_width,img_height,3), pooling=None)

# x = GlobalAveragePooling2D()(VGGmodel.output)

# pred = Dense(3, activation = 'softmax')(x)

# model = Model(inputs = VGGmodel.input, outputs = pred)

# model.compile(optimizer=
#          SGD(lr=LearningRate, decay=Decay, momentum=0.9, nesterov=True), 
#          loss='categorical_crossentropy',
#          metrics=['acc'])


# # make generators for data augmentation
# DATAGEN_TRAIN = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     data_format="channels_last",
#     validation_split=0.10)

# DATAGEN_TEST = ImageDataGenerator(
#     rescale=1./255,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     data_format="channels_last")

# # Generator의 instance 생성 (Train)
# TRAIN_GENERATOR = DATAGEN_TRAIN.flow_from_directory(
#     train_dir,
#     target_size = (img_width, img_height),
#     batch_size = batch_size,
#     class_mode= "categorical",
#     subset = "training")

# VALID_GENERATOR = DATAGEN_TRAIN.flow_from_directory(
#     train_dir,
#     target_size = (img_width, img_height),
#     batch_size = batch_size,
#     class_mode="categorical",
#     subset = "validation")

# # Generator의 instance 생성 (Test)
# TEST_GENERATOR = DATAGEN_TEST.flow_from_directory(
#     test_dir,
#     target_size = (img_width, img_height),
#     batch_size = batch_size,
#     shuffle = False,
#     class_mode='categorical')


# # setting callbacks
# check_points = ModelCheckpoint(filepath=model_dir+'VGG19-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',
#             monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# learning_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=3, verbose=1, min_lr=1e-8)

# callback = [check_points, learning_rate]


# # fitting and predicting
# model.fit_generator(TRAIN_GENERATOR, epochs = epochs, callbacks = callback, shuffle = True, validation_data = VALID_GENERATOR)
# test_pred = model.predict_generator(TEST_GENERATOR, verbose = 1)