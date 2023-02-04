# Burak Bozdağ
# 504211552

# Setup
import datetime
import os
import cv2
import sys
import random
import warnings
import numpy as np 
import pandas as pd
from time import time
from itertools import chain
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 
from skimage.transform import resize
from skimage.morphology import label
from skimage.io import imread, imshow, imread_collection, concatenate_images

import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (
    Dense, Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, 
    Activation, Add, multiply, add, concatenate, LeakyReLU, ZeroPadding2D, UpSampling2D, 
    BatchNormalization, SeparableConv2D, Flatten )

from sklearn.metrics import classification_report

### Logging suppression
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
###

MAIN_PATH = './chest_xray/'

# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.25,
                             zoom_range=0.1,
                             rotation_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

def get_transforms(data):
    
    if data == 'train':
        IMG_TRAIN = MAIN_PATH +'train/'
        train_generator = datagen.flow_from_directory(
            # dataframe = train,
            directory = IMG_TRAIN,
            # x_col = 'filename',
            # y_col = 'label',
            batch_size  = 8,
            shuffle=True,
            class_mode = 'categorical',
            target_size = (224, 224)
        )

        return train_generator

    elif data == 'valid':
        IMG_VAL = MAIN_PATH + 'val/'
        valid_generator = datagen.flow_from_directory(
            # dataframe = valid,
            directory = IMG_VAL,
            # x_col = 'filename',
            # y_col = 'label',
            batch_size = 8,
            shuffle = True,
            class_mode = 'categorical',
            target_size = (224, 224)
        )

        return valid_generator

    else :
        IMG_TEST = MAIN_PATH + 'test/'
        test_generator = test_datagen.flow_from_directory(
            # dataframe = test,
            directory = IMG_TEST,
            # x_col = 'filename',
            # y_col = None,
            batch_size = 8,
            shuffle = False,
            class_mode = None,
            target_size = (224, 224)
        )

        return test_generator

train = get_transforms('train')
valid = get_transforms('valid')
test = get_transforms('test')

# Callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='auto',
    min_delta=1e-10, cooldown=0, min_lr=0
)

early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=9, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)

ckpt = ModelCheckpoint(
    filepath = './saved_model_CNN/checkpoint/',
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True
)

log_dir = "logs_CNN/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [reduce_learning_rate, early_stopping, ckpt, tensorboard_callback]

########################################################

with tf.device("/GPU:0"):
    # Model
    image_size = 224

    model= tf.keras.Sequential()
    model.add(Conv2D(96,kernel_size=(11,11),strides=(4,4),activation='relu', input_shape=(image_size,image_size,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,kernel_size=(5,5),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1),activation='relu'))
    model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1),activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(2,activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.get_config())

    # Train
    history = model.fit(train, epochs=50, validation_data=valid, callbacks=callbacks, verbose=1)

    model.evaluate(valid, verbose=1)

    y_pred = model.predict(test, verbose=1)
    y_pred = np.argmax(y_pred, axis = 1)

    def create_df (dataset, label):
        filenames = []  
        labels = []
        for file in os.listdir(MAIN_PATH + f'{dataset}/{label}'):
            filenames.append(file)
            labels.append(label)
        return pd.DataFrame({'filename':filenames, 'label':labels})

    test_NORMAL = create_df('test', 'NORMAL')
    test_PNEUMONIA = create_df('test', 'PNEUMONIA')
    test_ori = test_NORMAL.append(test_PNEUMONIA, ignore_index=True)
    test_ori['label'] = test_ori['label'].apply(lambda x: 0 if x=='NORMAL' else 1)
    y_true = test_ori['label'].values

    print(classification_report(y_true, y_pred))
