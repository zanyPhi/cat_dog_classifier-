import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

def load_dataset():
    train_dir = "path/to/directory"
    val_dir = "path/to/directory"
    test_dir = "path/to/directory"
    
    train_data = image_dataset_from_directory(directory=train_dir,
                                                 image_size=IMG_SIZE,
                                                 labels='inferred',
                                                 label_mode='binary',
                                                 batch_size=BATCH_SIZE,
                                                 class_names=['cat', 'dog'],
                                                 shuffle=True)
    val_data = image_dataset_from_directory(directory=val_dir,
                                            image_size=IMG_SIZE,
                                            labels='inferred',
                                            label_mode='binary',
                                            batch_size=BATCH_SIZE,
                                            class_names=['cat', 'dog'],
                                            shuffle=True)
    test_data = image_dataset_from_directory(directory=test_dir,
                                            image_size=IMG_SIZE,
                                            labels='inferred',
                                            label_mode='binary',
                                            batch_size=BATCH_SIZE,
                                            class_names=['cat', 'dog'],
                                            shuffle=False)
    
    return train_data, val_data, test_data

def preprocess_input():    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    return preprocess_input

def data_augmentation():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
   
    return data_augmentation

def load_model(image_shape=IMG_SIZE, data_augmentation=data_augmentation()):
    
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                              include_top=False,
                                              weights='imagenet',
                                              classifier_activation='softmax')
    base_model.trainable = False
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    inputs = tf.keras.Input(shape=IMG_SHAPE)

    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(.2)(x)

    outputs = tfl.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model


def save_model(model, name):
    model.save(name) #this creates a "name" folder

def model_from_path(model_path):
    loaded_model = tf.keras.models.load_model(model_path)

    return loaded_model

# model parameters    
epochs = 3
lr = 0.001
loss = BinaryCrossentropy()
metrics = ['accuracy', 'AUC']
optimizer = Adam(lr)

# get dataset
train_data, val_data, test_data = load_dataset()

# get the model
model = load_model()

# training the model
model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
model.fit(train_data, 
          validation_data=val_data, 
          epochs=epochs, 
          verbose=1)

# predicting with test dataset
predictions = model.predict(x=test_data)

# evaluating model performance on test dataset
BCE_loss, accuracy, AUC = model.evaluate(test_data)

# saving the model
save_model(model=model, name="softmax_2units")

# loading the model
model_path = "softmax_2units"
loaded_model = model_from_path(model_path)