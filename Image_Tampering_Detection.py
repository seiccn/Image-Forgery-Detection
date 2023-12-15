#Original and Tampered Image Dataset

import pandas as pd
import numpy as np
import keras 
import keras.models as M
import keras.layers as L
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore as f
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization as IN

#Original Image
original=Image.open('../input/cg1050/TRAINING_CG-1050/TRAINING/ORIGINAL/Im100_2_cm.jpg')

#Tampered Image
tampered=Image.open('../input/cg1050/TRAINING_CG-1050/TRAINING/TAMPERED/Im100_cm2.jpg')

#Using Image Data Generator
img_shape=(150,150)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen_valid=ImageDataGenerator()
# Getting training data
training_data=datagen.flow_from_directory('../input/cg1050/TRAINING_CG-1050/TRAINING/',target_size=img_shape,color_mode='rgb',batch_size=64,seed=32,interpolation='bicubic')
validation_data=datagen_valid.flow_from_directory('../input/cg1050/VALIDATION_CG-1050/VALIDATION/',target_size=img_shape,color_mode='rgb',batch_size=64,seed=32,interpolation='bicubic')

#Defining Model
def make_model():
    model=M.Sequential()
    model.add(L.Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=2,activation="relu",input_shape=(150,150,3)))
    model.add(IN(axis=-1))
    model.add(L.MaxPooling2D(pool_size=(2,2)))
    model.add(L.Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=2))
    model.add(L.MaxPooling2D(pool_size=(2,2)))
    model.add(L.Dropout(0.4))
    model.add(L.Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=2))
    model.add(L.MaxPooling2D(pool_size=(2,2)))
    model.add(L.Flatten())
    model.add(L.Dense(100,'relu'))
    model.add(L.Dropout(0.4))
    model.add(L.Dense(2,'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
model=make_model()
model.summary()

#Fitting The Model
early_stop=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=4)
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',patience=2)
model.fit(training_data,validation_data=validation_data,epochs=40,callbacks=[early_stop,reduce_lr])

#Changes In Loss Over Epochs
loss=model.history.history['loss']
val_loss=model.history.history['val_loss']
acc=model.history.history['accuracy']
val_acc=model.history.history['val_accuracy']
epochs=[i for i in range(len(loss))]
plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.show();

#Changes In Accuracy Over Epochs
plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.show();

#Predicting some of the images
def predict_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    image=cv2.resize(image,(150,150))
    image=image.reshape(1,150,150,3)
    prediction=np.argmax(model.predict(image))
    labels=['Original','Tampered']
    return labels[prediction]

predict_image('../input/cg1050/VALIDATION_CG-1050/VALIDATION/ORIGINAL/Im100_1_cm.jpg')

predict_image('../input/cg1050/VALIDATION_CG-1050/VALIDATION/TAMPERED/Im100_cm1.jpg')
