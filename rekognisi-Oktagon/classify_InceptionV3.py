#!/usr/bin/env python
# coding: utf-8

# In[60]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3, preprocess_input


# In[61]:


data_path = "./dataset_two_class"
batch_size = 32
ExtractionModel = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


# In[62]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(8,8,2048), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[63]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model.load_weights('./weights/Two_Class.hdf5')


# In[64]:


datagenValid = ImageDataGenerator(rescale=1. / 255)
valid_generator = datagenValid.flow_from_directory(
        data_path+'/valid',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
bottleneck_features_validate = ExtractionModel.predict_generator(valid_generator, 1)
np.savez('inception_features_valid_2_class', features=bottleneck_features_validate)
valid_data = np.load('inception_features_valid_2_class.npz')['features']
valid_rounded_predictions = model.predict_classes(valid_data, batch_size=1,verbose=0)
prediction=np.round(model.predict(valid_data))
percentage=model.predict(valid_data)
if prediction[0][0]==0:
    predictionWord="Bukan Oktagon : "
    percentage=abs(percentage-1)
else :
    predictionWord="Oktagon : "
print(predictionWord+'%.0f'%(percentage[0][0]*100)+'%')

