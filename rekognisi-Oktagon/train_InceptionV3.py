#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning
# <img src="./tflearn.png" style="width:50%">

# In[1]:


import os
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


# In[2]:


displayIncorrectPictures = False #set True for display of incorrect picture at the end
keras.backend.clear_session()


# In[3]:


data_path = "./dataset_two_class"
batch_size = 32


# In[4]:


datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, vertical_flip=True, fill_mode="nearest")


# In[5]:


train_generator = datagen.flow_from_directory(
        data_path+'/train',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False)


# In[6]:


ExtractionModel = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


# In[7]:


bottleneck_features_train = ExtractionModel.predict_generator(train_generator, 1758/batch_size, verbose=1)
bottleneck_features_train.shape


# In[ ]:


np.savez('inception_features_train_2_class', features=bottleneck_features_train)


# In[7]:


test_generator = datagen.flow_from_directory(
        data_path+'/test',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


# In[10]:


bottleneck_features_test = ExtractionModel.predict_generator(test_generator, 1758/batch_size, verbose=1)
np.savez('inception_features_test_2_class', features=bottleneck_features_test)


# In[8]:


train_data = np.load('inception_features_train_2_class.npz')['features']


# In[9]:


print("Got Feature with Dimension")
train_data.shape


# In[10]:


train_labels = np.array([0] * 879 + [1] * 879)


# In[11]:


test_data = np.load('inception_features_test_2_class.npz')['features']
test_labels = np.array([0] * 879 + [1] * 879)


# In[12]:


classes = test_generator.class_indices
class_names = []
for key in classes.keys():
    class_names.append(key)
print(class_names)


# In[13]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_data.shape[1:], padding='same'))
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

checkpointer = ModelCheckpoint(filepath='./weights/Two_Class.hdf5', verbose=1, save_best_only=True)
keras.utils.print_summary(model)


# In[25]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

history = model.fit(train_data, train_labels,
          epochs=50,
          batch_size=batch_size,
          validation_split=0.3,
          verbose=2,
          callbacks=[checkpointer], shuffle=True)


# In[14]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model.load_weights('./weights/Two_Class.hdf5')
rounded_predictions = model.predict_classes(test_data, batch_size=32, verbose=1)
evaluation_result = model.evaluate(test_data,test_labels, verbose=1,batch_size=32)


# In[15]:


#Confusion Matrix
matrix = confusion_matrix(test_labels, rounded_predictions)

dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show


# In[28]:


# summarize history for accuracy
plt.plot(history.history['binary_accuracy'] )
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['binary_acc', 'val_bin_acc'], loc='upper left')
plt.show()


# In[29]:


# summarize history for loss
plt.plot(history.history['loss'] )
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print("Akurasi Model (Correctness Tested Out of All Test Dataset)")
print(evaluation_result[1])


# In[16]:


incorrect_labels = []
incorrect_classification_indices = []
for i in range(rounded_predictions.shape[0]):
    if not rounded_predictions[i] == test_labels[i]:
        incorrect_classification_indices.append(i)
        incorrect_labels.append(rounded_predictions[i][0])
incorrect_classification_file_paths = []
for i in range(len(incorrect_classification_indices)):
    if incorrect_classification_indices[i] < 879:
        incorrect_classification_file_paths.append('Non/'+ os.listdir('./dataset_two_class/test/Non')[i])
    else:
        incorrect_classification_file_paths.append('Oktagon/'+ os.listdir('./dataset_two_class/test/Oktagon')[i])
print(cv2.__version__)
img_path = data_path + '/test/'
def get_image(file_path):
    if os.path.isfile(img_path + file_path):
        image_bgr = cv2.imread(img_path + file_path,cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_resized = cv2.resize(image_rgb, (299, 299), interpolation = cv2.INTER_CUBIC)
        plt.imshow(image_rgb_resized)
        plt.axis("off")
        plt.show()


# In[ ]:


datagenValid = ImageDataGenerator(rescale=1. / 255)
valid_generator = datagenValid.flow_from_directory(
        data_path+'/valid',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False)
bottleneck_features_validate = ExtractionModel.predict_generator(valid_generator, batch_size, verbose=1)
np.savez('inception_features_valid_2_class', features=bottleneck_features_validate)


# In[ ]:


valid_data = np.load('inception_features_valid_2_class.npz')['features']
valid_rounded_predictions = model.predict_classes(valid_data, batch_size=32, verbose=1)


# In[ ]:


valid_rounded_predictions
j=0
for i in range (len(valid_rounded_predictions)):
    if valid_rounded_predictions[i][0]==1:
        j=j+1


# In[ ]:


print("Jumlah TVST Error")
print(j)


# In[ ]:


if displayIncorrectPictures==True:
    print ("Incorrect Pictures and Predictions")
    for i in range(0, len(incorrect_classification_file_paths)):
        if 'Non' in incorrect_classification_file_paths[i]:
            print("True:Non")
        else:
            print("True:Oktagon")

        if incorrect_labels[i] == 1:
            print("Pred:Oktagon")
        else:
            print("Pred:Non")
        get_image(incorrect_classification_file_paths[i])




