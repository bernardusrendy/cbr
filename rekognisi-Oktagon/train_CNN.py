# Importing Keras libraries dan packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Anti OSError: broken data stream when reading image file 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Menggunakan Google Colab 
# Akses data gambar
from google.colab import drive
drive.mount('/content/gdrive')

root_path = 'gdrive/My Drive/Colab Notebooks/DATA TRAIN'
path_uji = 'gdrive/My Drive/Colab Notebooks/DATA UJI'

# Inisialisasi CNN
classifier = Sequential()
# Tahap 1 - Konvolusi
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Tahap 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Menambah Konvolusi layer kedua
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Tahap 3 - Flattening
classifier.add(Flatten())
# Tahap 4 - Menyambungkan antar layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Membangun CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Bagian 2 - Memasukkan gambar pada CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(root_path,
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory(path_uji,
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

# Train modelnya
classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 20,
validation_data = test_set,
validation_steps= 20)

# Save weightnya
classifier.save_weights("model.h5")

import pickle
filename = 'bigbrain_model3.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Menguji kodenya
test_set = test_datagen.flow_from_directory(path_uji,
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
loss, acc = classifier.evaluate(test_set, verbose = 0)
print(acc * 100)

