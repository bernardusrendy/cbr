# Anti OSError: broken data stream when reading image file 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import Library.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize


## Menggunakan Google Colab
# Akses dan ambil data dari google colab

from google.colab import drive
drive.mount('/content/gdrive')
root_path = 'gdrive/My Drive/Colab Notebooks/DATA TRAIN COM'
path_uji = 'gdrive/My Drive/Colab Notebooks/DATA UJI'


# Fungsi untuk ambil gambar dan preprocessing

def load_image_files(container_path, dimension=(64, 64, 3)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

# Memanggil fungsi load_image_files
image_dataset = load_image_files(root_path)

# Membuat data train dan data test
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

# Membuat parameter
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# Membuat Model SVM
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)

# Memasukan data train
clf.fit(X_train, y_train)


# Melakukan prediksi
y_pred = clf.predict(X_test)

# Menampilkan hasil prediksi
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

