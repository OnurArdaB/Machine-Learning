# -*- coding: utf-8 -*-
"""fork-of-kernel1910868e43.ipynb

# Skin Cancer Classification

Skin cancer is a very common condition and visual-based scans are important in the diagnosis process. Diagnosis process starts with clinical screening first.It is then followed by dermoscopic analysis, a biopsy and histopathological examination.Achieving an autonomous structure is of great importance in the aforementioned processes and involves various difficulties.

The dataset given to us has a total of 10000 images, where each image is classified with one of 5 different types of skin cancer.This dataset is available to be used for training an effective machine learning algorithm.

The cancer classes mentioned are:

* Melanoma (MEL)
* Melanocytic nevus (NV)
* Basal cell carcinoma (BCC)
* Actinic keratosis (AK)
* Benign keratosis (BKL)

#### File descriptions

Train.csv - Training data and it consists of 10,000 images along with their labels (also known as the “ground truth”)
SkinCancerTest.csv - Testing data and it consist of 5,000 simages. Your final submission should be similar to this file.

In this core, models will be trained to accurately match these skin cancer lesions regarding the classes they adhere to.Convolutional Neural Networks are chosen as the model in order to classify the regarding data.

The kernel steps as followed:
1. Data Analysis and Preprocessing
    * Import necessary libraries
    * Read and store the raw data 
    * Create a label dictionary 
    * Image Pipeline
    * EDA (Exploratary Data Analysis)
2. Model Building
3. Model Training
4. Model Evaluation
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""## 1.Data Analysis and Preprocessing

Defaultly given by kaggle.
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from PIL import Image as img
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

"""## Read and store the raw data"""

img_prefix = "/kaggle/input/machinelearning412-skincancerclassification/Data_SkinCancer/Data_SkinCancer/"
df = pd.read_csv("/kaggle/input/machinelearning412-skincancerclassification/Train.csv")
testdf = pd.read_csv("/kaggle/input/machinelearning412-skincancerclassification/Test.csv")

df.head()

"""As we can see below,there is a serious class imbalance in the dataset."""

temp = df.groupby('Category').count()
temp.rename(columns={'Id':'Count'})

#Printing a random image, in order to see whether I'm doing correctly or not
#Also, I want to see image shape

for i in [str(j) for j in range(1,10)]:
    image = img.open(img_prefix+"Image_"+i+".jpg").convert("RGB")
    print("Image dimensions, ", np.asarray(image).shape)

"""Ok, Image dimensions are not fixed
We must fix the dimensions by whether shrinking the large ones or padding the small ones

## Create a label dictionary
"""

targetLabels={
    1: "MEL",
    2: "NV",
    3: "BCC",
    4: "AK",
    5: "BKL"
}
df["CategoryNames"]=df["Category"].apply(lambda col: targetLabels[col])

"""Labels that regards to the given Category is also inserted into the dataframe as CategoryNames columns.

## Image Pipeline

Despite the fact that there haven't been any EDA done on the dataset it is known that the images were not with the same sizes.
The following function takes an image as input, resizes and normalizes the image.Finally the labels are also one hot encoded in order to use softmax layer for the model.
"""

WIDTH = 128
HEIGHT = 128
def imagePipeline(imgPostFix):
    return tf.cast(np.array(
        img.open(img_prefix+imgPostFix+".jpg")
                      .resize((WIDTH,HEIGHT))
                      .convert("RGB")), tf.float32)/255.0


images = np.array(df["Id"])
target = np.array(df["Category"])
target=target-1 #[1,5] ==> [0,4]
target = to_categorical(target)#One hot encoding labels
target = [tf.cast(i, tf.int64) for i in target]
target=tf.stack(target)
images = tf.stack([imagePipeline(i) for i in images])
df.head()

"""Load=>Resize=>Convert To Rgb=>Create A Tensor=>Normalize

Therefore image shapes become (10000, WIDTH, HEIGHT, 3)
Label shapes become (10000, 5)
"""

print("Image dimensions, ", np.asarray(images).shape)

"""Image dimensions are fixed after the preprocessing operation.

## Activate the TPU
"""

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

"""## Exploratory Data Analysis (EDA)

Lets start with checking how the classes are distributed in the training set.
"""

temp = df.groupby('Category').count().drop(columns=["CategoryNames"]).rename(columns={"Id":"Count"})
plt.bar(targetLabels.values(),temp["Count"])

"""We can see that the dataset is unbalanced so it must be considered when checking the performance of the trained model in the further steps."""

n_samples = 4
fig, m_axs = plt.subplots(5, n_samples, figsize = (4*n_samples, 3*5))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         df.sort_values(['CategoryNames']).groupby('CategoryNames')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(img.open(img_prefix+c_row['Id']+".jpg").resize((WIDTH,HEIGHT))
                      .convert("RGB"))
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

"""Taking the TPU"""

# instantiating the model in the strategy scope creates the model on the TPU
""" 
with tpu_strategy.scope():
    
    model = tf.keras.Sequential()
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same',input_shape=(WIDTH,HEIGHT,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    

    model.add(Conv2D(128,(3,3), strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(128,(3,3), strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(5,activation='softmax'))
    model.summary()
   """ 
"""
with tpu_strategy.scope():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same',input_shape=(WIDTH,HEIGHT,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(5,activation='softmax'))
    model.summary()
"""


"""
with tpu_strategy.scope():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,kernel_size=3, strides=6,activation='relu',padding='Same',input_shape=(WIDTH,HEIGHT,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    #model.add(Dropout(0.25))
    model.add(Conv2D(32,kernel_size=3, strides=4,activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=3, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    #model.add(Dropout(0.25))
    

    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(5,activation='softmax'))
    model.summary()

"""
"""
with tpu_strategy.scope():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same',input_shape=(WIDTH,HEIGHT,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Conv2D(32,kernel_size=3, strides=2,activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,kernel_size=3, strides=2, activation='relu',padding='Same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(5,activation='softmax'))
    model.summary()
"""
"""

with tpu_strategy.scope():
    model = tf.keras.Sequential()
    

    model.add(ResNet50(input_shape=(WIDTH, HEIGHT, 3), include_top=False, pooling='avg', weights="imagenet"))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(5,activation='softmax'))
    model.layers[0].trainable = False
    model.summary()



"""

optimizer=Adam(lr=0.01,beta_1=0.8,beta_2=0.999,epsilon=1e-7,decay=0.0,amsgrad=False)
learning_reductor = ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
with tpu_strategy.scope():
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=["accuracy"])
    history  = model.fit(images, target, validation_split=0.2,epochs=150,verbose=1,callbacks=[learning_reductor])

"""This looks promising. Thus for the next step training the model with all the data is required.

Reason for this, the model is splitted to validation in order to get a adequate model.

When adequate model is choosen, as it is deployed in unknown data, the train data should

not be wasted.
"""

optimizer=Adam(lr=0.01,beta_1=0.8,beta_2=0.999,epsilon=1e-7,decay=0.0,amsgrad=False)
learning_reductor = ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
with tpu_strategy.scope():
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=["accuracy"])
    history  = model.fit(images, target, epochs=50,verbose=1,callbacks=[learning_reductor])

testids = np.array(testdf["Id"]).tolist()
preds = np.array([], dtype=np.int32)
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

for x in batch(testids, 100):
    x = tf.stack([imagePipeline(i) for i in x])
    prediction = model.predict(x)
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction+1
    preds=np.concatenate((preds, prediction))

import csv
header = ["Id", "Category"]
lines = [[i, j] for i, j in zip(testids, preds)]
with open("Results.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header)
    for l in lines:
        writer.writerow(l)