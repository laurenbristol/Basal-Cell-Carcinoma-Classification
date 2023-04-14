#imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow_datasets as tfds
import pandas as pd
import os
import cv2
#from skimage.io import imread
#from skimage.transform import resize
import keras.layers as kl

from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180
#import image directories
data_dir = '/Users/laure/Desktop/VS Code/melanoma/melanoma_cancer_dataset1'

batch_size = 32
img_height = 180
img_width = 180

#create train dataset as 80% of total images
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, #automatically assigns training split as 1-validation_split
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

""" for e in train_ds.as_numpy_iterator():
    print(e[1:5]) """



# Plot the image resolutions        
""" fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
points = ax.scatter(train_ds.Width, train_ds.Height, color='blue', alpha=0.5, s=train_ds["Aspect Ratio"]*100, picker=True)
ax.set_title("Press enter to after selecting the points.")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)
 """
#create validation dataset as 20% of total images
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#assign class names based on folder labels
class_names = train_ds.class_names
#print(class_names)

#tring to print images because I dont think theyre being read in
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

""" for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape) """
    
    #image_batch is a tensor of the shape (32, 180, 180, 3)
        # last dimension refers to color channels RGB
    #label_batch is a tensor of the shape (32,), these are corresponding labels to the 32 images.

#standardize/normalize the data
#rescale data from RGB (0, 255) to B/W (0, 1)
#this is because we want smaller data values to make the data easier to work with and easier for neural net to pick features out
#this is done in the model below in k1.Rescaling

#making the modes (using Keras sequential model)
#copied and pasted this just as a starting point
num_classes = len(class_names)
model = Sequential([
  kl.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  kl.Conv2D(16, 3, padding='same', activation='relu'),
  kl.MaxPooling2D(),
  kl.Conv2D(32, 3, padding='same', activation='relu'),
  kl.MaxPooling2D(),
  kl.Conv2D(64, 3, padding='same', activation='relu'),
  kl.MaxPooling2D(),
  kl.Flatten(),
  #using relu activation function
  kl.Dense(128, activation='relu'),
  kl.Dense(num_classes)
])

#compiling the model
#copied and pasted this just as a starting point
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#adam optimizer: stoichastic gradient descent

#checkpoint: program compiled without any obvious errors here

#training the model 
#epoch: one complete pass through training data 

#staring small with 10 epochs

""" epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
) """