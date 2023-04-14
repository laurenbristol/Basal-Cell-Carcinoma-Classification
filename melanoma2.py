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
import glob as gb

from keras.layers import Input, Activation, Lambda, Dense, Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

image_dir = 'melanoma_cancer_dataset1'
SIZE = 150 #150*150 pixels

data = [] #input
labels = [] #output

malignant_images = '/Users/laure/Desktop/VS Code/melanoma/melanoma_cancer_dataset1/malignant'
benign_images = '/Users/laure/Desktop/VS Code/melanoma/melanoma_cancer_dataset1/benign'

#loading each image, read it using opencv, resize images and convert to numpy array
#append label of malignant as 1, benign as 0
""" for i, image_name in enumerate(malignant_images):
    if(image_name.split('.') == 'jpg'):
        image = cv2.imread(image_dir + 'malignant' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(SIZE, SIZE)
        data.append(np.array(image))
        labels.append(1)


for i, image_name in enumerate(benign_images):
    if(image_name.split('.')[0] == 'jpg'):
        image = cv2.imread(image_dir + 'benign' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(SIZE, SIZE)
        data.append(np.array(image))
        labels.append(0)

print(data)
print(labels) """

#compiled without error past here

import torch
from torchvision import transforms
count = 0 

for images in os.listdir(benign_images):
    img_address = benign_images + '/' + images
    img = Image.open(img_address)
    convert_tensor = transforms.ToTensor()
    t = convert_tensor(img)
    #count+=1
    #print(count)
    #5500 in benign

count1 = 0 
for images in os.listdir(malignant_images):
    img_address = malignant_images + '/' + images
    img = Image.open(img_address)
    convert_tensor = transforms.ToTensor()
    t = convert_tensor(img)
    #count1+=1
    #print(count1)
    #5105 in malignant




data = np.array(data)
labels = np.array(labels)

print(labels)




""" img = Image.open('C:/Users/laure/Desktop/Pictures/IMG_3090 - Copy.jpg')
img
convert_tensor = transforms.ToTensor()
tens = convert_tensor(img)
print(tens) """






