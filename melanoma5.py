#imports
#import melanoma5plot

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow import keras
import pandas as pd
import os
import cv2
import keras.layers as kl
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

from keras.callbacks import CSVLogger

from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

#number of samples that will be passed through to the network at one time
#"Generally batch size of 32 or 25 is good, with epochs = 100 unless you have large dataset"
batch_size = 20
#img_height = 300
#img_width = 300
#import image directories
#data_dir = '/Users/laure/Desktop/melanoma/melanoma_cancer_dataset2'

#label benign images with 0, malignant images as 1

class_names = ['benign', 'malignant']

class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

num_classes = len(class_names)

IMAGE_SIZE = 200

#function to load data
def load_data():
    DIRECTORY = '/Users/laure/Desktop/melanoma_cancer_dataset2'
    CATEGORY = ['test', 'train']

    output = []

  

    def countFiles(directory_path):
        count = 0
        # Iterate directory
        for file in os.listdir(directory_path):
        # check if current path is a file
            if (file.endswith('jpg')):
                count += 1
        return count

    
    print("train benign has " + str(countFiles('/Users/laure/Desktop/melanoma_cancer_dataset2/train/benign')))
    print("train malignant has " + str(countFiles('/Users/laure/Desktop/melanoma_cancer_dataset2/train/malignant')))

    num_train_benign = countFiles('/Users/laure/Desktop/melanoma_cancer_dataset2/train/benign')

    num_train_malignant = countFiles('/Users/laure/Desktop/melanoma_cancer_dataset2/train/malignant')

    for category in CATEGORY:
        print("loading category " + category + "\n")

        path = os.path.join(DIRECTORY, category)
        #path is now set to /test or /train
        images = []
        labels = []

        for folder in os.listdir(path):
            label = class_names_label[folder]   
            #i.e, recall that label is 0 for benign, 1 for malignant 
            print("in folder " + folder+"\n")

            #iterate through each image in folder
            for file in os.listdir(os.path.join(path, folder)):
                #path is now /test/malignant or other combination od test, train, malignant, benign
                
                #create img path
                img_path = os.path.join(path, folder, file)
                #path is now, e.g. /test/benign/image47.jpg

                #open image and resize
                    #read image
                image = cv2.imread(img_path)
                    #convert to RGB color from cv2 default color value byte system
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #resize image to desired size
                image = cv2.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

                #append image and its label
                    #image is appended as its array representation, label is benign or malignant
                images.append(image)
                labels.append(label)
                #print("appended\n")
        #convert to numpy array
        images = np.array(images, dtype='float32') 
        labels = np.array(labels, dtype = 'int32') #can use int because working with 0 and 1 values

        #the output will be a list of images and their respective label in tuple form
        output.append((images, labels))

    #return the output list
    return output



#call load_data() function to create train and test dataset
(train_images, train_labels), (test_images, test_labels) = load_data()

print("data loaded\n")

#shuffling train images and labels trains the model better
train_images, train_labels = shuffle(train_images, train_labels, random_state = 25)

#standardize/normalize the data
#rescale data from RGB (0, 255) to B/W (0, 1)
#this is because we want smaller data values to make the data easier to work with and easier for neural net to pick features out
#this is done in the model below in k1.Rescaling

#making the modes (using Keras sequential model)
#copied and pasted this just as a starting point
model = Sequential([
  # the 3s refer to 3 colors (RGB)
  # relu

  #kl.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  #2D refers to image
  #using relu activation function (default)
  # params:
    # Conv2D( number of feature options that the NN is looking for, ( window length and width if when the NN is taking "steps" through the image ) )
      # the "window" size AKA kernel size is the "art": mess around with it

  kl.Conv2D(16, (3, 3), padding='same', activation='relu'),
  kl.Conv2D(32, (3, 3), padding='same', activation='relu'),
  kl.MaxPooling2D(2, 2),
  #hidden layers
  kl.Conv2D(32, (3, 3), padding='same', activation='relu'),
  kl.MaxPooling2D(2, 2),
  kl.Conv2D(64, (3, 3), padding='same', activation='relu'),
  kl.MaxPooling2D(2, 2),
  kl.Conv2D(128, (3, 3), padding='same', activation='relu'),
  kl.MaxPooling2D(2, 2),
  #flatten outputs to reduce number of features
  kl.Flatten(),
  #output layers
  kl.Dense(128, activation='relu'),
  kl.Dense(num_classes)
])

#compiling the model
#copied and pasted this just as a starting point
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("model compiled\n")

#adam optimizer: stoichastic gradient descent, default
# loss: 
# "everybody looks at accuracy"



#training the model 
#epoch: one complete pass through training data 

csv_logger = CSVLogger('training.log')
history = model.fit(train_images, train_labels, batch_size = 50, epochs = 2, callbacks = [csv_logger], validation_split=.25)
print('model has been trained\n')

print(model.summary())


#check with test images
print("testing model accuracy on test image and label set")
test_loss = model.evaluate(test_images, test_labels)

#this is the only part i dont understand yet but we'll get there i just wanted to see what it did
predictions = model.predict(test_images)    #vector of probabilities
pred_labels = np.argmax(predictions, axis=1) #take highest probability

print(classification_report(test_labels, pred_labels))

#print confusion matrix

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)

cm_data = pd.DataFrame(cm)

cm_data.to_csv("confusion_matrix_data.csv", encoding = 'utf-8', index = False)

#disp.plot()
#plt.show()

# loss_train = history.history['loss_train']
# loss_val = history.history['loss_val']
# epochs = range(1,35)
# plt.plot(epochs, loss_train, 'g', label='Training Loss')
# plt.plot(epochs, loss_val, 'b', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

history_data = pd.DataFrame(history.history)
history_data.to_csv("history_data.csv", encoding = 'utf-8', index = False)

#filename='history_data_logger.csv'
#history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)


#
# 
# pd.DataFrame(history.history).plot(figsize=(8,5))
#melanoma5plot.printAccuracyAndLoss()
