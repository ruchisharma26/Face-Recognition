# -*- coding: utf-8 -*-
"""MLProject-Phase3-Bush.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ciH9zdim1yYja5QqVRnbRqzF1PJNG975
"""

from google.colab import drive
drive.mount("/content/gdrive")

import numpy as np
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import keras
from keras.models import load_model
from keras.models import Sequential,Input,Model
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import keras_metrics

X = pd.read_csv("gdrive/My Drive/Fall2018/ML/data/X.csv",sep=' ',header=None,dtype=float)
X = X.values

y = pd.read_csv("gdrive/My Drive/Fall2018/ML/data/y_bush_vs_others.csv", sep = ' ', header=None, dtype=float)
y_bush = y.values.ravel()

#Reshaping input to a 64x64 
X_mod = X.reshape((X.shape[0], 64,64,1))

X_train, X_test, y_train, y_test = train_test_split(X_mod, y_bush, test_size = 1./3, random_state=4435,  shuffle = True, stratify=y_bush)

model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1))) 
model.add(Activation('tanh')) 
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Conv2D(64, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy',keras_metrics.precision(), keras_metrics.recall()])

model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test))

score_test = model.evaluate(X_test, y_test, verbose = 0)
print(model.metrics_names)
print(score_test)

f1_test = (2*score_test[2]*score_test[3])/(score_test[2]+score_test[3])
print(f1_test)

score_train = model.evaluate(X_train, y_train, verbose = 0)
print(model.metrics_names)
print(score_train)

f1_train = (2*score_train[2]*score_train[3])/(score_train[2]+score_train[3])
print(f1_train)