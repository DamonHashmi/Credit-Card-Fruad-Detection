# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:07:00 2020

@author: Damon
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('F:\Pyhton Practice\Fruad Dectection\creditcard.csv')
X = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
# Adding input layer or first first Hidden layer.
classifier.add(Dense(output_dim= 15, init = "uniform", activation ="relu", input_dim = 16))
# Adding input layer or first Hidden layer.
classifier.add(Dense(activation="relu", input_dim=15, units=8, kernel_initializer="uniform"))
# Adding input layer and first Second Hidden layer.
classifier.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=1))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 11, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
Prediction= classifier.predict(sc.transform(np.array([ ...]])))
print(Prediction>0.5)







