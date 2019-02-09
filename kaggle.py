###############################################################################
# CS 155 Miniproject 1
###############################################################################

import numpy as np
#import tensorflow as tf
#import keras
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten, Dropout
#from keras.datasets import mnist

## Importing the MNIST dataset using Keras
train = np.genfromtxt('train_2008.csv', delimiter=',')
X_train = train[1:, :-1]
y_train = train[1:, -1]
X_test = np.genfromtxt('test_2008.csv', delimiter=',')[1:,:]
#print (X_train[0])
#print (y_train)
#print (X_test[0])

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

printf("model accuracy: {tree.score(X_train, y_train)}")
