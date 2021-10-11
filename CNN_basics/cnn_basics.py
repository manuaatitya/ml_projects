# CNN Basics

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape


# Load the datasets
from tensorflow.keras.datasets import fashion_mnist

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

print("The shape of trainX is {}".format(trainX.shape))
# There are 60,000 (28,28) greyscale images in the dataset for training

print("The shape of testX is {}".format(testX.shape))
# There are 10,000 (28, 28) greyscale images in the dataset for testing

