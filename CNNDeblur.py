import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.models import Model, Input
from keras.optimizers import Adam
import keras.backend as K
import glob
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tensorflow.keras import datasets, layers, models


from PIL import Image

BATCH_SIZE = 2

height = 50
width = 50
dim = (width, height)

blurred = os.listdir('./data/blurred/')
blurred.sort()

sharp = os.listdir('./data/sharp/')
sharp.sort()

x_blur = []
for i in range(len(blurred)):
    image = Image.open('./data/blurred/' + blurred[i])
    res = cv2.resize(np.float32(image), dim, interpolation=cv2.INTER_LINEAR)
    x_blur.append(res)

y_sharp = []
for i in range(len(sharp)):
    image = Image.open('./data/sharp/' + sharp[i])
    res = cv2.resize(np.float32(image), dim, interpolation=cv2.INTER_LINEAR)
    y_sharp.append(res)

(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

print(len(x_train))
print(len(y_train))

im = cv2.imread('./data/blurred/image_7.jpg')
print(im.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

#compile model using accuracy to measure model performance
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs = 3 )