import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import pickle
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


BUFFER_SIZE = 60000
BATCH_SIZE = 256

height = 50
width = 50
dim = (width, height)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(2058,220,220,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[2058, 220, 220]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

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

generator = make_generator_model()
discriminator = make_discriminator_model()

discriminator.save_weights('discriminator.h5')

generator.save_weights('generator.h5')

gan = simple_gan(generator, discriminator, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,player_params=[generator.trainable_weights, discriminator.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')

# history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=10, batch_size=batch_size)


