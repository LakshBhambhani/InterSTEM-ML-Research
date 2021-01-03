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

BUFFER_SIZE = 60000
BATCH_SIZE = 256

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(2058,220,220,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

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

discriminator = make_discriminator_model()
discriminator.save_weights('discriminator.h5')

generator = make_generator_model()
generator.save_weights('generator.h5')

image_files = []

for origImg in glob.glob('./LungCancerDetection/data/test/*'):
    # image = Image.open(origImg)
    height = 220
    width = 220
    dim = (width, height)
    # res = cv2.resize(np.float32(image), dim, interpolation=cv2.INTER_LINEAR)
    image_files.append(origImg)


print(len(image_files))

generator = make_generator_model()
discriminator = make_discriminator_model()

gan = simple_gan(generator, discriminator, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,player_params=[generator.trainable_weights, discriminator.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')

# history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=10, batch_size=batch_size)


