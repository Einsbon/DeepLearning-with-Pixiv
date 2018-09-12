from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os

import numpy as np
import pickle
import cv2

LOAD = True


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.sampleNoise = np.random.normal(0, 1, (25, self.latent_dim))

        if os.path.exists(os.path.abspath(os.path.dirname(__file__))+'/save/sampleNoise.p') == True:
            with open(os.path.abspath(os.path.dirname(__file__))+'/save/sampleNoise.p', 'rb') as file:
                self.sampleNoise = pickle.load(file)
            print('load sampleNoise')
        else:
            with open(os.path.abspath(os.path.dirname(__file__))+'/save/sampleNoise.p', 'wb') as file:
                pickle.dump(self.sampleNoise, file)
            print('save sampleNoise')

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        if LOAD == True:
            self.load_model()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        X_train = []
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        print(X_train.shape)
        input()

        X_train = []

        list_of_imgs = []
        img_dir = r'D:\picture\miku_square'
        imagesList = os.listdir(img_dir)

        for imgPath in imagesList:
            imgRead = cv2.imread(img_dir+'\\'+imgPath, cv2.IMREAD_COLOR)
            b, g, r = cv2.split(imgRead)
            imgRead = cv2.merge([r, g, b])
            imgRead = cv2.resize(imgRead, (self.img_rows, self.img_rows))
            # list_of_imgs.append(imgRead.flatten())
            list_of_imgs.append(imgRead)
        X_train = np.array(list_of_imgs)
        print(X_train.shape)
        input()
        # Rescale -1 to 1
        X_train = (X_train) / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        print(X_train[0].shape)
        input()
        gen_imgs = 0.5 * X_train[0] + 0.5
        plt.imshow(gen_imgs)
        plt.show()

        epoch = 3060
        while epoch < (epochs):
            epoch += 1
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # self.showImage()
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_model(epoch)
            if epoch % 5 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # gen_imgs = self.generator.predict(noise)
        gen_imgs = self.generator.predict(self.sampleNoise)
        # print('shape')
        # print(gen_imgs.shape)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.abspath(os.path.dirname(__file__))+'/save/graph_'+str(epoch)+'.png', dpi=300)
        plt.close()

    def showImage(self):
        r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # gen_imgs = self.generator.predict(noise)
        gen_imgs = self.generator.predict(self.sampleNoise)
        print('shape')
        print(gen_imgs.shape)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        plt.close()

    def save_model(self, epoch):
        with open(os.path.abspath(os.path.dirname(__file__))+'/save/sampleNoise.p', 'rb') as file:
            self.sampleNoise = pickle.load(file)

        self.discriminator.save_weights(os.path.abspath(os.path.dirname(__file__)) +
                                        '/save/discriminator_weights.h5')
        self.generator.save_weights(os.path.abspath(os.path.dirname(__file__)) +
                                    '/save/generator_weights.h5')
        self.discriminator.save_weights(os.path.abspath(os.path.dirname(__file__)) +
                                        '/save/discriminator_weights' + str(epoch) + '.h5')
        self.generator.save_weights(os.path.abspath(os.path.dirname(__file__)) +
                                    '/save/generator_weights' + str(epoch) + '.h5')

    def load_model(self):
        self.discriminator.load_weights(os.path.abspath(os.path.dirname(__file__)) +
                                        '/save/discriminator_weights.h5')
        self.generator.load_weights(os.path.abspath(os.path.dirname(__file__)) +
                                    '/save/generator_weights.h5')


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=5000, batch_size=32, save_interval=5)
