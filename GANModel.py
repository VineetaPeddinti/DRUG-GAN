from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class GANModel():
    def __init__(self,max_len,table_len):
        self.img_rows = max_len
        self.img_cols = table_len
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(0.9, 0.999)

        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.generator()
        self.generator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator.trainable = False

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        

        validity = self.discriminator(img) #Validate images as real/ fake

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='softmax'))
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)
        # return model
    def discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)
        # return model
    def train(self,epochs, batch_size, sample_interval, X_train):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_data = self.generator.predict(noise)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            g_loss = self.combined.train_on_batch(noise, valid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    def sample_images(self):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        #gen_imgs = 0.5 * gen_imgs + 0.5
        print(gen_imgs)
        return gen_imgs
        