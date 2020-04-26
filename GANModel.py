from keras.layers import Input, Dense,LSTM, Reshape, Flatten, Dropout,CuDNNLSTM,Bidirectional
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from numpy import savetxt
from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np

class GANModel():
    def __init__(self,max_len,table_len,int_to_smi):
        self.img_rows = max_len
        self.img_cols = table_len
        self.channels = 1
        self.smile_length = max_len
        self.smile_shape = (self.smile_length, 1)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.table_len = table_len
        self.latent_dim = 100
        self.int_to_smi = int_to_smi
        optimizer = Adam(0.0002, 0.5)


        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img) #Validate images as real/ fake
        self.gan = Model(z, validity)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

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
        model.add(Dense(np.prod(self.smile_shape), activation='tanh'))
        model.add(Reshape(self.smile_shape))
        model.summary()
        
        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)
    def discriminator(self):
        model = Sequential()
        model.add(LSTM(512, input_shape=self.smile_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.smile_shape)
        validity = model(seq)

        return Model(seq, validity)
    def train(self,epochs, batch_size, sample_interval, X_train):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in tqdm(range(epochs)):
            print("\nEpoch .................",epoch)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_data = self.generator.predict(noise)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            g_loss = self.gan.train_on_batch(noise, valid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        self.discriminator.trainable = False
        self.gan.save("gan.h5")
        self.discriminator.trainable = True
        self.generator.save("generator.h5")
        self.discriminator.save("discriminator.h5")
    def sample_images(self):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print(type(gen_imgs))
        print(gen_imgs.shape)
        print(gen_imgs)
        f = open("pred_smiles.txt","w")
        for i in gen_imgs:
            pred_smiles = [((self.table_len/2) * x) + (self.table_len/2) for x in i]
            pred_smiles = [self.int_to_smi[int(x)] for x in pred_smiles]
            x = "".join(pred_smiles)
            print(x)
            f.write(x)
            f.write("\n")
        f.close()
        print(pred_smiles)
    def load_models(self):
        discriminator = load_model('discriminator.h5')
        generator = load_model('generator.h5')
        gan = load_model('gan.h5')
        gan.summary()
        discriminator.summary()
        generator.summary()

        
        