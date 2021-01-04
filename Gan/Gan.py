from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.initializers import RandomNormal

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

class GAN():
    def __init__(self
        , input_dim
        , discriminator_conv_filters
        , discriminator_conv_kernel_size
        , discriminator_conv_strides
        , generator_initial_dense_layer_size
        , generator_upsample
        , generator_conv_filters
        , generator_conv_kernel_size
        , generator_conv_strides
        , z_dim
        ):

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides

        self.z_dim = z_dim
        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)


        self._build()
    def _build(self):
        
        ## Discriminator ##
        discriminator_input = Input(shape=self.input_dim)
        x =discriminator_input

        for i in range(self.n_layers_discriminator):
            conv_layer = Conv2D(
                filters= self.discriminator_conv_filters[i],
                kernel_size = self.discriminator_conv_kernel_size[i],
                strides = self.discriminator_conv_strides[i],
                padding = 'same', 
            )
            x = conv_layer(x)
            x = Activation('relu')(x)
            x = Dropout(rate=0.4)(x)

        x = Flatten()(x)
        discriminator_output = Dense(1,activation='sigmoid',kernel_initializer='random_normal')(x)

        self.discriminator = Model(discriminator_input,discriminator_output)

        ## Generator ##

        generator_input =Input(shape =(self.z_dim,))
        x = generator_input

        x = Dense(np.prod(self.generator_initial_dense_layer_size))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        x = Dropout(rate=0.4)(x)

        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                conv_layer = Conv2D(
                    filters = self.generator_conv_filters[i],
                    kernel_size = self.generator_conv_kernel_size[i],
                    padding = 'same'
                )
                x = conv_layer(x)
            else:
                conv_layer = Conv2DTranspose(
                    filters = self.generator_conv_filters[i],
                    kernel_size = self.generator_conv_kernel_size[i],
                    padding = 'same'
                )
                x = conv_layer(x)
            if i < self.n_layers_generator -1:
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            else:
                x = Activation('tanh')(x)
        
        generator_output = x
        self.generator = Model(generator_input,generator_output)


        ## Make Training Model ##

        
        ## Discriminator Compile ##
        self.discriminator.trainable = True
        self.discriminator.compile(
            optimizer = RMSprop(lr = 0.0008),
            loss = 'binary_crossentropy'
        )
        ## Generator Compile
        self.generator.compile(
            optimizer=RMSprop(lr=0.0008),
            loss='binary_crossentropy'
        )
        self.discriminator.trainable =False
        # For freezing discriminator Weight , which mean 
        model_input = Input(shape=(self.z_dim,))
        model_output = self.discriminator(self.generator(model_input))
        self. model = Model(model_input,model_output)


        self.model.compile(
            optimizer=RMSprop(lr=0.0004),
            loss = 'binary_crossentropy',
            metrics =['accuracy']
        )

        ## train GAN ##

    def train_discriminator(self,x_train,batch_size):
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        idx = np.random.randint(0,x_train.shape[0],batch_size)
        true_image = x_train[idx]
        #print(true_image.shape)
        self.discriminator.train_on_batch(true_image,valid)

        noise = np.random.normal(0,1,(batch_size,self.z_dim))
        self.gen_image = self.generator.predict(noise)
        #print(gen_image.shape)
        self.discriminator.train_on_batch(self.gen_image,fake)

    def train_generator(self,batch_size):
        valid  = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        self.model.train_on_batch(noise,valid)

    def train(self, x_train, epoch, batch_size):
        for i in range(epoch):
            self.train_discriminator(x_train,batch_size)
            self.train_generator(batch_size)
            print(i,end=" ")
            
            if i%100 == 0 :
                znew = np.random.normal(size=(1, gan.z_dim))
                plt.imshow(gan.generator.predict(znew)[0],cmap='gray')
                plt.show()

gan = GAN(input_dim = (28,28,1)
        , discriminator_conv_filters = [64,64,128,128]
        , discriminator_conv_kernel_size = [5,5,5,5]
        , discriminator_conv_strides = [2,2,2,1]
        , generator_initial_dense_layer_size = (7, 7, 64)
        , generator_upsample = [2,2, 1, 1]
        , generator_conv_filters = [128,64, 64,1]
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [1,1, 1, 1]
        , z_dim = 100
        )

data = np.load("/home/sjo506/Gan_practice/test/camel/camel.npy")
data = data/255.0
data = np.reshape(data,(len(data),28,28,1))
gan.train(data,2000,64)
plt.imshow(data[0],cmap = 'gray')
plt.show()
