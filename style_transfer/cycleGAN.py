from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

###### Import Line ######

class cyclegan():
    def __init__(self,
        input_dim,
        gen_n_filters,
        disc_n_filters,
        learning_rate,
        lamda_validity,
        lamda_reconst,
        lamda_id
        ):
        self.input_dim = input_dim
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters
        self.learning_rate = learning_rate
        self.lamda_validity = lamda_validity
        self.lamda_reconst = lamda_reconst
        self.lamda_id = lamda_id


        self._build()
    def downsample(self,layer_input,filters,f_size = 4):
        d = Conv2D(filters = filters,kernel_size = f_size, strides =2 , padding ="same")(layer_input)
        d = InstanceNormalization(axis =-1 ,center =False, scale = False)(d)
        d = Activation('relu')(d)

        return d

    def upsample(self,layer_input,skip_input,filters,f_size =4):
        u = UpSampling2D(size =2)(layer_input)
        u = Conv2D(filters = filters,kernel_size = f_size, strides = 1, padding ="same")(u)
        u = InstanceNormalization(axis = -1, center = False, scale = False)(u)
        u = Activation('relu')(u)

        u = Concatenate()([u,skip_input])

        return u

    def conv4(self,layer_input,filters,strides,norm):
        y = Conv2D(filters =filters, kernel_size = 4, strides =strides,padding ="same")(layer_input)

        if norm == True:
            y = InstanceNormalization(axis =-1, center = False,scale = False)(y)

        y =LeakyReLU()(y)

        return y

    def residual(self,layer_input,filters):
        short_cut = layer_input
        y = Conv2D(filters = filters, kernel_size = (3,3),strides =1 , padding = 'same')(layer_input)
        y = InstanceNormalization(axis = -1 ,center=False, scale = False)(y)
        y = Activation('relu')(y)
        y = Conv2D(filters = filters, kernel_size = (3,3),strides =1 , padding = 'same')(y)
        y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
        
        return add([short_cut,y])

    def u_net_generator_model(self):
        img = Input(shape = self.input_dim)

        d1 = self.downsample(img,self.gen_n_filters)
        d2 = self.downsample(d1, self.gen_n_filters*2)
        d3 = self.downsample(d2, self.gen_n_filters*4)
        d4 = self.downsample(d3, self.gen_n_filters*8)

        u1 = self.upsample(d4,d3,self.gen_n_filters*4)
        u2 = self.upsample(u1,d2,self.gen_n_filters*2)
        u3 = self.upsample(u2,d1,self.gen_n_filters)
        u4 = UpSampling2D(size =2)(u3)

        output = Conv2D(filters = self.input_dim[-1], kernel_size=4, strides=1, padding="same", activation = 'tanh')(u4)

        return Model(img,output)

    def discriminator_model(self):
        img = Input( shape = self.input_dim)

        y = self.conv4(img,self.disc_n_filters,strides=2, norm = False)
        y = self.conv4(y, self.disc_n_filters, strides=2, norm = True)
        y = self.conv4(y, self.disc_n_filters, strides=2, norm = True)
        y = self.conv4(y, self.disc_n_filters, strides=2, norm = True)

        output = Conv2D(filters =1, kernel_size =4 ,strides = 1, padding = 'same')(y)

        return Model(img,output)

    def generator_discrimin_model(self):
        self.d_A = self.discriminator_model()
        self.d_B = self.discriminator_model()
        self.g_AB = self.u_net_generator_model()
        self.g_BA = self.u_net_generator_model()


        img_A = Input(shape = self.input_dim)
        img_B = Input(shape = self.input_dim)
        
        fake_A = self.g_BA(img_B)
        fake_B = self.g_AB(img_A)

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        # For validity

        reconst_A = self.g_BA(fake_B)
        reconst_B = self.g_AB(fake_A)
        # For Reconstruction

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        # For Identity

        return Model(inputs = [img_A,img_B], outputs = [valid_A,valid_B,reconst_A,reconst_B,img_A_id,img_B_id])
        
    def discriminator_compile(self):
 

        self.d_A.compile(loss = 'mse',
            optimizer = Adam(self.learning_rate),
            metrics =['accuracy'])
        self.d_A.compile(loss='mse',
            optimizer=Adam(self.learning_rate),
            metrics=['accuracy'])

    def generator_discrimin_compile(self):
        self.d_A.trainable = False
        self.d_B.trainable = False

        self.generator_discrimin = self.generator_discrimin_model()

        self.generator_discrimin.compile(
            loss = ['mse', 'mse', 'mse', 'mse', 'mse', 'mse'],
            loss_weights = [
                self.lamda_validity,
                self.lamda_validity,
                self.lamda_reconst,
                self.lamda_reconst,
                self.lamda_id,
                self.lamda_id
            ],
            optimizer = Adam(self.learning_rate)
            )

    def train(self,img_A,img_B,epochs,batch_size =1):
        patch = int(self.input_dim[1]/2**4)
        ## 1/2 convoluion 4 times   
        self.disc_patch =(patch,patch,1)

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            fake_A = self.g_BA.predict(img_B)
            fake_B = self.g_AB.predict(img_A)

            dA_loss_real = self.d_A.train_on_batch(img_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)

            dB_loss_real = self.d_A.train_on_batch(img_B, valid)
            dB_loss_fake = self.d_A.train_on_batch(fake_B, fake)

            g_loss = self.generator_discrimin.train_on_batch(
                [img_A, img_B], [valid, valid, img_A, img_B, img_A, img_B])

        print(epoch, dA_loss_real,dA_loss_fake,dB_loss_real,dB_loss_fake,g_loss)

    def _build(self):
        self.u_net_generator_model()
        self.discriminator_model()
        self.generator_discrimin_model()
        self.discriminator_compile()
        self.generator_discrimin_compile()


gan = cyclegan(
    input_dim= (256,256,3),
    gen_n_filters=32,
    disc_n_filters=64,
    learning_rate=0.0002,
    lamda_validity = 1,
    lamda_reconst = 10,
    lamda_id = 5 
)

    
