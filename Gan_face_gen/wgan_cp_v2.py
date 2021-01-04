import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, MaxPooling2D, ELU, concatenate

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
import random

########## Import line #############
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
#tf.config.experimental_run_functions_eagerly(True)


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class GradientPenalty(tf.keras.layers.Layer):
    def call(self,inputs):
        (target,wrt) = inputs
        grad = tf.gradients(target,wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)),axis =1, keepdims = True))-1

    def compute_output_shape(self,input_shape):
        return (input_shape[1][0],1)

class GAN():

    def __init__(self,input_dim,batch_size):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.z_dim =100
        self.generator_initial_dense_layer_size = (4, 4, 512)
        self.weight_save_dir ="./data/weight/face_make/256/"
        self.img_save_dir = "./data/face_make/256/"
        self._build()
        
    def wasserstein(self, y_true, y_pred):
        return K.mean(y_true * y_pred,axis=-1)
    
    def conv_reg(self,img,filter,kernel,strides,conv_trans = False,batch_norm = True,active ="leakyrelu"):
        x = img 
        if conv_trans== True:
            x = Conv2DTranspose(
                filters = filter,
                kernel_size = kernel,
                strides = strides,
                padding = 'same'
            )(x)

        elif conv_trans == "upsample":
            x = UpSampling2D()(x)
            x = Conv2D(filters = filter,
                    kernel_size = kernel,
                    strides =1,
                    padding ='same'
            )(x)
        else:
            x = Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=strides,
                padding='same'
            )(x)
        if batch_norm == True:
            x = BatchNormalization()(x)


        if active == "leakyrelu":
            x = LeakyReLU()(x)
        else:
            x = Activation('tanh')(x)
        return x
    def generator_model(self):
        generator_input = Input(shape=(self.z_dim,))
        x = generator_input
        x = Dense(np.prod(self.generator_initial_dense_layer_size))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        #x = Dropout(rate=0.4)(x)
        x = self.conv_reg(x,256,5,2,"upsample",True)
        x = self.conv_reg(x,128,5,2,"upsample",True)
        x = self.conv_reg(x,64,5,2,"upsample",True)
        #x = self.conv_reg(x,64,5,2,True,True)
        x = self.conv_reg(x,32,5,2,"upsample",True)
        #x = self.conv_reg(x,32,5,2,True,True)
        x = self.conv_reg(x,16,5,2,"upsample",True)
        x = self.conv_reg(x,3,5,1,"upsample",False,"tanh")
        # 16 *(2^6) = 1024 # output_img_size
        generator_output = x
        self.generator = Model(generator_input, generator_output)

    def discriminator_model(self):
        discriminator_input = Input(shape=self.input_dim)
        x =discriminator_input

        x = self.conv_reg(x,16,5,2,False,False)
        x = self.conv_reg(x,16,5,2,False,False)
        x = self.conv_reg(x,32,5,2,False,False)
        #x = self.conv_reg(x,32,5,2,False,True)
        x = self.conv_reg(x,64,5,2,False,False)
        #x = self.conv_reg(x,64,5,2,False,True)
        x = self.conv_reg(x,256,5,2,False,False)
        x = self.conv_reg(x,512,5,1,False,False)

        x = Flatten()(x)

        discriminator_output = Dense(1,activation="linear")(x)

        self.discriminator = Model(discriminator_input,discriminator_output) 

    def critic_model(self):
        real_img = Input(shape=self.input_dim)
        z_disc = Input(shape =(self.z_dim, ))
        fake_img = self.generator(z_disc)
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)
        # fake = fake_image's discriminator value
        # valid = real_image's discriminator value

        interpolated_img = RandomWeightedAverage(self.batch_size)(inputs = [real_img,fake_img])
        valid_inter =self.discriminator(interpolated_img)
        # validity_interpolated = interpolate's image discimiator value
        self.gp = GradientPenalty()([valid_inter,interpolated_img])
        
        self.critic = Model(inputs=[real_img, z_disc], outputs=[valid, fake, self.gp])

    def generator_discrimin_model(self):
        model_input = Input(shape=(self.z_dim,))
        model_output = self.discriminator(self.generator(model_input))
        self.generator_discrimin = Model(model_input,model_output)
        # In this model = Random noise -> generate -> discriminate 
        # Mean : generator's fake image discriminating

    def critic_compile(self):
        self.generator.trainable = False
        self.discriminator.trainable = True

        self.critic.compile(
            loss=[self.wasserstein, self.wasserstein, "mse"],
            optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.9),
            loss_weights=[1, 1, 10],
        )

    def generator_discrimin_compile(self):
        self.generator.trainable = True
        self.discriminator.trainable = False

        self.generator_discrimin.compile(
            loss=self.wasserstein,
            optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)
        )

    def critic_train(self,img):
        batch_size = self.batch_size
        valid_out = np.ones((batch_size,1),dtype = np.float32)
        fake_out = -np.ones((batch_size,1),dtype = np.float32)
        dummy_out = np.zeros((batch_size,1),dtype = np.float32)

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        #Latent_vector
        return self.critic.fit([img,noise], [valid_out, fake_out, dummy_out],batch_size=self.batch_size,epochs = 5)

    def generator_train(self):
        batch_size = self.batch_size
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0,1,(batch_size,self.z_dim))

        return self.generator_discrimin.fit(noise, valid,batch_size = self.batch_size,epochs = 1)

    def train(self,epoch,train_data):
        
        for epoch_num in range(0,epoch+1):
            train_img = next(train_data)[0]
            if train_img.shape[0] != self.batch_size:
                train_img = next(train_data)[0]
            
            d_loss =self.critic_train(train_img)

            g_loss = self.generator_train()
            #print("{} d:{}, g:{}".format(epoch_num, d_loss, g_loss))
            if epoch_num%100 == 0:
                print(epoch_num,end=" : ")
                self.save_img(epoch_num)
                self.save_weight()

    def _build(self):
        self.generator_model()
        self.discriminator_model()

        self.critic_model()
        self.generator_discrimin_model()

        # self.critic.summary()
        # self.generator_discrimin.summary()

        self.critic_compile()
        self.generator_discrimin_compile()

    def save_weight(self):
        self.generator_discrimin.save_weights(self.weight_save_dir + "generator.h5")
        self.critic.save_weights(self.weight_save_dir + "critic.h5")
        print("weight save")

    def load_weight(self):
        try:
            self.generator_discrimin.load_weights(self.weight_save_dir + "generator.h5")
            self.critic.load_weights(self.weight_save_dir + "critic.h5")
        except:
            self.save_weight()
            self.load_weight()

        print("weight loaded")

    def save_img(self,epoch):
        noise = np.random.normal(0,1,(1,self.z_dim,))
        fake_img = self.generator.predict(noise)
        save_img(self.img_save_dir+"face_img"+str(epoch)+".jpg",fake_img[0])
        fake_img = np.clip((0.5*(fake_img+1)),0,1)
        save_img(self.img_save_dir+"clipped_face_img"+str(epoch)+".jpg",fake_img[0])


def coco_data():
    DATA_PATH = "./data/cocodata"
    BATCH_SIZE = 12
    data_gen = ImageDataGenerator(rescale =1./255)
    data_flow =data_gen.flow_from_directory(
        DATA_PATH,
        target_size = [256,256],
        batch_size = BATCH_SIZE,
        shuffle = False,
        class_mode = "input",
        subset = "training"
        )
    return data_flow

def celeba_data():
    DATA_PATH = './data/celeba/'
    BATCH_SIZE = 16
    data_gen = ImageDataGenerator(rescale=1./255) 
    data_flow = data_gen.flow_from_directory(DATA_PATH,
    target_size = [256,256],
    batch_size = BATCH_SIZE,
    shuffle= True,
    class_mode='input',
    subset='training'
        )
    return data_flow

gan = GAN(
    input_dim=(256, 256, 3),
    batch_size=16,
    #using_generator=True
)
data = celeba_data()
gan.load_weight()
print("-------------------------Train-------------------------")
gan.train(20000,data)
