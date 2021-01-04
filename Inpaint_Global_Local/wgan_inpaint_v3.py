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
########## Import line #############
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
#tf.config.experimental_run_functions_eagerly(True)
class GAN():

    def __init__(self,input_dim,blank_dim,batch_size):
        self.input_dim = input_dim
        self.blank_dim = blank_dim
        self.batch_size = batch_size
        self.mask_dim = (4,)
        self.weight_save_dir ="./weight_wgan/"
        self.img_save_dir = "./data/inpaint/"
        self._build()
        
    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)
    
    # def compute_gradients(self, tensor, val_list):
    #     grads = tf.gradients(tensor, val_list)
    #     return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(val_list, grads)]

    # def gradient_penalty_loss(self, y_true, y_pred, samples, sample_weight=None):
    #     gradients = self.compute_gradients(y_pred, [samples])[0]
    #     gradient_l2_norm = K.sqrt(
    #         K.sum(
    #             K.square(gradients),
    #             axis=list(range(1, len(gradients.shape))
    #                       )
    #         )
    #     )
    #     return K.mean(K.square(1 - gradient_l2_norm))
    def conv_bn(self,img, kernel_size,dilated, strides, filters,normalize = True, activation = "leakyrelu",padding="same"):
        x = img
        if strides <0:
            x = UpSampling2D()(x)
            strides = 1
    
        x =Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilated,
            padding=padding,
        )(x)

        if normalize == True:
            x = BatchNormalization()(x)
        if activation == "leakyrelu":
            x = LeakyReLU()(x)
        else:
            x = Activation("tanh")(x)

        return x
            
    @tf.function
    def random_crop_layer(self,real_inputs):

        tmp_box = np.ones((self.batch_size,self.mask_dim[0]),dtype="float32")
        black_box = np.ones((self.batch_size,self.input_dim[0],self.input_dim[1],3),dtype="float32")
        white_box = np.zeros((self.batch_size,self.input_dim[0],self.input_dim[1],3),dtype="float32")
        for i in range(self.batch_size):
            a_1 = np.random.randint(0, self.input_dim[0]-self.blank_dim[0]+1)
            b_1 = np.random.randint(0, self.input_dim[1]-self.blank_dim[1]+1)
            
            a_2, b_2 = a_1 + self.blank_dim[0], b_1 + self.blank_dim[1]
            black_box[i,a_1:a_2,b_1:b_2,:]=[0.,0.,0.]
            white_box[i,a_1:a_2,b_1:b_2,:]=[1.,1.,1.]
            a_1 = a_1 / self.input_dim[0]
            a_2 = a_2 / self.input_dim[0]
            b_1 = b_1 / self.input_dim[1]
            b_2 = b_2 / self.input_dim[1]

            tmp_box[i,:] = tmp_box[i,:]*[a_1, a_2, b_1, b_2]


        mask = np.array(tmp_box)
        mask = tf.convert_to_tensor(mask,dtype="float32")
        masked_img = Lambda(lambda x :tf.math.multiply(x[0],x[1]))((real_inputs,black_box))
        
        return real_inputs, masked_img, mask,black_box,white_box
    @tf.function
    def masking_layer(self, img, mask):
        box_indices = np.arange(self.batch_size)
        patch = tf.image.crop_and_resize(img,mask,box_indices,(self.blank_dim[0],self.blank_dim[1]))
        return img, patch 

    def stage_1_model(self):
        kernel_norm = 12
        input_img = Input(shape=self.input_dim)
        x = input_img
        x = self.conv_bn(x, 5, 1, 1, kernel_norm)
        x = self.conv_bn(x, 3, 1, 2, kernel_norm*2) 
        #x = MaxPooling2D()(x)
        x = self.conv_bn(x, 3, 1, 2, kernel_norm*4)
        #x = MaxPooling2D()(x)
        x = self.conv_bn(x, 3, 2, 1, kernel_norm*4)
        x = self.conv_bn(x, 3, 4, 1, kernel_norm*4)
        x = self.conv_bn(x, 3, 8, 1, kernel_norm*4)
        x = self.conv_bn(x, 3, 16, 1, kernel_norm*4)
        #x = MaxPooling2D()(x)
        x = self.conv_bn(x, 3, 1, -2, kernel_norm*4)
        x = self.conv_bn(x, 3, 1, 1, kernel_norm*4)
        x = self.conv_bn(x, 3, 1, -2, kernel_norm*1)
        #x = MaxPooling2D()(x)
        x = self.conv_bn(x, 3, 1, 1, 24)
        x = self.conv_bn(x, 3, 1, 1, 3,True,activation='tanh')
        
        
        self.stage_1 = Model(input_img,x)

    def generator_model(self):
        fake_img = Input(shape=self.input_dim)
        black_box = Input(shape=self.input_dim)
        white_box = Input(shape=self.input_dim)
        ############
        # Image Crop and mix with fake img(pass stage_1)
        ###########
        
        stage_out = self.stage_1(fake_img)
        output_img = tf.math.multiply(fake_img,white_box)+tf.math.multiply(stage_out,black_box)
       
        self.generator = Model([fake_img,black_box,white_box], output_img)

    def local_discriminator_model(self):
        input_img = Input(shape=self.blank_dim)
        x = input_img
        x = self.conv_bn(x,5,1,2,64,False)
        x = self.conv_bn(x,5,1,2,256,False)
        x = self.conv_bn(x,5,1,2,256,False)
        x = self.conv_bn(x,5,1,2,64,False)
        #In discriminator, batch Normalization doesn't needful

        x = Flatten()(x)

        output_img = x
        self.local_discriminator = Model(input_img,output_img)

    def global_discriminator_model(self):
        input_img = Input(shape=self.input_dim)
        x = input_img
        x = self.conv_bn(x,5,1,2,64,False)
        x = self.conv_bn(x,5,1,2,128,False)
        x = self.conv_bn(x,5,1,2,256,False)
        x = self.conv_bn(x,5,1,2,256,False)
        x = self.conv_bn(x,5,1,2,256,False)
        x = self.conv_bn(x,5,1,2,64,False)
        x = Flatten()(x)

        output_img = x
        self.global_discriminator = Model(input_img,output_img)

    def discriminator_model(self):
        input_img = Input(shape=self.input_dim)
        #mask_input = Input(shape=self.mask_dim)
        #img, patch = self.masking_layer(input_img,mask_input)
        patch = Input(shape = self.blank_dim)
        g_out = self.global_discriminator(input_img)
        l_out = self.local_discriminator(patch)
        concat_out = concatenate([g_out,l_out])
        
        concat_out = Flatten()(concat_out)
        concat_out = Dense(1, activation="linear")(concat_out)
        self.discriminator = Model([input_img, patch], concat_out)

    def critic_model(self):
        input_img = Input(shape=self.input_dim)
        real, crop_img, mask,black_box,white_box = Lambda(self.random_crop_layer)(input_img)
        fake = self.generator([crop_img,black_box,white_box])

        interpolate = RandomWeightedAverage(self.batch_size)(inputs=[real,fake])        
        real, real_patch = Lambda(lambda x:self.masking_layer(x[0],x[1]))((real,mask))
        fake, fake_patch = Lambda(lambda x:self.masking_layer(x[0],x[1]))((fake, mask))
        inter, inter_patch = Lambda(lambda x:self.masking_layer(x[0],x[1]))((interpolate, mask))
        valid_real = self.discriminator([real,real_patch])
        valid_fake = self.discriminator([fake,fake_patch])
        valid_inter = self.discriminator([inter,inter_patch])

        self.gp = GradientPenalty()([valid_inter,inter_patch])

        self.critic = Model(input_img,[valid_real,valid_fake,self.gp])

    def generator_discrimin_model(self):
        input_img = Input(shape=self.input_dim)
        real, crop_img, mask,black_box,white_box = Lambda(self.random_crop_layer)(input_img)
        fake = self.generator([crop_img,black_box,white_box])
        fake, patch = Lambda(lambda x: self.masking_layer(x[0], x[1]))((fake, mask))
        valid_fake = self.discriminator([fake,patch])

        self.generator_discrimin = Model(input_img,valid_fake)


    def critic_compile(self):
        self.generator.trainable = False
        self.discriminator.trainable = True

        self.critic.compile(
            loss=[self.wasserstein, self.wasserstein, "mse"],
            optimizer=Adam(lr=0.004),
            loss_weights=[1, 1, 10],
        )

    def generator_discrimin_compile(self):
        self.generator.trainable = True
        self.discriminator.trainable = False

        self.generator_discrimin.compile(
            loss=self.wasserstein,
            optimizer=Adam(lr=0.004)
        )
  
    def _build(self):
        self.stage_1_model()
        self.generator_model()
        self.local_discriminator_model()
        self.global_discriminator_model()
        self.discriminator_model()

        self.critic_model()
        self.generator_discrimin_model()

        # self.critic.summary()
        # self.generator_discrimin.summary()

        self.critic_compile()
        self.generator_discrimin_compile()
        
    def critic_train(self,img):
        batch_size = self.batch_size
        valid_out = np.ones((batch_size,1),dtype = np.float32)
        fake_out = -np.ones((batch_size,1),dtype = np.float32)
        dummy_out = np.zeros((batch_size,1),dtype = np.float32)

        return self.critic.fit(img, [valid_out, fake_out, dummy_out],batch_size=self.batch_size,epochs = 5)

    def generator_train(self, img):
        batch_size = self.batch_size
        valid = np.ones((batch_size, 1), dtype=np.float32)

        return self.generator_discrimin.fit(img, valid,batch_size = self.batch_size)

    def train(self,epoch,train_data):
        
        for epoch_num in range(1,epoch+1):
            train_img = next(train_data)[0]
            if train_img.shape[0] != self.batch_size:
                train_img = next(train_data)[0]
            
            d_loss =self.critic_train(train_img)
        
            g_loss = self.generator_train(train_img)


            #print("{} s:{}, d:{}, g:{}".format(epoch_num,s_1_loss, d_loss, g_loss))
            if epoch_num%100 == 0:
                print("{} :, d:{}, g:{}".format(epoch_num,d_loss,g_loss))
                self.save_img(train_img[0],epoch_num)
                self.save_weight()

    def save_img(self,img,epoch):
        real,fake,mask,black_box,white_box = self.random_crop_layer(img)
        save_img(self.img_save_dir+"blanked_img"+str(epoch)+".jpg",fake)
        save_img(self.img_save_dir+"reconst"+str(epoch)+".jpg",self.generator.predict([fake,black_box,white_box]))

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
    DATA_PATH = '/home/sjo506/Gan_practice/test/celeba'
    BATCH_SIZE = 1
    data_gen = ImageDataGenerator(rescale=1./255) 
    data_flow = data_gen.flow_from_directory(DATA_PATH,
    target_size = [512,512],
    batch_size = BATCH_SIZE,
    shuffle= True,
    class_mode='input',
    subset='training'
        )
    return data_flow

gan = GAN(
    input_dim=[512, 512, 3],
    blank_dim=[64, 64, 3],
    batch_size=1,
    #using_generator=True
)
data = celeba_data()
gan.train(2000,data)
