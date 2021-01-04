import sys, os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='1'

import tensorflow as tf
import numpy as np
import glob
import pathlib
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from functools import partial

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


print('''+++ env info +++
Python version : {},
Tensorflow version : {},
Keras version : {}
'''.format(sys.version, tf.__version__, tf.keras.__version__))

print('TF GPU available test :', tf.config.list_physical_devices('GPU'))

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model




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
        return K.sqrt(tf.math.reduce_sum(K.batch_flatten(K.square(grad)),axis =1, keepdims = True))-1

    def compute_output_shape(self,input_shape):
        return (input_shape[1][0],1)

# For Gradient Penalty and interpolation

class inpainting():
    def __init__(self,
            data,
            global_shape = (128,128,3),
            local_shape = (64,64,3),
            batch_size = 16,
            weight_save_dir = "./data/weight/inpaint_3/",
            img_save_dir = "./data/inpaint_3/"):
        self.train_data = data
        self.local_shape, self.global_shape = local_shape,global_shape

        self.img_h, self.img_w, self.channel = global_shape[0],global_shape[1],global_shape[2]
        self.inpainted_h, self.inpainted_w = local_shape[0],local_shape[1]

        self.batch_size = batch_size
        self.margin = 5
        self.patch_size = local_shape[0]

        self.weight_save_dir = weight_save_dir
        self.img_save_dir = img_save_dir

        self.sample_interval =100


        self.compile()

    def wasserstein(self, y_true, y_pred):
        return K.mean(y_true * y_pred,axis=-1)

    def block_patch(self,input_, mode, patch_size=50, margin=10):
        input_ = tf.convert_to_tensor(input_)
        shape = input_.get_shape().as_list()

        if mode == 'central_box':
            #create patch in central size
            pad_size = tf.constant([patch_size, patch_size], dtype=tf.int32)
            patch = tf.zeros([shape[0], pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)

            h_ = tf.constant([int(shape[1]/4)], dtype=tf.int32)[0]
            w_ = tf.constant([int(shape[2]/4)], dtype=tf.int32)[0]
            coord = h_, w_
            padding = [[0, 0], [h_, shape[1]-h_-pad_size[0]], [w_, shape[2]-w_-pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            result = tf.multiply(input_, padded)

        if mode == 'random_box':
            #create patch in random box
            pad_size = tf.constant([patch_size, patch_size], dtype=tf.int32)
            patch = tf.zeros([shape[0], pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)

            h_ = tf.random.uniform([1], minval=margin, maxval=shape[1]-pad_size[0]-margin, dtype=tf.int32)[0]
            w_ = tf.random.uniform([1], minval=margin, maxval=shape[2]-pad_size[1]-margin, dtype=tf.int32)[0]
            coord = h_, w_
            padding = [[0, 0], [h_, shape[1]-h_-pad_size[0]], [w_, shape[2]-w_-pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            result = tf.multiply(input_, padded)    
            
        return result, padded, coord, pad_size

    def inpaint_crop(self,img, coord, pad_size):
        crop_box = tf.image.crop_to_bounding_box(img, coord[0], coord[1], pad_size[0], pad_size[1])
        crop_bbox = tf.image.resize(crop_box, (self.inpainted_h, self.inpainted_w))
        return crop_bbox

    def build_generator(self,global_shape):
        
        inputs = tf.keras.Input(shape=self.global_shape) # (None, 128, 128, 3)
        
        # Encoder
        x = Conv2D(64, kernel_size=5, strides=1, padding='same', activation='tanh')(inputs) # (None, 64, 64, 64)
        
        x = Conv2D(128, kernel_size=3, strides=2, padding='same', activation='tanh')(x)
        x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        
        x = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        
        # Dilated_Conv
        x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=2, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=4, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=8, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=16, padding='same', activation='tanh')(x)

        x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        
        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
        x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
        x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        x = Conv2D(self.channel, kernel_size=3, strides=1, padding='same', activation='tanh')(x)    

        x = tf.keras.Model(inputs=inputs, outputs=x, name='G')

        return x
        
    def build_discriminator(self,local_shape, global_shape):

        local_inputs = tf.keras.Input(shape=self.local_shape, name='local_input') # (None, 64, 64, 3)
        global_inputs = tf.keras.Input(shape=self.global_shape, name='global_input') # (None, 64, 64, 3)
        
        local = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='tanh')(local_inputs) 
        local = Conv2D(128, kernel_size=5, strides=2, padding='same', activation='tanh')(local) 
        local = Conv2D(256, kernel_size=5, strides=2, padding='same', activation='tanh')(local) 
        local = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='tanh')(local) 
        local = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='tanh')(local) 
        local = Flatten()(local) 
        local = Dense(1024)(local)

        global_ = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='tanh')(global_inputs) 
        global_ = Conv2D(128, kernel_size=5, strides=2, padding='same', activation='tanh')(global_) 
        global_ = Conv2D(256, kernel_size=5, strides=2, padding='same', activation='tanh')(global_) 
        global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='tanh')(global_) 
        global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='tanh')(global_) 
        global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='tanh')(global_) 
        global_ = Flatten()(global_) 
        global_ = Dense(1024)(global_)

        D_output = tf.keras.layers.concatenate([local, global_])
        D_output = Dense(1, activation='tanh')(D_output)
        D_output = tf.keras.Model(inputs=[local_inputs, global_inputs], outputs=D_output, name='D')
        
        return D_output

    def build_critic(self):

        real_img = Input(self.global_shape)
        real_field = Input(self.local_shape)
        fake_img = Input(self.global_shape)
        fake_field = Input(self.local_shape)
        interpolated_img = Input(self.global_shape)
        interpolated_field = Input(self.local_shape)


        real_validity = self.D([real_field, real_img])
        fake_validity = self.D([fake_field, fake_img])
        interpolated_validity = self.D([interpolated_field, interpolated_img])

        
        gp = GradientPenalty()([interpolated_validity,interpolated_field])
        critic = tf.keras.Model(inputs = [real_img, real_field, fake_img,fake_field,interpolated_img,interpolated_field], outputs = [real_validity,fake_validity,gp])

        return critic

    def build_generator_discrimin(self,coord,pad_size ):

        mask_img = Input(self.global_shape)
        inpainted_img = self.G(mask_img)
        inpainted_field = self.inpaint_crop(inpainted_img,coord,pad_size)    

        validity = self.D([inpainted_field, inpainted_img])

        return tf.keras.Model(mask_img , [inpainted_img, validity], name='GL')

    def compile(self):
        self.D = self.build_discriminator(self.local_shape, self.global_shape)
        self.G = self.build_generator(self.global_shape)

        self.C = self.build_critic()


        self.D.trainable = True
        self.G.trainable = False 
        self.C.compile(loss = [self.wasserstein,self.wasserstein,'mse'],loss_weights = [1,1,10],optimizer =Adam(lr=0.0001,beta_1=0.5,beta_2=0.9))#,metrics=['accuracy'])


        sample = next(self.train_data)[0]
        result, mask, coord, pad_size = self.block_patch(sample, mode='central_box', patch_size=self.patch_size, margin=self.margin)
        self.GL = self.build_generator_discrimin(coord,pad_size)
        
        self.D.trainable = False
        self.G.trainable = True

        self.GL.compile(loss=['mse', self.wasserstein], loss_weights=[0.9996, 0.0004], optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)) #,metrics=['accuracy'])

    def train(self,epoch):

        # Adversarial ground truths
        real = np.ones((self.batch_size, 1))
        fake = - np.ones((self.batch_size, 1))
        #fake = np.zeros((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))

        for epoch_num in range(0,epoch+1):
            train_img = next(self.train_data)[0]
            if train_img.shape[0] != self.batch_size:
                train_img = next(self.train_data)[0]
            

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images

            masked_imgs, mask, coord, pad_size = self.block_patch(train_img, mode='central_box', patch_size=self.patch_size, margin=self.margin)
            y1, y2, x1, x2 = coord[0].numpy(), coord[0].numpy()+pad_size[0].numpy(), coord[1].numpy(), coord[1].numpy()+pad_size[1].numpy()
        
            # Generate a batch of new images
            inpainted_imgs = self.G.predict(masked_imgs)
            inpainted_fields = self.inpaint_crop(inpainted_imgs, coord, pad_size)

            real_fields = self.inpaint_crop(train_img, coord, pad_size)
            
            filled_in = tf.identity(masked_imgs).numpy() # tensor copy
            filled_in[:, y1:y2, x1:x2, :] = inpainted_fields.numpy()
            filled_in = tf.constant(filled_in)

            # Train the discriminator
            #D.trainable = True

            interpolated_img = RandomWeightedAverage(self.batch_size)(inputs = [train_img, inpainted_imgs])
            interpolated_field = self.inpaint_crop(interpolated_img, coord, pad_size)


            for _ in range(1):
                d_loss = self.C.train_on_batch([train_img, real_fields, inpainted_imgs,inpainted_fields,interpolated_img,interpolated_field],[real,fake,dummy])

            #print(self.C.predict([train_img, real_fields, inpainted_imgs,inpainted_fields,interpolated_img,interpolated_field])[2])
            print("D:",d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------
            #D.trainable = False
            
            #g_loss = self.GL.train_on_batch([inpainted_imgs,inpainted_fields], [real_fields, real])
            g_loss = self.GL.train_on_batch(masked_imgs, [train_img, real])
            print("G:",g_loss)

            print("\n{} epoch".format(epoch_num))
            # If at save interval => save generated image samples
            if epoch_num % self.sample_interval == 0:

                save_img(self.img_save_dir+"filled_in"+str(epoch_num) +".jpg",filled_in[0])
                
                self.save_weight()

    def save_weight(self):
        self.GL.save_weights(self.weight_save_dir + "generator.h5")
        self.C.save_weights(self.weight_save_dir + "critic.h5")
        print("weight save")

    def load_weight(self):
        try:
            self.GL.load_weights(self.weight_save_dir + "generator.h5")
            self.C.load_weights(self.weight_save_dir + "critic.h5")
        except:
            self.save_weight()
            self.load_weight()
        print("weight loaded")


def coco_data():
    DATA_PATH = "./data/cocodata"
    BATCH_SIZE = 16
    data_gen = ImageDataGenerator(rescale =1./255)
    data_flow =data_gen.flow_from_directory(
        DATA_PATH,
        target_size = [512,512],
        batch_size = BATCH_SIZE,
        shuffle = True,
        class_mode = "input",
        subset = "training"
        )
    return data_flow

def celeba_data():
    DATA_PATH = '/home/sjo506/Gan_practice/test/data/celeba'
    BATCH_SIZE = 1
    data_gen = ImageDataGenerator(rescale=1./255) 
    data_flow = data_gen.flow_from_directory(DATA_PATH,
    target_size = [512,512],
    batch_size = 1,
    shuffle= True,
    class_mode='input',
    subset='training'
        )
    return data_flow


#data = coco_data()
data = celeba_data()
NN = inpainting(data = data,
        global_shape = (512,512,3),
        local_shape = (256,256,3),
        batch_size = 1,
        weight_save_dir = "./data/weight/inpaint_3/",
        img_save_dir = "./data/inpaint_3/")
#NN.load_weight()
NN.train(50000)
