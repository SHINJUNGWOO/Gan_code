import tensorflow as tf
import numpy as np
import glob
import pathlib
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Lambda
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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




class inpainting():
    def __init__(self,
        batch_size,
        global_shape,
        local_shape,
        weight_save_dir,
        img_save_dir
        ):
        self.batch_size = batch_size
        self.global_shape = global_shape
        self.local_shape = local_shape
        self.mask_shape = (global_shape[0],global_shape[1],1)
        self.weight_save_dir = weight_save_dir
        self.img_save_dir = img_save_dir
        self.extract_rate =8
        self.build()

    def wasserstein(self, y_true, y_pred):
        return K.mean(y_true * y_pred,axis=-1)


    def block_patch(self,input_, mode = 'random_box',x=10,y=10, margin=10):
        input_ = tf.convert_to_tensor(input_)
        shape = input_.get_shape().as_list()


        if mode == 'select_box':
            #create patch in central size
            pad_size = tf.constant([self.local_shape[0], self.local_shape[1]], dtype=tf.int32)
            patch = tf.zeros([self.batch_size, self.local_shape[0], self.local_shape[1], self.global_shape[-1]], dtype=tf.float32)

            h_ = tf.constant([y], dtype=tf.int32)[0]
            w_ = tf.constant([x], dtype=tf.int32)[0]
            padding = [[0, 0], [h_, shape[-3]-h_-pad_size[0]], [w_, shape[-2]-w_-pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
            result = tf.multiply(input_, padded)

        if mode == 'random_box':
            #create patch in random box
            pad_size = tf.constant([self.local_shape[0], self.local_shape[1]], dtype=tf.int32)
            patch = tf.zeros([self.batch_size, self.local_shape[0], self.local_shape[1], self.global_shape[-1]], dtype=tf.float32)

            h_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[0]-self.local_shape[0]-margin, dtype=tf.int32)[0]
            w_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[1]-self.local_shape[1]-margin, dtype=tf.int32)[0]

            padding = [[0, 0], [h_, shape[1]-h_-pad_size[0]], [w_, shape[2]-w_-pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            result = tf.multiply(input_, padded)    
            
        mask = tf.math.logical_not(tf.cast(padded,dtype = tf.bool))
        mask = tf.expand_dims(mask[:, :, :, 0],axis =-1)
        return result, mask

    def masking(self,img,mask):
        mask = tf.squeeze(mask,axis = -1)
        x = tf.boolean_mask(img,mask)
        x = tf.reshape(x,shape = (-1,self.local_shape[0],self.local_shape[1],self.local_shape[2]))
        return x
    
    def paste_patch(self,real_img,fake_img,mask):
        inverted_mask = tf.math.logical_not(mask)
        mask = tf.cast(mask,dtype=tf.float32)
        inverted_mask = tf.cast(inverted_mask,dtype=tf.float32)
        
        return mask * fake_img + inverted_mask * real_img

    @tf.function
    def context_attention(self,img,mask):
        img_shape = img.get_shape().as_list()
        img = tf.image.resize(img,size=[self.global_shape[0],self.global_shape[1]])
        mask = tf.squeeze(mask,axis = -1)
        fore_ground = tf.boolean_mask(img,mask)
        fore_ground = tf.reshape(fore_ground,shape = (-1,self.local_shape[0],self.local_shape[1],img_shape[-1]))


        fore_ground_size = (img_shape[1]//(self.global_shape[0]//self.local_shape[0]),
                            img_shape[2]//(self.global_shape[1]//self.local_shape[1]))
        
        #img = tf.image.resize(img,size=[img_shape[1],img_shape[2]])
        #fore_ground = tf.image.resize(img, size=[fore_ground_size[0], fore_ground_size[1]])

        self.extract_rate =16
        out_channel_num = self.global_shape[0]//self.extract_rate

        inverted_mask = tf.math.logical_not(mask)
        inverted_mask = tf.cast(inverted_mask,dtype=tf.int32)
        inverted_mask = tf.expand_dims(inverted_mask,axis =-1)
        back_ground_patch = tf.image.extract_patches(img,sizes=(1,1,1,1),strides=(1,self.extract_rate,self.extract_rate,1),rates=(1,1,1,1),padding="SAME")
        inverted_mask = tf.image.extract_patches(inverted_mask,sizes=(1,1,1,1),strides=(1,self.extract_rate,self.extract_rate,1),rates=(1,1,1,1),padding="SAME")
        inverted_mask = tf.cast(inverted_mask,dtype=tf.bool)

        inverted_mask = tf.squeeze(inverted_mask, axis=-1)
        back_ground_patch = tf.boolean_mask(back_ground_patch,inverted_mask)
        back_ground_patch = tf.reshape(back_ground_patch, shape = (-1,out_channel_num,img_shape[-1]))
        back_ground_patch = tf.transpose(back_ground_patch,[0,2,1])
        def cosine_sim(argument):
            obj = argument[0]
            patch = argument[1]
            patch = tf.expand_dims(patch,axis = 0)
            patch = tf.expand_dims(patch, axis = 0)
            patch =tf.transpose(patch, [3,0,1,2])

            patch = tf.image.resize(patch, size=[self.local_shape[0], self.local_shape[1]])

            out = tf.map_fn(lambda x: losses.CosineSimilarity(
                axis=-1, reduction=losses.Reduction.NONE)(obj, x),patch)

            return out
        x = tf.map_fn(cosine_sim,(fore_ground,back_ground_patch),dtype = tf.float32)
        x = tf.transpose(x, [0,2,3,1])
        x = tf.image.resize(x, size=[fore_ground_size[0], fore_ground_size[1]])
        return x



    def coarse_model(self):
        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape,dtype=tf.bool)
        x = Conv2D(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_)
        x = Conv2D(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(64, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=2, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=4, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=8, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=16, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(32, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(16, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(3, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)

        x = tf.clip_by_value(x, clip_value_min = -1, clip_value_max = 1)


        x = self.paste_patch(input_, x ,input_mask)
        self.coarse = Model([input_, input_mask], x, name="coarse")

    def g_discrimin_model(self):
        input_ = Input(self.global_shape)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same', activation="tanh")(input_)
        x = Conv2D(128, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Flatten()(x)
        x = Dense(1,activation = "tanh")(x)

        self.g_disrimin = Model(input_, x, name="g_discrim")

    def l_discrimin_model(self):
        input_ = Input(self.local_shape)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same', activation="tanh")(input_)
        x = Conv2D(128, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Conv2D(512, kernel_size=5, strides=2, padding='same', activation="tanh")(x)
        x = Flatten()(x)
        x = Dense(1, activation="tanh")(x)
 
        self.l_disrimin = Model(input_, x,  name="l_discrim")
    
    def refine_model(self):
        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape,dtype=bool)
        mask = tf.identity(input_mask)

        x = Conv2D(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_)
        x = Conv2D(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)


        #context = self.context_attention(x,mask)
        context = Lambda(lambda inp:self.context_attention(inp[0],inp[1]))((x,mask))
        #context = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)


        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(context)
        x = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x)
        # contexture attention 


        y = Conv2D(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_)
        y = Conv2D(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y)
        y = Conv2D(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)
        y = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y)
        y = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)
        y = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)       
        
        while(y.get_shape().as_list()[1] > x.get_shape().as_list()[1]):
            y = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y)
            y = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)

        y = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)
        y = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y)

        out = tf.concat([x,y], axis = -1)

        while(input_.get_shape().as_list()[1] > out.get_shape().as_list()[1]):
            out = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out)
            out = Conv2D(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out)
            out = UpSampling2D()(out)

        out = Conv2D(32, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out)
        out = Conv2D(16, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out)
        out = Conv2D(3, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out)

        out = tf.clip_by_value(out, clip_value_min = -1, clip_value_max = 1)
        
        out = self.paste_patch(input_, out, input_mask)
        self.refine = Model([input_, input_mask], out,  name="refine")

    def spartial_discounted_loss(self,x,y):
        loss_mask = [[0.99**((min(i,self.local_shape[1]-i)+min(j,self.local_shape[0]-j))//2) for i in range(self.local_shape[1])] for j in range(self.local_shape[0])]
        loss_mask =  np.array(loss_mask)
        diff = tf.abs(x - y)
        diff = tf.math.reduce_sum(diff,axis =-1)
        loss = loss_mask * diff
        loss = tf.math.reduce_mean(loss)
        return loss
        
    def generator_discrimin_model(self):

        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape,dtype=tf.bool)

        coarse_result = self.coarse([input_,input_mask])
        refine_result = self.refine([coarse_result,input_mask])

        coarse_patch = self.masking(coarse_result, input_mask)
        refine_patch = self.masking(refine_result,input_mask)
        g_valid = self.g_disrimin(refine_result)
        l_valid = self.l_disrimin(refine_patch)

        self.generator_discrimin = Model([input_,input_mask],[coarse_patch,refine_patch,g_valid,l_valid],name="generator")

    def critic_model(self):

        input_real = Input(self.global_shape)
        input_fake = Input(self.global_shape)
        input_mask = Input(self.mask_shape,dtype=tf.bool)

        coarse_result = self.coarse([input_fake,input_mask])
        refine_result = self.refine([coarse_result,input_mask])
        inter_result = RandomWeightedAverage(self.batch_size)(inputs = [input_real, refine_result])

        real_patch = self.masking(input_real,input_mask)
        fake_patch = self.masking(refine_result,input_mask)
        inter_patch = self.masking(inter_result, input_mask)

        g_real_valid = self.g_disrimin(input_real)
        l_real_valid = self.l_disrimin(real_patch)

        g_fake_valid = self.g_disrimin(refine_result)
        l_fake_valid = self.l_disrimin(fake_patch)

        g_inter_valid = self.g_disrimin(inter_result)
        l_inter_valid = self.l_disrimin(inter_patch)

        g_gp = GradientPenalty()([g_inter_valid, inter_result])
        l_gp = GradientPenalty()([l_inter_valid,inter_patch])

        self.critic = Model([input_real,input_fake,input_mask],[g_real_valid,l_real_valid,g_fake_valid,l_fake_valid,g_gp,l_gp],name="critic")

    def compile(self):

        self.coarse.trainable = True
        self.refine.trainable = True
        self.g_disrimin.trainable = False
        self.l_disrimin.trainable = False

        self.generator_discrimin.compile(
            loss = [self.spartial_discounted_loss,self.spartial_discounted_loss,self.wasserstein,self.wasserstein],
            loss_weights = [1,1,0.0004,0.0004],
            optimizer =Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)
            )

        self.coarse.trainable = False
        self.refine.trainable = False
        self.g_disrimin.trainable = True
        self.l_disrimin.trainable = True

        self.critic.compile(
            loss = [self.wasserstein,self.wasserstein,self.wasserstein,self.wasserstein,"mse","mse"],
            loss_weights = [1,1,1,1,10,10],
            optimizer =Adam(lr=0.0001,beta_1=0.5,beta_2=0.9)
            )


    def generator_model(self):
        input_ = Input(self.global_shape,name = "input")
        mask = Input(self.mask_shape,dtype=tf.bool)
        out = self.refine([self.coarse([input_, mask]),mask])
        out = tf.identity(out, name="output")
        self.generator = Model([input_, mask], out)

    def build(self):

        self.coarse_model()
        self.refine_model()
        self.g_discrimin_model()
        self.l_discrimin_model()
        self.generator_discrimin_model()
        self.critic_model()
        self.compile()
        # self.g_disrimin.summary()
        # self.critic.summary()
        self.generator_model()

    def train(self, train_data, epochs):
        real = np.ones((self.batch_size, 1))
        fake = - np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))


        for i in range(epochs):

            real_img = next(train_data)[0]
            if real_img.shape[0] != self.batch_size:
                real_img = next(train_data)[0]

            masked_img, mask = self.block_patch(real_img, mode='random_box')
            real_patch = self.masking(real_img,mask)
            for _ in range(1):
                D = self.critic.train_on_batch([real_img, masked_img, mask], [
                                           real, real, fake, fake, dummy,dummy])

            G = self.generator_discrimin.train_on_batch(
                [masked_img, mask], [real_patch, real_patch, real, real])


            print("epoch = {} D ={} G={}".format(i,D,G))

            if i %100 == 0:
                self.save_img(masked_img, mask, i)
                self.save_weight()
                #plt.imshow(coarse_result[0])
                #plt.show()
                #plt.imshow(refine_result[0])
                #plt.show()

    def save_weight(self):
        self.generator_discrimin.save_weights(self.weight_save_dir + "generator.tf")
        self.critic.save_weights(self.weight_save_dir + "critic.tf")
        print("weight save")

    def load_weight(self):
        try:
            self.generator_discrimin.load_weights(self.weight_save_dir + "generator.tf")
            self.critic.load_weights(self.weight_save_dir + "critic.tf")
        except:
            self.save_weight()
            self.load_weight()
        print("weight loaded")

    def save_img(self, masked_img,mask, epoch):
        coarse_result = self.coarse.predict([masked_img, mask])
        refine_result = self.refine.predict([coarse_result, mask])
        save_img(self.img_save_dir+"blanked_img"+str(epoch)+".jpg", masked_img[0])
        save_img(self.img_save_dir+"coarse"+str(epoch) +".jpg", coarse_result[0])
        save_img(self.img_save_dir+"refine"+str(epoch) +".jpg", refine_result[0])


    def test_img(self,img_dir,axis):
        x,y= axis[0],axis[1]
        test_img = PIL.Image.open(img_dir)
        test_img = test_img.resize((256, 256))
        test_img = np.array(test_img)
        test_img = test_img / 255
        test_img = test_img.astype("float32")
        test_img, test_mask = test.block_patch(test_img, "select_box", x=x, y=y)
        plt.imshow(test_img[0])
        plt.show()
        # coarse_result = self.coarse.predict([test_img, test_mask])
        # refine_result = self.refine.predict([coarse_result, test_mask])
        out = self.generator.predict([test_img,test_mask])
        plt.imshow(out[0])
        plt.show()

# def coco_data():
#     DATA_PATH = "./data/cocodata"
#     BATCH_SIZE = 8
#     data_gen = ImageDataGenerator(rescale=1./255)
#     data_flow = data_gen.flow_from_directory(
#         DATA_PATH,
#         target_size=[128, 128],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         class_mode="input",
#         subset="training"
#     )
#     return data_flow
#
# def celeba_data():
#     DATA_PATH = '/home/sjo506/Gan_practice/test/data/celeba'
#     BATCH_SIZE = 1
#     data_gen = ImageDataGenerator(rescale=1./255)
#     data_flow = data_gen.flow_from_directory(DATA_PATH,
#     target_size = [128,128],
#     batch_size = 1,
#     shuffle= True,
#     class_mode='input',
#     subset='training'
#         )
#     return data_flow
#
# def place_data():
#     DATA_PATH = "./data/place365"
#     BATCH_SIZE = 4
#     data_gen = ImageDataGenerator(rescale=1./255)
#     data_flow = data_gen.flow_from_directory(
#         DATA_PATH,
#         target_size=[256, 256],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         class_mode="input",
#         subset="training"
#     )
#     return data_flow
#
# def place_test_data():
#     DATA_PATH = "./data/place_test"
#     BATCH_SIZE = 4
#     data_gen = ImageDataGenerator(rescale=1./255)
#     data_flow = data_gen.flow_from_directory(
#         DATA_PATH,
#         target_size=[128, 128],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         class_mode="input",
#         subset="training"
#     )
#     return data_flow
#
#
# def castle_data():
#     DATA_PATH = "./data/castle"
#     BATCH_SIZE = 4
#     data_gen = ImageDataGenerator(rescale=1./255)
#     data_flow = data_gen.flow_from_directory(
#         DATA_PATH,
#         target_size=[256, 256],
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         class_mode="input",
#         subset="training"
#     )
#     return data_flow





test = inpainting(
    batch_size = 1,
    global_shape = (256,256,3),
    local_shape=(64, 64, 3),
    weight_save_dir="./data/weight/inpaint_attention_2/",
    img_save_dir = "./data/inpaint_attention_4/")

#data = castle_data()
test.load_weight()
#test.train(data,100000)
test.test_img("./test_castle.jpg",(30,180))
test.generator.summary()


test.generator.save_weights('./data/generator_check')
#batch_size = 8
#data = place_test_data()
#for i in range(60000,61000):
#    real_img = next(data)[0]
#    if real_img.shape[0] != batch_size:
#        real_img = next(data)[0]
#
#    masked_img, mask = test.block_patch(real_img, mode='random_box')
#
#    test.save_img(masked_img,mask,i)
#    print("i")
