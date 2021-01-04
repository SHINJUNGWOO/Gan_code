from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D,Concatenate, Cropping2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.initializers import RandomNormal

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from functools import partial
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,save_img
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as tfback
########## Import line #############


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



class RandomWeightedAverage(_Merge):
    def __init__(self):
        super().__init__()
        #self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform(( 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


######## RandomWeightedAverage #############
# Inheritance in Keras.layer.merge._merge, #
# For making Interpolared image            #
############################################



class GAN():

    def __init__(self,input_dim,blank_dim,batch_size,using_generator):
        self.input_dim = input_dim
        self.blank_dim = blank_dim
        self.batch_size = batch_size
        self.weight_save_dir= "./data/"
        self.mask_dim = (4,)
        self.using_generator = using_generator
        self.weight_save_dir = "./data/weight/"
        self.img_save_dir = "./data/inpaint/"
        self.generator_conv_layer =[
        [5,1,1,64],
        [3,1,2,128],
        [3,1,1,128],
        [3,1,2,256],
        [3,1,1,256],
        [3,1,1,256],
        [3,2,1,256],
        [3,4,1,256],
        [3,8,1,256],
        [3,16,1,256],
        [3,1,1,256],
        [3,1,1,256]
        ]
        self.generator_deconv_layer =[
        [4,1,2,128],
        [3,1,1,128],
        [4,1,2,64],
        [3,1,1,32],
        [3,1,1,3]
        ]
        self.local_discriminator_layer=[
        [5,2,64],
        [5,2,128],
        [5,2,256],
        [5,2,512],
        [5,2,512]
        ]
        self.global_discriminator_layer=[
        [5,2,64],
        [5,2,128],
        [5,2,256],
        [5,2,512],
        [5,2,512],
        [5,2,512]
        ]


        self._build()

    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def compute_gradients(self,tensor,val_list):
        grads = tf.gradients(tensor,val_list)
        return [grad if grad is not None else tf.zeros_like(var) for var,grad in zip(val_list,grads)]

    def gradient_penalty_loss(self, y_true, y_pred, samples, sample_weight=None):
        gradients = self.compute_gradients(y_pred, [samples])[0]
        gradient_l2_norm = K.sqrt(
            K.sum(
                K.square(gradients),
                axis=list(range(1, len(gradients.shape))
                          )
            )
        )
        return K.mean(K.square(1 - gradient_l2_norm))

    def make_blank(self, real):
        #img = cv2.rectangle(img,(30,30),(50,50),(255,255,255),3)
        mask = []
        img = real.copy()
        for i in range(self.batch_size):
            a_1, b_1 = np.random.randint(
                0, self.input_dim[0]-self.blank_dim[0]+1, 2)
            a_2, b_2 = a_1 + self.blank_dim[0], b_1 + self.blank_dim[0]
            img[i, a_1:a_2, b_1:b_2, :] = (0.0, 0.0, 0.0)

            a_1 = a_1 / self.input_dim[0]
            a_2 = a_2 / self.input_dim[0] 
            b_1 = b_1 / self.input_dim[0]
            b_2 = b_2 / self.input_dim[0]
        
            mask.append([a_1, a_2, b_1, b_2])
        # img = img *mask
        return np.array(img,dtype="float32"), np.array(mask,dtype="float32")

    def blank_masking_model(self):
        img = Input(shape = self.input_dim,dtype="float32")
        mask = Input(shape=self.mask_dim,dtype="float32")
        
        
        out_img,output = Lambda(lambda param : self.masking(param) )([img,mask])
        
        self.blank_masking = Model(inputs = [img,mask],outputs = [out_img,output])

    def masking(self,inputs):
        x = inputs[0]
        mask =inputs[1]
        box_index = Lambda(lambda x: tf.cast( x[:,0],dtype = tf.int32))(mask)
        test = Lambda(lambda x: tf.range(tf.size(x[:,0]),dtype=tf.int32))(mask)
        output = tf.image.crop_and_resize(x,mask,test,(self.blank_dim[0],self.blank_dim[1]))
        return [x,output]


    def generator_model(self):
        generator_input = Input(shape = self.input_dim,dtype="float32")
        mask = Input(shape =self.mask_dim ,dtype="float32")
        x = generator_input
        for (kernel,dilation,stride,output) in self.generator_conv_layer:
            x = Conv2D(filters = output, kernel_size = kernel, padding="same", dilation_rate = dilation,strides =stride)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        #convolution
        for (kernel,dilation,stride,output) in self.generator_deconv_layer:
            if stride == 2:
                x = UpSampling2D()(x)
            x = Conv2D(filters = output, kernel_size =kernel, padding = "same", dilation_rate =dilation,strides = 1)(x)
            if output != 3:
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
            else:
                pass

        #deconvolution
        generator_output, masked_block = self.blank_masking([x, mask])
        self.generator = Model([generator_input,mask],[generator_output,masked_block])
    

    def local_discriminator_model(self):
        discriminator_input = Input(shape=self.blank_dim, dtype="float32")
        x = discriminator_input
        for (kernel,stride,output) in self.local_discriminator_layer:
            x = Conv2D(filters = output, kernel_size = kernel, strides = stride, padding = "same")(x)
            x = LeakyReLU()(x)
            x = Dropout(rate = 0.2)(x)

        x = Flatten()(x)
        discriminator_output = x
        self.local_discriminator = Model(discriminator_input,discriminator_output)
    


    def global_discriminator_model(self):
        discriminator_input = Input(shape=self.input_dim, dtype="float32")
        x = discriminator_input
        for (kernel,stride,output) in self.global_discriminator_layer:
            x = Conv2D(filters = output, kernel_size = kernel, strides = stride, padding = "same")(x)
            x = LeakyReLU()(x)
            x = Dropout(rate = 0.2)(x)

        x = Flatten()(x)
        discriminator_output = x
        self.global_discriminator = Model(discriminator_input,discriminator_output)

    def discriminator_model(self):
        input_discrimin = Input(shape = self.input_dim,dtype="float32")
        blank_discrimin = Input(shape=self.blank_dim, dtype="float32")

        concat_out = Concatenate()([self.global_discriminator(
            input_discrimin), self.local_discriminator(blank_discrimin)])
        concat_out = Dense(1,activation = None)(concat_out)
        self.discriminator = Model([input_discrimin,blank_discrimin],concat_out)

    def critic_model(self):
        real_input = Input(shape = self.input_dim,dtype="float32")
        fake_input = Input(shape = self.input_dim,dtype="float32")
        mask = Input(shape = self.mask_dim)

        valid = self.discriminator(self.blank_masking([real_input, mask]))
        fake = self.discriminator(self.generator([fake_input, mask]))
        interpolared_img = RandomWeightedAverage()([real_input,fake_input])
        interpolared_img, interpolared_blank = self.blank_masking([interpolared_img, mask])
        validity_interpolated = self.discriminator([interpolared_img,interpolared_blank])
        
        self.gp_loss = partial(self.gradient_penalty_loss,samples =interpolared_img)

        
        self.critic = Model(inputs = [real_input,fake_input,mask], outputs = [valid,fake,validity_interpolated])

    def generator_discrimin_model(self):
        img_input = Input(shape = self.input_dim)
        mask_input = Input(shape = self.mask_dim)
        x,x_mask = self.generator([img_input,mask_input])
        model_output = self.discriminator([x,x_mask])
        self.generator_discrimin = Model([img_input,mask_input],model_output)


    def generator_compile(self):
        self.generator.compile(
                optimizer = RMSprop(lr =0.004),
                loss = self.wasserstein
                )

    def critic_compile(self):
        self.generator.trainable = False
        self.discriminator.trainable = True

        self.critic.compile(
            loss = [self.wasserstein,self.wasserstein,self.gp_loss],
            optimizer = RMSprop(lr = 0.004),
            loss_weights = [1,1,10]
            )

    def generator_discrimin_compile(self):
        self.discriminator.trainable = False
        self.generator.trainable = True

        self.generator_discrimin.compile(
            optimizer = RMSprop(lr =0.004),
            loss = self.wasserstein
            )

    def critic_train(self,real_img,fake_img,mask):
        batch_size = self.batch_size
        valid_out = np.ones((batch_size,1),dtype = np.float32)
        fake_out = -np.ones((batch_size,1),dtype = np.float32)
        dummy_out = np.zeros((batch_size,1),dtype = np.float32)
        return self.critic.train_on_batch([real_img,fake_img,mask],[valid_out,fake_out,dummy_out])


    def generator_train(self,fake_img,mask):
        batch_size = self.batch_size
        valid = np.ones((batch_size,1),dtype =np.float32)
        
        return self.generator_discrimin.train_on_batch([fake_img,mask],valid)

    def train(self,train,epoch):
        for i in range(epoch):
            if self.using_generator == True:
                img = next(train)[0]
                if img.shape[0] != self.batch_size:
                    img = next(train)[0]
            else:
                img = train[0]

            real = img
            fake, mask = self.make_blank(real)
            for _ in range(5):
                d_loss = self.critic_train(real,fake,mask)
            g_loss = self.generator_train(fake,mask)
            if i%100 == 0:
                print(i,d_loss,g_loss)
                save_img(self.img_save_dir+"reconst"+str(i)+".jpg",self.generator.predict([fake,mask])[0][0])
                self.save_weight()
                

    def _build(self):
        self.blank_masking_model()
        self.blank_masking = multi_gpu_model(self.blank_masking,gpus=2)
        self.generator_model()
        self.local_discriminator_model()
        self.global_discriminator_model()
        self.discriminator_model()
        self.critic_model()
        self.critic = multi_gpu_model(self.critic, gpus=3)
        self.generator_discrimin_model()
        self.generator_discrimin = multi_gpu_model(self.generator_discrimin, gpus=3)
        self.generator_compile()
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
            
        except Exception as ex:
            print(ex)
            self.save_weight()
            self.load_weight()

        print("weight loaded")

gan = GAN(
    input_dim=[256,256,3],
    blank_dim=[32,32,3],
    batch_size=12,
    using_generator = True
)

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


def caleba_data():
    DATA_PATH = '/home/sjo506/Gan_practice/test/celeba'
    BATCH_SIZE = 32
    data_gen = ImageDataGenerator(rescale=1./255) 
    data_flow = data_gen.flow_from_directory(DATA_PATH,
    target_size = [128,128],
    batch_size = BATCH_SIZE,
    shuffle= True,
    class_mode='input',
    subset='training'
        )
    return data_flow

data = coco_data()
#gan.load_weight()
gan.train(data,30000)
