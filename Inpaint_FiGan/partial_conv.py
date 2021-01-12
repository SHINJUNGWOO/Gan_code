import tensorflow as tf
import numpy as np
import glob
import pathlib
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise,Concatenate
from tensorflow.keras.layers import MaxPooling2D, Lambda
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
import random

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

    def call(self, inputs):
        (target, wrt) = inputs
        grad = tf.gradients(target, wrt)[0]
        return K.sqrt(tf.math.reduce_sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True)) - 1

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], 1)


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)

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
        self.mask_shape = (global_shape[0], global_shape[1], 1)
        self.weight_save_dir = weight_save_dir
        self.img_save_dir = img_save_dir
        self.extract_rate = 8
        self.build()

    def wasserstein(self, y_true, y_pred):
        return K.mean(y_true * y_pred, axis=-1)

    def block_patch(self, input_, mode='random_box', x=10, y=10, margin=10):
        input_ = tf.convert_to_tensor(input_)
        shape = input_.get_shape().as_list()

        if mode == 'select_box':
            # create patch in central size
            pad_size = tf.constant([self.local_shape[0], self.local_shape[1]], dtype=tf.int32)
            patch = tf.zeros([self.batch_size, self.local_shape[0], self.local_shape[1], self.global_shape[-1]],
                             dtype=tf.float32)

            h_ = tf.constant([y], dtype=tf.int32)[0]
            w_ = tf.constant([x], dtype=tf.int32)[0]
            padding = [[0, 0], [h_, shape[-3] - h_ - pad_size[0]], [w_, shape[-2] - w_ - pad_size[1]], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
            result = tf.multiply(input_, padded)


        if mode == 'random_box':
            # create patch in random box


            h_size = tf.random.uniform([1], minval=margin, maxval=self.local_shape[0],
                                   dtype=tf.int32)[0]
            w_size = tf.random.uniform([1], minval=margin, maxval=self.local_shape[1],
                                       dtype=tf.int32)[0]

            patch = tf.zeros([self.batch_size, h_size, w_size, self.global_shape[-1]],
                             dtype=tf.float32)

            h_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[0] - h_size - margin,
                                   dtype=tf.int32)[0]
            w_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[1] - w_size - margin,
                                   dtype=tf.int32)[0]

            padding = [[0, 0], [h_, shape[-3] - h_ - h_size], [w_, shape[-2] - w_ - w_size], [0, 0]]
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            for _ in range(random.randint(1,5)):
                h_size = tf.random.uniform([1], minval=margin, maxval=self.local_shape[0],
                                           dtype=tf.int32)[0]
                w_size = tf.random.uniform([1], minval=margin, maxval=self.local_shape[1],
                                           dtype=tf.int32)[0]

                patch = tf.zeros([self.batch_size, h_size, w_size, self.global_shape[-1]],
                                 dtype=tf.float32)

                h_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[0] - h_size - margin,
                                       dtype=tf.int32)[0]
                w_ = tf.random.uniform([1], minval=margin, maxval=self.global_shape[1] - w_size - margin,
                                       dtype=tf.int32)[0]

                padding = [[0, 0], [h_, shape[-3] - h_ - h_size], [w_, shape[-2] - w_ - w_size], [0, 0]]
                padded *= tf.pad(patch, padding, "CONSTANT", constant_values=1)

            result = tf.multiply(input_, padded)

        mask = tf.cast(padded, dtype=tf.float32)
        mask = tf.expand_dims(mask[:, :, :, 0], axis=-1)
        return result, mask



    def paste_patch(self, real_img, fake_img, mask):
        return (1- mask) * fake_img + mask * real_img

    def partial_conv(self,filters, kernel_size, strides, dilation_rate, padding='same', activation='elu'):

        def func(x,mask):
            x = Conv2D(filters =filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same', activation='elu')(x)
            mask = Conv2D(filters =1, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same', activation='sigmoid')(mask)
            x = x * mask
            return x,mask

        return func

    def context_attention(self,img,mask):
        img_shape = img.get_shape().as_list()
        back_ground = mask * img
        bool_mask = tf.cast(1-mask,dtype=tf.bool)
        bool_mask = tf.squeeze(bool_mask,axis=-1)
        fore_ground = tf.boolean_mask(img, bool_mask)
        fore_ground = tf.reshape(fore_ground,(self.batch_size,-1,img_shape[-1]))
        fore_ground = tf.transpose(fore_ground, [0, 2, 1])
        back_ground = tf.expand_dims(back_ground,axis=-1)
        fore_ground = tf.expand_dims(fore_ground, axis=1)
        fore_ground = tf.expand_dims(fore_ground, axis=1)

        out = tf.losses.CosineSimilarity(axis=3, reduction=losses.Reduction.NONE)(fore_ground, back_ground)

        mask = mask[:,:,:,0]
        mask = tf.expand_dims(mask, axis=-1)
        out = (1-mask) * out
        out = tf.argmax(out,axis=-1)
        out = tf.expand_dims(out, axis=-1)
        out = tf.cast(out,dtype=tf.float32)


        return out

    def coarse_model(self):
        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape)
        x,mask = self.partial_conv(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_,input_mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(64, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=2, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=4, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=8, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=16, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x = UpSampling2D()(x)
        mask = UpSampling2D()(mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x = UpSampling2D()(x)
        mask = UpSampling2D()(mask)
        x,mask = self.partial_conv(32, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(16, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(3, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)

        x = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)

        x = self.paste_patch(input_, x, input_mask)
        self.coarse = Model([input_, input_mask], x, name="coarse")


    def refine_model(self):
        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape)


        x,mask = self.partial_conv(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_,input_mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x, mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)


        x = self.context_attention(x,mask)
        # context = Conv2D(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(x)

        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        x,mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(x,mask)
        # contexture attention

        y,mask_y = self.partial_conv(32, kernel_size=5, strides=1, dilation_rate=1, padding='same', activation='elu')(input_,input_mask)
        y,mask_y = self.partial_conv(64, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
        y,mask_y = self.partial_conv(64, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
        y,mask_y = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
        y,mask_y = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
        y,mask_y = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)

        while (y.get_shape().as_list()[1] > x.get_shape().as_list()[1]):
            y,mask_y = self.partial_conv(128, kernel_size=3, strides=2, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
            y,mask_y = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)

        y,mask_y = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)
        y,mask_y = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(y,mask_y)

        out = Concatenate(axis=-1)([x, y])
        out_mask = Concatenate(axis=-1)([mask, mask_y])
        while (input_.get_shape().as_list()[1] > out.get_shape().as_list()[1]):
            out,out_mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out,out_mask)
            out,out_mask = self.partial_conv(128, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out,out_mask)
            out = UpSampling2D()(out)
            out_mask = UpSampling2D()(out_mask)

        out,out_mask = self.partial_conv(32, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out,out_mask)
        out,out_mask = self.partial_conv(16, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out,out_mask)
        out,out_mask = self.partial_conv(3, kernel_size=3, strides=1, dilation_rate=1, padding='same', activation='elu')(out,out_mask)

        out = tf.clip_by_value(out, clip_value_min=-1, clip_value_max=1)

        out = self.paste_patch(input_, out, input_mask)
        self.refine = Model([input_, input_mask], out, name="refine")

    def discrimin_model(self):
        input_ = Input(self.global_shape)
        mask = Input(self.mask_shape)
        x = Concatenate(axis=-1)([input_,mask])
        x = SpectralNormalization(Conv2D(64, kernel_size=5, strides=1, padding='same', activation="tanh"))(x)
        x = SpectralNormalization(Conv2D(128, kernel_size=5, strides=2, padding='same', activation="tanh"))(x)
        x = SpectralNormalization(Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh"))(x)
        x = SpectralNormalization(Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh"))(x)
        x = SpectralNormalization(Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh"))(x)
        x = SpectralNormalization(Conv2D(256, kernel_size=5, strides=2, padding='same', activation="tanh"))(x)
        print(x)
        self.discrimin = Model([input_,mask],x , name = "discriminator")

    def spartial_discounted_loss(self, x, y):
        loss_mask = [[0.99 ** ((min(i, self.local_shape[1] - i) + min(j, self.local_shape[0] - j)) // 2) for i in
                      range(self.local_shape[1])] for j in range(self.local_shape[0])]
        loss_mask = np.array(loss_mask)
        diff = tf.abs(x - y)
        diff = tf.math.reduce_sum(diff, axis=-1)
        loss = loss_mask * diff
        loss = tf.math.reduce_mean(loss)
        return loss

    def l1_loss(self,x,y):
        return tf.abs(x-y)

    def generator_discrimin_model(self):

        input_ = Input(self.global_shape)
        input_mask = Input(self.mask_shape)
        coarse_result = self.coarse([input_, input_mask])
        refine_result = self.refine([coarse_result, input_mask])
        valid = self.discrimin([refine_result,input_mask])

        self.generator_discrimin = Model([input_, input_mask], [coarse_result, refine_result, valid],
                                         name="generator")

    def critic_model(self):

        input_real = Input(self.global_shape)
        input_fake = Input(self.global_shape)
        input_mask = Input(self.mask_shape)

        coarse_result = self.coarse([input_fake, input_mask])
        refine_result = self.refine([coarse_result, input_mask])
        inter_result = RandomWeightedAverage(self.batch_size)(inputs=[input_real, refine_result])

        real_valid = self.discrimin([input_real,input_mask])
        fake_valid = self.discrimin([refine_result, input_mask])
        inter_valid = self.discrimin([inter_result,input_mask])
        gp = GradientPenalty()([inter_valid, inter_result])

        self.critic = Model([input_real, input_fake, input_mask],
                            [real_valid,fake_valid, gp], name="critic")

    def compile(self):

        self.coarse.trainable = True
        self.refine.trainable = True
        self.discrimin.trainable = False

        self.generator_discrimin.compile(
            loss=[self.l1_loss, self.l1_loss, self.wasserstein],
            loss_weights=[1, 1, 0.0004 ],
            optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
        )

        self.coarse.trainable = False
        self.refine.trainable = False
        self.discrimin.trainable = True

        self.critic.compile(
            loss=[self.wasserstein, self.wasserstein, "mse"],
            loss_weights=[1, 1, 10],
            optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
        )

    def build(self):

        self.coarse_model()
        self.refine_model()
        self.discrimin_model()
        self.generator_discrimin_model()
        self.critic_model()
        self.compile()

    def train(self, train_data, epochs):
        real = np.ones((self.batch_size,self.global_shape[0]//32,self.global_shape[1]//32,256 ))
        fake = - np.ones((self.batch_size,self.global_shape[0]//32,self.global_shape[1]//32,256 ))
        dummy = np.zeros((self.batch_size,))

        for i in range(epochs):

            real_img = next(train_data)[0]
            if real_img.shape[0] != self.batch_size:
                real_img = next(train_data)[0]

            masked_img, mask = self.block_patch(real_img, mode='random_box')
            for _ in range(1):
                D = self.critic.train_on_batch([real_img, masked_img, mask], [
                    real,fake,  dummy])

            G = self.generator_discrimin.train_on_batch(
                [masked_img, mask], [real_img, real_img, real])

            print("epoch = {} D ={} G={}".format(i, D, G))

            if i % 100 == 0:
                #self.save_img(masked_img, mask, i)
                #self.save_weight()
                plt.imshow(masked_img[0])
                plt.show()
                plt.imshow(self.refine.predict([self.coarse.predict([masked_img,mask]),mask])[0])
                plt.show()

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

    def save_img(self, masked_img, mask, epoch):
        coarse_result = self.coarse.predict([masked_img, mask])
        refine_result = self.refine.predict([coarse_result, mask])
        save_img(self.img_save_dir + "blanked_img" + str(epoch) + ".jpg", masked_img[0])
        save_img(self.img_save_dir + "coarse" + str(epoch) + ".jpg", coarse_result[0])
        save_img(self.img_save_dir + "refine" + str(epoch) + ".jpg", refine_result[0])

    def test_img(self, test_img, test_mask):
        coarse_result = self.coarse.predict([test_img, test_mask])
        refine_result = self.refine.predict([coarse_result, test_mask])
        plt.imshow(refine_result[0])
        plt.show()




def coco_data():
    DATA_PATH = "./data/cocodata"
    BATCH_SIZE = 8
    data_gen = ImageDataGenerator(rescale=1./255)
    data_flow = data_gen.flow_from_directory(
        DATA_PATH,
        target_size=[128, 128],
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="input",
        subset="training"
    )
    return data_flow
def castle_data():
    DATA_PATH = "./data/castle"
    BATCH_SIZE = 4
    data_gen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=10,
                                  zoom_range=[0.7, 1.0],
                                  brightness_range=[0.7, 1.3],
                                  channel_shift_range=0.3)
    data_flow = data_gen.flow_from_directory(
        DATA_PATH,
        target_size=[256, 256],
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="input",
        subset="training"
    )
    return data_flow

def castle_test_data():
    DATA_PATH = "./test_data"
    BATCH_SIZE = 4
    data_gen = ImageDataGenerator(rescale=1./255)
    data_flow = data_gen.flow_from_directory(
        DATA_PATH,
        target_size=[256, 256],
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="input",
        subset="training"
    )
    return data_flow

test = inpainting(

    batch_size=1,

    global_shape=(256, 256, 3),
    local_shape=(80, 80, 3),
    weight_save_dir="./data/weight/partial_conv/",
    img_save_dir="./data/partial_conv/")





data = castle_test_data()
#test.load_weight()
#test.train(data, 100)

test_img = PIL.Image.open("./test_castle.jpg")
test_img = test_img.resize((256, 256))
test_img = np.array(test_img)
test_img = test_img / 255
test_img = test_img.astype("float32")
plt.imshow(test_img)
plt.show()
test_img,test_mask = test.block_patch(test_img, mode='random_box')
test.test_img(test_img,test_mask)
# input shape 는 (1,256,256,3) mask는 (1,256,256,1)의 형태
# test.refine.save("hello")


