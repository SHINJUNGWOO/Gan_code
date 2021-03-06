import sys, os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='1'

import tensorflow as tf
import numpy as np
import glob
import pathlib
import matplotlib.pyplot as plt
import PIL
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

print('''+++ env info +++
Python version : {},
Tensorflow version : {},
Keras version : {}
'''.format(sys.version, tf.__version__, tf.keras.__version__))

print('TF GPU available test :', tf.config.list_physical_devices('GPU'))
#print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    @tf.RegisterGradient("ResizeBilinearGrad")
    def _ResizeBilinearGrad(op,grads):
        return (array_ops.zeros(shape=array_ops.shape(op.inputs[0]), dtype=op.inputs[0].dtype),tf.raw_ops.ResizeBilinearGrad(grads=grads, original_image=op.inputs[1]))

    def call(self,inputs):
        (target,wrt) = inputs
        grad = tf.gradients(target,wrt)[0]
        return K.sqrt(tf.math.reduce_sum(K.batch_flatten(K.square(grad)),axis =1, keepdims = True))-1

    def compute_output_shape(self,input_shape):
        return (input_shape[1][0],1)

# For Gradient Penalty and interpolation


def wasserstein( y_true, y_pred):
    return K.mean(y_true * y_pred,axis=-1)

# For Wasserstein Loss

def block_patch(input_, mode, patch_size=50, margin=10):
	
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
	
def configure_for_performance(ds, batch_size, buffer_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=buffer_size)
    return ds
	
def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) * 2 - 1
    img = tf.image.resize(img, [img_h, img_w])
    return img
	
def sample_images(epoch, imgs, save_path):
    
    masked_imgs, mask, coord, pad_size = block_patch(imgs, mode='random_box', patch_size=patch_size, margin=margin)
    y1, y2, x1, x2 = coord[0].numpy(), coord[0].numpy()+pad_size[0].numpy(), coord[1].numpy(), coord[1].numpy()+pad_size[1].numpy()
    inpainted_imgs = G.predict(masked_imgs)
    inpainted_fields = inpaint_crop(inpainted_imgs, coord, pad_size)
    
    imgs = 0.5 * imgs + 0.5
    masked_imgs = 0.5 * masked_imgs + 0.5
    inpainted_imgs = 0.5 * inpainted_imgs + 0.5
    inpainted_fields = 0.5 * inpainted_fields + 0.5

    r, c = 4, 6
    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0,i].imshow(imgs[i])
        axs[0,i].axis('off')
        axs[1,i].imshow(masked_imgs[i])
        axs[1,i].axis('off')
        axs[2,i].imshow(inpainted_imgs[i])
        axs[2,i].axis('off')
        filled_in = tf.identity(masked_imgs)[i].numpy()
        filled_in[y1:y2, x1:x2, :] = inpainted_fields[i].numpy()
        axs[3,i].imshow(filled_in)
        axs[3,i].axis('off')
        
    if not os.path.exists('images/{}'.format(save_path)):
        os.mkdir('images/{}'.format(save_path))
    
    fig.savefig("images/{}/{}.png".format(save_path,epoch))
    plt.close()

def inpaint_crop(img, coord, pad_size):
    crop_box = tf.image.crop_to_bounding_box(img, coord[0], coord[1], pad_size[0], pad_size[1])
    crop_bbox = tf.image.resize(crop_box, (inpainted_h, inpainted_w))
    return crop_bbox
	
local_shape, global_shape = (64,64,3), (128,128,3)

img_h, img_w, channel = 128, 128, 3
inpainted_h, inpainted_w = 64, 64

batch_size = 15
AUTOTUNE = tf.data.experimental.AUTOTUNE
buffer_size = AUTOTUNE
epochs = 20000
sample_interval = 10

margin = 5
patch_size = 64

#data_dir = '/home/user_5/bg/python/GDL_code/data/celeb/img_align_celeba/img_align_celeba'
data_dir = '/home/user_6/Gan_practice/data/cocodata/val2017'
save_path = '/home/user_6/Gan_practice/data/img_make'

# load Data
image_count = len(list(glob.glob(data_dir+'/*.jpg')))
list_ds = tf.data.Dataset.list_files(str(data_dir+"/*"), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
list_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_size = int(image_count * 0.001)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_len = tf.data.experimental.cardinality(train_ds).numpy()
val_len = tf.data.experimental.cardinality(val_ds).numpy()
print(train_len)
print(val_len)

train_ds = configure_for_performance(train_ds, batch_size, buffer_size)
val_ds = configure_for_performance(val_ds, batch_size, buffer_size)

for i in iter(train_ds):
    sample = i
    break
	
result, padded, coord, pad_size = block_patch(sample, mode='random_box', patch_size=patch_size, margin=margin)

plt.figure(figsize=(10, 5))
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    plt.imshow(result[i].numpy()*0.5+0.5)
    plt.axis("off")
#plt.close()

result.shape, padded.shape, coord, pad_size

def build_generator(global_shape):
    
    inputs = tf.keras.Input(shape=global_shape) # (None, 128, 128, 3)
    
    # Encoder
    x = Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs) # (None, 64, 64, 64)
    
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    
    x = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    
    # Dilated_Conv
    x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=2, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=4, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=8, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, dilation_rate=16, padding='same', activation='relu')(x)

    x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(channel, kernel_size=3, strides=1, padding='same', activation='tanh')(x)    

    x = tf.keras.Model(inputs=inputs, outputs=x, name='G')

    return x
	
def build_discriminator(local_shape, global_shape):

    local_inputs = tf.keras.Input(shape=local_shape, name='local_input') # (None, 64, 64, 3)
    global_inputs = tf.keras.Input(shape=global_shape, name='global_input') # (None, 64, 64, 3)
    
    local = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(local_inputs) 
    local = Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu')(local) 
    local = Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(local) 
    local = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(local) 
    local = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(local) 
    local = Flatten()(local) 
    local = Dense(1024)(local)

    global_ = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(global_inputs) 
    global_ = Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu')(global_) 
    global_ = Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(global_) 
    global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(global_) 
    global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(global_) 
    global_ = Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(global_) 
    global_ = Flatten()(global_) 
    global_ = Dense(1024)(global_)

    D_output = tf.keras.layers.concatenate([local, global_])
    D_output = Dense(1, activation='tanh')(D_output)
    D_output = tf.keras.Model(inputs=[local_inputs, global_inputs], outputs=D_output, name='D')
    
    return D_output
	
D = build_discriminator(local_shape, global_shape)
D.compile(loss= wasserstein, optimizer=Adam(lr = 0.0001))#, metrics=['accuracy'])


G = build_generator(global_shape)
G_input = Input(global_shape)
inpainted_img = G(G_input)
inpainted_field = inpaint_crop(inpainted_img, coord, pad_size)

interpolated_img = RandomWeightedAverage(batch_size)(inputs = [G_input,inpainted_img])
interpolated_field = inpaint_crop(interpolated_img, coord, pad_size)

validity = D([inpainted_field, inpainted_img])
interpolated_validity = D([interpolated_field, interpolated_img])
gp = GradientPenalty()([interpolated_validity,interpolated_img])

critic = tf.keras.Model(inputs = G_input, outputs = gp)
critic.compile(loss = 'mse',optimizer = Adam(lr=0.0001,beta_1=0.5,beta_2=0.9))


D.trainable = False
GL = tf.keras.Model(G_input , [inpainted_field, validity], name='GL')
GL.compile(loss=['mse', wasserstein], loss_weights=[0.9996, 0.0004], optimizer=Adam())

GL.summary()


D_losses = []
D_acc = []
G_losses = []
G_mse = []

# Adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    
    for train_batch in iter(train_ds):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of images

        masked_imgs, mask, coord, pad_size = block_patch(train_batch, mode='random_box', patch_size=patch_size, margin=margin)
        y1, y2, x1, x2 = coord[0].numpy(), coord[0].numpy()+pad_size[0].numpy(), coord[1].numpy(), coord[1].numpy()+pad_size[1].numpy()
        
        # Generate a batch of new images
        inpainted_imgs = G.predict(masked_imgs)
        inpainted_fields = inpaint_crop(inpainted_imgs, coord, pad_size)
        real_fields = inpaint_crop(train_batch, coord, pad_size)
        
        filled_in = tf.identity(masked_imgs).numpy() # tensor copy
        filled_in[:, y1:y2, x1:x2, :] = inpainted_fields.numpy()
        filled_in = tf.constant(filled_in)
        
        # Train the discriminator
        D.trainable = True
        GL.trainable = False 
        d_loss_real = D.train_on_batch([real_fields, train_batch], real)

        d_loss_fake = D.train_on_batch([inpainted_fields, filled_in], -real)
        #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print("D:",d_loss_real , d_loss_fake)
        for _ in range(10): 
            gradient = critic.train_on_batch(masked_imgs, fake)
        print("Gradient",gradient)
        # ---------------------
        #  Train Generator
        # ---------------------
        D.trainable = False
        GL.trainable = True  
        g_loss = GL.train_on_batch(masked_imgs, [real_fields, real])
        print("G:",g_loss)
        # Plot the progress
        #D_losses.append(d_loss[0])
        #D_acc.append(d_loss[1])
        #G_losses.append(g_loss[0])
        #G_mse.append(g_loss[1])
        #print(train_batch.shape)

    print("\n{} epoch".format(epoch))
        # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        val_batch = next(iter(val_ds))
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
        val = val_batch[:6]
        #sample_images(epoch, val, save_path)
