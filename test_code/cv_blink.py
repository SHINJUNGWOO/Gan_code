from keras.preprocessing import image
from keras.layers import Cropping2D,Input,Lambda
from keras.models import Model

import random
import numpy as np
import tensorflow as tf

img = image.load_img("./data/done.jpg",target_size = (100,100))
img2 = image.load_img("./data/mone.jpg",target_size = (100,100))

img = np.array(img).reshape((100,100,3))
#img = img.reshape((1,100,100,3))
print(img.shape)
mask =np.array([0.2,0.6,0.2,0.6])
print(mask.shape)

x_1 = Input(shape = img.shape)

x_2 = Input(shape = mask.shape)

box_index = np.array([0])
out = Lambda(lambda a: tf.image.crop_and_resize(x_1, x_2,box_index,(32,32)))(x_1)
model = Model([x_1,x_2],out)

img = image.img_to_array(img)
print(model([x_1,x_2]))
        


#print(x.shape)
#print(y.shape)
