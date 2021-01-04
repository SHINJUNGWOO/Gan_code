import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, LeakyReLU, Dropout

(x_tr, y_tr), (x_te, y_te) = cifar10.load_data()

NUM_CLASSES = 10
# CIFAR-10 DATASET HEVE 10 LABEL

x_tr = x_tr.astype('float32') / 255.0
x_te = x_te.astype('float32') / 255.0

y_tr = to_categorical(y_tr, NUM_CLASSES)
y_te = to_categorical(y_te, NUM_CLASSES)

input_layer = Input(shape=(32,32,3))
x = Conv2D(
    filters=10,
    kernel_size=(4, 4),
    strides=2,
    padding='same')(input_layer)

for i in range(1,3):    
    x = Conv2D(
        filters=32*i,
        kernel_size=(3,3),
        strides=2,
        padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate =0.5)(x)


output_layer = Dense(units=10,activation = 'softmax')(x)

model = Model(input_layer,output_layer)
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.fit(x_tr, y_tr, batch_size=32, epochs=10, shuffle=True)
print(model.evaluate(x_te,y_te,batch_size=1000))

