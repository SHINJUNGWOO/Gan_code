import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input,Flatten,Dense
from keras.models import Model
from keras.optimizers import Adam

(x_tr,y_tr),(x_te,y_te) = cifar10.load_data()

NUM_CLASSES =10
# CIFAR-10 DATASET HEVE 10 LABEL

x_tr = x_tr.astype('float32') /255.0
x_te = x_te.astype('float32') / 255.0

y_tr = to_categorical(y_tr,NUM_CLASSES)
y_te = to_categorical(y_te,NUM_CLASSES)



# Make Layer 

input_layer = Input(shape=(32,32,3))
x= Flatten()(input_layer)
x= Dense(units=200, activation='relu')(x)
x= Dense(units=10000, activation='relu')(x)
output_layer = Dense(units=10, activation='softmax')(x)

model = Model(input_layer,output_layer)

# Set loss function and optimizer

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

# Train
model.fit(x_tr,y_tr,batch_size=32,epochs=10,shuffle=True)

#Accuracy Test
print(model.evaluate(x_te,y_te))

# Test check
CLASSES = np.array(list(range(10)))
preds = model.predict(x_te)
preds_single = CLASSES[np.argmax(preds,axis=-1)]
actual_single = CLASSES[np.argmax(y_te,axis=-1)]
