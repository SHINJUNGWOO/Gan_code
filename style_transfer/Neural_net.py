import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


class net():
    def __init__(self,input_dim,base_image_path,style_reference_image_path,weights):
        self.input_dim = input_dim
        self.img_nrows = self.input_dim[0]
        self.img_ncols = self.input_dim[1]
        self.channels = self.input_dim[2]
        self.base_image_path = base_image_path
        self.style_reference_image_path = style_reference_image_path
        self.content_weight = weights[0]
        self.style_weight = weights[1]
        self.total_weight = weights[2]
        vgg_model = vgg19.VGG19(weights ="imagenet", include_top =False)
        outputs_dict = dict([(layer.name, layer.output) for layer in vgg_model.layers])
        self.feature_extractor = keras.Model(inputs = vgg_model.inputs, outputs = outputs_dict)
        self._build()

    def preprocess_image(self,image_path):
        img = keras.preprocessing.image.load_img(image_path,target_size=(self.img_nrows,self.img_ncols))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(self,x):
        x = x.reshape((self.img_nrows, self.img_ncols,3))
        #In Vgg19 subtraction of average pixel(103.939,116.779,123.68)
        # vgg19 => BGR 
        x[:,:,0] += 103.939
        x[:,:,1] += 116.779
        x[:,:,2] += 123.68
        x = x[:,:,::-1]
        x= np.clip(x,0,255).astype("uint8")
        return x

    def gram_matix(self,x):
        x = tf.transpose(x,(2,0,1))
        features = tf.reshape(x,(tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(self,style,combination):
        S = self.gram_matix(style)
        C = self.gram_matix(combination)
        size = self.img_nrows * self.img_ncols
        total = tf.reduce_sum(tf.square(S-C))
        total = total / (4.0 * (self.channels ** 2) * (size ** 2))
        return total

    def content_loss(self,base,combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(self,x):
        a = tf.square(
                x[:, : self.img_nrows - 1, : self.img_ncols - 1,:] 
                - x[:,1:, : self.img_ncols - 1,:]
                )
        b = tf.square(
                x[:, : self.img_nrows - 1, : self.img_ncols - 1,:]
                - x[:, : self.img_nrows -1, 1: , :]
                )
        return tf.reduce_sum(tf.pow(a+b, 1.25))

    def compute_loss(self,combination_image,base_image,style_reference_image):
        input_tensor = tf.concat(
                [base_image, style_reference_image, combination_image], axis = 0
                )
        features = self.feature_extractor(input_tensor)

        loss = tf.zeros(shape = ())
       
        # Content Loss calculate
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0,:,:,:]
        combination_features = layer_features[2,:,:,:]
        loss = loss + self.content_weight * self.content_loss(
                base_image_features, combination_features)


        # Style loss
        for layer_name in self.style_layer_name:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1,:,:,:]
            combination_features = layer_features[2,:,:,:]
            sl = self.style_loss(style_reference_features, combination_features)
            loss += (self.style_weight / len(self.style_layer_name)) * sl

        # Total Variable loss
        loss += self.total_weight * self.total_variation_loss(combination_image)

        return loss
    @tf.function
    def compute_loss_and_grads(self,combination_image,base_image,style_reference_image):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(combination_image,base_image,style_reference_image)
        grads = tape.gradient(loss, combination_image)
        
        return loss,grads

    def _build(self):

        self.style_layer_name = [
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1"
                ]
        self.content_layer_name = "block5_conv2"

        self.base_image = self.preprocess_image(self.base_image_path)
        self.style_reference_image = self.preprocess_image(self.style_reference_image_path)
        self.combination_image = tf.Variable(self.preprocess_image(self.base_image_path))

        self.optimizer = keras.optimizers.SGD(
                keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = 100.0, decay_steps =100, decay_rate = 0.96)
                )


    def train(self,epoch):
        for i in range(1,epoch+1):
            loss, grads = self.compute_loss_and_grads(self.combination_image, self.base_image, self.style_reference_image)
            self.optimizer.apply_gradients([(grads, self.combination_image)])             
            
            if i % 100 == 0:
                print(i,"%.2f"%loss)
                img = self.deprocess_image(self.combination_image.numpy())
                keras.preprocessing.image.save_img("./data/done_%d.png"%i,img)


base = "./data/full.jpg"
style = "./data/mone.jpg"
nst = net((512,512,3),base,style,[2.5e-8,1e-6,1e-6])
print("init_done")
nst.train(6000)
img = nst.deprocess_image(nst.combination_image.numpy())
keras.preprocessing.image.save_img("./data/done.jpg", img)



