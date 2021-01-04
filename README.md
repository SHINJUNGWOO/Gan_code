# Practice GAN

Practice Generative Adverarial Network for Image Inpainting and Denosing, Style Transfer

## System Requirements

1. Ubuntu 18.04
2. Python 3.6.9
3. Tensorflow 2.1.0
4. Keras 2.3.1
5. Jupyter-Notebook 



## Usage



#### Inpaint Image

###### wgan _inapint__*.py

​	=> OOM Error and Model blank error

​	

###### w_global_local.py

​	v_1: global_local model => doesn't work because of OOM

​	v_2: Gradient Penalty vanishing , Model doesn't work

​	v_3: Reconstruct Model => Can not train by dying Relu and Model Gradient

​	v_4: work for me



###### Inpaint.py

​	v_1: basic 모델 128*128

​	v_2: basic 모델 256*256

​	v_5: basic 모델 512*512

​	v_6: partial_convolution 



#### Generate Image

###### wgan_cp.py

​	v_1~v_4 : all model work for me in different Resolution, and another hyper parameter



#### Another

Study for Wgan and test GAN Model, RCNN, Style Transfer(Neural_net)

,VAE, etc





