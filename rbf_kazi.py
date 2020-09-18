"""
Created on Fri Jul  3 21:40:35 2020

@author: Kazi T Shahid and Ioannis D Schizas

This code is used for the script "autoencoder_main.py"

This is the RBF Layer, where the input is hyperspectral pixel vectors. The centers are estimated endmembers, and the output will be
estimated abundances for that mixed pixel input.


If you wish to use this code, please cite the URL given above for the dataset, and also the URL where this code was downloaded from:
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Autoencoder
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class RBFLayer(Layer):
    
    def __init__(self, num_classes, betas, centers):
        
        
        self.num_classes = tf.constant(num_classes)
        self.betas_init = betas
        self.centers_init = centers
        self.centers_shape = centers.shape
        
        super(RBFLayer, self).__init__()
        
    def build(self, input_shape):
        
        self.centers = self.add_weight("centers", shape = self.centers_shape,initializer=tf.constant_initializer(self.centers_init))
        self.betas = self.add_weight("betas", shape = (self.num_classes,),initializer=tf.constant_initializer(self.betas_init))

        
        super().build(input_shape)
        
    def call(self, x):
        
        C = tf.expand_dims(self.centers, -1)
        H = tf.transpose(C-tf.transpose(x)) 
        return tf.exp(-1 * self.betas * tf.math.reduce_sum(H**2,axis=1))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)
