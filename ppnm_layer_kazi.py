# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:48:35 2020

@author: Kazi T Shahid and Ioannis D Schizas

This code is used for the script "autoencoder_main.py"

This is the Final Layer, where the input is the vector containing abundances, and their cross products.

Input:
initial_endmembers = Weights that are initialized with the endmember matrices. These weights will determine the final estimated values of the endmembers.

Weights:
linear_endmembers = Named so, because they make up the endmembers in the linear component of the mixture, and their cross-products are later used to 
estimate the nonlinear components of the mixture model.
b_s = What value is used to control the contribution of the nonlinear components of the mixture model, w.r.t. to the entire mixture. This determines how
strong the nonlinear component is, compared to the linear component.

If you wish to use this code, please cite the URL given above for the dataset, and also the URL where this code was downloaded from:
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Autoencoder

Also, please cite the paper published, which can be found in this link
https://ieeexplore.ieee.org/document/9432042

"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class PPNM_Layer(Layer):
    
    def __init__(self,initial_endmembers):
        
        self.initial_endmembers_init = initial_endmembers
        self.initial_endmembers_shape = initial_endmembers.shape
        
        super(PPNM_Layer, self).__init__()
        
    def build(self, input_shape):
        
        self.linear_endmembers = self.add_weight("linear_endmembers", shape = self.initial_endmembers_shape,initializer=tf.constant_initializer(self.initial_endmembers_init),constraint = tf.keras.constraints.NonNeg()))        
        
        self.b_s = self.add_weight("b_s", shape = (input_shape[0],), initializer=tf.keras.initializers.Zeros())
        
        
        super().build(input_shape)

    
    def call(self, x):
        
        mat = tf.matmul(x,self.linear_endmembers)
        b_s_mat = tf.linalg.diag(self.b_s)
        
        output = mat + tf.transpose( tf.matmul( tf.transpose(tf.math.multiply(mat,mat)) , b_s_mat) )
        
        return output
        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.initial_endmembers_shape[1])
