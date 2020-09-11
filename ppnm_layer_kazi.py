# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:48:35 2020

@author: Shounak
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class PPNM_Layer(Layer):
    
    def __init__(self,initial_endmembers):
        
        self.initial_endmembers_init = initial_endmembers
        self.initial_endmembers_shape = initial_endmembers.shape
        
        super(PPNM_Layer, self).__init__()
        
    def build(self, input_shape):
        
        self.linear_endmembers = self.add_weight("linear_endmembers", shape = self.initial_endmembers_shape,initializer=tf.constant_initializer(self.initial_endmembers_init))        
        
        self.b_s = self.add_weight("b_s", shape = (input_shape[0],), initializer=tf.keras.initializers.Zeros())
        
        
        super().build(input_shape)

    
    def call(self, x):
        
        mat = tf.matmul(x,self.linear_endmembers)
        b_s_mat = tf.linalg.diag(self.b_s)
        
        output = mat + tf.transpose( tf.matmul( tf.transpose(tf.math.multiply(mat,mat)) , b_s_mat) )
        
        return output
        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.initial_endmembers_shape[1])