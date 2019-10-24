# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:38:06 2019

@author: lawle
"""

'''
conv2d 및 conv2d_transpose 함수를 numpy로 구현
'''

import tensorflow as tf
import numpy as np


def conv_2d_to_2d(a, b):
    a_h, a_w, a_c = a.shape
    b_h, b_w, b_c = b.shape
    
    assert a_h == b_h and a_w == b_w and a_c == b_c
    
    sum_all=0
    for c in range(a_c):
        for i in range(a_h):
            for j in range(a_w):
                sum_all += a[i,j,c] * b[i,j,c]
                
    return sum_all


def conv2d(Input, Kernel, stride=2, padding="valid"):
    # 1. Calculate output shape
    _, i_h, i_w, i_c = Input.shape
    k_h, k_w, _, k_c = Kernel.shape
    
    if padding == "valid":
        o_w = int((i_w-k_w)/stride)+1
        o_h = int((i_h-k_h)/stride)+1
    
    elif padding == "same":
        o_w = int((i_w-1)/stride)+1
        o_h = int((i_h-1)/stride)+1
        
        need_i_h = k_h + (o_h-1)*stride
        need_i_w = k_w + (o_w-1)*stride
    
    o_c = k_c
    
    # 2. Padding input
    if padding == "same":
        pad_h_left = int(np.ceil((need_i_h - i_h)/2))
        pad_h_right = int(np.floor((need_i_h - i_h)/2))
        pad_w_left = int(np.ceil((need_i_w - i_w)/2))
        pad_w_right = int(np.floor((need_i_w - i_w)/2))
        
        input_after_padding = []
        for input_s in Input:
            input_padding_c = [] 
            for c in range(i_c):
                input_padding_c.append(np.pad(input_s[:,:,c], [[pad_h_left, pad_h_right],[pad_w_left, pad_w_right]], "constant"))
            input_after_padding.append(input_padding_c)
            
        Input = np.array(input_after_padding, dtype=np.float32)
        Input = np.transpose(Input, [0,2,3,1])
    
    
    # conv2d 1) 3d * 3d
    output = []
    for out_c in range(o_c):
        output_c = np.zeros([o_h, o_w])
        for i in range(o_h):
            for j in range(o_w):
                input_slice = Input[0, i*stride:i*stride+k_h, j*stride:j*stride+k_w, :]
                output_c[i][j] = conv_2d_to_2d(input_slice, Kernel[:,:,:,out_c])
            
        output.append(output_c)
        
    output = np.array(output, np.float32)
    output = np.transpose(output, [1,2,0])
        
    # conv2d 2) Using Sparse matrix
    _, i_h, i_w, i_c = Input.shape
    starting_point = [(i*stride,j*stride) for i in range(o_h) for j in range(o_w)]  
    
    output2 = []
    for j in range(o_c):
        out_c = np.zeros([o_h* o_w])
        for c in range(i_c):
            M = []
            for x, y in starting_point:
                M_a = np.zeros([i_h, i_w])
                M_a[x:x+k_h,y:y+k_w] = np.copy(Kernel[:,:,c,j])
                M.append(M_a.flatten())
                    
            M = np.array(M, dtype=np.float32)
            out = np.matmul(M, Input[0,:,:,c].flatten())
            out_c += out
                
        output2.append(out_c.reshape([o_h, o_w]))
    
    output2 = np.array(output2, dtype=np.float32)
    output2 = np.transpose(output2, [1,2,0])
    
    return output, output2
        

def conv2d_transpose(I, K, stride=2, padding="valid"):
    # 1. Define mask
    _, i_h, i_w, i_c = I.shape
    k_h, k_w, _, o_c = K.shape
    
    if padding == "valid":
        out_w = k_w + (i_w-1)*stride
        out_h = k_h + (i_h-1)*stride
    
    elif padding == "same":
        out_w = k_w * stride
        out_h = k_h * stride
    
    starting_point = [(i*stride,j*stride) for i in range(i_h) for j in range(i_w)]  
    
    output = []
    for j in range(o_c):
        out_c = np.zeros([out_h* out_w])
        for c in range(i_c):
            M = []
            for x, y in starting_point:
                M_a = np.zeros([out_h, out_w])
                M_a[x:x+k_h,y:y+k_w] = np.copy(Kernel[:,:,c,j])
                M.append(M_a.flatten())
                    
            M = np.array(M, dtype=np.float32)
            out = np.matmul(M.T, Input[0,:,:,c].flatten())
            out_c += out
                
        output.append(out_c.reshape([out_h, out_w]))
    
    output = np.array(output, dtype=np.float32)
    output = np.transpose(output, [1,2,0])
    
    return output
    

Input = np.random.rand(1*25*12*3).reshape([1,25,12,3])
Kernel = np.random.rand(7*7*3*6).reshape([7,7,3,6])
s_v, v_v = conv2d(Input, Kernel, stride=2, padding="valid")
s_s, v_s = conv2d(Input, Kernel, stride=2, padding="same")

a = tf.constant(Input, dtype=np.float32)
b = tf.constant(Kernel, dtype=np.float32)
c_v = tf.nn.conv2d(a, filter=b, strides=(1,2,2,1), padding="VALID")
c_s = tf.nn.conv2d(a, filter=b, strides=(1,2,2,1), padding="SAME")
tf.layers.conv2d_transpose(a, 6, (7,7), strides=(2,2), padding="valid").shape.as_list()
conv2d_transpose(Input, Kernel, stride=2, padding="valid").shape


sess = tf.Session()
m_v, m_s = sess.run([c_v, c_s])

s_v == v_v
s_v == m_v
