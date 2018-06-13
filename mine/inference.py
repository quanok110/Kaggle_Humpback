import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
#def inference(x):
#
#
#    #tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
#    #tfy = tf.placeholder(tf.float32, [None,1])
#
#    conv1_1 = conv_layer(x, "conv1_1")
#    conv1_2 = conv_layer(conv1_1, "conv1_2")
#    pool1 = max_pool(conv1_2, 'pool1')
#
#    conv2_1 = conv_layer(pool1, "conv2_1")
#    conv2_2 = conv_layer(conv2_1, "conv2_2")
#    pool2 = max_pool(conv2_2, 'pool2')
#
#    conv3_1 = conv_layer(pool2, "conv3_1")
#    conv3_2 = conv_layer(conv3_1, "conv3_2")
#    conv3_3 = conv_layer(conv3_2, "conv3_3")
#    pool3 = max_pool(conv3_3, 'pool3')
#
#    conv4_1 = conv_layer(pool3, "conv4_1")
#    conv4_2 = conv_layer(conv4_1, "conv4_2")
#    conv4_3 = conv_layer(conv4_2, "conv4_3")
#    pool4 = max_pool(conv4_3, 'pool4')
#
#    conv5_1 = conv_layer(pool4, "conv5_1")
#    conv5_2 = conv_layer(conv5_1, "conv5_2")
#    conv5_3 = conv_layer(conv5_2, "conv5_3")
#    pool5 = max_pool(conv5_3, 'pool5')
#
#
#    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
#    fc6 = tf.layers.dense(flatten, 5120, tf.nn.relu, name='fc6')
#    out = tf.layers.dense(fc6, 4250, name='out')
#
#    return out

def npy_dict():
    npy_path = './vgg16.npy'
    data_dict = np.load(npy_path, encoding='latin1').item()
    return data_dict

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, name):
    data_dict = npy_dict()
    with tf.variable_scope(name):  # CNN's filter is constant, NOT Variable that can be trained
        conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
        return lout


if __name__ == "__main__":
    x = np.array([[0],[1],[2],[1],[3]])
    x2 = np.array([0,1,1])
    mask = _get_triplet_mask(x2)
    with tf.Session() as sess:
        mask = sess.run(mask)

    print(mask)