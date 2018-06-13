import numpy as np
import pandas as pd
import tensorflow as tf
from input_fn import *
from triplet_loss import *
from resnet_model import *

train_csv = '../dataset/train.csv'
n_classes = 4250
margin = 100
square = False
learning_rate = 0.001
trainning_round = 1000
triplet_strategy = "batch_hard"

def train(triplet_strategy):
    x = tf.placeholder(tf.float32, [None,224,224,3], name='x_input')
    y = tf.placeholder(tf.int32, [None, 1], name='y_input')
    out_pre = resnet50(x)
    out = tf.nn.l2_normalize(out_pre, dim=1)
    if triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(y,out,margin,square)
    elif triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(y,out,margin,square)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.initialize_all_variables()

    #read_data
    train_gen = train_input_fn(train_csv)
    #session config
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth =True
    #train
    with tf.Session(config=config) as sess:
        sess.run(init)
        for i in range(trainning_round):
            [train_batch, label_batch] = train_gen.__next__()
            _, _tra_loss = sess.run([train_op,loss], feed_dict={x:train_batch, y: label_batch})
            print(_tra_loss)

    #output


train(triplet_strategy)