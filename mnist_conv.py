# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06:16:46 2019

@author: Aman

"""

from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
#importing the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Reset the default graph state
tf.reset_default_graph()

#Test images and leabels
x_test = mnist.test.images
y_test = mnist.test.labels

##HELPER Functions

#Helper Function for weigths
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist, name = "W")

#Helper Function fro baises
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals, name = "b")


#Convolutional Layer
def convolutional_layer(input_x, shape, name = "Convolutional"):
    with tf.name_scope(name):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        conv = tf.nn.conv2d(input_x, W, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("Activations", act)
        return act

#Fully Connnected Layer    
def normal_full_layer(input_layer, size, name = "Full_Layer"):
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size])
        b = init_bias([size])
        act =  tf.matmul(input_layer, W) + b
        return act

        
#Placeholders    
x = tf.placeholder(tf.float32, shape = [None, 784], name = "x")
y_true = tf.placeholder(tf.float32, shape = [None, 10], name = "labels")


#Layers
x_image = tf.reshape(x, [-1, 28, 28, 1])

convo_1 = convolutional_layer(x_image, shape = [5, 5, 1, 32], name = "conv1")
convo_1_pooling = tf.nn.max_pool(convo_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

convo_2 = convolutional_layer(convo_1_pooling, shape = [5, 5, 32, 64], name = "conv2")
convo_2_pooling = tf.nn.max_pool(convo_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024, name = "fc1"))

#DROPOUT Layer
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob = hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10, name = "fc2")

#LOSS Function
with tf.name_scope("Cross_Entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

with tf.name_scope("Train"):
    train = optimizer.minimize(cross_entropy)

#Name scope for Accuracy
with tf.name_scope("Accuracy"):
    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

#Scalar Summaries
tf.summary.scalar(name = 'Cross_Entropy', tensor = cross_entropy)
tf.summary.scalar(name = 'Accuracy', tensor = acc)
tf.summary.image('input_x', x_image, 3)

#Embedding Visualizer

# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 784 # Dimensionality of the embedding.
LOG_DIR = os.getcwd()
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.sprite.image_path = os.path.join(LOG_DIR + "./embedding files", "sprite_images.png") #'mnistdigits.png'
embedding.sprite.single_image_dim.extend([28,28])
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR + "./embedding files", "metadata.tsv")


#Session
sess = tf.InteractiveSession() #Interactive Session

#Global Variable initializer
init = tf.global_variables_initializer()
sess.run(init)
    
#Merged Sumary 
merged_summmary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tboard_output", sess.graph)
projector.visualize_embeddings(writer, config)

#saving the model
saver = tf.train.Saver()

#The steps to run
steps = 5000
   
for i in range(steps):
    
    batch_x, batch_y = mnist.train.next_batch(50)
    
    #For every 500 Steps our model wiil be saved
    if i%500==0:
        saver.save(sess, os.path.join(LOG_DIR + "./tboard_output", "model.ckpt"), i)
    
    #For every 5 steps our summaries will be added    
    if i%5 == 0:
        
        s = sess.run(merged_summmary, feed_dict = {x: batch_x, y_true: batch_y, hold_prob: 0.5})
        writer.add_summary(s, i)
    
    #For every 500 steps our accuracy will be printed
    if i%500 == 0:
        
        print("ON STEP : {}".format(i))
        print("ACCURACY : ")
            
        #Accuracy function    
        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
        print(sess.run(acc, feed_dict = {x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))
    
    #Actual model run    
    sess.run(train, feed_dict = {x: batch_x, y_true: batch_y, hold_prob: 0.5})

#CLosing the session
sess.close()   

