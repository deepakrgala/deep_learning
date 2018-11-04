# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:30:46 2017

@author: Deepak

Part A - Network A
"""

import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import tensorflow as tf
import time, shutil, os
from fdl_examples.datatools import input_data
import matplotlib.pyplot as plt
import scipy
from scipy import misc

# read in MNIST data --------------------------------------------------
mnist = input_data.read_data_sets("../../data/", one_hot=True)

# run network ----------------------------------------------------------
    
# Parameters
learning_rate = 0.01
training_epochs = 50 # NOTE: you'll want to eventually change this 
batch_size = 100
display_step = 1


def inference(x,W,b):
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    
    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("output", output)
    
    return output

def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row 
    # into a single value. In general, reduction along axis i 
    # collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, axis=1)
     
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):

    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy

if __name__ == '__main__':
#    if os.path.exists("logistic_logs/"):
#        shutil.rmtree("logistic_logs/")

    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

        init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", [784, 10],
                             initializer=init)    
        b = tf.get_variable("b", [10],
                             initializer=init)

        output = inference(x,W,b)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        train_op = training(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

 #       summary_writer = tf.summary.FileWriter("logistic_logs/",
 #                                           graph_def=sess.graph_def)

        
        init_op = tf.global_variables_initializer()

        sess.run(init_op)


        # PLOTTING EACH DIGIT
        #x[7]=0, x[6]=1, x[13]=2, x[1]=3, x[2]=4, x[27]=5, x[26]=6, x[25]=7, x[9]=8, x[8]=9 
        mini_x, mini_y = mnist.train.next_batch(30)

        num=mini_x[7]
        num0=num.reshape(28,28)

        num=mini_x[6]
        num1=num.reshape(28,28)
        
        num=mini_x[13]
        num2=num.reshape(28,28)
        
        num=mini_x[1]
        num3=num.reshape(28,28)
        
        num=mini_x[2]
        num4=num.reshape(28,28)
        
        num=mini_x[27]
        num5=num.reshape(28,28)
        
        num=mini_x[26]
        num6=num.reshape(28,28)
        
        num=mini_x[25]
        num7=num.reshape(28,28)
        
        num=mini_x[9]
        num8=num.reshape(28,28)
        
        num=mini_x[8]
        num9=num.reshape(28,28)
                    
#       plt.imshow(num1)
#            
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/0.jpg', num0)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/1.jpg', num1)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/2.jpg', num2)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/3.jpg', num3)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/4.jpg', num4)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/5.jpg', num5)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/6.jpg', num6)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/7.jpg', num7)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/8.jpg', num8)
        scipy.misc.imsave('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/9.jpg', num9)

#        scipy.misc.imsave('outfile.jpg', image_array)

        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(batch_size):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                
            # Fit training using batch data
            sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
        # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

                print("Validation Error:", (1 - accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                #summary_writer.add_summary(summary_str, sess.run(global_step))

                #saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
           

        print("Optimization Finished!")
        
        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        
        #PLOTTING FIRST 10 WEIGHTS
        for wi in range (9):
            weights=sess.run(W)
            im=weights[:,wi]
            wim=im.reshape(28,28)
            filename = "C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/W_%d.jpg"%wi
            scipy.misc.imsave(filename, wim)
            
            
        print("Test Accuracy:", accuracy)
        
        with open('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part A1 Output/Accuracy.txt', 'w') as f:
            print('Accuracy = ',accuracy, file=f)
        
        f.close()
 