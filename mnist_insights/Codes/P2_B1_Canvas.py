# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:21:35 2017

@author: Deepak

Part B - Netwok A
"""

import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import tensorflow as tf
#import time, shutil, os
from fdl_examples.datatools import input_data
import matplotlib.pyplot as plt
#import scipy
from scipy.ndimage.interpolation import rotate
import random
#from sklearn.metrics import confusion_matrix
#import numpy

# read in MNIST data --------------------------------------------------
mnist = input_data.read_data_sets("../../data/", one_hot=True)

# run network ----------------------------------------------------------

# Parameters
learning_rate = 0.01
training_epochs = 10 # NOTE: you'll want to eventually change this 
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

def rotateImage(im):
    im_size=round(np.size(im)/784)
    for j in range(im_size):
        img=im[j]
        img=img.reshape(28,28)
        rot_image=rotate(img, random.randint(0,360),reshape=False)
        rot_image=rot_image.reshape(784)
        im[j]=rot_image
    return im

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
#                                            graph_def=sess.graph_def)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        argMax_y=tf.argmax(y,1)
        argMax_output=tf.argmax(output,1)
        
        total_batch = int(mnist.train.num_examples/batch_size)
        z_x1,z_y=mnist.train.next_batch(batch_size*total_batch)

        # Displaying original image
        num=z_x1[0]
        num0=num.reshape(28,28)
        plt.subplot(211)
        plt.title('Original image in training set', fontsize=25)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(num0)
        z_x=rotateImage(z_x1)
        

        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            z_x,z_y=mnist.train.next_batch(batch_size*total_batch)
            
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x=z_x[i*100:(i+1)*100,0:784]
                minibatch_y=z_y[i*100:(i+1)*100,0:784]

                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))                
                accuracy = sess.run(eval_op, feed_dict={x: rotateImage(mnist.validation.images), y: mnist.validation.labels})
                print("Validation Error:", (1 - accuracy))
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                #summary_writer.add_summary(summary_str, sess.run(global_step))
                #saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
        num1=z_x[0]
        num1=num.reshape(28,28)

        plt.subplot(212)
        plt.imshow(num1)
        plt.title('Rotated image in training set', fontsize=25)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part B1 Output/Original_Rotated.jpg')
        print("Optimization Finished!")

        accuracy = sess.run(eval_op, feed_dict={x: rotateImage(mnist.test.images), y: mnist.test.labels})
        print("Test Accuracy:", accuracy)

        with open('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part B1 Output/Accuracy.txt', 'w') as f:
            print('Accuracy = ',accuracy, file=f)        
        f.close()
         
        # CONFUSION MATRIX GENERATION               
        y_test = sess.run(argMax_y, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        output_test = sess.run(argMax_output, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        C_matrix = np.zeros((10,10))
        count=len(y_test)
        for i in range(0,count):
            true_value = y_test[i]
            est_value = output_test[i]
            C_matrix[true_value,est_value] += 1
                    
        plt.figure(figsize = (15,15))
        rlabels = ['  0  ','  1  ','  2  ','  3  ','  4  ','  5  ','  6  ','  7  ','  8  ','  9  ']
        clabels = ['0','1','2','3','4','5','6','7','8','9']
        ytable = plt.table(cellText=np.int_(C_matrix), loc='center', rowLabels=rlabels,colLabels=clabels)
        ytable.set_fontsize(14)
        
        table_props = ytable.properties()
        table_cells = table_props['child_artists']
        for cell in table_cells: 
            cell.set_height(0.09)
            cell.set_width(0.09)
            
 #       plt.rcParams['axes.labelweight'] = 'bold'
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel('True Values', fontsize=25)
        plt.xlabel('Estimated Values', fontsize=25)
        plt.title('Confusion Matrix', fontsize=45)
        plt.show()
        plt.savefig('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part B1 Output/Confusion_Matrix.jpg')