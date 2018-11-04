# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:11:42 2017

@author: Deepak

Part C - Network B
"""

import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import tensorflow as tf
import time, shutil, os
from fdl_examples.datatools import input_data
mnist = input_data.read_data_sets("../../data/", one_hot=True)
import matplotlib.pyplot as plt
import scipy
import math
from scipy.ndimage.interpolation import zoom

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.01
training_epochs = 5 # NOTE: you'll want to eventually change this 
batch_size = 100
display_step = 1

#def layer(input, weight_shape, bias_shape):
def layer(input, W, b):
    return tf.nn.relu(tf.matmul(input, W) + b)

#def inference(x,W1,b1,W2,b2):
def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, W1,b1)
     
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1,W2,b2)
     
    with tf.variable_scope("output"):
        #output = layer(hidden_2, [n_hidden_2, 10], [10])
        output = layer(hidden_2, W3,b3)

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)    
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
    tf.summary.scalar("validation", accuracy)
    return accuracy

def ScaleImage(im):
#    print(im.shape)
#    all_scaled_image=[]
    im_size=round(np.size(im)/784)
    for j in range(im_size):
        img=im[j]
        img=img.reshape(28,28)
        scale_amount=np.random.uniform(low=0.5, high=1.0)
#        print(scale_amount)
        scaled_image=zoom(img,scale_amount)
        scaled_image_size=[round(math.sqrt(scaled_image.size)),round(math.sqrt(scaled_image.size))]
        size = (28,28)
        background =np.zeros(size)
        offset = [round((size[0] - scaled_image_size[0]) / 2), round((size[1] - scaled_image_size[1]) / 2)]
#                offset_y = (size[1] - scaled_image_size[1]) / 2
        background[offset[0]:offset[0] + scaled_image_size[0], offset[1]:offset[1] + scaled_image_size[1]]=scaled_image
        scaled_padded_image =background
        scaled_padded_image=scaled_padded_image.reshape(784)
        im[j]=scaled_padded_image
#    print(im.shape)
    return im

if __name__ == '__main__':
    
#    if os.path.exists("mlp_logs/"):
#        shutil.rmtree("mlp_logs/")

    with tf.Graph().as_default():

        with tf.variable_scope("mlp_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

            weight_init1 = tf.random_normal_initializer(stddev=(2.0/784)**0.5)
            weight_init2 = tf.random_normal_initializer(stddev=(2.0/n_hidden_1)**0.5)
            weight_init3 = tf.random_normal_initializer(stddev=(2.0/n_hidden_2)**0.5)
            bias_init = tf.constant_initializer(value=0.)
            
            W1 = tf.get_variable("W1", [784, n_hidden_1],
                                initializer=weight_init1)
            b1 = tf.get_variable("b1", [n_hidden_1],
                                initializer=bias_init)
            W2 = tf.get_variable("W2", [n_hidden_1, n_hidden_2],
                                initializer=weight_init2)
            b2 = tf.get_variable("b2", [n_hidden_2],
                                initializer=bias_init)
            W3 = tf.get_variable("W3", [n_hidden_2, 10],
                                initializer=weight_init3)
            b3 = tf.get_variable("b3", [10],
                                initializer=bias_init)

            #output = inference(x,W1,b1,W2,b2)
            output = inference(x)

            cost = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op = evaluate(output, y)

            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            sess = tf.Session()

#            summary_writer = tf.summary.FileWriter("mlp_logs/",
#                                                graph_def=sess.graph_def)

            
            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            argMax_y=tf.argmax(y,1)
            argMax_output=tf.argmax(output,1)

            # saver.restore(sess, "mlp_logs/model-checkpoint-66000")
            total_batch = int(mnist.train.num_examples/batch_size)
            z_x1,z_y=mnist.train.next_batch(batch_size*total_batch)
                    # Displaying original image
            num=z_x1[10]
            num0=num.reshape(28,28)
            plt.subplot(211)
            plt.title('Original image in training set', fontsize=25)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(num0)
    
            z_x=ScaleImage(z_x1)
            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.

                # Loop over all batches
                for i in range(total_batch):
#                    minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                    minibatch_x=z_x[i*100:(i+1)*100,0:784]
                    minibatch_y=z_y[i*100:(i+1)*100,0:784]
#                    num=minibatch_x[0]
#                    num0=num.reshape(28,28)
#                    plt.imshow(num0)
                    
#                    minibatch_x1=minibatch_x
                
#                    minibatch_x=ScaleImage(minibatch_x)
 
                    # Fit training using batch data
                    sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    # Compute average loss
                    avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                    accuracy = sess.run(eval_op, feed_dict={x: ScaleImage(mnist.validation.images), y: mnist.validation.labels})

                    print("Validation Error:", (1 - accuracy))

#                    summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
#                    summary_writer.add_summary(summary_str, sess.run(global_step))

 #                   saver.save(sess, "mlp_logs/model-checkpoint", global_step=global_step)


        num1=z_x[10]
        num1=num.reshape(28,28)

        plt.subplot(212)
        plt.imshow(num1)
        plt.title('Scaled image in training set', fontsize=25)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part C2 Output/Original_Scaled.jpg')



        print("Optimization Finished!")


        accuracy = sess.run(eval_op, feed_dict={x: ScaleImage(mnist.test.images), y: mnist.test.labels})

        print("Test Accuracy:", accuracy)

        with open('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part C2 Output/Accuracy.txt', 'w') as f:
            print('Accuracy = ',accuracy, file=f)
        
        f.close()
       
        
        # CONFUSION MATRIX GENERATION   
        
        y_test = sess.run(argMax_y, feed_dict={x: ScaleImage(mnist.test.images), y: mnist.test.labels})
        output_test = sess.run(argMax_output, feed_dict={x: ScaleImage(mnist.test.images), y: mnist.test.labels})
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
 
#        ax = plt.axes()
#        ax.xaxis.set_ticks_position('none') 
#            tb = plt.gca()
#            tb.set_xticks([])
#            tb.set_yticks([])

        plt.show()
        plt.savefig('C:/Users/Deepak/Dropbox/Deep Learning/Project 2/Part C2 Output/Confusion_Matrix.jpg')

 #           py.iplot(table, filename='Confusion_Matrix')
     
    
    

        
