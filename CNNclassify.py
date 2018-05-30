from __future__ import print_function
import numpy as np
import tensorflow as tf
import math,time
class CNNsentiment(object):
    # class_numbers is number of features
    # vocalulary_size
    # length/ seqence length
    # filters
    # l2 control
    def __init__(self,vocabulary_size,length,filters,n_filters,embeddings_size,l2_control=0.0,classes_numbers=12):
        # Word Vectors
        self.tweet_embeddings = tf.placeholder(tf.int32,[None,length],name="X_values")
        # [0,1,0,0,0,0]
        self.y_label = tf.placeholder(tf.float32,[None,classes_numbers],name="Y_labels")
        self.dropout_prob = tf.placeholder(tf.float32,name='dropout_probability')

        # l2 regularization for Vector Embedding
        l2_loss = tf.constant(0.0)
        l2_reg_lambda = 0.01
        with tf.device('/cpu:0'),tf.name_scope("word_embeddings"):
            self.WV = tf.Variable(tf.random_uniform([vocabulary_size,embeddings_size],-1.0,1.0),name="WV")
            self.embeddings_characters = tf.nn.embedding_lookup(self.WV,self.tweet_embeddings)
            self.embed_expanded = tf.expand_dims(self.embeddings_characters,-1)

        # create a CNN layer
        maxpool_outputs = []
        for i,filter_size in enumerate(filters):
            with tf.name_scope("covnul-maxpool-%s"%filter_size):
                # CNN layer
                filter_shape = [filter_size,embeddings_size,1,n_filters]
                # filter matrix for words
                WV = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="WV")
                # baises matrix
                b = tf.Variable(tf.constant(0.1,shape=[n_filters]),name="b")

                
                #convolutional input layer
                covn = tf.nn.conv2d(self.embed_expanded,WV,strides=[1,1,1,1],padding="VALID",name="covnet")
                # applied linearity added with bais
                h = tf.nn.relu(tf.nn.bias_add(covn,b),name="relu")
                # max pooling of the outputs
                pooling = tf.nn.max_pool(h,ksize=[1,length-filter_size+1,1,1],
                                        strides=[1,1,1,1],
                                        padding ="VALID",
                                        name="pooling")
                maxpool_outputs.append(pooling)
        # combine the pooled features
        n_filters_total = n_filters * len(filters)
        self.h_pool = tf.concat(maxpool_outputs,3)
        # flatten the pooled output matrix to get feature vector
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,n_filters_total])

        # Dropout layer 0.5 at training and 1 to evaluate

        with tf.name_scope('dropout'):
            self.drop_set = tf.nn.dropout(self.h_pool_flat,self.dropout_prob)
        with tf.name_scope("result"):
            # WV = tf.get_variable("WV",shape=[n_filters_total,classes_numbers],initializer=tf.contrib.layers.xavier_initializer())
            # WV = tf.Variable(tf.constant(0.0, shape=[n_filters_total, classes_numbers]), name="WV")
            WV = tf.Variable(tf.constant(0.0, shape=[n_filters_total, classes_numbers]), name="WV")
            b = tf.Variable(tf.constant(0.0,shape=[classes_numbers]),name="bais")
            l2_loss += tf.nn.l2_loss(WV)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.drop_set,WV,b,name="result")
            self.predictions = tf.argmax(self.scores,1,name="predictions")
            # self.predictions = tf.round(tf.nn.sigmoid(self.scores),name="predictions")

        # calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.y_label)
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,labels=self.y_label)
            self.loss = tf.reduce_mean(losses)+ 0.15 * l2_loss
        # calculate the engine accuracy
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions,tf.round(self.y_label))
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.y_label,1))
            # all_labels_true = tf.reduce_min(tf.cast(correct_predictions,"float"),1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="Accuracy")
            # self.accuracy_all= tf.reduce_mean(all_labels_true,name="Accuracy_All")
            # reca, rec_op = tf.metrics.recall(labels=y_label,predictions=self.predictions)
