#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time,ast
import datetime
from CNNclassify import CNNsentiment
from tensorflow.contrib import learn

dev_sample_percentage = 0.19
embedding_dim = 300
filter_sizes = "3,4,5"
num_filters = 100
dropout_keep_prob = 0.5
l2_reg_lambda = 0.15

batch_size = 32
num_epochs = 50
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
allow_soft_placement = True
log_device_placement = False
learning_rate = 0.001

load_pre_trained = True
word2vec_path = "/home/ashmit/Documents/wiki.ne/wiki.ne.vec"

from docprocessing import load_data,batch_iteration

# Load data
print("Loading data...")
x_text,y = load_data()
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# Build vocabulary
# max_document_length = max([len(x) for x in x_text])
max_document_length = max([len(x.split(" ")) for x in x_text])
# max_document_length = len(x_text[0])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
y = np.array(y)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
time.sleep(4)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNNsentiment(
            length=x_train.shape[1],
            # num_classes=y_train.shape[1],
            vocabulary_size=len(vocab_processor.vocabulary_),
            embeddings_size=embedding_dim,
            filters=list(map(int, filter_sizes.split(","))),
            n_filters=num_filters,
            l2_control=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev/validation summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        if load_pre_trained:
            initialize_WM = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_),embedding_dim))
            print ("Loading Word2Vec file")
            with open(word2vec_path,"r") as f:
                data = f.read().split("\n")[1:]
                for each in data:
                    word = each.split(" ")[0]
                    wv = []
                    wv = np.array([ast.literal_eval(every) for every in each.split(" ")[1:301]])
                    idx = vocab_processor.vocabulary_.get(word)
                    if idx !=0:
                        initialize_WM[idx] = wv
            sess.run(cnn.WV.assign(initialize_WM))
            del data
            del initialize_WM
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.tweet_embeddings: x_batch,
              cnn.y_label: y_batch,
              cnn.dropout_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy= sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            return [step,loss,accuracy]

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.tweet_embeddings: x_batch,
              cnn.y_label: y_batch,
              cnn.dropout_prob: 1.0
            }
            print ("entered DEV")
            step, summaries, loss, accuracy, predicted = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        # print ("tensorshape",cnn.y_label.get_shape())
        time.sleep(4)
        batches = batch_iteration(
            list(zip(x_train, y_train)), batch_size,num_epochs)
        # Training loop. For each batch...
        total_loss = float(0.00)
        total_accuracy = float(0.00)
        for batch in batches:
            x_batch, y_batch = zip(*batch)

            step,loss,accuracy = train_step(x_batch, y_batch)
            total_loss+= loss
            total_accuracy+=accuracy
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                print ("Average loss :",total_loss/100)
                print ("Average accuracy :",total_accuracy/100)
                x_dev_split = len(x_dev) // 2
                dev_step(x_dev[:x_dev_split], y_dev[:x_dev_split], writer=dev_summary_writer)
                time.sleep(30)
                dev_step(x_dev[x_dev_split:], y_dev[x_dev_split:], writer=dev_summary_writer)
                print("")
                total_loss = float(0.00)
                total_accuracy = float(0.00)
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # path = saver.save(sess, checkpoint_prefix, global_step=current_step,write_meta_graph=False)
                print("Saved model checkpoint to {}\n".format(path))
# print (embedding_dim,num_epochs,learning_rate,"ada grad")