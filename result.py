# import tensorflow as tf
# import numpy as np
# import os
# import time
# import datetime
# from CNNclassify import CNNsentiment
# from tensorflow.contrib import learn
# from docprocessing import batch_iteration,load_data_tf
# checkpoint_dir = "./runs/1511460977/checkpoints/"
# test_file_path = "data_text.txt"

# batch_size = 50
# allow_soft_placement = True
# log_device_placement = False

# # x_raw,y_data = load_data_tf("data_text.txt")
# vocab_path = os.path.join(checkpoint_dir,"..","vocab")
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

# # x_text = np.array(list(vocab_processor.transform(x_raw)))
# # print (x_text)
# checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
# graph = tf.Graph()
# # all_predictions = 0
# with graph.as_default():
#     session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
#     sess=tf.Session(config=session_conf)
#     with sess.as_default():
#         saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         saver.restore(sess,checkpoint_file)


#         tweet_embeddings = graph.get_operation_by_name("X_values").outputs[0]
#         y = graph.get_operation_by_name("Y_labels").outputs[0]
#         print (y)
#         dropout_prob = graph.get_operation_by_name("dropout_probability").outputs[0]

#         predictions = graph.get_operation_by_name("result/predictions").outputs
#         # predictions = graph.get_operation_by_name("result/predictions").outputs[0]
#         print (predictions)
#         # batches = batch_iteration(x_text,batch_size,1,shuffle=False)

#         all_predictions = []
#         # for x_text_batches in batches:
#             # batch_predictions = sess.run(predictions,{tweet_embeddings:x_text_batches,dropout_prob:1.0})
#             # print (batch_predictions)
#             # print (tf.argmax(batch_predictions,1))
#             # all_predictions= np.concatenate([all_predictions,batch_predictions])

# # print (all_predictions)

