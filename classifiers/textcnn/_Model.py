#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf
import nlp_tensorflow as TF

class _Model(object):
  def __init__(self,
               max_seq_length,
               num_classes,
               vob_size,
               embedding_size,
               kernels: list,
               filter_num,
               l2_reg_lambda=0.0):
    self.input_x = tf.placeholder(tf.int32, [None, max_seq_length], 
                                  name="input_x")
    self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
    print(f"name(input_x):{self.input_x.name}")
    print(f"name(input_y):{self.input_y.name}")
    print(f"name(dropout_keep_prob):{self.dropout_keep_prob.name}")
    
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      W = tf.Variable(tf.random_uniform([vob_size, embedding_size], -1.0, 1.0))
      embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
      embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    
    pooled_outputs = []
    for idx, kernel in enumerate(kernels):
      with tf.name_scope(f"conv-maxpool-{idx}-{kernel}"):
        filter_shape = [kernel, embedding_size, 1, filter_num]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[filter_num]))
        conv = tf.nn.conv2d(embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID")

        h = tf.nn.relu(tf.nn.bias_add(conv, b))
        pooled = tf.nn.max_pool(h,
                                ksize=[1, max_seq_length - kernel + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID')
        pooled_outputs.append(pooled)

    num_filters_total = filter_num * len(kernels)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

    l2_loss = tf.constant(0.0)
    with tf.name_scope("output"):
      W = tf.get_variable("W",
                          shape=[num_filters_total, num_classes],
                          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      class_scores = tf.nn.xw_plus_b(h_drop, W, b)
      #output
      self.class_probs = tf.keras.activations.softmax(class_scores)
      print(f"name(class_probs):{self.class_probs.name}")
      #output
      self.predicted_class = tf.argmax(class_scores, 1,
                                       name="predictions",
                                       output_type=tf.int32)
      print(f"name(predicted_class):{self.predicted_class.name}")

      input_y = tf.one_hot(self.input_y, num_classes)
      #output
      losses = tf.losses.softmax_cross_entropy(input_y, class_scores)
      self.loss = losses + l2_reg_lambda * l2_loss
      print(f"name(loss):{self.loss.name}")
      
      self.accuracy = TF.accuracy(self.predicted_class, self.input_y,
                                  "accuracy")
      print(f"name(accuracy):{self.accuracy.name}")
    
