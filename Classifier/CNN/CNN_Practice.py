#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import tensorflow as tf

class CNN(object):
  """
  CNN for text classification
  """
  def __init__(
    self, sequence_length, num_classes, vocab_size, embedding_size,
    filter_sizes, num_filters, l2_reg_lambda=0.0
  ):
    # placeholders for variables
    self.input_x = tf.placeholder(
      tf.int32, [None, sequence_length], name="input_x"
    )
    # float32 in original version, however, no need to be float
    self.input_y = tf.placeholder(
      tf.int32, [None, num_classes], name="input_y"
    )
    # define the dropout keep probability
    self.dropout_keep_prob = tf.placeholder(
      tf.float32, name="dropout_keep_prob"
    )

    # Keep track of l2 regularizer
    l2_loss = tf.constant(0.0)

    # Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      self.weights = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="weights"
      )
      self.embedded_chars = tf.nn.embedding_lookup(self.weights, self.input_x)
      self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    # create convolution and maxpool layer for each filter size
    pooled_outputs = []
    for index, filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        # convolution layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                              name="weights")
        bias = tf