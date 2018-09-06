#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import tensorflow as tf

class _Model(object):
  def __init__(self,
               max_seq_length,
               num_classes,
               vob_size,
               embedding_size,
               kernels: list,
               filter_num,
               l2_reg_lambda=0.0):
    self.input_x = tf.placeholder(
      tf.int32, [None, max_seq_length], name="input_x")
    self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32,
                                            name="dropout_keep_prob")
    
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      W = tf.Variable(
        tf.random_uniform([vob_size, embedding_size], -1.0, 1.0), name="W")
      embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
      embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    
    pooled_outputs = []
    for kernel in kernels:
      with tf.name_scope(f"conv-maxpool-{kernel}"):
        filter_shape = [kernel, embedding_size, 1, filter_num]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                        name="W")
        b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
        conv = tf.nn.conv2d(
          embedded_chars_expanded,
          W,
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="conv")
        
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
          h,
          ksize=[1, max_seq_length - kernel + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="pool")
        pooled_outputs.append(pooled)

    num_filters_total = filter_num * len(kernels)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.name_scope("dropout"):
      h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

    l2_loss = tf.constant(0.0)
    with tf.name_scope("output"):
      W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      self.class_scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
      self.predicted_class = tf.argmax(self.class_scores, 1, name="predictions",
                                       output_type=tf.int32)

    loss_fun = tf.nn.softmax_cross_entropy_with_logits
    input_y = tf.one_hot(self.input_y, num_classes)
    losses = loss_fun(logits=self.class_scores, labels=input_y)
    self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    correct = tf.equal(self.predicted_class, self.input_y)
    self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy"
    )
    
