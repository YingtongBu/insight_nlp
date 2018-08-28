#coding: utf8
#author: Shuang Zhao (shuang.zhao11@pactera.com)

import tensorflow as tf
import numpy as np

class TextCNN(object):
  def __init__(self, sequenceLength, numClasses, vocabSize,
               embeddingSize, kernelSizes, numKernels, l2RegLambda=0.0):
    self.inputX = tf.placeholder(tf.int32, [None, sequenceLength],name='inputX')
    self.inputY = tf.placeholder(tf.float32, [None, numClasses], name='inputY')
    self.dropoutKeepProb = tf.placeholder(tf.float32, name='dropoutKeepProb')
    l2Loss = tf.constant(0.0)
    self.w = tf.Variable(tf.random_uniform([vocabSize, embeddingSize], -1, 1))
    self.embeddedChars = tf.nn.embedding_lookup(self.w, self.inputX)
    self.embeddedCharsExpanded = tf.expand_dims(self.embeddedChars, -1)

    pooledOutputs = []
    for idx, kernelSize in enumerate(kernelSizes):
      kernelShape = [kernelSize, embeddingSize, 1, numKernels]
      w = tf.Variable(tf.truncated_normal(kernelShape, stddev=0.1), name='w')
      b = tf.Variable(tf.constant(0.1, shape=[numKernels]), name='b')
      conv = tf.nn.conv2d(
        self.embeddedCharsExpanded, w, strides=[1, 1, 1, 1],
        padding='VALID', name='conv')

      h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

      pooled = tf.nn.max_pool(
        h, ksize = [1, sequenceLength - kernelSize + 1, 1, 1],
        strides=[1, 1, 1, 1], padding='VALID', name='pooled')
      pooledOutputs.append(pooled)

    numKernelsTotal = numKernels * len(kernelSizes)
    self.hPool = tf.concat(pooledOutputs, 3)
    self.hPoolFlat = tf.reshape(self.hPool, [-1, numKernelsTotal])

    self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

    w = tf.get_variable('w', shape=[numKernelsTotal, numClasses],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[numClasses]), name='b')
    l2Loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    self.scores = tf.nn.xw_plus_b(self.hDrop, w, b, name='scores')
    self.predictions = tf.argmax(self.scores, 1, name='predictions')

    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                     labels=self.inputY)
    self.loss = tf.reduce_mean(losses) + l2RegLambda * l2Loss

    correctPredictions = tf.equal(self.predictions, tf.argmax(self.inputY, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correctPredictions, 'float'),
                                   name='accuracy')
if __name__ == '__main__':
  pass