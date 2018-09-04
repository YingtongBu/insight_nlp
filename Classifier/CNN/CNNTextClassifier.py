#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import re
from sklearn import preprocessing
from Insight_NLP.Classifier.CNN.ModelCNN import _CNNModel
from Insight_NLP.Common import *
from Insight_NLP.Vocabulary import Vocabulary
from tensorflow.contrib import learn

class CNNTextClassifier(object):
  def __init__(self,
               train_file='Insight_NLP/Classifier/CNN/Sample.Train.data',
               test_file='Insight_NLP/Classifier/CNN/Sample.Test.data',
               num_classes=44, embedding_dim=128,
               kernel_sizes='1,1,1,2,3', num_kernels=128, dropout_keep_prob=0.5,
               l2_reg_lambda=0.0, max_words_len=64, batch_size=1024,
               num_epochs=2,evaluate_frequency=100, GPU=3):
    self.GPU = str(GPU)
    self.train_data = open(train_file, 'r', encoding='latin').readlines()[1:]
    self.test_data = open(test_file, 'r', encoding='latin').readlines()[1:]
    self.num_classes = num_classes
    self.embedding_dim = embedding_dim
    self.kernel_sizes = kernel_sizes
    self.num_kernels = num_kernels
    self.dropout_keep_prob = dropout_keep_prob
    self.l2_reg_lambda = l2_reg_lambda
    self.max_words_len = max_words_len
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.evaluate_frequency = evaluate_frequency
    # select the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU

  def train(self):
    '''
    step: word --> word Idx
    step: create batch data : treat as a function(): random.sample(data, N)
    step: create the model: self_model = CNNModel(...), and optimizer ...
          _create_model(....)
    '''
    x_train, y_train, vocab_size = \
      self._pre_process(self.train_data, self.num_classes, self.max_words_len)

    # Generate batches
    batches = batch_iter(
      list(zip(x_train, y_train)), self.batch_size, self.num_epochs)

    # initial tensorflow session
    sess = tf.Session()

    # create the model
    self.model = _CNNModel(
      sequence_length=x_train.shape[1],
      num_classes=y_train.shape[1],
      vocab_size=vocab_size,
      embedding_size=self.embedding_dim,
      filter_sizes=list(map(int, self.kernel_sizes.split(","))),
      num_filters=self.num_kernels,
      l2_reg_lambda=self.l2_reg_lambda)

    # Define Training procedure and optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(self.model.loss)
    train_optimizer = optimizer.apply_gradients(grads_and_vars,
                                         global_step=global_step)

    # Output directory for saving models
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(
      os.path.join(os.path.curdir, "models", timestamp))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Checkpoint directory for model saving
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train_step(x_batch, y_batch):
      """
      A single training step
      """
      feed_dict = {
        self.model.input_x          : x_batch,
        self.model.input_y          : y_batch,
        self.model.dropout_keep_prob: self.dropout_keep_prob
      }
      _, step, loss, accuracy = sess.run(
        [train_optimizer, global_step, self.model.loss, self.model.accuracy],
        feed_dict)
      time_str = datetime.datetime.now().isoformat()
      # print("{}: step {}, loss {:g}, acc {:g}".format(time_str,
      # step, loss, accuracy))

    # Training loops
    for batch in batches:
      x_batch, y_batch = zip(*batch)
      train_step(x_batch, y_batch)
      current_step = tf.train.global_step(sess, global_step)
      if current_step % self.evaluate_frequency == 0:
        print(f"\nstep number till now: {current_step}")

    # save model
    print("training finished, saving the model ...")
    final_step = tf.train.global_step(sess, global_step)
    path = saver.save(sess, checkpoint_prefix, global_step=final_step)
    print("Saved model checkpoint to {}\n".format(path))

  def predict(self, model_path):
    """
    :param model_path: 'models/1536034094/checkpoints/model-8.meta'
    :return:
    """
    x_dev, y_dev, vocab_size = \
      self._pre_process(self.test_data, self.num_classes, self.max_words_len)
    # start tf session
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.pardir(model_path)))
    feed_dict = {
      self.model.input_x          : x_dev,
      self.model.input_y          : y_dev,
      self.model.dropout_keep_prob: 1.0
    }
    loss, accuracy, predictions = sess.run(
      [self.model.loss, self.model.accuracy,self.model.predictions], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print(f"{time_str}: validation loss {loss}, acc {accuracy}")
    # TODO: try store all the files together with checkpoint
    # print('Generating the truth & prediction table ...')
    # y_batch = [np.where(r == 1)[0][0] for r in y_batch]
    # truPred = list(zip(y_batch, predictions, x_ori_dev))
    # with open('TruPred', 'w') as newFile:
    #   for index, item in enumerate(truPred):
    #     newFile.write(str(index) + '\t' +
    #                   '\t'.join(str(v) for v in item) + '\n')
    # print('File generated!')

  def _pre_process(self, data, num_classes, max_words_len):
    print('Loading Data ...')
    x_text, y = self._load_data(data, num_classes)
    #Build vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_words_len)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # summer's
    # vocab_builder = Vocabulary()
    # vocab_builder.create_vob_from_data(x_text)
    # vocab_builder.add_word("<empty>")
    # vocab_builder.add_word("<oov>")
    # x_text = [vocab_builder.convert_to_word_ids(tokens) for tokens in x_text]
    # for index, tokens in enumerate(x_text):
    #   empty_num_add = max_words_len - len(tokens)
    #   if empty_num_add < 0:
    #     empty_num_add = 0
    #     x_text[index] = x_text[index] + empty_num_add * ["<empty>"]
    # print(x_text)
    # x = np.array(x_text)
    # vocab_size = vocab_builder.size()

    # Randomly shuffle data
    vocab_size = len(vocab_processor.vocabulary_)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled, vocab_size

  def _load_data(self, data, num_of_class: int=45):
    """
    :param data: [[10, 'Ping An is in Palo Alto'], [44, 'welcome to ping an']]
    :param num_of_class: from 0 to num_of_class
    :return:
    """
    x_text, train_y, y = [], [], []
    data = [sample.strip() for sample in data]
    for row in data:
      row = row.split('\t')
      x_text.append(row[1].replace('\ufeff', ''))
      train_y.append(row[0])
    # Split by words
    x_text = [token_str(sent) for sent in x_text]
    # clean y
    for item in train_y:
      try:
        item = int(item)
        if item > num_of_class:
          item = 0
      except:
        item = 0
      y.append(item)

    # Generate labels
    y = [[item] for item in y]
    enc = preprocessing.OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return x_text, y