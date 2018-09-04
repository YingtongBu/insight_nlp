#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import re
from sklearn import preprocessing
from Insight_NLP.Classifier.CNN._Model_CNN import _CNNModel
from Insight_NLP.Common import *
from Insight_NLP.Vocabulary import Vocabulary
from tensorflow.contrib import learn

class CNNTextClassifier(object):
  def __init__(self,
               train_file,
               validation_file,
               num_classes,
               embedding_dim=128,
               kernel_sizes='1,1,1,2,3',
               num_kernels=128,
               dropout_keep_prob=0.5,
               l2_reg_lambda=0.0,
               max_words_len=64,
               batch_size=1024,
               num_epochs=2,
               evaluate_frequency=100,
               GPU=3):
    self._GPU = str(GPU)
    self._train_data = open(train_file).readlines()[1:]
    self._validation_data = open(validation_file).readlines()[1:]
    self._num_classes = num_classes
    self._embedding_dim = embedding_dim
    self._kernel_sizes = kernel_sizes
    self._num_kernels = num_kernels
    self._dropout_keep_prob = dropout_keep_prob
    self._l2_reg_lambda = l2_reg_lambda
    self._max_words_len = max_words_len
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._evaluate_frequency = evaluate_frequency
    # select the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU

  def train(self):

    x_train, y_train, vocab_size = self._load_data(self._train_data)

    # Generate batches
    batches = batch_iter(
      list(zip(x_train, y_train)), self._batch_size, self._num_epochs)

    # initial tensorflow session
    self._sess = tf.Session()

    # create the model
    self._model = _CNNModel(
      sequence_length=x_train.shape[1],
      num_classes=y_train.shape[1],
      vocab_size=vocab_size,
      embedding_size=self._embedding_dim,
      filter_sizes=list(map(int, self._kernel_sizes.split(","))),
      num_filters=self._num_kernels,
      l2_reg_lambda=self._l2_reg_lambda)

    # Define Training procedure and optimizer
    self._global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients(
      grads_and_vars, global_step=self._global_step)

    # Output directory for saving models
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(
      os.path.join(os.path.curdir, "models", timestamp))

    # Initialize all variables
    self._sess.run(tf.global_variables_initializer())

    # Checkpoint directory for model saving
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # Training loops
    for batch in batches:
      x_batch, y_batch = zip(*batch)
      self._train_step(self._sess, x_batch, y_batch)
      current_step = tf.train.global_step(self._sess, self._global_step)
      if current_step % self._evaluate_frequency == 0:
        self.predict(self._validation_data)
        print(f"\nstep number till now: {current_step}")

    # save model
    print("training finished, saving the model ...")
    final_step = tf.train.global_step(self._sess, self._global_step)
    path = saver.save(self._sess, checkpoint_prefix, global_step=final_step)
    print(f"Saved model checkpoint to {path}\n")

  def _train_step(self, sess, x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      self._model.input_x          : x_batch,
      self._model.input_y          : y_batch,
      self._model.dropout_keep_prob: self._dropout_keep_prob
    }
    _, step, loss, accuracy = sess.run(
      [self._train_optimizer,
       self._global_step,
       self._model.loss,
       self._model.accuracy],
      feed_dict
    )
    time_str = datetime.datetime.now().isoformat()

  def load_model(self, model_path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.pardir(model_path)))
    self._sess = sess

  def predict(self, sample_file):
    x_dev, y_dev, vocab_size = self._load_data(sample_file)
    # start tf session
    feed_dict = {
      self._model.input_x          : x_dev,
      self._model.input_y          : y_dev,
      self._model.dropout_keep_prob: 1.0
    }
    loss, accuracy, predictions = self._sess.run(
      [self._model.loss,
       self._model.accuracy,
       self._model.predictions],
      feed_dict
    )
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

  def _load_data(self, data_file):
    """
    :param data_file: the path to the data_file
    :return: x_shuffled, y_shuffled, vocab_size
    """
    print('Loading Data ...')
    data = open(data_file, encoding='latin').readlines()[1:]
    x_text, train_y, y = [], [], []
    data = [sample.strip() for sample in data]
    for row in data:
      row = row.split('\t')
      x_text.append(row[1].replace('\ufeff', ''))
      train_y.append(row[0])
    # Split by spaces
    x_text = [self._tokenize_string(sample) for sample in x_text]
    # clean y
    for item in train_y:
      try:
        item = int(item)
        if item > self._num_classes:
          item = 0
      except:
        item = 0
      y.append(item)

    # Generate one-hot labels
    y = [[item] for item in y]
    enc = preprocessing.OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

    #Build vocabulary
    vocab_processor = \
      learn.preprocessing.VocabularyProcessor(self._max_words_len)
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

  def _tokenize_string(self, string: str):
    """
    Tokenization/string cleaning for Chinese and English data
    """
    string = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5()（）！？，,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # clean for chinese character
    new_string = ""
    for char in string:
      if re.findall(r"[\u4e00-\u9fa5]", char) != []:
        char = " " + char + " "
      new_string += char
    return new_string.strip().lower().replace('\ufeff', '')