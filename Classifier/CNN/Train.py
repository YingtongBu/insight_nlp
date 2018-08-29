#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from  Insight_NLP.Classifier.CNN import PreProcess
from Insight_NLP.Classifier.CNN.CNNClassifier import CNNClassifier
from tensorflow.contrib import learn
import optparse

def preprocess(options):
  print('Loading Data ...')
  if options.language_type == 'ENG':
    x_text, y, x_original = PreProcess.load_data_eng(options.train_data,
                                                     options.num_classes)
    # Build vocabulary
    #TODO: to see if the performance is good or not
    #max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = options.num_words
    vocab_processor = \
      learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    x_original = np.array(x_original)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    x_ori_shuffled = x_original[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(options.dev_sample_percentage * float(len(y)))
    x_train, x_dev = \
      x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = \
      y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    x_ori_dev = x_ori_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, x_ori_dev

  elif options.language_type == 'CHI':
    data, label, dictLength, wordDict, rawText = \
      PreProcess.load_data_chi(
        options.train_data, lowRate=0, len_sentence=options.num_words,
        num_of_class=options.num_classes
      )
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(label)))
    x_shuffled = data[shuffle_indices]
    y_shuffled = label[shuffle_indices]
    text_shuffled = np.array(rawText)[shuffle_indices]

    # Split train/dev set
    dev_sample_index = -1 * int(
      options.dev_sample_percentage * float(len(y_shuffled)))
    x_train, x_dev = \
      x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = \
      y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    x_ori_dev = text_shuffled[dev_sample_index:]
    del x_shuffled, y_shuffled, data, label, text_shuffled
    return x_train, y_train, 5000, x_dev, y_dev, x_ori_dev

  else:
    assert False

def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_ori_dev, options):
  # Training
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      if options.language_type == 'ENG':
        vocab_len = len(vocab_processor.vocabulary_)
      elif options.language_type == 'CHI':
        vocab_len = 5000
      else:
        assert False
      cnn = CNNClassifier(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=vocab_len,
        embedding_size=options.embedding_dim,
        filter_sizes=list(map(int, options.kernel_sizes.split(","))),
        num_filters=options.num_kernels,
        l2_reg_lambda=options.l2_reg_lambda)

      # Define Training procedure
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(1e-3)
      grads_and_vars = optimizer.compute_gradients(cnn.loss)
      train_op = optimizer.apply_gradients(grads_and_vars,
                                           global_step=global_step)

      # Output directory for models and summaries
      timestamp = str(int(time.time()))
      out_dir = os.path.abspath(
        os.path.join(os.path.curdir, "runs", timestamp))
      print("Writing to {}\n".format(out_dir))

      # Initialize all variables
      sess.run(tf.global_variables_initializer())

      def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: options.dropout_keep_prob
        }
        _, step, loss, accuracy = sess.run(
          [train_op, global_step, cnn.loss, cnn.accuracy],
          feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str,
        # step, loss, accuracy))

      def dev_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy = sess.run(
          [global_step, cnn.loss, cnn.accuracy],
          feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str,
                                                        step, loss, accuracy))

      def final_dev_step(x_batch, y_batch, x_ori_dev):
        """
        Evaluates model on a dev set, generate the output
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy, predictions = sess.run(
          [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
          feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str,
                                                        step, loss, accuracy))
        print('Generating the truth & prediction table ...')
        y_batch = [np.where(r == 1)[0][0] for r in y_batch]
        truPred = list(zip(y_batch, predictions, x_ori_dev))
        with open('TruPred', 'w') as newFile:
          for index, item in enumerate(truPred):
            newFile.write(str(index) + '\t' + '\t'.join(str(v) for
                                                        v in
                                                        item) + '\n')
        print('File generated!')
      # Generate batches
      batches = PreProcess.batch_iter(
        list(zip(x_train, y_train)), options.batch_size, options.num_epochs)
      # Training loop. For each batch...
      for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % options.evaluate_every == 0:
          print("\nEvaluation:")
          dev_step(x_dev, y_dev)
          print("")

      print("\nFinal Evaluation:")
      final_dev_step(x_dev, y_dev, x_ori_dev)
      print("")
