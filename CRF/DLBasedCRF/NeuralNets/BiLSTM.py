#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from __future__ import print_function
from CRF.DLBasedCRF.util import BIOF1Validation
import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging
from CRF.DLBasedCRF.NeuralNets.ChainCRF import ChainCRF

class BiLSTM:
  '''
  BiLSTM class to accomplish the BiLSTM model
  '''
  def __init__(self, params=None):
    self.models = None
    self.model_save_path = None
    self.results_save_path = None

    default_params = {'dropout': (0.5, 0.5), 'classifier': ['Softmax'],
                     'LSTM-Size': (100,), 'customClassifier': {},
                     'optimizer': 'adam',
                     'charEmbeddings': None, 'charEmbeddingsSize': 30, 
                     'charFilterSize': 30, 'charFilterLength': 3, 
                     'charLSTMSize': 25, 'maxCharLength': 25,
                     'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                     'earlyStopping': 5, 'miniBatchSize': 32,
                     'featureNames': ['tokens', 'casing'], 
                     'addFeatureDimensions': 10}
    if params is not None:
      default_params.update(params)
    self.params = default_params

  def set_mappings(self, mappings, embeddings):
    self.embeddings = embeddings
    self.mappings = mappings

  def set_dataset(self, datasets, data):
    self.datasets = datasets
    self.data = data
    self.main_model_name = None
    self.epoch = 0
    self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
    self.model_names = list(self.datasets.keys())
    self.evaluate_model_names = []
    self.label_keys = {}
    self.idx_to_labels = {}
    self.train_mini_batch_ranges = None
    self.train_sentence_length_ranges = None

    for model_name in self.model_names:
      label_key = self.datasets[model_name]['label']
      self.label_keys[model_name] = label_key
      self.idx_to_labels[model_name] = ({v: k for k, v in
                                         self.mappings[label_key].items()})
            
      if self.datasets[model_name]['evaluate']:
        self.evaluate_model_names.append(model_name)
            
      logging.info("--- %s ---" % model_name)
      logging.info("%d train sentences" % 
                   len(self.data[model_name]['trainMatrix']))
      logging.info("%d dev sentences" % 
                   len(self.data[model_name]['devMatrix']))
      logging.info("%d test sentences" % 
                   len(self.data[model_name]['testMatrix']))
        
    if len(self.evaluate_model_names) == 1:
      self.main_model_name = self.evaluate_model_names[0]
             
    self.casing_to_idx = self.mappings['casing']

    if (self.params['charEmbeddings'] not in 
       [None, "None", "none", False, "False", "false"]):
      logging.info("Pad words to uniform length for characters embeddings")
      all_sentences = []
      for dataset in self.data.values():
        for data in ([dataset['trainMatrix'], 
                     dataset['devMatrix'], 
                     dataset['testMatrix']]):
          for sentence in data:
            all_sentences.append(sentence)

      self.pad_characters(all_sentences)
      logging.info("Words padded to %d characters" % (self.max_char_len))

  def build_model(self):
    self.models = {}

    tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
    tokens = Embedding(input_dim=self.embeddings.shape[0], 
                       output_dim=self.embeddings.shape[1], 
                       weights=[self.embeddings], trainable=True, 
                       name='word_embeddings')(tokens_input)

    input_nodes = [tokens_input]
    merge_input_layers = [tokens]

    for feature_name in self.params['featureNames']:
      if feature_name == 'tokens' or feature_name == 'characters':
        continue

      feature_input = Input(shape=(None,), dtype='int32',
                           name=feature_name + '_input')
      feature_embedding = Embedding(input_dim=len(self.mappings[feature_name]),
                                   output_dim=self.params
                                   ['addFeatureDimensions'], 
                                   name=feature_name +
                                   '_emebddings')(feature_input)

      input_nodes.append(feature_input)
      merge_input_layers.append(feature_embedding)
        
    if (self.params['charEmbeddings'] not in 
       [None, "None", "none", False, "False", "false"]):
      charset = self.mappings['characters']
      char_embeddings_size = self.params['charEmbeddingsSize']
      max_char_len = self.max_char_len
      char_embeddings = []
      for _ in charset:
        limit = math.sqrt(3.0 / char_embeddings_size)
        vector = np.random.uniform(-limit, limit, char_embeddings_size)
        char_embeddings.append(vector)

      char_embeddings[0] = np.zeros(char_embeddings_size)
      char_embeddings = np.asarray(char_embeddings)

      chars_input = Input(shape=(None, max_char_len), dtype='int32',
                         name='char_input')
      mask_zero = (self.params['charEmbeddings'].lower() == 'lstm')
      chars = TimeDistributed(
          Embedding(input_dim=char_embeddings.shape[0],
                    output_dim=char_embeddings.shape[1],
                    weights=[char_embeddings],
                    trainable=True, 
                    mask_zero=mask_zero),
          name='char_emd')(chars_input)

      if self.params['charEmbeddings'].lower() == 'lstm': 
        char_lstm_size = self.params['charLSTMSize']
        chars = TimeDistributed(Bidirectional(LSTM(char_lstm_size,
                                return_sequences=False)), 
                                name="char_lstm")(chars)
      else:  
        char_filter_size = self.params['charFilterSize']
        char_filter_length = self.params['charFilterLength']
        chars = TimeDistributed(Conv1D(char_filter_size,
                                       char_filter_length,
                                       padding='same'), 
                                name="char_cnn")(chars)
        chars = TimeDistributed(GlobalMaxPooling1D(), 
                                name="char_pooling")(chars)

      self.params['featureNames'].append('characters')
      merge_input_layers.append(chars)
      input_nodes.append(chars_input)

    if self.params['useTaskIdentifier']:
      self.add_task_identifier()
            
      task_id_input = Input(shape=(None,), dtype='int32', name='task_id_input')
      task_id_matrix = np.identity(len(self.model_names), dtype='float32')
      task_id_outputlayer = Embedding(input_dim=task_id_matrix.shape[0],
                                    output_dim=task_id_matrix.shape[1],
                                    weights=[task_id_matrix],
                                    trainable=False, 
                                    name='task_id_embedding')(task_id_input)
        
      merge_input_layers.append(task_id_outputlayer)
      input_nodes.append(task_id_input)
      self.params['featureNames'].append('taskID')

    if len(merge_input_layers) >= 2:
      merged_input = concatenate(merge_input_layers)
    else:
      merged_input = merge_input_layers[0]

    shared_layer = merged_input
    logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
    cnt = 1
    for size in self.params['LSTM-Size']:      
      if isinstance(self.params['dropout'], (list, tuple)):  
        shared_layer = Bidirectional(LSTM(size, return_sequences=True,
                                    dropout=self.params['dropout'][0], 
                                    recurrent_dropout=self.params['dropout']
                                    [1]), 
                                    name='shared_varLSTM_' + 
                                    str(cnt))(shared_layer)
      else:    
        shared_layer = Bidirectional(LSTM(size, return_sequences=True),
                                    name='shared_LSTM_' + 
                                    str(cnt))(shared_layer)
        if self.params['dropout'] > 0.0:
          shared_layer = TimeDistributed(Dropout(self.params['dropout']),
                                        name='shared_dropout_' + 
                                        str(self.params['dropout']) + "_" + 
                                        str(cnt))(shared_layer)
      cnt += 1
                 
    for model_name in self.model_names:
      output = shared_layer
            
      model_classifier = (self.params['customClassifier'][model_name]
                         if model_name in self.params['customClassifier']
                         else self.params['classifier'])

      if not isinstance(model_classifier, (tuple, list)):
        model_classifier = [model_classifier]
            
      cnt = 1
      for classifier in model_classifier:
        n_class_labels = len(self.mappings[self.label_keys[model_name]])

        if classifier == 'Softmax':
          output = TimeDistributed(Dense(n_class_labels, activation='softmax'),
                                   name=model_name + '_softmax')(output)
          loss_fct = 'sparse_categorical_crossentropy'
        elif classifier == 'CRF':
          output = TimeDistributed(Dense(n_class_labels, activation=None),
                                   name=model_name +
                                        '_hidden_lin_layer')(output)
          crf = ChainCRF(name=model_name + '_crf')
          output = crf(output)
          loss_fct = crf.sparse_loss
        elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':
                            
          size = classifier[1]
          if isinstance(self.params['dropout'], (list, tuple)): 
            output = Bidirectional(LSTM(size, return_sequences=True, 
                                   dropout=self.params['dropout'][0], 
                                   recurrent_dropout=self.params['dropout'][1]), 
                                   name=model_name + '_varLSTM_' +
                                   str(cnt))(output)
          else:
            output = Bidirectional(LSTM(size, return_sequences=True), 
                                   name=model_name + '_LSTM_'
                                        + str(cnt))(output)
            if self.params['dropout'] > 0.0:
              output = TimeDistributed(Dropout(self.params['dropout']), 
                                       name=model_name + '_dropout_' +
                                       str(self.params['dropout']) + "_" + 
                                       str(cnt))(output)                    
        else:
          assert(False)  
                    
        cnt += 1

      optimizer_params = {}
      if ('clipnorm' in self.params and 
         self.params['clipnorm'] is not None and 
         self.params['clipnorm'] > 0):
        optimizer_params['clipnorm'] = self.params['clipnorm']
            
      if ('clipvalue' in self.params and 
         self.params['clipvalue'] is not None and 
         self.params['clipvalue'] > 0):
        optimizer_params['clipvalue'] = self.params['clipvalue']
      if self.params['optimizer'].lower() == 'adam':
        opt = Adam(**optimizer_params)
      elif self.params['optimizer'].lower() == 'nadam':
        opt = Nadam(**optimizer_params)
      elif self.params['optimizer'].lower() == 'rmsprop': 
        opt = RMSprop(**optimizer_params)
      elif self.params['optimizer'].lower() == 'adadelta':
        opt = Adadelta(**optimizer_params)
      elif self.params['optimizer'].lower() == 'adagrad':
        opt = Adagrad(**optimizer_params)
      elif self.params['optimizer'].lower() == 'sgd':
        opt = SGD(lr=0.1, **optimizer_params)
                  
      model = Model(inputs=input_nodes, outputs=[output])
      model.compile(loss=loss_fct, optimizer=opt)
            
      model.summary(line_length=125)        
      self.models[model_name] = model

  def train_model(self):
    self.epoch += 1
        
    if (self.params['optimizer'] in self.learning_rate_updates and
       self.epoch in self.learning_rate_updates[self.params['optimizer']]):
      logging.info("Update Learning Rate to %f" % 
                   (self.learning_rate_updates[self.params['optimizer']]
                    [self.epoch]))
      for model_name in self.model_names:
        K.set_value(self.models[model_name].optimizer.lr,
                    self.learning_rate_updates[self.params['optimizer']]
                    [self.epoch]) 
                   
    for batch in self.minibatch_iterate_dataset():
      for model_name in self.model_names:
        nn_labels = batch[model_name][0]
        nn_input = batch[model_name][1:]
        self.models[model_name].train_on_batch(nn_input, nn_labels)
        
  def minibatch_iterate_dataset(self, model_names=None):
    if self.train_sentence_length_ranges is None:
      self.train_sentence_length_ranges = {}
      self.train_mini_batch_ranges = {}
      for model_name in self.model_names:
        train_data = self.data[model_name]['trainMatrix']
        train_data.sort(key=lambda x: len(x['tokens']))
        train_ranges = []
        old_sent_length = len(train_data[0]['tokens'])
        idx_start = 0

        for idx in range(len(train_data)):
          sent_length = len(train_data[idx]['tokens'])
                    
          if sent_length != old_sent_length:
            train_ranges.append((idx_start, idx))
            idx_start = idx
                    
          old_sent_length = sent_length
        train_ranges.append((idx_start, len(train_data)))
        mini_batch_ranges = []
        for batch_range in train_ranges:
          range_len = batch_range[1] - batch_range[0]

          bins = int(math.ceil(range_len / float(self.params['miniBatchSize'])))
          bin_size = int(math.ceil(range_len / float(bins)))
                    
          for bin_nr in range(bins):
            start_idx = bin_nr * bin_size + batch_range[0]
            end_idx = min(batch_range[1], (bin_nr + 1) * bin_size +
                          batch_range[0])
            mini_batch_ranges.append((start_idx, end_idx))
                  
        self.train_sentence_length_ranges[model_name] = train_ranges
        self.train_mini_batch_ranges[model_name] = mini_batch_ranges
                
    if model_names is None:
      model_names = self.model_names

    for model_name in model_names:
      x = self.data[model_name]['trainMatrix']
      for data_range in self.train_sentence_length_ranges[model_name]:
        for i in reversed(range(data_range[0] + 1, data_range[1])):
          j = random.randint(data_range[0], i)
          x[i], x[j] = x[j], x[i]
                   
      random.shuffle(self.train_mini_batch_ranges[model_name])
        
    if self.main_model_name is not None:
      range_length = len(self.train_mini_batch_ranges[self.main_model_name])
    else:
      range_length = min([len(self.train_mini_batch_ranges[model_name])
                         for model_name in model_names])
        
    batches = {}
    for idx in range(range_length):
      batches.clear()
            
      for model_name in model_names:
        train_matrix = self.data[model_name]['trainMatrix']
        data_range = (self.train_mini_batch_ranges[model_name]
                     [idx % len(self.train_mini_batch_ranges[model_name])])
        labels = np.asarray([train_matrix[idx][self.label_keys[model_name]] for
                             idx in range(data_range[0], data_range[1])])
        labels = np.expand_dims(labels, -1)
                
        batches[model_name] = [labels]
                
        for feature_name in self.params['featureNames']:
          input_data = np.asarray([train_matrix[idx][feature_name] for
                                 idx in range(data_range[0], data_range[1])])
          batches[model_name].append(input_data)
            
      yield batches   
            
  def store_results(self, results_filepath):
    if results_filepath is not None:
      directory = os.path.dirname(results_filepath)
      if not os.path.exists(directory):
        os.makedirs(directory)
                
      self.results_save_path = open(results_filepath, 'w')
    else:
      self.results_save_path = None
        
  def fit(self, epochs):
    if self.models is None:
      self.build_model()

    total_train_time = 0
    max_dev_score = {model_name: 0 for model_name in self.models.keys()}
    max_test_score = {model_name: 0 for model_name in self.models.keys()}
    no_improvement_since = 0
        
    for epoch in range(epochs):      
      sys.stdout.flush()           
      logging.info("\n--------- Epoch %d -----------" % (epoch + 1))
            
      start_time = time.time()
      self.train_model()
      time_diff = time.time() - start_time
      total_train_time += time_diff
      logging.info("%.2f sec for training (%.2f total)" % 
                   (time_diff, total_train_time))
            
      start_time = time.time()
      for model_name in self.evaluate_model_names:
        logging.info("-- %s --" % (model_name))
        dev_score, test_score = self.compute_score(model_name,
                                                 self.data[model_name]
                                                ['devMatrix'],
                                                 self.data[model_name]
                                                ['testMatrix'])
        if dev_score > max_dev_score[model_name]:
          max_dev_score[model_name] = dev_score
          max_test_score[model_name] = test_score
          no_improvement_since = 0
          if self.model_save_path is not None:
            self.save_model(model_name, epoch, dev_score, test_score)
        else:
          no_improvement_since += 1
                    
        if self.results_save_path is not None:
          self.results_save_path.write("\t".join(map(str,
                                                     [epoch + 1, model_name,
                                                      dev_score,
                                      test_score, max_dev_score[model_name],
                                      max_test_score[model_name]])))
          self.results_save_path.write("\n")
          self.results_save_path.flush()
                
        logging.info("Max: %.4f dev; %.4f test" % (max_dev_score[model_name],
                                                   max_test_score[model_name]))
        logging.info("")
                
      logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
      if (self.params['earlyStopping'] > 0 and 
         no_improvement_since >= self.params['earlyStopping']):
        logging.info("!!! Early stopping, no improvement after " + 
                     str(no_improvement_since) + " epochs !!!")
        break
            
  def tag_sentences(self, sentences):
    if 'characters' in self.params['featureNames']:
      self.pad_characters(sentences)

    labels = {}
    for model_name, model in self.models.items():
      padded_pred_labels = self.predict_labels(model, sentences)
      pred_labels = []
      for idx in range(len(sentences)):
        unpadded_pred_labels = []
        for token_idx in range(len(sentences[idx]['tokens'])):
          if sentences[idx]['tokens'][token_idx] != 0:
            unpadded_pred_labels.append(padded_pred_labels[idx][token_idx])

        pred_labels.append(unpadded_pred_labels)

      idx_to_label = self.idx_to_labels[model_name]
      labels[model_name] = ([[idx_to_label[tag] for tag in tag_sentence] for
                           tag_sentence in pred_labels])

    return labels
            
  def get_sentence_lengths(self, sentences):
    sentence_lengths = {}
    for idx in range(len(sentences)):
      sentence = sentences[idx]['tokens']
      if len(sentence) not in sentence_lengths:
        sentence_lengths[len(sentence)] = []
      sentence_lengths[len(sentence)].append(idx)
        
    return sentence_lengths

  def predict_labels(self, model, sentences):
    pred_labels = [None] * len(sentences)
    sentence_lengths = self.get_sentence_lengths(sentences)
        
    for indices in sentence_lengths.values():
      nn_input = []
      for feature_name in self.params['featureNames']:
        input_data = np.asarray([sentences[idx][feature_name]
                                 for idx in indices])
        nn_input.append(input_data)
            
      predictions = model.predict(nn_input, verbose=False)
      predictions = predictions.argmax(axis=-1)            
      pred_idx = 0
      for idx in indices:
        pred_labels[idx] = predictions[pred_idx]
        pred_idx += 1
        
    return pred_labels
    
  def compute_score(self, model_name, dev_matrix, test_matrix):
    if (self.label_keys[model_name].endswith('_BIO') or
       self.label_keys[model_name].endswith('_IOBES') or
       self.label_keys[model_name].endswith('_IOB')):
      return self.compute_f1_scores(model_name, dev_matrix, test_matrix)
    else:
      return self.compute_acc_scores(model_name, dev_matrix, test_matrix)

  def compute_f1_scores(self, model_name, dev_matrix, test_matrix):
    dev_pre, dev_rec, dev_f1 = self.compute_f1(model_name, dev_matrix)
    logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % 
                 (dev_pre, dev_rec, dev_f1))
        
    test_pre, test_rec, test_f1 = self.compute_f1(model_name, test_matrix)
    logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % 
                 (test_pre, test_rec, test_f1))
        
    return dev_f1, test_f1
    
  def compute_acc_scores(self, model_name, dev_matrix, test_matrix):
    dev_acc = self.compute_acc(model_name, dev_matrix)
    test_acc = self.compute_acc(model_name, test_matrix)
        
    logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
    logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
    return dev_acc, test_acc
        
  def compute_f1(self, model_name, sentences):
    label_key = self.label_keys[model_name]
    model = self.models[model_name]
    idx_to_label = self.idx_to_labels[model_name]
        
    correct_labels = [sentences[idx][label_key] for
                      idx in range(len(sentences))]
    pred_labels = self.predict_labels(model, sentences)

    label_key = self.label_keys[model_name]
    encoding_scheme = label_key[label_key.index('_') + 1:]
        
    pre, rec, f1 = BIOF1Validation.compute_f1(pred_labels,
                                              correct_labels,
                                              idx_to_label, 'O',
                                              encoding_scheme)
    pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(pred_labels, correct_labels,
                                                 idx_to_label, 'B',
                                                 encoding_scheme)
        
    if f1_b > f1:
      logging.debug("Setting wrong tags to pre_b- improves from %.4f to %.4f" %
                    (f1, f1_b))
      pre, rec, f1 = pre_b, rec_b, f1_b
        
    return pre, rec, f1
    
  def compute_acc(self, model_name, sentences):
    correct_labels = ([sentences[idx][self.label_keys[model_name]] for
                      idx in range(len(sentences))])
    pred_labels = self.predict_labels(self.models[model_name], sentences)
        
    num_labels = 0
    num_corr_labels = 0
    for sentence_id in range(len(correct_labels)):
      for token_id in range(len(correct_labels[sentence_id])):
        num_labels += 1
        if (correct_labels[sentence_id][token_id] ==
           pred_labels[sentence_id][token_id]):
          num_corr_labels += 1

    return num_corr_labels / float(num_labels)
    
  def pad_characters(self, sentences):
    max_char_len = self.params['maxCharLength']
    if max_char_len <= 0:
      for sentence in sentences:
        for token in sentence['characters']:
          max_char_len = max(max_char_len, len(token))

    for sentence_idx in range(len(sentences)):
      for token_idx in range(len(sentences[sentence_idx]['characters'])):
        token = sentences[sentence_idx]['characters'][token_idx]

        if len(token) < max_char_len:
          sentences[sentence_idx]['characters'][token_idx] = np.pad(token,
                                                                  (0, 
                                                                   max_char_len
                                                                   -
                                                                   len(token)),
                                                                  'constant')
        else:  
          sentences[sentence_idx]['characters'][token_idx] = \
            token[0:max_char_len]
    
    self.max_char_len = max_char_len
        
  def add_task_identifier(self):
    task_id = 0
    for model_name in self.model_names:
      dataset = self.data[model_name]
      for data_name in ['trainMatrix', 'devMatrix', 'testMatrix']:
        for sentence_idx in range(len(dataset[data_name])):
          dataset[data_name][sentence_idx]['taskID'] = ([task_id] *
                                                      len(dataset
                                                      [data_name]
                                                      [sentence_idx]
                                                      ['tokens']))
            
      task_id += 1

  def save_model(self, model_name, epoch, dev_score, test_score):
    import json
    import h5py

    if self.model_save_path is None:
      raise ValueError('modelSavePath not specified.')

    save_path = (self.model_save_path.replace("[DevScore]", "%.4f" % dev_score).
                replace("[TestScore]", "%.4f" % test_score).
                replace("[Epoch]", str(epoch + 1)).
                replace("[ModelName]", model_name))

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
      os.makedirs(directory)

    if os.path.isfile(save_path):
      logging.info("Model " + save_path +
                   " already exists. Model will be overwritten")

    self.models[model_name].save(save_path, True)

    with h5py.File(save_path, 'a') as h5file:
      h5file.attrs['mappings'] = json.dumps(self.mappings)
      h5file.attrs['params'] = json.dumps(self.params)
      h5file.attrs['modelName'] = model_name
      h5file.attrs['labelKey'] = self.datasets[model_name]['label']

  @staticmethod
  def load_model(model_path):
    import h5py
    import json
    from CRF.DLBasedCRF.NeuralNets.KerasLayers.ChainCRF import \
      create_custom_objects

    model = keras.models.load_model(model_path,
                                    custom_objects=create_custom_objects())

    with h5py.File(model_path, 'r') as f:
      mappings = json.loads(f.attrs['mappings'])
      params = json.loads(f.attrs['params'])
      model_name = f.attrs['modelName']
      label_key = f.attrs['labelKey']

    bilstm = BiLSTM(params)
    bilstm.set_mappings(mappings, None)
    bilstm.models = {model_name: model}
    bilstm.label_keys = {model_name: label_key}
    bilstm.idx_to_labels = {}
    bilstm.idx_to_labels[model_name] = ({v: k for k, v in
                                        bilstm.mappings[label_key].items()})
    return bilstm