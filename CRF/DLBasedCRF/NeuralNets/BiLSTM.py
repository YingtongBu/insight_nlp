#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
from Insight_NLP.CRF.DLBasedCRF.util import BIOF1Validation
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
from Insight_NLP.CRF.DLBasedCRF.NeuralNets.KerasLayers.ChainCRF import ChainCRF

class BiLSTM:
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
    self.idx2_labels = {}
    self.train_mini_batch_ranges = None
    self.train_sentence_length_ranges = None

    for model_name in self.model_names:
      label_key = self.datasets[model_name]['label']
      self.label_keys[model_name] = label_key
      self.idx2_labels[model_name] = ({v: k for k, v in
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
             
    self.casing2_idx = self.mappings['casing']

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
        nClassLabels = len(self.mappings[self.label_keys[model_name]])

        if classifier == 'Softmax':
          output = TimeDistributed(Dense(nClassLabels, activation='softmax'), 
                                   name=model_name + '_softmax')(output)
          lossFct = 'sparse_categorical_crossentropy'
        elif classifier == 'CRF':
          output = TimeDistributed(Dense(nClassLabels, activation=None),
                                   name=model_name +
                                        '_hidden_lin_layer')(output)
          crf = ChainCRF(name=model_name + '_crf')
          output = crf(output)
          lossFct = crf.sparse_loss
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

      optimizerParams = {}
      if ('clipnorm' in self.params and 
         self.params['clipnorm'] is not None and 
         self.params['clipnorm'] > 0):
        optimizerParams['clipnorm'] = self.params['clipnorm']
            
      if ('clipvalue' in self.params and 
         self.params['clipvalue'] is not None and 
         self.params['clipvalue'] > 0):
        optimizerParams['clipvalue'] = self.params['clipvalue']
      if self.params['optimizer'].lower() == 'adam':
        opt = Adam(**optimizerParams)
      elif self.params['optimizer'].lower() == 'nadam':
        opt = Nadam(**optimizerParams)
      elif self.params['optimizer'].lower() == 'rmsprop': 
        opt = RMSprop(**optimizerParams)
      elif self.params['optimizer'].lower() == 'adadelta':
        opt = Adadelta(**optimizerParams)
      elif self.params['optimizer'].lower() == 'adagrad':
        opt = Adagrad(**optimizerParams)
      elif self.params['optimizer'].lower() == 'sgd':
        opt = SGD(lr=0.1, **optimizerParams)
                  
      model = Model(inputs=input_nodes, outputs=[output])
      model.compile(loss=lossFct, optimizer=opt)
            
      model.summary(line_length=125)        
      self.models[model_name] = model

  def train_model(self):
    self.epoch += 1
        
    if (self.params['optimizer'] in self.learning_rate_updates and
       self.epoch in self.learning_rate_updates[self.params['optimizer']]):
      logging.info("Update Learning Rate to %f" % 
                   (self.learning_rate_updates[self.params['optimizer']]
                    [self.epoch]))
      for modelName in self.model_names:
        K.set_value(self.models[modelName].optimizer.lr, 
                    self.learning_rate_updates[self.params['optimizer']]
                    [self.epoch]) 
                   
    for batch in self.minibatch_iterate_dataset():
      for modelName in self.model_names:
        nnLabels = batch[modelName][0]
        nnInput = batch[modelName][1:]
        self.models[modelName].train_on_batch(nnInput, nnLabels)  
        
  def minibatch_iterate_dataset(self, modelNames=None):
    if self.train_sentence_length_ranges is None:
      self.train_sentence_length_ranges = {}
      self.train_mini_batch_ranges = {}
      for modelName in self.model_names:
        trainData = self.data[modelName]['trainMatrix']
        trainData.sort(key=lambda x: len(x['tokens']))  
        trainRanges = []
        oldSentLength = len(trainData[0]['tokens'])            
        idxStart = 0

        for idx in range(len(trainData)):
          sentLength = len(trainData[idx]['tokens'])
                    
          if sentLength != oldSentLength:
            trainRanges.append((idxStart, idx))
            idxStart = idx
                    
          oldSentLength = sentLength
        trainRanges.append((idxStart, len(trainData)))
        miniBatchRanges = []
        for batchRange in trainRanges:
          rangeLen = batchRange[1] - batchRange[0]

          bins = int(math.ceil(rangeLen / float(self.params['miniBatchSize'])))
          binSize = int(math.ceil(rangeLen / float(bins)))
                    
          for binNr in range(bins):
            startIdx = binNr * binSize + batchRange[0]
            endIdx = min(batchRange[1], (binNr + 1) * binSize + batchRange[0])
            miniBatchRanges.append((startIdx, endIdx))
                  
        self.train_sentence_length_ranges[modelName] = trainRanges
        self.train_mini_batch_ranges[modelName] = miniBatchRanges
                
    if modelNames is None:
      modelNames = self.model_names

    for modelName in modelNames:      
      x = self.data[modelName]['trainMatrix']
      for dataRange in self.train_sentence_length_ranges[modelName]:
        for i in reversed(range(dataRange[0] + 1, dataRange[1])):
          j = random.randint(dataRange[0], i)
          x[i], x[j] = x[j], x[i]
                   
      random.shuffle(self.train_mini_batch_ranges[modelName])
        
    if self.main_model_name is not None:
      rangeLength = len(self.train_mini_batch_ranges[self.main_model_name])
    else:
      rangeLength = min([len(self.train_mini_batch_ranges[modelName])
                        for modelName in modelNames])
        
    batches = {}
    for idx in range(rangeLength):
      batches.clear()
            
      for modelName in modelNames:   
        trainMatrix = self.data[modelName]['trainMatrix']
        dataRange = (self.train_mini_batch_ranges[modelName]
                     [idx % len(self.train_mini_batch_ranges[modelName])])
        labels = np.asarray([trainMatrix[idx][self.label_keys[modelName]] for
                             idx in range(dataRange[0], dataRange[1])])
        labels = np.expand_dims(labels, -1)
                
        batches[modelName] = [labels]
                
        for featureName in self.params['featureNames']:
          inputData = np.asarray([trainMatrix[idx][featureName] for 
                                 idx in range(dataRange[0], dataRange[1])])
          batches[modelName].append(inputData)
            
      yield batches   
            
  def store_results(self, resultsFilepath):
    if resultsFilepath is not None:
      directory = os.path.dirname(resultsFilepath)
      if not os.path.exists(directory):
        os.makedirs(directory)
                
      self.results_save_path = open(resultsFilepath, 'w')
    else:
      self.results_save_path = None
        
  def fit(self, epochs):
    if self.models is None:
      self.build_model()

    totalTrainTime = 0
    maxDevScore = {modelName: 0 for modelName in self.models.keys()}
    maxTestScore = {modelName: 0 for modelName in self.models.keys()}
    noImprovementSince = 0
        
    for epoch in range(epochs):      
      sys.stdout.flush()           
      logging.info("\n--------- Epoch %d -----------" % (epoch + 1))
            
      startTime = time.time() 
      self.train_model()
      timeDiff = time.time() - startTime
      totalTrainTime += timeDiff
      logging.info("%.2f sec for training (%.2f total)" % 
                   (timeDiff, totalTrainTime))
            
      startTime = time.time() 
      for modelName in self.evaluate_model_names:
        logging.info("-- %s --" % (modelName))
        devScore, testScore = self.compute_score(modelName,
                                                 self.data[modelName]
                                                ['devMatrix'],
                                                 self.data[modelName]
                                                ['testMatrix'])
        if devScore > maxDevScore[modelName]:
          maxDevScore[modelName] = devScore
          maxTestScore[modelName] = testScore
          noImprovementSince = 0
          if self.model_save_path is not None:
            self.save_model(modelName, epoch, devScore, testScore)
        else:
          noImprovementSince += 1
                    
        if self.results_save_path is not None:
          self.results_save_path.write("\t".join(map(str,
                                                     [epoch + 1, modelName,
                                                      devScore,
                                      testScore, maxDevScore[modelName], 
                                      maxTestScore[modelName]])))
          self.results_save_path.write("\n")
          self.results_save_path.flush()
                
        logging.info("Max: %.4f dev; %.4f test" % (maxDevScore[modelName], 
                                                   maxTestScore[modelName]))
        logging.info("")
                
      logging.info("%.2f sec for evaluation" % (time.time() - startTime))
            
      if (self.params['earlyStopping'] > 0 and 
         noImprovementSince >= self.params['earlyStopping']):
        logging.info("!!! Early stopping, no improvement after " + 
                     str(noImprovementSince) + " epochs !!!")
        break
            
  def tag_sentences(self, sentences):
    if 'characters' in self.params['featureNames']:
      self.pad_characters(sentences)

    labels = {}
    for modelName, model in self.models.items():
      paddedPredLabels = self.predict_labels(model, sentences)
      predLabels = []
      for idx in range(len(sentences)):
        unpaddedPredLabels = []
        for tokenIdx in range(len(sentences[idx]['tokens'])):
          if sentences[idx]['tokens'][tokenIdx] != 0:  
            unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

        predLabels.append(unpaddedPredLabels)

      idx2Label = self.idx2_labels[modelName]
      labels[modelName] = ([[idx2Label[tag] for tag in tagSentence] for 
                           tagSentence in predLabels])

    return labels
            
  def get_sentence_lengths(self, sentences):
    sentenceLengths = {}
    for idx in range(len(sentences)):
      sentence = sentences[idx]['tokens']
      if len(sentence) not in sentenceLengths:
        sentenceLengths[len(sentence)] = []
      sentenceLengths[len(sentence)].append(idx)
        
    return sentenceLengths

  def predict_labels(self, model, sentences):
    predLabels = [None] * len(sentences)
    sentenceLengths = self.get_sentence_lengths(sentences)
        
    for indices in sentenceLengths.values():   
      nnInput = []                  
      for featureName in self.params['featureNames']:
        inputData = np.asarray([sentences[idx][featureName] for idx in indices])
        nnInput.append(inputData)
            
      predictions = model.predict(nnInput, verbose=False)
      predictions = predictions.argmax(axis=-1)            
      predIdx = 0
      for idx in indices:
        predLabels[idx] = predictions[predIdx]    
        predIdx += 1   
        
    return predLabels
    
  def compute_score(self, modelName, devMatrix, testMatrix):
    if (self.label_keys[modelName].endswith('_BIO') or
       self.label_keys[modelName].endswith('_IOBES') or
       self.label_keys[modelName].endswith('_IOB')):
      return self.compute_f1_scores(modelName, devMatrix, testMatrix)
    else:
      return self.compute_acc_scores(modelName, devMatrix, testMatrix)

  def compute_f1_scores(self, modelName, devMatrix, testMatrix):
    devPre, devRec, devF1 = self.compute_f1(modelName, devMatrix)
    logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % 
                 (devPre, devRec, devF1))
        
    testPre, testRec, testF1 = self.compute_f1(modelName, testMatrix)
    logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % 
                 (testPre, testRec, testF1))
        
    return devF1, testF1
    
  def compute_acc_scores(self, modelName, devMatrix, testMatrix):
    devAcc = self.compute_acc(modelName, devMatrix)
    testAcc = self.compute_acc(modelName, testMatrix)
        
    logging.info("Dev-Data: Accuracy: %.4f" % (devAcc))
    logging.info("Test-Data: Accuracy: %.4f" % (testAcc)) 
    return devAcc, testAcc       
        
  def compute_f1(self, modelName, sentences):
    labelKey = self.label_keys[modelName]
    model = self.models[modelName]
    idx2Label = self.idx2_labels[modelName]
        
    correctLabels = [sentences[idx][labelKey] for idx in range(len(sentences))]
    predLabels = self.predict_labels(model, sentences)

    labelKey = self.label_keys[modelName]
    encodingScheme = labelKey[labelKey.index('_') + 1:]
        
    pre, rec, f1 = BIOF1Validation.compute_f1(predLabels,
                                              correctLabels,
                                              idx2Label, 'O',
                                              encodingScheme)
    preB, recB, f1B = BIOF1Validation.compute_f1(predLabels, correctLabels,
                                                 idx2Label, 'B',
                                                 encodingScheme)
        
    if f1B > f1:
      logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % 
                    (f1, f1B))
      pre, rec, f1 = preB, recB, f1B
        
    return pre, rec, f1
    
  def compute_acc(self, modelName, sentences):
    correctLabels = ([sentences[idx][self.label_keys[modelName]] for
                      idx in range(len(sentences))])
    predLabels = self.predict_labels(self.models[modelName], sentences)
        
    numLabels = 0
    numCorrLabels = 0
    for sentenceId in range(len(correctLabels)):
      for tokenId in range(len(correctLabels[sentenceId])):
        numLabels += 1
        if (correctLabels[sentenceId][tokenId] == 
           predLabels[sentenceId][tokenId]):
          numCorrLabels += 1

    return numCorrLabels / float(numLabels)
    
  def pad_characters(self, sentences):
    max_char_len = self.params['maxCharLength']
    if max_char_len <= 0:
      for sentence in sentences:
        for token in sentence['characters']:
          max_char_len = max(max_char_len, len(token))

    for sentenceIdx in range(len(sentences)):
      for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
        token = sentences[sentenceIdx]['characters'][tokenIdx]

        if len(token) < max_char_len:
          sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, 
                                                                  (0, 
                                                                   max_char_len
                                                                   -
                                                                   len(token)),
                                                                  'constant')
        else:  
          sentences[sentenceIdx]['characters'][tokenIdx] = token[0:max_char_len]
    
    self.max_char_len = max_char_len
        
  def add_task_identifier(self):
    taskID = 0
    for modelName in self.model_names:
      dataset = self.data[modelName]
      for dataName in ['trainMatrix', 'devMatrix', 'testMatrix']:            
        for sentenceIdx in range(len(dataset[dataName])):
          dataset[dataName][sentenceIdx]['taskID'] = ([taskID] * 
                                                      len(dataset
                                                      [dataName]
                                                      [sentenceIdx]
                                                      ['tokens']))
            
      taskID += 1

  def save_model(self, modelName, epoch, devScore, testScore):
    import json
    import h5py

    if self.model_save_path is None:
      raise ValueError('modelSavePath not specified.')

    savePath = (self.model_save_path.replace("[DevScore]", "%.4f" % devScore).
                replace("[TestScore]", "%.4f" % testScore).
                replace("[Epoch]", str(epoch + 1)).
                replace("[ModelName]", modelName))

    directory = os.path.dirname(savePath)
    if not os.path.exists(directory):
      os.makedirs(directory)

    if os.path.isfile(savePath):
      logging.info("Model " + savePath + 
                   " already exists. Model will be overwritten")

    self.models[modelName].save(savePath, True)

    with h5py.File(savePath, 'a') as h5file:
      h5file.attrs['mappings'] = json.dumps(self.mappings)
      h5file.attrs['params'] = json.dumps(self.params)
      h5file.attrs['modelName'] = modelName
      h5file.attrs['labelKey'] = self.datasets[modelName]['label']

  @staticmethod
  def load_model(modelPath):
    import h5py
    import json
    from .KerasLayers.ChainCRF import create_custom_objects

    model = keras.models.load_model(modelPath, 
                                    custom_objects=create_custom_objects())

    with h5py.File(modelPath, 'r') as f:
      mappings = json.loads(f.attrs['mappings'])
      params = json.loads(f.attrs['params'])
      modelName = f.attrs['modelName']
      labelKey = f.attrs['labelKey']

    bilstm = BiLSTM(params)
    bilstm.set_mappings(mappings, None)
    bilstm.models = {modelName: model}
    bilstm.label_keys = {modelName: labelKey}
    bilstm.idx2_labels = {}
    bilstm.idx2_labels[modelName] = ({v: k for k, v in
                                      bilstm.mappings[labelKey].items()})
    return bilstm