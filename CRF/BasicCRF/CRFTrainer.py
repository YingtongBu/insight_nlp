#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite

import pycrfsuite
from CRF.BasicCRF._DataProcessing import _DataProcessing

class CRFTrainer(object):
  '''
  call 'train' function to train a CRF model
  '''
  def __init__(self, model_name, data_file, feature_extractor, c1=0.1, c2=0.01,
               max_iterations=200):
    self._model_name = model_name
    self._data_file = data_file
    self._feature_extractor = feature_extractor
    self._c1 = c1 # coefficient for L1 penalty
    self._c2 = c2 # coefficient for L2 penalty
    self._max_iterations = max_iterations # stop earlier

  def _train(self, X_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=False)
    for x_seq, y_seq in zip(X_train, y_train):
      trainer.append(x_seq, y_seq)
    trainer.set_params({
      'c1': self._c1,
      'c2': self._c2,
      'max_iterations': self._max_iterations,
      'feature.possible_transitions': True
      # include transitions that are possible, but not observed
    })
    trainer.train(self._model_name)

  def train(self):
    '''
    :return: return None but generate a specific CRF model
    '''
    data_processing = _DataProcessing(self._data_file)
    data = data_processing._process_train_data()
    X = [self._feature_extractor._extract_features(sample) for sample in data]
    y = [data_processing._get_labels(sample) for sample in data]
    self._train(X, y)