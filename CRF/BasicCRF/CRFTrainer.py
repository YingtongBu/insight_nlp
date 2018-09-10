#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite

import pycrfsuite
from Insight_NLP.CRF.BasicCRF.DataProcessing import DataProcessing

class CRFTrainer(object):

  def __init__(self, model_name, data, feature_extractor):
    self.model_name = model_name
    self.data = data
    self.feature_extractor = feature_extractor

  def _train(self, X_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=False)
    for x_seq, y_seq in zip(X_train, y_train):
      trainer.append(x_seq, y_seq)
    trainer.set_params({
      'c1': 0.1,
      'c2': 0.01,
      'max_iterations': 200,
      'feature.possible_transitions': True
    })
    trainer.train(self.model_name)

  def train(self):
    data_processing = DataProcessing(self.data)
    sample = data_processing.process_train_data()
    X = [self.feature_extractor.extract_features(sample) for sample in sample]
    y = [data_processing.get_labels(sample) for sample in sample]
    self._train(X, y)