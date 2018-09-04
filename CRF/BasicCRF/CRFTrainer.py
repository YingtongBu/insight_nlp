#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite
import pycrfsuite
from Insight_NLP.CRF.BasicCRF.DataPreprocessing import DataPreprocessing

class CRFTrainer(object):

  #code review: doc_path, feature_extractor
  def __init__(self, model_name, doc_path, feature_extraction):
    self.model_name = model_name
    self.doc_path = doc_path
    self.feature_extraction = feature_extraction

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

  #code review: train(self):
  def crf_trainer(self):
    data_preprocessing = DataPreprocessing(self. doc_path)
    data = data_preprocessing.train_doc_process()
    X = [self.feature_extraction.extract_features(sample) for sample in data]
    y = [data_preprocessing.get_labels(sample) for sample in data]
    self._train(X, y)