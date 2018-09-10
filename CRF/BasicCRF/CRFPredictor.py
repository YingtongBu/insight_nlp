#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite

import pycrfsuite
from Insight_NLP.CRF.BasicCRF.DataProcessing import DataProcessing

class CRFPredictor(object):

  def __init__(self, model_name, data, feature_extractor, output_file,
               is_batch_process):
    self.model_name = model_name
    self.data = data
    self.feature_extractor = feature_extractor
    self.output_file = output_file
    self.is_batch_process = is_batch_process

  def _write_to_doc(self, result, probability):
    f = open(self.output_file, 'w+')
    for i in range(len(result)):
      f.write(str(result[i]) + '\t' + str(probability[i]) + '\n')
    f.close()

  def _predict(self, X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open(self.model_name)
    y_pred = list()
    probability = list()
    for x_seq in X_test:
      pred = (tagger.tag(x_seq))
      y_pred.append(pred)
      probability.append(tagger.probability(pred))
    result = list()
    for i in range(len(y_pred)):
      # result.append(self.data_processing.get_longest_label(
      #   [x[1].split("=")[1] for x in X_test[i]], y_pred[i]))
      result.append([[x[1].split("=")[1] for x in X_test[i]], y_pred[i]])
    print(result)
    self._write_to_doc(result, probability)

  def predict(self):
    self.data_processing = DataProcessing(self.data)
    if self.is_batch_process == True:
      sample = self.data_processing.process_test_data_batch()
    else:
      sample = self.data_processing.process_test_data()
    X = [self.feature_extractor.extract_features(sample) for sample in sample]
    self._predict(X)