#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite

import pycrfsuite
from CRF.BasicCRF._DataProcessing import _DataProcessing
from CRF.BasicCRF.FeatureExtraction import FeatureExtraction

class CRFPredictor(object):
  '''
  call 'predict_file' function to do prediction in batch
  call 'predict' function to do prediction in a scale of single sentence
  '''
  def __init__(self, model_name, feature_extractor):
    self._model_name = model_name
    self._feature_extractor = feature_extractor

  def _write_to_doc(self, result, probability, output_file):
    f = open(output_file, 'w+')
    for i in range(len(result)):
      f.write(str(result[i]) + '\t' + str(probability[i]) + '\n')
    f.close()

  def _predict(self, X_test, output_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(self._model_name)
    y_pred = list()
    probability = list()
    for x_seq in X_test:
      pred = (tagger.tag(x_seq))
      y_pred.append(pred)
      probability.append(tagger.probability(pred))
    result = list()
    for i in range(len(y_pred)):
      result.append([[x[1].split("=")[1] for x in X_test[i]], y_pred[i]])
    self._write_to_doc(result, probability, output_file)

  def predict_file(self, file_name, output_file):
    '''
    :param file_name: input data collection to do prediction
    :param output_file: pre-assigned output file path
    :return: prediction results directed to the output file
    '''
    self.data_processing = _DataProcessing(file_name)
    data = self.data_processing._process_test_data_batch()
    X = [self._feature_extractor._extract_features(sample) for sample in data]
    self._predict(X, output_file)

  def predict(self, prediction_content, output_file):
    '''
    :param prediction_content: single item of input data
    :param output_file: pre-assigned output file path
    :return: prediction results directed to the output file
    '''
    self.data_processing = _DataProcessing(prediction_content)
    data = [self.data_processing._process_line(prediction_content)]
    X = [self._feature_extractor._extract_features(sample) for sample in data]
    self._predict(X, output_file)