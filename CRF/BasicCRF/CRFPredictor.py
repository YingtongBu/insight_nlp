#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

#install python-crfsuite
import pycrfsuite
from Insight_NLP.CRF.BasicCRF.DataPreprocessing import DataPreprocessing
class CRFPredictor(object):

  def __init__(self, model_name, doc_path, feature_extraction, output_file):
    self.model_name = model_name
    self.doc_path = doc_path
    self.feature_extraction = feature_extraction
    self.output_file = output_file

  def _write_to_doc(self, result, probability):
    f = open(self.output_file, 'w+')
    for i in range(len(result)):
      f.write(result[i] + '\t' + str(probability[i]) + '\n')
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
      result.append(self.data_preprocessing.get_longest([x[1].split("=")[1]
                                                         for x in X_test[i]],
                                                        y_pred[i]))
    self._write_to_doc(result, probability)

  def crf_predictor(self):
    self.data_preprocessing = DataPreprocessing(self. doc_path)
    data = self.data_preprocessing.test_doc_process()
    X = [self.feature_extraction.extract_features(sample) for sample in data]
    self._predict(X)