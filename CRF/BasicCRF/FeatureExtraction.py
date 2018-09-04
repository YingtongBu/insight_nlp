#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from Insight_NLP.CRF.BasicCRF.DataPreprocessing import DataPreprocessing

class FeatureExtraction(object):
  def __init__(self, doc_path):
    self.doc_path = doc_path

  def extract_features(self, data):
    return [self._word_to_features(data, i)
            for i in range(len(data))]

  #code review: add some sample codes.
  #to override
  def _word_to_features(self, sample, pos):
    features = list()
    return features