#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

import optparse
from Insight_NLP.CRF.BasicCRF.FeatureExtraction import FeatureExtraction
from Insight_NLP.CRF.BasicCRF.CRFTrainer import CRFTrainer
from Insight_NLP.CRF.BasicCRF.CRFPredictor import CRFPredictor

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-m', '--model_name',
                    default='company')
  parser.add_option('-t', '--data_for_train',
                    default='./Data/train.pydict')
  parser.add_option('-e', '--data_for_test',
                    default='./Data/test.pydict')
  parser.add_option('-o', '--result_output_file',
                    default='./output.txt')
  (options, args) = parser.parse_args()
  #train
  crf_trainer = CRFTrainer(options.model_name,
                           options.data_for_train,
                           FeatureExtraction(options.data_for_train))
  crf_trainer.train()

  #predict
  crf_predictor = CRFPredictor(options.model_name,
                               options.data_for_test,
                               FeatureExtraction(options.data_for_test),
                               options.result_output_file)
  crf_predictor.predict()