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
  parser.add_option('--model_name',
                    default='company')
  parser.add_option('--data_for_train',
                    default='./Data/train.pydict')
  parser.add_option('--file_for_prediction',
                    default='./Data/test.data')
  (options, args) = parser.parse_args()

  #train
  crf_trainer = CRFTrainer(options.model_name,
                           options.data_for_train,
                           FeatureExtraction(), 0.1, 0.01, 200)
  crf_trainer.train()

  #predictor_init
  crf_predictor = CRFPredictor(options.model_name, FeatureExtraction())

  #predict a sample
  crf_predictor.predict('1.当事人:尉氏县第三人民医院', 'output_text.result')

  #predict in batch
  crf_predictor.predict_file(options.file_for_prediction, 'output_file.result')