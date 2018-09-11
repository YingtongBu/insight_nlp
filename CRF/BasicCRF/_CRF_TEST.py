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
  parser.add_option('-e', '--data_for_prediction',
                    default='./Data/test.data')
  parser.add_option('-o', '--result_output_file',
                    default='./output.txt')
  parser.add_option('-i', '--prediction_content',
                    default='1.当事人:尉氏县第三人民医院')
  parser.add_option('--c1', default=0.1)
  parser.add_option('--c2', default=0.01)
  parser.add_option('--max_iterations', default=200)
  parser.add_option('--feature_possible_transitions', default=True)
  (options, args) = parser.parse_args()

  #train
  crf_trainer = CRFTrainer(options.model_name,
                           options.data_for_train,
                           FeatureExtraction(), options.c1, options.c2,
                           options.max_iterations)
  crf_trainer.train()

  #predictor_init
  #code review: FeatureExtraction()
  crf_predictor = CRFPredictor(options.model_name, options.result_output_file)

  #override feature extraction
  feature_extraction = FeatureExtraction()

  #predict a sample
  #code review
  crf_predictor.predict(options.prediction_text, feature_extraction)

  #predict in batch
  #code review: output_file
  crf_predictor.predict_file(options.file_for_prediction,
                             feature_extraction)
  crf_predictor.predict_file("test1.pydict", feature_extraction)
  crf_predictor.predict_file("test2.pydict", feature_extraction)
