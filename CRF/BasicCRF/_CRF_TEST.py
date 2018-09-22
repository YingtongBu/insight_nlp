#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

import optparse
from CRF.BasicCRF.FeatureExtraction import FeatureExtraction
from CRF.BasicCRF.CRFTrainer import CRFTrainer
from CRF.BasicCRF.CRFPredictor import CRFPredictor
from Common import *

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  (options, args) = parser.parse_args()
  
  model_name = "model.company"
  data_path = os.path.join(
    get_module_path("Common"),
    "CRF/BasicCRF/SampleData"
  )
  
  train_data = os.path.join(data_path, "train.pydict")
  test_data  = os.path.join(data_path, "test.data")

  #train
  crf_trainer = CRFTrainer(model_name,
                           train_data,
                           FeatureExtraction(), 0.1, 0.01, 200)
  crf_trainer.train()

  #predictor_init
  crf_predictor = CRFPredictor(model_name, FeatureExtraction())

  #predict a sample
  crf_predictor.predict('1.当事人:尉氏县第三人民医院', 'output_text.result')

  #predict in batch
  crf_predictor.predict_file(test_data, 'output_file.result')