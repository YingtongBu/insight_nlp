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
  
  train_data = os.path.join(data_path, "why.train.pydict")
  # train_data = os.path.join(data_path, "train.pydict")
  # test_data  = os.path.join(data_path, "why.test.txt")

  #train
  crf_trainer = CRFTrainer(model_name,
                           train_data,
                           FeatureExtraction(), 0.1, 0.01, 200)
  crf_trainer.train()

  #predictor_init
  crf_predictor = CRFPredictor(model_name, FeatureExtraction())

  #predict a sample
  # text = "招标人:漳州市交通建设投资开发有限公司"
  text = "亚马逊与哪些公司有相同员工"
  crf_predictor.predict(text, 'output_text.result')
  print(open("output_text.result").read())

  #predict in batch
  # crf_predictor.predict_file(test_data, 'output_file.result')