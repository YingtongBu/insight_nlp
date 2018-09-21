#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Predictor import *
from Common import *

if __name__ == '__main__':
  data_path = "SampleData"
  test_file = os.path.join(data_path, "test.pydict")

  model_path = os.path.join(data_path, "../model")

  predictor = Predictor(model_path)
  predictor.predict_dataset(test_file)
