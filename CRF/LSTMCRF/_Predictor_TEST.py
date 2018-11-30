#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from CRF.LSTMCRF.Predictor import *
from Common import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Common"),
    "CRF/LSTMCRF/SampleData"
  )

  model_path = "model"
  predictor = Predictor(model_path)

  train_file = os.path.join(data_path, "train.pydict")
  predictor.predict_dataset(train_file)
  
  test_file = os.path.join(data_path, "test.pydict")
  predictor.predict_dataset(test_file)

  word_list = ['公', '司', '名', '称', ':', '山', '西', '潞', '宝', '兴', '海',
               '新', '材', '料', '有', '限', '公', '司', '地', '址', ':', '山',
               '西', '潞', '城', '市', '店', '上', '镇', '潞',
               '宝', '工', '业', '园', '区', '注', '册', '资',
               '本', ':', '5', '0', ',', '0', '0', '0', '万', '元']
  tran, prob = predictor.predict_one_sample(word_list)
  print(tran)
  print(prob)
  
