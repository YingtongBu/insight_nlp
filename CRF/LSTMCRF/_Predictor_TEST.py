#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from CRF.LSTMCRF.Predictor import *
from Common import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Common"),
    "CRF/LSTMCRF/SampleData"
  )

  train_file = os.path.join(data_path, "train.pydict")
  test_file = os.path.join(data_path, "test.pydict")

  model_path = os.path.join(data_path, "../model")

  predictor = Predictor(model_path)
  predictor.predict_dataset(train_file)
  predictor.predict_dataset(test_file)

  word_list = ['公', '司', '名', '称', ':', '山', '西', '潞', '宝', '兴', '海',
               '新', '材', '料', '有', '限', '公', '司', '地', '址', ':', '山',
               '西', '潞 ', ' 城 ', ' 市 ', ' 店 ', ' 上 ', ' 镇 ', ' 潞 ',
               '宝', ' 工 ', ' 业 ', ' 园 ', ' 区 ', ' 注 ', ' 册 ', ' 资 ',
               ' 本 ', ': ', '5', '0', ',', '0', '0', '0', '万', '元']
  word_ids = predictor.vob.convert_to_word_ids(word_list)
  predict_seq, _, _ = predictor.predict([word_ids], None)
  predict_seq = predict_seq[0]
  
  print(predictor.translate(word_list, predict_seq))
