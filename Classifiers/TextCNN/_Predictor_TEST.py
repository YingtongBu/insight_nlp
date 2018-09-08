#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Classifiers.TextCNN.Predictor import Predictor
from Insight_NLP.Common import *
from Insight_NLP.Classifiers.TextCNN.Data import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Insight_NLP.Common"),
    "Insight_NLP/Classifiers/TextCNN/SampleData"
  )

  test_file = os.path.join(data_path, "data.1.test.pydict")
  test_norm_file = normalize_data_file(test_file, normalize_text)
  
  model_path = os.path.join(data_path, "../model")

  predictor = Predictor(model_path)
  predictor.predict_dataset(test_norm_file)
