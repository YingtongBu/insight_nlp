#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.classifiers.textcnn.predictor import Predictor
from pa_nlp.common import *
from pa_nlp.classifiers.textcnn.data import normalize_data_file
from pa_nlp.chinese import split_and_norm_string

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("common"),
    "classifiers/textcnn/test_data"
  )

  test_file = os.path.join(data_path, "data.1.test.pydict")
  test_norm_file = normalize_data_file(test_file, split_and_norm_string)
  
  model_path = "model"

  predictor = Predictor(model_path)
  predictor.predict_dataset(test_norm_file)
  
  text = '"Second Boer War: Boers attempt to end the Siege of Ladysmith, ' \
         'which leads to the Battle of Platrand."'
  word_list = split_and_norm_string(text)
  print(predictor.predict_one_sample(text))
