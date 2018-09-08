#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifiers.TextCNN.Classifier import *
from Insight_NLP.Common import *

def train_model(param):
  trainer = Classifier()
  trainer.train(param)
  print("Training is Done")
  
def apply_to_test(model_path, test_file):
  predictor = Classifier()
  predictor.load_model(model_path)
  output_file = test_file.replace(".pydict", ".prediction.pydict")
  predictor.predict_dataset(test_file, output_file)
  
def preprocess(data_file):
  assert data_file.endswith(".pydict")
  norm_file = data_file.replace(".pydict", ".norm.pydict")
  normalize_data_file(data_file, norm_file)
  
  return norm_file
  
if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Insight_NLP.Common"),
    "Insight_NLP/Classifiers/TextCNN/SampleData"
  )

  train_file = os.path.join(data_path, "data.1.train.pydict")
  vali_file = os.path.join(data_path, "data.1.train.pydict")
  test_file = os.path.join(data_path, "data.1.test.pydict")
  
  train_norm_file = preprocess(train_file)
  vali_norm_file = preprocess(vali_file)
  test_norm_file = preprocess(test_file)

  param = create_classifier_parameter(
    train_file=train_norm_file,
    vali_file=vali_norm_file,
    vob_file="vob.txt",
    num_classes=45,
    max_seq_length=64,
    epoch_num=5,
    batch_size=32,
    embedding_size=128,
    kernels="1,2,3,4,5",
    filter_num=128,
    dropout_keep_prob=0.5,
    learning_rate=0.001,
    l2_reg_lambda=0,
    evaluate_frequency=100,
    remove_OOV=False,
    GPU=-1,
    model_dir="model")
 
  '''create_vocabulary(param["train_file"],
                    min_freq=2,
                    out_file=param["vob_file"])'''
  
  # train_model(param)
  apply_to_test(param["model_dir"], test_norm_file)
