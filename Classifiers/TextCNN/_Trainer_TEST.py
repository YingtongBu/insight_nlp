#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Classifiers.TextCNN.Trainer import *
from Insight_NLP.Classifiers.TextCNN.Data import normalize_text
from Insight_NLP.Common import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Insight_NLP.Common"),
    "Insight_NLP/Classifiers/TextCNN/SampleData"
  )

  train_file = os.path.join(data_path, "data.1.train.pydict")
  vali_file = os.path.join(data_path, "data.1.train.pydict")
  
  train_norm_file = normalize_data_file(train_file, normalize_text)
  vali_norm_file = normalize_data_file(vali_file, normalize_text)

  param = create_classifier_parameter(
    train_file=train_norm_file,
    vali_file=vali_norm_file,
    vob_file="vob.txt",
    num_classes=45,
    max_seq_length=32,
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
 
  create_vocabulary(param["train_file"],
                    min_freq=3,
                    out_file=param["vob_file"])
  
  Trainer().train(param)
  print("Training is Done")
