#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Classifiers.TextCNN.Trainer import *
from Chinese import split_and_norm_string
from Common import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("Common"),
    "Classifiers/TextCNN/SampleData"
  )

  train_file = os.path.join(data_path, "data.1.train.pydict")
  vali_file = os.path.join(data_path, "data.1.train.pydict")
  
  train_norm_file = normalize_data_file(train_file, split_and_norm_string)
  vali_norm_file = normalize_data_file(vali_file, split_and_norm_string)

  param = create_parameter(
    train_file=train_norm_file,
    vali_file=vali_norm_file,
    num_classes=45,
    vob_file="vob.data",
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
 
  create_vocabulary(param["train_file"], 1, param["vob_file"])
  
  Trainer().train(param)
  print("Training is Done")
