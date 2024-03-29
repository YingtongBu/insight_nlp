#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.classifiers.textcnn.trainer import *
from pa_nlp.chinese import split_and_norm_string
from pa_nlp.common import *

if __name__ == '__main__':
  data_path = os.path.join(
    get_module_path("common"),
    "classifiers/textcnn/test_data"
  )

  train_file = os.path.join(data_path, "data.1.train.pydict")
  vali_file = os.path.join(data_path, "data.1.test.pydict")
  
  train_norm_file = normalize_data_file(train_file, split_and_norm_string)
  vali_norm_file = normalize_data_file(vali_file, split_and_norm_string)

  param = create_parameter(
    train_file=train_norm_file,
    vali_files=[vali_norm_file],
    num_classes=45,
    vob_file="vob.data",
    neg_sample_ratio=1,
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
    GPU=-1
  )

  create_vocabulary(param["train_file"], 1, param["vob_file"])
  
  Trainer(param).train()
  print("Training is Done")
