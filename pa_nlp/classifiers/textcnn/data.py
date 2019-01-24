#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.vocabulary import *
import typing

'''The format of data is defined as
1. each line is a python dict string, denoting a sample dict.
2. each sample dict has basic "label" and "text" keys, whose types are "int" and
"string" respectively.
3. In the function "normalize_data_file", a new key "word_list" would be added.
4. In the function "create_batch_iterator", a new key "word_ids" would be added,
which would be fed in a DL model.
'''

def create_parameter(
  train_file: str,
  vali_files: typing.Union[list, None],  # can be None
  num_classes: int,
  vob_file: str,
  neg_sample_ratio: float=1,  # usually equlas=#neg/#pos.
  max_seq_length=64,
  epoch_num=1,
  batch_size=1024,
  embedding_size=128,
  kernels="1,1,1,2,3",
  filter_num=128,
  dropout_keep_prob=0.5,
  learning_rate=0.001,
  l2_reg_lambda=0.0,
  evaluate_frequency=100,  # must divided by 100.
  remove_OOV=True,
  GPU: int=-1,  # which_GPU_to_run: [0, 4), and -1 denote CPU.
):
  
  assert os.path.isfile(train_file)
  if vali_files is not None:
    for vali_file in vali_files:
      assert os.path.isfile(vali_file)
  
  return {
    "train_file": os.path.realpath(train_file),
    "vali_files": [os.path.realpath(f) for f in vali_files],
    "num_classes": num_classes,
    "vob_file": vob_file,
    "neg_sample_ratio": neg_sample_ratio,
    "max_seq_length": max_seq_length,
    "epoch_num": epoch_num,
    "batch_size": batch_size,
    "embedding_size": embedding_size,
    "kernels": list(map(int, kernels.split(","))),
    "filter_num": filter_num,
    "learning_rate": learning_rate,
    "dropout_keep_prob": dropout_keep_prob,
    "l2_reg_lambda": l2_reg_lambda,
    "evaluate_frequency":  evaluate_frequency,
    "remove_OOV": remove_OOV,
    "GPU":  GPU,
  }

class DataSet:
  def __init__(self, data_file, num_class, vob: Vocabulary):
    self._data = []
    self.data_file = data_file
    samples = read_pydict_file(data_file)
    self._data_name = os.path.basename(data_file)
    for sample in samples:
      if not 0 <= sample["label"] < num_class:
        print(f"ERROR: {data_file}: label={sample['label']}")
        continue
      
      word_ids = vob.convert_to_word_ids(sample["word_list"])
      self._data.append([word_ids, sample["label"]])
      
  def size(self):
    return len(self._data)
      
  def create_batch_iter(self, batch_size, epoch_num, shuffle: bool):
    return create_batch_iter_helper(
      self._data_name, self._data, batch_size, epoch_num, shuffle
    )

def normalize_data_file(file_name, split_and_norm_text_func)-> str:
  '''
  :return: normalized file name
  '''
  data = read_pydict_file(file_name)
  for sample in data:
    text  = sample["text"]
    sample["word_list"] = split_and_norm_text_func(text)
  
  out_file_name = file_name.replace(".pydict", ".norm.pydict")
  write_pydict_file(data, out_file_name)
  return out_file_name
  
