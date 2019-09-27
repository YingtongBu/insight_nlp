#!/usr/bin/env python3
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.tf import *
from pa_nlp.tf.estimator.param import ParamBase

class PredictorBase(abc.ABC):
  def __init__(self, param: ParamBase, model_cls, data_reader_cls):
    self._param = param
    self._data_reader_cls = data_reader_cls
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = model_cls(param, False)
    self._sess = tf.Session(graph=self._graph)

  def load_model(self, model_path: str):
    nlp_tf.model_load(self._graph, self._sess, model_path)

  @abc.abstractmethod
  def predict_sample(self, batch_data):
    pass

  @abc.abstractmethod
  def calc_measure(self, pred_labels: list, correct_labels: list):
    pass

  def predict_dataset(self, data_file: str):
    print("-" * 80)
    time_start = time.time()

    reader = self._data_reader_cls(data_file, self._param, False)
    pred_labels = []
    correct_labels = []
    for _, batch_data in reader.get_batch_data():
      pred_labels.extend(self.predict_sample(batch_data))
      correct_labels.extend(batch_data[1])

    measure = self.calc_measure(pred_labels, correct_labels)
    duration = time.time() - time_start
    print(f"[evaluate]: '{data_file}', {measure.items()}, {duration:.2f} sec.")
    print("-" * 80)

