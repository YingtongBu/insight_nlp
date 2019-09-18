#!/usr/bin/env python3
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.tf import *
from pa_nlp.tf.estimator.predict import PredictorBase
from pa_nlp.tf.estimator.param import Param

class TrainerBase(abc.ABC):
  def __init__(self, param: Param, model_cls, predictor_cls, data_reader_cls):
    self._param = param
    self._model_cls = model_cls
    self._predictor_cls = predictor_cls
    self._data_reader_cls = data_reader_cls

    random.seed()
    nlp.ensure_folder_exists(param.path_work)

    self._model = model_cls(param, True)
    self._lr = tf.placeholder(dtype=tf.float32, shape=[])
    self._train_op = nlp_tf.construct_optimizer(
      self._model.loss, self._lr
    )

  def _get_batch_id(self):
    return self._sess.run(tf.train.get_global_step())

  def _save_model(self):
    nlp_tf.model_save(
      self._saver, self._sess, self._param.path_model, "model", self._batch_id,
    )

  def _evaluate(self):
    self._save_model()

    predictor = self.__dict__.setdefault(
      "_predictor",
      self._predictor_cls(self._param, self._model_cls, self._data_reader_cls)
    )

    predictor.load_model(self._param.path_model)
    for data_file in self._param.eval_files:
      predictor.predict_dataset(data_file)

  @abc.abstractmethod
  def _run_one_batch(self, epoch_id, batch_data):
    pass

  def train(self):
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    param = self._param

    reader = self._data_reader_cls(param.train_file, param, True)
    for epoch_id, batch_data in reader.get_batch_data():
      self._run_one_batch(epoch_id, batch_data)
      self._batch_id = self._get_batch_id()

      if param.evaluate_freq is not None and  \
        (self._batch_id + 1) % param.evaluate_freq == 0:
        self._evaluate()

    self._save_model()

