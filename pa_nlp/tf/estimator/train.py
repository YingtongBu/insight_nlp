#!/usr/bin/env python3
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.tf import *
from pa_nlp.tf.estimator.param import ParamBase

class _ModelBuff:
  def __init__(self, model_path: str, capacity: int):
    '''bug: multiple files'''
    self._all_records = defaultdict(list)
    self._capacity = capacity
    self._model_path = model_path 
    
  def update(self, batch_id: int, data_losses: list):
    '''
    :param data_losses: [(data_file, loss), ...]
    :return: 
    '''
    all_records = self._all_records
    kept_batch_ids = set()
    removed_batch_ids = set()
    for data_file, loss in data_losses:
      records  = all_records[data_file]
      records.append((loss, batch_id, data_file))
      if len(records) > self._capacity:
        records.sort()
        
        for p, (_, his_batch_id, _) in enumerate(records):
          if p < self._capacity:
            kept_batch_ids.add(his_batch_id)
          else:
            removed_batch_ids.add(his_batch_id)
            
        records.pop(self._capacity)      
          
    for his_batch_id in removed_batch_ids - kept_batch_ids:
      self._remove_model(his_batch_id)

  def _remove_model(self, batch_id: int):
    file = os.path.join(self._model_path, f"model-{batch_id}.*")
    nlp.execute_cmd(f"rm {file}")

class TrainerBase(abc.ABC):
  def __init__(self, param: ParamBase, model_cls, predictor_cls, data_reader_cls):
    self._param = param
    self._model_cls = model_cls
    self._predictor_cls = predictor_cls
    self._data_reader_cls = data_reader_cls

    random.seed()
    nlp.ensure_folder_exists(param.path_work)

    self._model = model_cls(param, True)
    self._lr = tf.placeholder(dtype=tf.float32, shape=[])
    self._train_op = nlp_tf.construct_optimizer2(
      self._model.loss, learning_rate=self._lr
    )

    max_to_keep = 3 if nlp.is_none_or_empty(self._param.eval_files) else 1000
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
    self._model_buff = _ModelBuff(self._param.path_model, 2)

    self._predictor = None
    if not nlp.is_none_or_empty(self._param.eval_files):
      pred_param = copy.deepcopy(self._param)
      pred_param.epoch_num = 1
      self._predictor = self._predictor_cls(
        pred_param, self._model_cls, self._data_reader_cls
      )

    nlp_tf.get_network_parameter_num()

  def _get_batch_id(self):
    return self._sess.run(tf.train.get_global_step())

  def _save_model(self):
    nlp_tf.model_save(
      self._saver, self._sess, self._param.path_model, "model", self._batch_id,
    )

  def _evaluate(self):
    self._save_model()

    self._predictor.load_model(self._param.path_model)
    data_losses = []
    for data_file in self._param.eval_files:
      key_measure = self._predictor.predict_dataset(data_file)
      data_losses.append((data_file, key_measure))
      
    self._model_buff.update(self._get_batch_id(), data_losses)
    for records in self._model_buff ._all_records.values():
      loss, batch_id, data_file = records[0]
      print(f"[optimal]: '{data_file}', batch_id: {batch_id}, measure={loss}")
    print()

  @abc.abstractmethod
  def _run_one_batch(self, epoch_id, batch_data):
    pass

  def train(self):
    self._sess = nlp_tf.get_new_session()
    self._sess.run(tf.global_variables_initializer())
    param = self._param

    reader = self._data_reader_cls(param.train_file, param, True)
    for epoch_id, batch_data in reader.get_batch_data():
      self._run_one_batch(epoch_id, batch_data)
      self._batch_id = self._get_batch_id()

      if param.evaluate_freq is not None and  \
        (self._batch_id + 1) % param.evaluate_freq == 0:
        self._evaluate()

    self._save_model()

