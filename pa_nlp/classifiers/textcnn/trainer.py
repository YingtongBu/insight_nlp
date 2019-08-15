#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.tensorflow import *
from pa_nlp.classifiers.textcnn._model import Model
from pa_nlp.classifiers.textcnn.data import *
from pa_nlp.classifiers.textcnn.predictor import Predictor

class Trainer:
  def __init__(self, param):
    assert param["evaluate_frequency"] % 100 == 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param["GPU"])
    self._param = param

    execute_cmd(f"rm -rf model; mkdir model")
    write_pydict_file([param], "model/param.pydict")

    vob = Vocabulary(param["remove_OOV"], param["max_seq_length"])
    vob.load_model(param["vob_file"])

    self._model = Model(
      max_seq_len=param["max_seq_length"],
      num_classes=param["num_classes"],
      vob_size=vob.size(),
      embedding_size=param["embedding_size"],
      kernels=param["kernels"],
      filter_num=param["filter_num"],
      neg_sample_weight=1 / param["neg_sample_ratio"],
      is_train=True,
      l2_reg_lambda=param["l2_reg_lambda"]
    )

    self._predictor = Predictor()

    optimizer = tf.train.AdamOptimizer(param["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients(grads_and_vars)

    self._train_data = DataSet(data_file=self._param["train_file"],
                               num_class=self._param["num_classes"],
                               vob=vob)

    self._vali_datasets = [self._train_data]
    if not is_none_or_empty(self._param["vali_files"]):
      self._vali_datasets.extend([
        DataSet(data_file=f, num_class=self._param["num_classes"], vob=vob)
        for f in self._param["vali_files"]]
      )

    self._model_path = "model"
    self._model_path_prefix = "model/cnn"

    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())
    self._saver = tf.train.Saver(max_to_keep=50)

  def train(self):
    batch_iter = self._train_data.create_batch_iter(
      batch_size=self._param["batch_size"],
      epoch_num=self._param["epoch_num"],
      shuffle=True
    )

    step = 0
    for step, [x_batch, y_batch] in enumerate(batch_iter):
      start_time = time.time()
      _, loss, accuracy = self._sess.run(
        fetches=[
          self._train_optimizer,
          self._model.loss,
          self._model.accuracy
        ],
        feed_dict={
          self._model.input_x          : x_batch,
          self._model.input_y          : y_batch,
          self._model.dropout_keep_prob: self._param["dropout_keep_prob"],
        }
      )

      duration = time.time() - start_time
      print(f"batch: {step + 1}, "
            f"loss: {loss:.4f}, "
            f"accuracy: {accuracy:.4f}, "
            f"time: {duration:.2} secs")

      if (step + 1) % self._param["evaluate_frequency"] == 0:
        self._save_model(step)
        self._validate()

    self._save_model(step)
    self._validate()

  def _save_model(self, step: int):
    self._saver.save(self._sess, self._model_path_prefix, step)

  def _validate(self):
    self._predictor.load_model(self._model_path)

    for dataset in self._vali_datasets:
      self._predictor.predict_dataset(dataset)

