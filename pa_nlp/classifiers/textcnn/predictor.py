#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.tensorflow import *
from pa_nlp.classifiers.textcnn.config import Model
from pa_nlp.classifiers.textcnn.data import *
from pa_nlp.measure import Measure

class Predictor(object):
  def __init__(self):
    self._param = read_pydict_file("model/param.pydict")[0]

    self._vob = Vocabulary(self._param["remove_OOV"],
                           self._param["max_seq_length"])
    self._vob.load_model(self._param["vob_file"])

    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = Model(
        max_seq_len=self._param["max_seq_length"],
        num_classes=self._param["num_classes"],
        vob_size=self._vob.size(),
        embedding_size=self._param["embedding_size"],
        kernels=self._param["kernels"],
        filter_num=self._param["filter_num"],
        is_train=False,
        neg_sample_weight=1,
        l2_reg_lambda=self._param["l2_reg_lambda"]
      )

    self._sess = tf.Session(graph=self._graph)

  def load_model(self, model_path: str):
    print(f"loading model from '{model_path}'")
    with self._graph.as_default():
      tf.train.Saver().restore(
        self._sess,
        tf.train.latest_checkpoint(model_path)
      )

  def predict_dataset(self, data_set: DataSet):
    time_start = time.time()
    pred_file = data_set.data_file.replace(".pydict", ".pred.pydict")
    all_true_labels, all_pred_labels = [], []
    with open(pred_file, "w") as fou:
      for batch_x, batch_y in data_set.create_batch_iter(32, 1, False):
        pred_y, pred_prob = self._sess.run(
          fetches=[
            self._model.predicted_class,
            self._model.class_probs,
          ],
          feed_dict={
            self._model.input_x: batch_x,
            self._model.input_y: batch_y,
          }
        )

        all_true_labels.extend(batch_y)
        all_pred_labels.extend(pred_y)

        for idx, y in enumerate(batch_y):
          pred = {
            "label": y,
            "predicted_label": pred_y[idx],
            "class_probilities": list(pred_prob[idx]),
          }
          print(pred, file=fou)

    eval = Measure.calc_precision_recall_fvalue(all_true_labels,
                                                all_pred_labels)
    print(f"evaluate({data_set.data_file}): "
          f"#sample: {len(all_true_labels)} {eval}")

    duration = time.time() - time_start
    print(f"evaluation takes {duration:.2f} seconds.")
    print("-" * 80)

  def predict_one_sample(self, normed_word_list: list):
    '''
    :param a list of normed_word_list:
    :return: label, probs
    '''
    batch_x = []
    for normed_words in normed_word_list:
      word_ids = self._vob.convert_to_word_ids(normed_words)
      batch_x.append(word_ids)

    preds, probs = self.predict(batch_x)
    return preds, probs

  def predict(self, batch_x):
    pred_y, pred_prob = self._sess.run(
      fetches=[
        self._model.predicted_class,
        self._model.class_probs,
      ],
      feed_dict={
        self._model.input_x: batch_x,
      }
    )
    return pred_y, pred_prob

