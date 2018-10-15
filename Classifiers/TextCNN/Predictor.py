#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Tensorflow import *
from Classifiers.TextCNN._Model import _Model
from Classifiers.TextCNN.Data import *
from Classifiers.TextCNN.Trainer import Trainer

class Predictor(object):
  def __init__(self, model_path):
    ''' We only load the best model in {model_path}
    '''
    def extract_id(model_file):
      return int(re.findall(r"iter-(.*?).index", model_file)[0])

    param_file = os.path.join(model_path, "param.pydict")
    self.param = read_pydict_file(param_file)[0]

    self._vob = Vocabulary(self.param["remove_OOV"],
                           self.param["max_seq_length"])
    self._vob.load_model(self.param["vob_file"])

    names = [extract_id(name) for name in os.listdir(model_path)
             if name.endswith(".index")]
    best_iter = max(names)

    self._sess = tf.Session()
    model_prefix = f"{model_path}/iter-{best_iter}"
    print(f"loading model: '{model_prefix}'")
    saver = tf.train.import_meta_graph(f"{model_prefix}.meta")
    saver.restore(self._sess, f"{model_prefix}")

  def predict_dataset(self, file_name):
    data = DataSet(data_file=file_name,
                   num_class=self.param["num_classes"],
                   vob=self._vob)
    data_iter = data.create_batch_iter(batch_size=self.param["batch_size"],
                                       epoch_num=1,
                                       shuffle=False)
    fou = open(file_name.replace(".pydict", ".pred.pydict"), "w")
    correct = 0.
    for batch_x, batch_y in data_iter:
      preds, accuracy, class_probs = self.predict(batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
      for idx, y in enumerate(batch_y):
        pred = {
          "label": y,
          "predicted_label": preds[idx],
          "class_probilities": list(class_probs[idx]),
        }
        print(pred, file=fou)
    fou.close()
    
    accuracy = correct / data.size()
    print(f"Test: '{file_name}': {accuracy:.4f}")
    
  def predict_one_sample(self, normed_word_list: list):
    '''
    :param normed_word_list:
    :return: label, probs
    '''
    word_ids = self._vob.convert_to_word_ids(normed_word_list)
    preds, accuracy, probs = self.predict([word_ids], None)
    return preds[0], probs[0]
    
  def predict(self, batch_x, batch_y):
    return Trainer.predict(self._sess, batch_x, batch_y)

  def _create_model(self):
    self._model = _Model(
      max_seq_length=self.param["max_seq_length"],
      num_classes=self.param["num_classes"],
      vob_size=self._vob.size(),
      embedding_size=self.param["embedding_size"],
      kernels=self.param["kernels"],
      filter_num=self.param["filter_num"],
      l2_reg_lambda=self.param["l2_reg_lambda"])
      
    optimizer = tf.train.AdamOptimizer(self.param["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients(grads_and_vars)
