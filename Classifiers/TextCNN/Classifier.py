#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Tensorflow import *
from Insight_NLP.Classifiers.TextCNN._Model import _Model
from Insight_NLP.Classifiers.TextCNN.Data import *

def create_classifier_parameter(
  train_file,
  vali_file,  # can be None
  vob_file,
  num_classes,
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
  model_dir: str= "model"):
  
  assert os.path.isfile(train_file)
  assert os.path.isfile(vali_file)
  assert os.path.isfile(vob_file)
  
  return {
    "train_file": os.path.realpath(train_file),
    "vali_file": os.path.realpath(vali_file),
    "vob_file": os.path.realpath(vob_file),
    "num_classes": num_classes,
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
    "model_dir": os.path.realpath(model_dir),
  }

class Classifier(object):
  def train(self, param):
    self.param = param
    assert param["evaluate_frequency"] % 100 == 0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param["GPU"])
    
    self.vob = Vocabulary()
    self.vob.load_from_file(param["vob_file"])
   
    self._create_model()
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    
    train_data = DataSet(data_file=param["train_file"],
                         num_class=param["num_classes"],
                         max_seq_length=param["max_seq_length"],
                         vob=self.vob,
                         remove_OOV=param["remove_OOV"])
    batch_iter = train_data.create_batch_iter(batch_size=param["batch_size"],
                                              epoch_num=param["epoch_num"],
                                              shuffle=True)
    
    if is_none_or_empty(param["vali_file"]):
      vali_data = None
    else:
      vali_data = DataSet(data_file=param["vali_file"],
                          num_class=param["num_classes"],
                          max_seq_length=param["max_seq_length"],
                          vob=self.vob,
                          remove_OOV=param["remove_OOV"])

    self._best_vali_accuracy = None

    model_dir = param["model_dir"]
    execute_cmd(f"rm -rf {model_dir}; mkdir {model_dir}")
    self._model_prefix = os.path.join(model_dir, "iter")
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    param_file = os.path.join(model_dir, "param.pydict")
    write_pydict_file([param], param_file)
   
    display_freq = 10
    accum_loss = 0.
    last_display_time = time.time()
    accum_run_time = 0
    for step, [x_batch, y_batch] in enumerate(batch_iter):
      start_time = time.time()
      loss, accuracy = self._go_a_step(x_batch, y_batch,
                                       param["dropout_keep_prob"])
      duration = time.time() - start_time
      
      accum_loss += loss
      accum_run_time += duration
      if (step + 1) % display_freq == 0:
        accum_time = time.time() - last_display_time
        avg_loss = accum_loss / display_freq
        
        print(f"step: {step + 1}, avg loss: {avg_loss:.4f}, "
              f"accuracy: {accuracy:.4f}, "
              f"time: {accum_time:.4} secs, "
              f"data reading time: {accum_time - accum_run_time:.4} sec.")
        
        accum_loss = 0.
        last_display_time = time.time()
        accum_run_time = 0

      if ((step + 1) % param["evaluate_frequency"] == 0 and
        vali_data is not None):
        self._validate(vali_data, step)

    if vali_data is None:
      self._save_model(step)
    else:
      self._validate(vali_data, step)
      
  def _save_model(self, step):
    self._saver.save(self._sess, self._model_prefix, step)

  def _validate(self, vali_data, step):
    vali_iter = vali_data.create_batch_iter(128, 1, False)
    correct = 0
    for batch_x, batch_y in vali_iter:
      _, accuracy, _ = self.predict(batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
    accuracy = correct / vali_data.size()
    if self._best_vali_accuracy is None or accuracy > self._best_vali_accuracy:
      self._best_vali_accuracy = accuracy
      self._save_model(step)
      
    print(f"evaluation: accuracy: {accuracy:.4f} "
          f"best: {self._best_vali_accuracy:.4f}")
    
  def predict_dataset(self, file_name, out_file):
    data = DataSet(data_file=file_name,
                        num_class=self.param["num_classes"],
                        max_seq_length=self.param["max_seq_length"],
                        vob=self.vob,
                        remove_OOV=self.param["remove_OOV"])
    data_iter = data.create_batch_iter(batch_size=self.param["batch_size"],
                                       epoch_num=1,
                                       shuffle=False)
    fou = open(out_file, "w")
    correct = 0.
    for batch_x, batch_y in data_iter:
      preds, accuracy, class_probs = self.predict(batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
      for idx, y in enumerate(batch_y):
        pred = {
          "label": y,
          "predicted_label": preds[idx],
          "class_probilities": class_probs[idx],
        }
        print(pred, file=fou)
    fou.close()
    
    accuracy = correct / data.size()
    print(f"Test: '{file_name}': {accuracy:.4f}")

  def predict(self, batch_x, batch_y=None):
    if batch_y is None:
      batch_y = [-1] * len(batch_x)
  
    preds, accuracy, class_probs = self._sess.run(
      [
        self._model.predicted_class,
        self._model.accuracy,
        self._model.class_probs,
      ],
      feed_dict={
        self._model.input_x          : batch_x,
        self._model.input_y          : batch_y,
        self._model.dropout_keep_prob: 1.
      }
    )
    
    return preds, accuracy, class_probs

  def _go_a_step(self, x_batch, y_batch, dropout_keep_prob):
    _, loss, accuracy = self._sess.run(
      [
        self._train_optimizer,
        self._model.loss,
        self._model.accuracy
      ],
      feed_dict={
        self._model.input_x          : x_batch,
        self._model.input_y          : y_batch,
        self._model.dropout_keep_prob: dropout_keep_prob,
      }
    )
    
    return loss, accuracy

  def _create_model(self):
    self._model = _Model(
      max_seq_length=self.param["max_seq_length"],
      num_classes=self.param["num_classes"],
      vob_size=self.vob.size(),
      embedding_size=self.param["embedding_size"],
      kernels=self.param["kernels"],
      filter_num=self.param["filter_num"],
      l2_reg_lambda=self.param["l2_reg_lambda"])
      
    self._global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(self.param["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients(
      grads_and_vars, global_step=self._global_step)

  def load_model(self, model_path):
    ''' We only load the best model in {model_path}
    '''
    def extract_id(model_file):
      return int(re.findall(r"iter-(.*?).index", model_file)[0])

    names = [extract_id(name) for name in os.listdir(model_path)
             if name.endswith(".index")]
    best_iter = max(names)
    
    param_file = os.path.join(model_path, "param.pydict")
    self.param = read_pydict_file(param_file)[0]
    
    self.vob = Vocabulary()
    self.vob.load_from_file(self.param["vob_file"])

    self._create_model()
    
    self._sess = tf.Session()
    model_prefix = f"{model_path}/iter-{best_iter}"
    saver = tf.train.import_meta_graph(f"{model_prefix}.meta")
    saver.restore(self._sess, f"{model_prefix}")
    
