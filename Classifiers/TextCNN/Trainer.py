#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Tensorflow import *
from Insight_NLP.Classifiers.TextCNN._Model import _Model
from Insight_NLP.Classifiers.TextCNN.Data import *

class Trainer(object):
  def __init__(self, param):
    self.param = param
    assert param["evaluate_frequency"] % 100 == 0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param["GPU"])
    
    self.vob = Vocabulary()
    self.vob.load_from_file(param["vob_file"])
   
    self._create_model()
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    
  def train(self):
    param = self.param
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
    self._saver = tf.train.Saver(max_to_keep=5)
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
      _, accuracy, _ = Trainer.predict(self._sess, batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
    accuracy = correct / vali_data.size()
    if self._best_vali_accuracy is None or accuracy > self._best_vali_accuracy:
      self._best_vali_accuracy = accuracy
      self._save_model(step)
      
    print(f"evaluation: accuracy: {accuracy:.4f} "
          f"best: {self._best_vali_accuracy:.4f}\n")

  @staticmethod
  def predict(sess, batch_x, batch_y=None):
    if batch_y is None:
      batch_y = [-1] * len(batch_x)
  
    graph = sess.graph
    preds, accuracy, class_probs = sess.run(
      [
        graph.get_tensor_by_name("output/predictions:0"),
        graph.get_tensor_by_name("output/accuracy:0"),
        graph.get_tensor_by_name("output/Softmax:0")
      ],
      feed_dict={
        graph.get_tensor_by_name("input_x:0"): batch_x,
        graph.get_tensor_by_name("input_y:0"): batch_y,
        graph.get_tensor_by_name("dropout:0"): 1
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
    
