#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Tensorflow import *
from Insight_NLP.Classifiers.TextCNN._Model import _Model
from Insight_NLP.Classifiers.TextCNN.Data import *

class Classifier(object):
  def train(self,
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
            l2_reg_lambda=0.0,
            evaluate_frequency=100,  # must divided by 100.
            remove_OOV=True,
            GPU: int=-1,  # which_GPU_to_run: [0, 4), and -1 denote CPU.
            save_model_dir: str="model"
            ):
    
    assert train_file.endswith(".pydict")
    assert is_none_or_empty(vali_file) or vali_file.endswith(".pydict")
    assert evaluate_frequency % 100 == 0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    
    vob = Vocabulary()
    vob.load_from_file(vob_file)
   
    kernels = list(map(int, kernels.split(",")))
    self._create_model(max_seq_length, num_classes, vob.size(),
                       embedding_size, kernels, filter_num, l2_reg_lambda)
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    
    train_data = DataSet(train_file, num_classes, max_seq_length, vob,
                         remove_OOV)
    batch_iter = train_data.create_batch_iter(batch_size, epoch_num, True)
    
    if is_none_or_empty(vali_file):
      vali_data = None
    else:
      vali_data = DataSet(vali_file, num_classes, max_seq_length, vob,
                          remove_OOV)
    self._best_vali_accuracy = None
    
    self._checkpoint_dir = save_model_dir
    try:
      os.rmdir(save_model_dir)
    except:
      pass
    os.mkdir(self._checkpoint_dir)
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
   
    display_freq = 10
    accum_loss = 0.
    last_display_time = time.time()
    accum_run_time = 0
    for step, [x_batch, y_batch] in enumerate(batch_iter):
      start_time = time.time()
      loss = self._go_a_step(x_batch, y_batch, dropout_keep_prob)
      duration = time.time() - start_time
      
      accum_loss += loss
      accum_run_time += duration
      if (step + 1) % display_freq == 0:
        accum_time = time.time() - last_display_time
        avg_loss = accum_loss / display_freq
        
        print(f"step: {step + 1}, avg loss: {avg_loss:.4f} "
              f"time: {accum_time:.4} secs "
              f"data reading time: {accum_time - accum_run_time:.4} sec")
        
        accum_loss = 0.
        last_display_time = time.time()
        accum_run_time = 0
        
      if (step + 1) % evaluate_frequency == 0 and vali_data is not None:
        self._validate(vali_data, step)

    if vali_data is None:
      self._save_model(step)
      pass  #save model
    else:
      self._validate(vali_data, step)
      
  def _save_model(self, step):
    # self._saver.save(self._sess, self._checkpoint_dir, step)
    pass

  def _validate(self, vali_data, step):
    vali_iter = vali_data.create_batch_iter(128, 1, False)
    correct = 0
    for batch_x, batch_y in vali_iter:
      _, accuracy = self.predict(batch_x, batch_y)
      correct += accuracy * len(batch_x)
      
    accuracy = correct / vali_data.size()
    if self._best_vali_accuracy is None or accuracy > self._best_vali_accuracy:
      self._best_vali_accuracy = accuracy
      self._save_model(step)
      
    print(f"evaluation: accuracy: {accuracy:.4f} "
          f"best: {self._best_vali_accuracy:.4f}")

  def predict(self, batch_x, batch_y=None):
    if batch_y is None:
      batch_y = [-1] * len(batch_x)
      
    preds, accuracy = self._sess.run(
      [
        self._model.predicted_class,
        self._model.accuracy,
      ],
      feed_dict={
        self._model.input_x          : batch_x,
        self._model.input_y          : batch_y,
        self._model.dropout_keep_prob: 1.
      }
    )
    
    return preds, accuracy

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
    
    return loss

  def _create_model(self,
                    max_seq_length,
                    num_classes,
                    vob_size,
                    embedding_size,
                    kernels,
                    filter_num,
                    l2_reg_lambda):
    self._model = _Model(
      max_seq_length=max_seq_length,
      num_classes=num_classes,
      vob_size=vob_size,
      embedding_size=embedding_size,
      kernels=kernels,
      filter_num=filter_num,
      l2_reg_lambda=l2_reg_lambda)
      
    self._global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(self._model.loss)
    self._train_optimizer = optimizer.apply_gradients(
      grads_and_vars, global_step=self._global_step)

  def load_model(self, model_path):
    ''' We only load the best model in {model_path}
    '''
    self._sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    #todo:
    # saver.restore(sess, tf.train.latest_checkpoint(os.path.pardir(model_path)))
    saver.restore(self._sess, model_path)
