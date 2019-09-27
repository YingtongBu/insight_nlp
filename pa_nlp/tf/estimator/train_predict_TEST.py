#!/usr/bin/env python3
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.tf import *
from pa_nlp.tf.estimator.dataset import DataReaderBase
from pa_nlp.tf.estimator.model import ModelBase
from pa_nlp.tf.estimator.predict import PredictorBase
from pa_nlp.tf.estimator.param import ParamBase
from pa_nlp.tf.estimator.train import TrainerBase

class Model(ModelBase):
  def construct(self):
    self.x = tf.placeholder(tf.float32, [None])
    self.y = tf.placeholder(tf.float32, [None])
    a = tf.get_variable("a", [], tf.float32, nlp_tf.init_rand(-1, 1))
    b = tf.get_variable("b", [], tf.float32, nlp_tf.init_rand(-1, 1))
    self.pred_y = a * self.x + b
    self.param = [a, b]
    self.loss = tf.losses.mean_squared_error(self.y, self.pred_y)

class DataReader(DataReaderBase):
  def parse_example(self, serialized_example):
    data_fields = {
      "x": tf.FixedLenFeature((), tf.float32, 0),
      "y": tf.FixedLenFeature((), tf.float32, 0),
    }
    parsed = tf.parse_single_example(serialized_example, data_fields)

    x = parsed["x"]
    y = parsed["y"]

    return x, y

class Trainer(TrainerBase):
  def _run_one_batch(self, epoch_id, batch_data):
    start_time = time.time()
    _, loss, batch, weights = self._sess.run(
      fetches=[
        self._train_op,
        self._model.loss,
        tf.train.get_or_create_global_step(),
        self._model.param,
      ],
      feed_dict={
        self._model.x: batch_data[0],
        self._model.y: batch_data[1],
        self._lr: self._param.lr,
      }
    )
    duration = time.time() - start_time
    # print("weights:", weights)
    print(f"batch[{batch}], loss={loss}, time={duration} sec.")

class Predictor(PredictorBase):
  def predict_sample(self, batch_data):
    return self._sess.run(
      fetches=self._model.pred_y,
      feed_dict={
        self._model.x: batch_data[0],
        self._model.y: batch_data[1],
      }
    )

  def calc_measure(self, pred_labels: list, correct_labels: list):
    loss = sum([(y1 - y2) * (y1 - y2)
                for y1, y2 in zip(pred_labels, correct_labels)])
    loss /= (len(pred_labels) + nlp.EPSILON)
    measure = {
      "MSE": loss
    }

    return measure

def gen_train_data(tf_file: str):
  class Serializer:
    def __call__(self, seg_samples: list):
      for sample in seg_samples:
        x, y = sample
        feature = {
          "x": nlp_tf.tf_feature_float(x),
          "y": nlp_tf.tf_feature_float(y),
        }
        example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
        )

        yield example_proto.SerializeToString()

  def get_file_record():
    a, b = 10, 5
    for _ in range(1000):
      x = random.random()
      y = a * x + b + 2 * (random.random() - random.random())
      # y = a * x + b
      yield [(x, y)]

  nlp_tf.write_tfrecord(get_file_record(), Serializer(), tf_file)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  train_file = "_debug.tfrecord"
  # gen_train_data(train_file)

  param = ParamBase("debug_model")
  param.train_file = train_file
  param.eval_files = [train_file]
  param.epoch_num = 20
  param.batch_size = 3
  param.evaluate_freq = 10
  param.lr = 0.1
  param.verify()

  trainer = Trainer(param, Model, Predictor, DataReader)
  trainer.train()
  predictor = Predictor(param, Model, DataReader)
  predictor.load_model(param.path_model)
  predictor.predict_dataset(train_file)

if __name__ == '__main__':
  main()

