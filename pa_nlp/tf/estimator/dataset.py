#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.tf import *
from pa_nlp.tf.estimator.param import Param

# Buffer size for reading records from a TFRecord file. Each training file is
# 7.2 MB, so 8 MB allows an entire file to be kept in memory.
_READ_RECORD_BUFFER = 64 * 1000 * 1000

class DataReader(abc.ABC):
  def __init__(self,
               tfrecord_file: str, param: Param,
               training: bool, debug: bool=False):
    self._tf_file = tfrecord_file
    self._param = param
    self._shuffle = training
    self._debug = debug

  @abc.abstractmethod
  def parse_example(self, serialized_example):
    '''return parsed tensors
    '''
    pass

  def _batch_examples(self, dataset):
    '''you could define how to group examples'''

    dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
        map_func=lambda example: self.parse_example(example),
        batch_size=self._param.batch_size,
        drop_remainder=False,
      )
    )

    return dataset

  def _read_and_batch_from_files(self):
    dataset = tf.data.Dataset.list_files(self._tf_file, shuffle=self._shuffle)

    dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
        lambda tf_file: tf.data.TFRecordDataset(
          tf_file, buffer_size=_READ_RECORD_BUFFER
        ),
        sloppy=self._shuffle, cycle_length=4
      )
    )

    dataset = self._batch_examples(dataset)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    data_iter = dataset.make_initializable_iterator()
    sample = data_iter.get_next()

    return data_iter.initializer, sample

  def get_batch_data(self):
    graph = tf.Graph()
    with graph.as_default():
      initializer, sample = self._read_and_batch_from_files()
    sess = tf.Session(graph=graph)

    for epoch_id in range(self._param.epoch_num):
      sess.run(initializer)

      while True:
        try:
          start_time = time.time()
          batch_data = sess.run(sample)
          duration = time.time() - start_time
          if self._debug:
            print(f"batch fetching time: {duration:.4f} seconds.")

          yield epoch_id, batch_data

        except tf.errors.OutOfRangeError:
          break

