import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import librosa
import common as nlp

class DataGraphMFCC:
  window_size: int=800
  stride: int=480

  def __init__(self, dct_coef_count: int=13):
   '''
   By default settings, the returned dimension would be [100, 13], [100, 13],
   [100, 13]
   '''
   self._graph = tf.Graph()
   with self._graph.as_default():
     self._wav_file_ts = tf.placeholder(tf.string, [], name='wav_filename')
     self._frame_num = tf.placeholder(tf.int32, [])
     wav_loader = io_ops.read_file(self._wav_file_ts)
     wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
     audio_clamp = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
     spectrogram = contrib_audio.audio_spectrogram(
       audio_clamp,
       window_size=self.window_size,
       stride=self.stride,
       magnitude_squared=True)

     feat_ts = contrib_audio.mfcc(
       spectrogram=spectrogram,
       sample_rate=wav_decoder.sample_rate,
       dct_coefficient_count=dct_coef_count,
     )
     self._feat_ts = feat_ts[0]
     self._real_length = tf.shape(self._feat_ts)[0]

     self._expanded_feat_ts = tf.pad(
       self._feat_ts,
       [[0, self._frame_num - self._real_length], [0, 0]],
     )

   self._sess = tf.Session(graph=self._graph)

  def run(self, wav_file: str, target_frame_num: int=-1):
    assert nlp.get_file_extension(wav_file) == "wav"
    if target_frame_num <= 0:
      mfcc, real_length = self._sess.run(
        fetches=[self._feat_ts, self._real_length],
        feed_dict={
          self._wav_file_ts: wav_file
        }
      )
    else:
      mfcc, real_length = self._sess.run(
        fetches=[self._expanded_feat_ts, self._real_length],
        feed_dict={
          self._wav_file_ts: wav_file,
          self._frame_num: target_frame_num
        }
      )

    mfcc_delta1 = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta1)

    return [mfcc, mfcc_delta1, mfcc_delta2, real_length]
