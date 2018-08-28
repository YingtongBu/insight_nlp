#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import absolute_import
import keras
from keras import backend as K
from keras import regularizers
from keras import constraints
from keras import initializers
from keras.engine import Layer, InputSpec
import tensorflow as tf

def path_energy(y, x, U, bStart=None, bEnd=None, mask=None):
  x = add_boundary_energy(x, bStart, bEnd, mask)
  return path_energy0(y, x, U, mask)

def path_energy0(y, x, U, mask=None):
  nClasses = K.shape(x)[2]
  yOneHot = K.one_hot(y, nClasses)
  energy = K.sum(x * yOneHot, 2)
  energy = K.sum(energy, 1)
  yT = y[:, :-1]
  yTp1 = y[:, 1:]
  UFlat = K.reshape(U, [-1])
  flatIndices = yT * nClasses + yTp1
  UYTTp1 = K.gather(UFlat, flatIndices)
  if mask is not None:
    mask = K.cast(mask, K.floatx())
    yTMask = mask[:, :-1]
    yTp1Mask = mask[:, 1:]
    UYTTp1 *= yTMask * yTp1Mask
  energy += K.sum(UYTTp1, axis=1)
  return energy

def sparse_chain_crf_loss(y, x, U, bStart=None, bEnd=None, mask=None):
  x = add_boundary_energy(x, bStart, bEnd, mask)
  energy = path_energy0(y, x, U, mask)
  energy -= free_energy0(x, U, mask)
  return K.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, bStart=None, bEnd=None, mask=None):
  ySparse = K.argmax(y, -1)
  ySparse = K.cast(ySparse, 'int32')
  return sparse_chain_crf_loss(ySparse, x, U, bStart, bEnd, mask)

def add_boundary_energy(x, bStart=None, bEnd=None, mask=None):
  if mask is None:
    if bStart is not None:
      x = K.concatenate([x[:, :1, :] + bStart, x[:, 1:, :]], axis=1)
    if bEnd is not None:
      x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + bEnd], axis=1)
  else:
    mask = K.cast(mask, K.floatx())
    mask = K.expand_dims(mask, 2)
    x *= mask
    if bStart is not None:
      maskR = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]], 
                            axis=1)
      startMask = K.cast(K.greater(mask, maskR), K.floatx())
      x = x + startMask * bStart
    if bEnd is not None:
      maskL = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])], 
                            axis=1)
      endMask = K.cast(K.greater(mask, maskL), K.floatx())
      x = x + endMask * bEnd
  return x

def viterbi_decode(x, U, bStart=None, bEnd=None, mask=None):
  x = add_boundary_energy(x, bStart, bEnd, mask)

  alpha0 = x[:, 0, :]
  gamma0 = K.zeros_like(alpha0)
  initialStates = [gamma0, alpha0]
  _, gamma = _forward(x,
                      lambda B: [K.cast(K.argmax(B, axis=1),
                                 K.floatx()), K.max(B, axis=1)],
                      initialStates,
                      U,
                      mask)
  y = _backward(gamma, mask)
  return y

def free_energy(x, U, bStart=None, bEnd=None, mask=None):
  x = add_boundary_energy(x, bStart, bEnd, mask)
  return free_energy0(x, U, mask)

def free_energy0(x, U, mask=None):
  initialStates = [x[:, 0, :]]
  lastAlpha, _ = _forward(x,
                          lambda B: [K.logsumexp(B, axis=1)],
                          initialStates,
                          U,
                          mask)
  return lastAlpha[:, 0]

def _forward(x, reduceStep, initialStates, U, mask=None):
  def _forward_step(energyMatrixT, states):
    alphaTm1 = states[-1]
    newStates = reduceStep(K.expand_dims(alphaTm1, 2) + energyMatrixT)
    return newStates[0], newStates

  UShared = K.expand_dims(K.expand_dims(U, 0), 0)
  if mask is not None:
    mask = K.cast(mask, K.floatx())
    maskU = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
    UShared = UShared * maskU
  inputs = K.expand_dims(x[:, 1:, :], 2) + UShared
  inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1)
  last, values, _ = K.rnn(_forward_step, inputs, initialStates)
  return last, values

def batch_gather(reference, indices):
  refShape = K.shape(reference)
  batchSize = refShape[0]
  nClasses = refShape[1]
  flatIndices = K.arange(0, batchSize) * nClasses + K.flatten(indices)
  return K.gather(K.flatten(reference), flatIndices)


def _backward(gamma, mask):
  gamma = K.cast(gamma, 'int32')

  def _backward_step(gammaT, states):
    yTm1 = K.squeeze(states[0], 0)
    yT = batch_gather(gammaT, yTm1)
    return yT, [K.expand_dims(yT, 0)]

  initialStates = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]
  _, yRev, _ = K.rnn(_backward_step,
                     gamma,
                     initialStates,
                     go_backwards=True)
  y = K.reverse(yRev, 1)

  if mask is not None:
    mask = K.cast(mask, dtype='int32')
    y *= mask
    y += -(1 - mask)
  return y

class ChainCRF(Layer):
  def __init__(self, init='glorot_uniform',
               U_regularizer=None,
               b_start_regularizer=None,
               b_end_regularizer=None,
               U_constraint=None,
               b_start_constraint=None,
               b_end_constraint=None,
               weights=None,
               **kwargs):
    super(ChainCRF, self).__init__(**kwargs)
    self.init = initializers.get(init)
    self.U_regularizer = regularizers.get(U_regularizer)
    self.b_start_regularizer = regularizers.get(b_start_regularizer)
    self.b_end_regularizer = regularizers.get(b_end_regularizer)
    self.U_constraint = constraints.get(U_constraint)
    self.b_start_constraint = constraints.get(b_start_constraint)
    self.b_end_constraint = constraints.get(b_end_constraint)
    self.initial_weights = weights
    self.supports_masking = True
    self.uses_learning_phase = True
    self.input_spec = [InputSpec(ndim=3)]

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 3
    return (input_shape[0], input_shape[1], input_shape[2])

  def compute_mask(self, input, mask=None):
    if mask is not None:
      return K.any(mask, axis=1)
    return mask

  def _fetch_mask(self):
    mask = None
    if self._inbound_nodes:
      mask = self._inbound_nodes[0].input_masks[0]

    return mask

  def build(self, input_shape):
    assert len(input_shape) == 3
    n_classes = input_shape[2]
    n_steps = input_shape[1]
    assert n_steps is None or n_steps >= 2
    self.input_spec = [InputSpec(dtype=K.floatx(),
                                 shape=(None, n_steps, n_classes))]

    self.U = self.add_weight((n_classes, n_classes),
                             initializer=self.init,
                             name='U',
                             regularizer=self.U_regularizer,
                             constraint=self.U_constraint)

    self.b_start = self.add_weight((n_classes, ),
                                   initializer='zero',
                                   name='b_start',
                                   regularizer=self.b_start_regularizer,
                                   constraint=self.b_start_constraint)

    self.b_end = self.add_weight((n_classes, ),
                                 initializer='zero',
                                 name='b_end',
                                 regularizer=self.b_end_regularizer,
                                 constraint=self.b_end_constraint) 

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
   
    self.built = True

  def call(self, x, mask=None):
    y_pred = viterbi_decode(x, self.U, self.b_start, self.b_end, mask)
    nb_classes = self.input_spec[0].shape[2]
    y_pred_one_hot = K.one_hot(y_pred, nb_classes)
    return K.in_train_phase(x, y_pred_one_hot)

  def loss(self, y_true, y_pred):
    mask = self._fetch_mask()
    return chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, 
                          mask)

  def sparse_loss(self, y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_true = K.squeeze(y_true, 2)
    mask = self._fetch_mask()
    return sparse_chain_crf_loss(y_true, y_pred, self.U, self.b_start, 
                                 self.b_end, mask)

  def get_config(self):
    config = {
        'init': initializers.serialize(self.init),
        'U_regularizer': regularizers.serialize(self.U_regularizer),
        'b_start_regularizer': regularizers.serialize(self.
                                                      b_start_regularizer),
        'b_end_regularizer': regularizers.serialize(self.b_end_regularizer),
        'U_constraint': constraints.serialize(self.U_constraint),
        'b_start_constraint': constraints.serialize(self.
                                                    b_start_constraint),
        'b_end_constraint': constraints.serialize(self.b_end_constraint)
    }
    base_config = super(ChainCRF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def create_custom_objects():
  instanceHolder = {'instance': None}

  class ChainCRFClassWrapper(ChainCRF):
    def __init__(self, *args, **kwargs):
      instanceHolder['instance'] = self
      super(ChainCRFClassWrapper, self).__init__(*args, **kwargs)

  def loss(*args):
    method = getattr(instanceHolder['instance'], 'loss')
    return method(*args)

  def sparse_loss(*args):
    method = getattr(instanceHolder['instance'], 'sparse_loss')
    return method(*args)

  return {'ChainCRF': ChainCRFClassWrapper, 
          'ChainCRFClassWrapper': ChainCRFClassWrapper, 'loss': loss, 
          'sparse_loss': sparse_loss}