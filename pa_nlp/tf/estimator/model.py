#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.tf.estimator.param import Param
from pa_nlp.tf import *

class ModelBase(abc.ABC):
  def __init__(self, param: Param, training: bool):
    self.training = training
    self.param = param
    self.loss = None

    self.construct()

  @abc.abstractmethod
  def construct(self):
    pass



