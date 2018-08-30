#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import os
from Insight_NLP.Classifier.CNN.Train import pre_process
from Insight_NLP.Classifier.CNN.Train import train

class CNNTextClassifier(object):
  def __init__(self, GPU=3):
    self.GPU = str(GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU

  def train_CNN(self,
    dev_sample_percentage=0.3,train_data='CNN/TotalEvents.data',
    num_classes=44, embedding_dim=128, kernel_sizes='1,1,1,2,3',
    num_kernels=128, dropout_keep_prob=0.5, l2_reg_lambda=0.0, num_words=64,
    batch_size=1024, num_epochs=2, evaluate_every=100, language_type="ENG"):

    x_train, y_train, vocab_processor, x_dev, y_dev, origin_text_dev = \
      pre_process(dev_sample_percentage, train_data, num_classes,
                 embedding_dim, num_words, language_type)
    train(x_train, y_train, vocab_processor, x_dev, y_dev, origin_text_dev,
          embedding_dim, kernel_sizes, num_kernels, dropout_keep_prob,
          l2_reg_lambda, batch_size, num_epochs, evaluate_every, language_type)

  def predict(self):
    pass
