#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifiers.CNN.CNNTextClassifier import CNNTextClassifier

if __name__ == '__main__':
  cnn_project = \
    CNNTextClassifier(
                  train_file='Insight_NLP/Classifiers/CNN/Sample.Train.data',
                  validation_file='Insight_NLP/Classifiers/CNN/Sample.Test.data',
                  num_classes=44, GPU=3)
  cnn_project.train()
  cnn_project.predict('Insight_NLP/Classifiers/CNN/Sample.Test.data')
