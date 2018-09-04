#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifier.CNN.CNNTextClassifier import CNNTextClassifiergi

if __name__ == '__main__':
  cnn_project = CNNTextClassifier(GPU=3)
  cnn_project.train()