#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifier.CNN.CNNTextClassifier import CNNTextClassifier

#code review: remove Sample.Origin.data.

if __name__ == '__main__':
  #code review: put it in /tmp/, then test again!
  cnn_project = CNNTextClassifier(GPU=3)
  cnn_project.train()
  #code review: how to use cnn_project.predict()?
