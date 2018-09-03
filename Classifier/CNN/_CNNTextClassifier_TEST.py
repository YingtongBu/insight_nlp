#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifier.CNN.CNNTextClassifier import CNNTextClassifier
# from Insight_NLP.Common import get_module_path

if __name__ == '__main__':
  # get_module_path('Insight_NLP.Classifier.CNN.CNNTextClassifier')
  cnn_project = CNNTextClassifier(GPU=3)
  cnn_project.train()