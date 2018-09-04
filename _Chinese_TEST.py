#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Common import *
from Insight_NLP.Chinese import *

if __name__ == "__main__":
  text = "中国人民共和国今天成立了。"
  print(segment_sentence(text, False))
  print(segment_sentence(text, True))
