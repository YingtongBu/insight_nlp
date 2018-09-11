#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Common import *
from Chinese import *

if __name__ == "__main__":
  text = "中国人民共和国今天成立了。"
  print(segment_sentence(text, False))
  print(segment_sentence(text, True))
