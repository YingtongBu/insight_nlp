#!/usr/bin/env python
#coding: utf8

#Author: hong xu(hong.xu55@pactera.com)

import Insight_NLP.Translation.Translate as Trans

if __name__ == '__main__':
  src = "Beijing"
  tran = Trans.translate_sentence(src)
  print(f"{src} --> {tran}")
  
  src = "你好，平安银行"
  tran = Trans.translate_sentence(src, "en")
  print(f"{src} --> {tran}")
  
