from Common import *

import jieba
import jieba.posseg as pseg

def convertFullToHalf(s):
  '''全角转半角'''
  n = []
  for char in s:
    num = ord(char)
    if num == 12288:
      num = 32
    elif num == 12290:
      num = 46
    elif 65281 <= num <= 65374:
      num -= 65248
    num = chr(num)
    n.append(num)
  return ''.join(n)

def segmentSentence(text, posTagging=False):
  if posTagging:
    words, tags = [], []
    for token in pseg.cut(text):
      words.append(token.word)
      tags.append(token.flag)
    return words, tags
  else:
    return list(jieba.cut(text, cut_all=False))

if __name__ == "__main__":
  text = "中国人民共和国今天成立了。"
  print(segmentSentence(text, False))
  print(segmentSentence(text, True))

