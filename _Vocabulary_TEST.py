#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Vocabulary import *

if __name__ == "__main__":
  data = [
    ["welcome", "to", "ping'an"],
    ["欢迎", "来到", "平安"],
    ["欢迎", "来到", "平安"],
    ["欢", "迎", "来", "到", "平安"],
  ]

  vob = Vocabulary()
  vob.create_vob_from_data(data)
  vob.add_word("<empty>")
  vob.add_word("<oov>")
  vob.save_to_file("vob.data")
  vob.load_from_file("vob.data")
 
  for tokens in data:
    print(f"{tokens}:",
          vob.convert_to_word_ids(tokens,
                                  remove_OOV=False, mark_OOV="<oov>",
                                  output_length=10, mark_empty="<empty>"))
  
  tokens = ["欢", "迎", "来", "到", "平安吧"]
  print(f"{tokens}:",
        vob.convert_to_word_ids(tokens,
                                remove_OOV=False, mark_OOV="<oov>",
                                output_length=10, mark_empty="<empty>"))
  
  print(f"#vob: {vob.size()}")
