#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build
import csv
import os

class Searcher:
  def __init__(self, key, page_num_max=10):
    self.key = key
    # the search item for each page
    self.page_num = page_num_max

  def mkdir(self, path):
    folder_exist = os.path.exists(path)
    if not folder_exist:
      os.mkdir(path)
      print('Attention: folder created')
    else:
      print('Folder existed, data will be stored later')

  def search(self, item_num, search_keywords):
    loop_num = eval(item_num) // 10 + 1
    service = build("customsearch", "v1",
                    developerKey="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")
    print(f'loop_num: {loop_num}')
    print(search_keywords)
    res_dict = {}
    keyword_res_list = []
    for item in search_keywords:
      for i in range(loop_num):
        res = service.cse().list(
          q = item,
          cx = self.key,
          num = self.page_num,
          high_range = (i + 1) * 10
        ).execute()
        # print(res)
        # with open("result.json","w") as newfile:
        #   newfile.write(str(res))
        # start saving
        keyword_res_list = []
        for i in range(len(res['items'])):
          s = res['items'][i]['snippet'].replace('\n', '')
          link = res['items'][i]['displayLink']
          final_result = s + link
          keyword_res_list.append(final_result)
      res_dict[item] = keyword_res_list
    return res_dict
