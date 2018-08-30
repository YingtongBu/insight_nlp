#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import pprint
from googleapiclient.discovery import build
import csv
import os

def mkdir(path):
  folder_exist = os.path.exists(path)
  if not folder_exist:
      os.mkdir(path)
      print('Attention: CNN folder created')
  else:
      print('Folder existed, data will be stored later')

class Crawler:
  """ • Important optional parameters:
      ○ dateRestrict: Restricts results to URLs based on date.
      ○ exactTerms: Identifies a phrase that all documents in the search
      results must contain.
      ○ excludeTerms: Identifies a word or phrase that should not appear in any
       documents in the search results.
      ○ fileType
      ○ lowRange, highRange: Use lowRange and highRange to append an inclusive
      search range of lowRange...highRange  to the query.
      ○ lr: Restricts the search to documents written in a particular language
      (e.g., lr=lang_ja). 
      ○ Num: Number of search results to return.
      ○ orTerms: Provides additional search terms to check for in a document,
       where each document in the search results must contain at least one of
       the additional search terms.
      # Build a service object for interacting with the API. Visit
      # the Google APIs Console <http://code.google.com/apis/console>
      # to get an API key for your own application.
  """
  def __init__(self, key, num=10):
      self.key = key
      self.num = num

  def crawl(self, times, search_keywords, search_item):
      loop_num = eval(times) // 10 + 1
      service = build("customsearch", "v1",
                      developerKey="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")

      print('loop_num: ', loop_num)

      # create store folder path
      mkdir(search_item)
      print(search_keywords)
      for items in search_keywords:
          f = open(search_item + '/' + items + ".csv", "w+")
          for i in range(loop_num):
              res = service.cse().list(
                  q = items,
                  cx = self.key,
                  num = self.num,
                  high_range = (i + 1) * 10
              ).execute()
              #print(res)
              # with open("result.json","w") as newfile:
              #     newfile.write(str(res))
              #start saving
              myWriter = csv.writer(f, delimiter='\t')
              rlist = []
              for i in range(len(res['items'])):
                  s = res['items'][i]['snippet'].replace('\n', '')
                  link = res['items'][i]['displayLink']
                  final_result = s + link
                  rlist.append([final_result])
              myWriter.writerows(rlist)
          f.close()