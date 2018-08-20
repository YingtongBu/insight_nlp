#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build
import csv
import os

def mkdir(path):
  folder_exist = os.path.exists(path)
  if not folder_exist:
    os.mkdir(path)
    print('Attention: new folder created')
  else:
    print('Folder existed, data will be stored later')

class Crawler:
  """ • Important optional parameters:
      ○ dateRestrict: Restricts results to URLs based on date.
      ○ exactTerms: Identifies a phrase that all documents in the search results must contain.
      ○ excludeTerms: Identifies a word or phrase that should not appear in any documents in the search results.
      ○ fileType
      ○ lowRange, highRange: Use lowRange and highRange to append an inclusive search range of lowRange...highRange  to the query.
      ○ lr: Restricts the search to documents written in a particular language (e.g., lr=lang_ja). 
      ○ Num: Number of search results to return.
      ○ orTerms: Provides additional search terms to check for in a document, where each document in the search results must contain at least one of the additional search terms.
      # Build a service object for interacting with the API. Visit
      # the Google APIs Console <http://code.google.com/apis/console>
      # to get an API key for your own application.
  """

  def __init__(self, key, num = 10):
    self.key = key
    self.num = num

  def crawl(self, times, search_keywords, search_item):
    loop_num = eval(times) // 10 + 1
    service = build("customsearch", "v1",
                    developerKey="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")
    print('loop_num: ', loop_num)

    # create store folder path
    os.mkdir(search_item)
    for items in search_keywords:
      f = open(search_item + '/' + items + ".csv", "w+")
      for iter in range(loop_num):
        print('page' + str(iter))
        res = service.cse().list(
          q = items,
          cx = self.key,
          num = self.num,
          lowRange = (iter + 1) * 10
        ).execute()

        #start saving
        myWriter = csv.writer(f, delimiter = ',')
        rlist = []
        for iter in range(len(res['items'])):
          snippet = res['items'][iter]['snippet'].replace('\n', '')
          rlist.append([snippet])
        myWriter.writerows(rlist)
      f.close()