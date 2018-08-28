#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build
import csv
import os

def _mkdir(path):
  folder_exist = os.path.exists(path)
  if not folder_exist:
    os.mkdir(path)
    print('Attention: new folder created')
  else:
    print('Folder existed, continue adding new data')

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
                    developerKey="005808576341306023160:yojc6z7o63u")
    print('loop_num: ', loop_num)

    # create store folder path
    _mkdir(search_item)
    for items in search_keywords:
      newFile = open(search_item + '/' + items + ".csv", "w+")
      for iter in range(loop_num):
        print('page' + str(iter))
        res = service.cse().list(
          q = items,
          cx = self.key,
          num = self.num,
          low_range = (iter + 1) * 10
        ).execute()

        #start saving
        myWriter = csv.writer(newFile, delimiter = '\t')
        rlist = []
        for iter in range(len(res['items'])):
          snippet = res['items'][iter]['snippet'].replace('\n', '')
          link =res['item'][iter]['displayLink']
          final_result = snippet + link
          rlist.append([final_result])
        myWriter.writerows(rlist)
      newFile.close()