#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build

class SearchEngine(object):
  def __init__(self, key):
    self.key = key

  def search(self, search_keyword, item_num=20):
    loop_num = item_num // 10 + 1
    service = build("customsearch", "v1",
                    developerKey="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")
    keyword_res_list = []
    for i in range(loop_num):
      # q: search item
      # cx: search engine key
      # num: search items returned in each search
      # highRange: the last item number of search result, used for paging
      res = service.cse().list(
        q=search_keyword,
        cx=self.key,
        num=10,
        highRange=(i + 1) * 10
      ).execute()
      for i in range(len(res['items'])):
        snippet = res['items'][i]['snippet'].replace('\n', '')
        link = res['items'][i]['displayLink']
        single_result = snippet + link
        keyword_res_list.append(single_result)
    result = [search_keyword, keyword_res_list]
    return result
