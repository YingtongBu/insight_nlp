#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build

class SearchEngine(object):
  def __init__(self, key='005808576341306023160:yojc6z7o63u'):
    self.key = key

  def search(self, keywords:str, page_num=2):
    """
    :param keywords:
    :param page_num:
    :return: [{"link": "www.sina.com.cn", "snippet": "today...."}, {...}]
    """
    service = build("customsearch", "v1",
                    developerKey="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")
    result = []
    for i in range(page_num):
      # q: search item
      # cx: search engine key
      # num: search items returned in each search
      # highRange: the last item number of search result, used for paging
      res = service.cse().list(
        q=keywords,
        cx=self.key,
        num=10,
        highRange=(i + 1) * 10
      ).execute()
      for i in range(len(res['items'])):
        snippet = res['items'][i]['snippet'].replace('\n', '')
        link = res['items'][i]['displayLink']
        single_result = {
          "link": link,
          "snippet": snippet,
        }
        result.append(single_result)
    return result
