#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from googleapiclient.discovery import build

class SearchEngine(object):
  def __init__(self, custom_search_key='005808576341306023160:yojc6z7o63u',
               developer_key="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps"):
    self.customSearchkey = custom_search_key
    self.developerKey = developer_key


  def search(self, keywords:str, item_nums=100):
    """
    :param keywords:
    :param item_num:
    :return: [{"link": "www.sina.com.cn", "snippet": "today...."}, {...}]
    """
    service = build("customsearch", "v1",developerKey=self.developerKey)
    result = []
    page_nums = item_nums // 10
    if item_nums % 10 > 0:
      page_nums = page_nums + 1
    for i in range(page_nums):
      # q: search item
      # cx: search engine key
      # num: search items returned in each search
      # highRange: the last item number of search result, used for paging
      res = service.cse().list(
        q=keywords,
        cx=self.customSearchkey,
        num=10,
        highRange=(i+1)*10
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
