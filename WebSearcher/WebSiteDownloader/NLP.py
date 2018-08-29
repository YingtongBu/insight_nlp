#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import gensim.summarization

def news_title_filter(title):
  return True

def news_summary_filter(summary):
  return True

class Processor:
  def __init__(self, news):
    self.list_news = news

  def summarize(self):
    for news in self.list_news:
      news['summaryGensim'] = gensim.summarization.summarize(news['article'],
                                                             word_count=100)

  @property
  def news_list(self):
    return self.list_news

if __name__ == '__main__':
    print(news_summary_filter('This is a fake news.'))
    print(news_title_filter('Today is a good day.'))