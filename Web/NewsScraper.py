#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import requests
from newspaper import Article
import newspaper
from bs4 import BeautifulSoup

class NewsScraper(object):
  def __init__(self, url, language='zh'):
    self.url = url
    self._article = Article(self.url, language=language)
    self._article.download()
    self._article.parse()
    self._article.nlp()
    self._publish_time = self._get_newstime()

  def _get_newstime(self):
    news_content = requests.get(self.url)
    news_content.encoding = 'utf-8'
    news_content = news_content.text
    soup = BeautifulSoup(news_content, 'html.parser')
    news_time = soup.find(class_="time")
    news_time = news_time.text
    return news_time

  def get_result(self):
    _news_title = self._article.title
    _news_summary = self._article.summary
    _news_datetime = self._publish_time
    _news_result = {
      'date': _news_datetime,
      'title': _news_title,
      'summary': _news_summary,
      'url': self.url,
      "text": self._article.text
    }
    return _news_result
