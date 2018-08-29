#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
"""
This modules implements the scraper_reuters which is used to scrap news from
Reuters
"""
import requests
import bs4
import datetime as dt
from Crawlers.Clean import clean_text
import logging
from NLP import news_title_filter, news_summary_filter

logger = logging.getLogger("NewsScraper.crawlers.reuters")

def scraper_reuters():
  list_news = list()
  url = 'https://www.reuters.com/news/archive/worldNews'
  try:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    for tag in soup.find_all('div', {'class': 'story-content'}):
      title = clean_text(tag.find('a').find('h3').get_text())
      if not news_title_filter(title):
        logger.debug('This news is not in interest based on title: {}'.format(
          title))
        continue
      this_news = dict()
      this_news['title'] = title
      this_news['url'] = 'https://www.reuters.com' + tag.find('a').get('href')
      this_news['summary'] = clean_text(tag.find('p').get_text())
      if not news_summary_filter(this_news['summary']):
        logger.debug('This news is not in interest based on summary: {}'.format(
          this_news['summary']))
        continue
      try:
        art_response = requests.get(this_news['url'])
        art_soup = bs4.BeautifulSoup(art_response.content, 'lxml')
        art_date = art_soup.find('div', {'class': 'date_V9eGk'}).get_text()
        this_news['newsDate'] = str(
          dt.datetime.strptime(art_date[:art_date.rfind(' /')],
                               '%B %d, %Y / %I:%M %p'))
        article = art_soup.find('div', {'class': 'body_1gnLA'})
        texts = [a.get_text() for a in article.find_all('p')]
        text = ''.join(texts)
        this_news['article'] = clean_text(text)
      except:
        logger.exception("Error in scraping {}".format(this_news['url']))
      list_news.append(this_news)
  except:
    logger.exception("Error in scraping {}".format(url))
  return list_news

if __name__ == '__main__':
    pass