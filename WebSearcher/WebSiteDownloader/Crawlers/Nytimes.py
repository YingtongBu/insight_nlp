#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
"""
This modules implements the scraperNytimes which is used to scrap news from
The New York Times
"""
import requests
import bs4
from Crawlers.Clean import clean_text
import datetime as dt
import logging
from NLP import news_title_filter, news_summary_filter

logger = logging.getLogger("NewsScraper.crawlers.nytimes")

def scraper_nytimes():
  list_news = list()
  url = 'https://www.nytimes.com/section/world'
  try:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    for tag in soup.find_all('div', {'class': 'story-meta'}):
      title = clean_text(tag.find('h2').get_text().lower())
      if not news_title_filter(title):
        logger.debug('This news is not in interest based on title: {}'.format(
          title))
        continue
      this_news = dict()
      this_news['title'] = title
      this_news['url'] = tag.parent.parent.find('a').get('href')
      this_news['summary'] = clean_text(tag.find('p', {'class':
                                                        'summary'}).get_text())
      if not news_summary_filter(this_news['summary']):
        logger.debug('This news is not in interest based on summary: {}'.format(
          this_news['summary']))
        continue
      try:
        art_response = requests.get(this_news['url'])
        art_soup = bs4.BeautifulSoup(art_response.content, 'lxml')
        art_date = art_soup.find('time').get('datetime')
        this_news['newsDate'] = art_date
        # article = art_soup.find('div', {'class': 'StoryBodyCompanionColumn'})
        texts = [a.get_text() for a in art_soup.find_all('p')]
        text = ''.join(texts)
        this_news['article'] = clean_text(text)
      except:
        logger.exception("Error in scraping {}".format(this_news['url']))
      list_news.append(this_news)
  except:
    logger.exception("Error in scraping {}".format(url))
  return list_news

def scraper_nytimesdaily(date):
  list_news = list()
  url = 'https://www.nytimes.com/issue/todayspaper/' + \
        dt.datetime.strftime(date, '%Y/%m/%d') + '/todays-new-york-times'
  try:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    STOPWORD = ["Corrections:", "Bulletin", "Review:"]
    for tag in soup.find_all('h2', {'class': 'headline'}):
      if not any(word in tag.get_text() for word in STOPWORD):
        this_news = dict()
        tag_title = tag.find('a')
        title = clean_text(tag_title.get_text())
        if not news_title_filter(title):
          continue
        this_news['title'] = title
        this_news['url'] = tag_title.get('href')
        if tag.parent.find('p', {'class': 'summary'}):
          this_news['summary'] = clean_text(
            tag.parent.find('p', {'class': 'summary'}).get_text())
          if not news_summary_filter(this_news['summary']):
            continue
        try:
          art_response = requests.get(this_news['url'])
          art_soup = bs4.BeautifulSoup(art_response.content, 'lxml')
          art_date = art_soup.find('time').get('datetime')
          this_news['newsDate'] = art_date
          texts = [a.get_text() for a in art_soup.find_all('p')]
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