#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait as web_driver_wait
import datetime as dt
from Crawlers.Nytimes import scraper_nytimes, scraper_nytimesdaily
from Crawlers.Reuters import scraper_reuters
from Crawlers.Wallstreetjournal import scraper_wsj
import logging

logger = logging.getLogger("NewsScraper.crawlers")

class Scraper:
  """Base class for scrapers.
  """
  def __init__(self, phantomjs_path, username, password):
    logger.info("Initialize crawler")
    # store each news as a dict(title, url, newsDate, article, summary, time,
    #  location, people, event)
    self.list_news = list()

    # WSJ driver setting
    try:
      self.driver = webdriver.PhantomJS(executable_path=phantomjs_path)
      loginUrl = \
        'https://accounts.wsj.com/login?target=https%3A%2F%2Fwww.wsj.com'
      self.driver.get(loginUrl)
      username_field = self.driver.find_element_by_id("username")
      password_field = self.driver.find_element_by_id("password")
      login_button = \
        self.driver.find_element_by_class_name("basic-login-submit")
      username_field.clear()
      password_field.clear()
      username_field.send_keys(username)
      password_field.send_keys(password)
      login_button.send_keys(Keys.ENTER)
      self.driver.wait = web_driver_wait(self.driver, 30)
      self.driver.wait.until(lambda driver:
                             driver.current_url == 'https://www.wsj.com/')
    except:
      logger.exception("Error in initializing WSJ crawler")

  def scraper_reuters(self):
    logger.info("Start scraping Reuters")
    self.list_news = scraper_reuters()
    logger.info('{} news scraped from Reuters'.format(len(self.list_news)))

  def scraper_nytimes(self):
    logger.info("Start scraping NYTimes on World News page")
    self.list_news = scraper_nytimes()
    logger.info('{} news scraped from NYTimes'.format(len(self.list_news)))

  def scraper_nytimesdaily(self, date):
    logger.info("Start scraping from NYTimes for " +
                dt.datetime.strftime(date, '%Y-%m-%d'))
    self.list_news = scraper_nytimesdaily(date)
    logger.info('{} news scraped from NYTimes for '.format(
      len(self.list_news)) + dt.datetime.strftime(date, '%Y-%m-%d'))

  def scraper_wsj(self, date):
    logger.info("Start scraping Wall Street Journal for " +
                dt.datetime.strftime(date, '%Y-%m-%d'))
    self.list_news = scraper_wsj(self.driver, date)
    logger.info('{} news scraped from Wall Street Journal for '.format(
      len(self.list_news)) + dt.datetime.strftime(date, '%Y-%m-%d'))

  def quit(self):
    self.driver.quit()

  @property
  def news_list(self):
    return self.list_news

if __name__ == '__main__':
    pass