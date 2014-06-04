from selenium import webdriver
from util import *
import lxml
from bs4 import BeautifulSoup
from models import Tournament


ATP_CALENDAR_URL = 'http://tennisabstract.com/reports/atp_calendar.html'


if __name__ == '__main__':

  # get page source
  driver = webdriver.PhantomJS()
  driver.get(ATP_CALENDAR_URL)
  source = driver.page_source

  # parse tournaments and add to database
  session = mysql_session(ensure_created=True)
  soup = BeautifulSoup(source, "lxml")
  rows = soup.find(id='reportable').find('tbody').find_all('tr')
  for row in rows:
    cells = row.find_all('td')
    t = Tournament()
    t.name = cells[1].text
    t.surface = cells[2].text
    t.level = cells[3].text
    session.add(t)
  session.commit()

  print 'Done.'
