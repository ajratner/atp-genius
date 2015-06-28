import lxml
from bs4 import BeautifulSoup
from models import Player, Match
from datetime import datetime
import re
from util import *
from sqlalchemy import or_, and_
import time


# general parse modules
def date_parse(s):
  try:
    d = re.search(r'(\d{1,2}).([a-z]+).(\d{4})', s, flags=re.I)
    return datetime.strptime(d.group(1)+'-'+d.group(2)+'-'+d.group(3), '%d-%b-%Y').date()
  except:
    return None


# player bio parse sub-modules
def player_name_parse(s):
  s = s.strip()
  c = re.search(r'\[([A-Z]+)\]', s)
  country = c.group(1) if c else None
  name = re.sub(r'(\[|\().*?(\]|\))', '', s).strip()
  return name, country

def player_twitter_parse(s):
  a = s.find('a')
  return a.text.strip() if a else None

def player_plays_parse(s):
  return re.sub(r'Plays:\s+', '', s).strip()

def player_atp_parse(s):
  d = re.search(r'\d+', s)
  if d:
    return int(d.group(0))
  else:
    return None

def player_peak_atp_parse(s):
  s = s.strip()
  d = date_parse(s)
  r = int(re.search(r'\d+', re.sub(r'\(.*?\)', '', s)).group(0))
  return r, d


# match parse sub-modules
ROUND_MAP = {
  'f':0,
  'sf':1,
  'qf':2,
  'r16':3,
  'r32':4,
  'r64':5,
  'r128':6,
  'q3':7,
  'q2':8,
  'q1':9,
  'rr':5
}
def match_round_parse(s):
  key = s.lower().strip()
  if ROUND_MAP.has_key(key):
    return ROUND_MAP[key]
  else:
    return 5

def match_players_parse(s, p_name, p_id):
  spans = s.find_all('span')
  names = []
  urls = []
  ids = []
  for ni in [1,-1,-2]:
    if len(names) == 2:
      break
    s = spans[ni]
    if re.match(r'\[.*?\]', s.text.strip()):
      continue
    elif (s.attrs.has_key('style') and re.search(r'bold',s['style'])) or s.find('bold'):
      names.append(p_name)
      ids.append(p_id)
    else:
      names.append(spans[ni].text.strip())
      urls.append(spans[ni].find('a')['href'])
      ids.append(None)
  return names[0], ids[0], names[1], ids[1], urls

def match_rank_parse(s):
  s = s.strip()
  if s == 'UNR' or s == '':
    return None
  else:
    return int(s)


URL_PATTERN = r'(https?://)?(www\.)?(tennisabstract\.com/)?(cgi-bin/player\.cgi\?p=[A-Za-z]+)'
URL_ROOT = 'http://www.tennisabstract.com/'
PATH_GROUP = 4
PATH_APPEND = '&f=ACareerqqo1'
def unique_urls(urls):
  out = []
  for url in urls:
    mu = re.match(URL_PATTERN, url)
    if mu:
      out.append(URL_ROOT + mu.group(PATH_GROUP) + PATH_APPEND)
  return list(set(out))


# function to parse a single tennisabstract player page into match instances + player metadata
def parse_player_page(source, url):

  print 'soupifying...'
  t0 = time.time()

  session = mysql_session(ensure_created=True)
  soup = BeautifulSoup(source, "lxml")

  t1 = time.time(); print 'Done in %s' % (t1-t0,); t0=t1
  print 'scraping player...'

  # scrape player bio and create player object
  p = Player()
  p.url = url
  bio = soup.find(id='bio')

  # NOTE: if no bio, skip!
  if bio.find('table') is None:
    return []

  cells = bio.find('table').find_all('td')
  p.name, p.country = player_name_parse(cells[0].text)
  p.twitter = player_twitter_parse(cells[1])
  p.dob = date_parse(cells[2].text)
  p.plays = player_plays_parse(cells[3].text)
  p.atp_rank = player_atp_parse(cells[4].text)
  try:
    p.peak_atp_rank, p.peak_date = player_peak_atp_parse(cells[5].text)
  except AttributeError as e:
    pass

  t1 = time.time(); print 'Done in %s' % (t1-t0,); t0=t1
  print 'saving player...'

  # commit player to database and get id
  # >> if player page already exists, replace!
  if len(session.query(Player).filter(Player.name == p.name).all()) > 0:
    session.query(Player).filter(Player.name == p.name).delete()
  session.add(p)
  session.commit()

  # delete matches first
  session.query(Match).filter(or_(Match.p1_name == p.name, Match.p2_name == p.name)).delete()

  t1 = time.time(); print 'Done in %s' % (t1-t0,); t0=t1
  print 'scraping matching...'

  # scrape matches list and create corresponding list of match objects
  # >> also track player name urls
  matches = soup.find(id='matches').find_all('tr')[2:]
  urls = []
  ms = []
  for match in matches:
    cells = match.find_all('td')
    m = Match()
    m.date = date_parse(cells[0].text)
    m.tournament = cells[1].text
    m.surface = cells[2].text
    m.tournament_round = match_round_parse(cells[3].text)
    m.p1_rank = match_rank_parse(cells[4].text)
    m.p2_rank = match_rank_parse(cells[5].text)
    m.p1_name, m.p1_id, m.p2_name, m.p2_id,m_urls = match_players_parse(cells[6], p.name, p.id)
    urls += m_urls

    # check for uniqueness then add to db
    #if session.query(Match).filter(and_(or_(and_(Match.p1_name == m.p1_name, Match.p2_name == m.p2_name), and_(Match.p1_name == m.p2_name, Match.p2_name == m.p1_name)), Match.date == m.date)).count() == 0:
    #  session.add(m)
    session.add(m)
    ms.append(m)

  t1 = time.time(); print 'Done in %s' % (t1-t0,); t0=t1
  print 'committing...'

  # keep only unique urls (to other player pages for crawler)
  urls = unique_urls(urls)

  # >> return urls for crawling
  session.commit()
  session.close()
  return urls


if __name__ == '__main__':
  with open('test.html', 'rb') as f:
    source = f.read()
  urls = parse_player_page(source)
