from scrape import *
import lxml
from bs4 import BeautifulSoup
from models import Player, Match
from datetime import datetime
import re


# general parse modules
def date_parse(s):
  d = re.search(r'(\d{1,2}).([a-z]+).(\d{4})', s, flags=re.I)
  return datetime.strptime(d.group(1)+'-'+d.group(2)+'-'+d.group(3), '%d-%b-%Y')


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
  return int(re.search(r'\d+', s).group(0))

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
  'q1':9
}
def match_round_parse(s):
  return ROUND_MAP[s.lower().strip()]

def match_players_parse(s, p_name, p_id):
  spans = s.find_all('span')
  names = []
  ids = []
  for ni in [1,-1]:
    s = spans[ni]
    if (s.attrs.has_key('style') and re.search(r'bold',s['style'])) or s.find('bold'):
      names.append(p_name)
      ids.append(p_id)
    else:
      names.append(spans[ni].text.strip())
      ids.append(None)
  return names[0], ids[0], names[1], ids[1]

def match_rank_parse(s):
  s = s.strip()
  if s == 'UNR':
    return None
  else:
    return int(s)


# function to parse a single tennisabstract player page into match instances + player metadata
def parse_player_page(source):
  soup = BeautifulSoup(source, "lxml")

  # scrape player bio and create player object
  p = Player()
  bio = soup.find(id='bio')
  cells = bio.find('table').find_all('td')
  p.name, p.country = player_name_parse(cells[0].text)
  p.twitter = player_twitter_parse(cells[1])
  p.dob = date_parse(cells[2].text)
  p.plays = player_plays_parse(cells[3].text)
  p.atp_rank = player_atp_parse(cells[4].text)
  p.peak_atp_rank, p.peak_date = player_peak_atp_parse(cells[5].text)

  # commit player to database and get id
  # NOTE: TO-DO
  p_id = -1

  # scrape matches list and create corresponding list of match objects
  matches = soup.find(id='matches').find_all('tr')[2:]
  ms = []
  for match in matches:
    cells = match.find_all('td')
    m = Match()
    m.date = date_parse(cells[0].text)
    m.tournament = cells[1].text
    m.surface = cells[2].text
    m.tournament_round = match_round_parse(cells[3].text)
    m.p1_atp_rank_at_time = match_rank_parse(cells[4].text)
    m.p2_atp_rank_at_time = match_rank_parse(cells[5].text)
    m.p1_name, m.p1_id, m.p2_name, m.p2_id = match_players_parse(cells[6], p.name, p_id)

    # check for uniqueness then add to db
    # NOTE: TO-DO

    ms.append(m)

  # >> return (for testing)
  return p, ms







