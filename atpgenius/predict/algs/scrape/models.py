from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, Text, Date, DateTime, Boolean, types
from sqlalchemy import Index, PrimaryKeyConstraint
from sqlalchemy import create_engine
from datetime import datetime
Base = declarative_base()


class Player(Base):
  __tablename__ = 'players'
  
  id = Column(Integer, primary_key=True)
  name = Column(String(100))
  country = Column(String(5))
  twitter = Column(String(100))
  dob = Column(Date)
  plays = Column(String(150))
  atp_rank = Column(Integer)
  peak_atp_rank = Column(Integer)
  peak_date = Column(Date)
  last_updated = Column(DateTime, default=datetime.now)
  url = Column(Text)
  
  def __repr__(self):
    return '<Player(name=%s, atp_rank=%s)>' % (self.name, self.atp_rank)


class Match(Base):
  __tablename__ = 'matches'

  id = Column(Integer, primary_key=True)
  date = Column(Date)
  p1_id = Column(Integer)
  p1_name = Column(String(100))
  p2_id = Column(Integer)
  p2_name = Column(String(100))
  p1_rank = Column(Integer)
  p2_rank = Column(Integer)
  tournament = Column(String(100))
  tournament_round = Column(Integer)
  surface = Column(String(25))

  def __repr__(self):
    return '<Match(%s, %s (%s) d. %s (%s))>' % (self.date, self.p1_name, self.p1_atp_rank_at_time, self.p2_name, self.p2_atp_rank_at_time)


class Tournament(Base):
  __tablename__ = 'tournaments'

  id = Column(Integer, primary_key=True)
  name = Column(String(100))
  surface = Column(String(25))
  level = Column(String(5))

  def __repr__(self):
    return '<Tournament(%s, %s, %s)>' % (self.id, self.name, self.level)
  


