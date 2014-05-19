from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base


def mysql_session(db_user='tennisbro',db_pass='scheme',db_host='localhost',db_name='atp_genius',ensure_created=False):
  engine = create_engine('mysql://%s:%s@%s/%s' % (db_user, db_pass, db_host, db_name))
  if ensure_created:
    Base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  return Session()
