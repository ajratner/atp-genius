import sys
from scrape.util import *
from scrape.models import Player, Match, Tournament
import random
import numpy as np
from sqlalchemy import or_, and_, desc
from scipy.sparse import csr_matrix, vstack
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from collections import defaultdict
from time import time
import cPickle


# get the matches from the db
def get_matches(pid=None):
  session = mysql_session()
  if pid:
    return session.query(Match).filter(or_(Match.p1_id == pid, Match.p2_id == pid)).all()
  else:
    return session.query(Match).all()


# Baseline #1- predict winner will be the one with higher rank
# >> 53%
def baseline1(pid=None):
  matches = [m for m in get_matches(pid) if m.p1_rank and m.p2_rank]
  correct = len([m for m in matches if m.p1_rank > m.p2_rank])
  return correct / float(len(matches))


# Baseline #2- random sampling based on the difference between ranks
# >> 68%
def baseline2(pid=None, i=100):
  return np.mean([baseline2_sub(pid) for n in range(i)])

def baseline2_sub(pid=None):
  matches = [m for m in get_matches(pid) if m.p1_rank and m.p2_rank]
  correct = 0
  deltas = [float(m.p1_rank - m.p2_rank) for m in matches]
  min_d = min(deltas)
  max_d = max(deltas)
  d_range = max_d - min_d
  for d in deltas:
    r = random.random()*d_range + min_d
    if r < d:
      correct += 1
  return correct / float(len(matches))


# Model #1- vectorize single-match features only, run through simple ML model
N_FEATURES = 12
LAST_N_MATCHES = 5
UNK_RANK = 1000
SURFACE_MAP = {'Clay':0, 'Hard':1, 'Grass':2, 'Carpet':2}
T_LEVEL_MAP = {'CH':0, 'ATP':1, 'M1000':2, 'GS':3, '':1}


# tournament level sub-function
def tournament_level(tournament, session=None):
  session = session if session else mysql_session()

  # try for direct match first
  tournaments = session.query(Tournament).filter(Tournament.name == tournament).all()
  if tournaments and len(tournaments) > 0:
    return T_LEVEL_MAP[tournaments[0].level.strip()]

  # else look for imilar and average
  else:
    tournaments=session.query(Tournament).filter(Tournament.name.like('%'+tournament+'%')).all()
    if tournaments and len(tournaments) > 0:
      return int(np.mean([T_LEVEL_MAP[t.level.strip()] for t in tournaments]))
    else:
      return 1

# create tournament level map
# NOTE: just do this once when scraping the data in the first place??
def create_tournament_level_map(matches, session=None):
  session = session if session else mysql_session()
  tl_map = {}
  for m in matches:
    if not tl_map.has_key(m.tournament):
      tl_map[m.tournament] = tournament_level(m.tournament, session)
  return tl_map


# core feature assembly function
# feature range = [-1,1]

# >> vs_p2  =  list of win/losses ~ [1, 0]

# >> lnm = last_n_matches  =  list of (win/loss ~ [1,0], rank-at-time)

# >> pNfr = 'player N faceted record'
# >> array of (w,l) for: overall (0), surfaces (1-3), tournament levels (4-7)

def get_match_features(p1_rank, p2_rank, tl, surface, vs_p2, lnm1, lnm2, p1fr, p2fr):
  f = np.zeros(N_FEATURES)

  # feature 1 - binary, is player higher rank?
  f[0] = 1.0 if p1_rank < p2_rank else -1.0
  
  # feature 2 - norm. + capped rank difference
  f[1] = max(min((p2_rank - p1_rank)/100.0, 1.0), -1.0)

  # feature 3 - vs. opponent history
  f[2] = 2.0*vs_p2 - 1.0

  # feature 4 - last n matches avg for player 1
  f[3] = 2.0*((sum([x[0] for x in lnm1]) / float(LAST_N_MATCHES)) - 0.5)

  # feature 5 - last n matches avg for player 2
  f[4] = 2.0*((sum([x[0] for x in lnm2]) / float(LAST_N_MATCHES)) - 0.5)

  # feature 6 - binary recent change in rank for p1
  mov_avg = np.mean([lnm1[i-1][1] - lnm1[i][1] for i in range(1, len(lnm1)) if lnm1[i][1] and lnm1[i-1][1]]) if len(lnm1) > 1 else 0
  f[5] = 1.0 if mov_avg > 0 else -1.0

  # feature 7 - binary recent change in rank for p2
  mov_avg = np.mean([lnm2[i-1][1] - lnm2[i][1] for i in range(1, len(lnm2)) if lnm2[i][1] and lnm2[i-1][1]]) if len(lnm2) > 1 else 0
  f[6] = 1.0 if mov_avg > 0 else -1.0

  # >> get surface int
  s = SURFACE_MAP[surface] if SURFACE_MAP.has_key(surface) else -1

  # feature 8 - p1 relative performance on this match's surface
  if sum(p1fr[0]) > 0 and sum(p1fr[s+1]) > 0:
    f[7] = (p1fr[s+1][0]/float(sum(p1fr[s+1]))) - (p1fr[0][0]/float(sum(p1fr[0])))

  # feature 9 - p2 relative performance on surface
  if sum(p2fr[0]) > 0 and sum(p2fr[s+1]) > 0:
    f[8] = (p2fr[s+1][0]/float(sum(p2fr[s+1]))) - (p2fr[0][0]/float(sum(p2fr[0])))

  # feature 10 - p1 vs p2 relative experience at tournament level
  if sum(p1fr[0]) > 0 or sum(p2fr[0]) > 0:
    f[9] = float(sum(p1fr[tl+4]) - sum(p2fr[tl+4])) / max(sum(p1fr[0]), sum(p2fr[0]))

  # feature 11 - p1 record at this tournament level
  if sum(p1fr[tl+4]) > 0:
    f[10] = 2.0*(p1fr[tl+4][0] / float(sum(p1fr[tl+4]))) - 1.0

  # feature 12 - p2 record at this tournament level
  if sum(p2fr[tl+4]) > 0:
    f[11] = 2.0*(p2fr[tl+4][0] / float(sum(p2fr[tl+4]))) - 1.0

  # features to (potentially) add:

  # NOTE: to-do: use tournament round?

  # NOTE: to-do: defending points
  # >> http://www.pinnaclesports.com/online-betting-articles/03-2014/tennis-ranking-points.aspx

  # XXX old features...
  """
  # feature 3 - norm. tournament round
  # f[2] = (5 - float(t_round)) / 5.0

  # feature 6 - norm. + capped winning streak (consecutive previous wins)
  # XXX nix this?
  streak = 0
  for i,m in enumerate(lnm[::-1]):
    if m[0] < 1 or i >= LAST_N_MATCHES:
      break
    streak += 1
  f[5] = float(streak) / LAST_N_MATCHES

  # feature 4 - binary clay surface?
  f[3] = 1.0 if surface == 'Clay' else -1.0

  """
  
  return f


# vectorize a single (eg unseen) match, based only on certain provided info
def vectorize_match(p1_name, p2_name, p1_rank, p2_rank, t_level, surface, t_round,session=None):
  session = mysql_session() if not session else session

  # get p1,p2 joint match history
  p1w = session.query(Match).filter(and_(Match.p1_name == p1_name, Match.p2_name == p2_name)).count()
  p2w = session.query(Match).filter(and_(Match.p1_name == p2_name, Match.p2_name == p1_name)).count()
  vs_p2 = float(p1w) / (p1w + p2w) if p1w + p2w > 2 else 0.5

  # get all matches for both players
  p1_matches = session.query(Match).filter(or_(Match.p1_name == p1_name, Match.p2_name == p1_name)).order_by(desc(Match.date)).all()
  p2_matches = session.query(Match).filter(or_(Match.p1_name == p2_name, Match.p2_name == p2_name)).order_by(desc(Match.date)).all()

  # compile pfr dicts
  tl_map = create_tournament_level_map(p1_matches+p2_matches, session)

  # p1
  p1fr = [[0,0] for i in range(8)]
  for m in p1_matches:
    w = 0 if  m.p1_name == p1_name else 1
    
    # overall record
    p1fr[0][w] += 1

    # surface-specific record
    try:
      s = SURFACE_MAP[m.surface]
      p1fr[s+1][w] += 1
    except KeyError:
      pass

    # tournament-level specific record
    t = tl_map[m.tournament]
    p1fr[t+4][w] += 1

  # p2
  p2fr = [[0,0] for i in range(8)]
  for m in p2_matches:
    w = 0 if  m.p1_name == p2_name else 1
    
    # overall record
    p2fr[0][w] += 1

    # surface-specific record
    try:
      s = SURFACE_MAP[m.surface]
      p2fr[s+1][w] += 1
    except KeyError:
      pass

    # tournament-level specific record
    t = tl_map[m.tournament]
    p2fr[t+4][w] += 1

  # get last n matches
  lnm1 = []
  for m in p1_matches[:LAST_N_MATCHES]:
    if m.p1_name == p1_name:
      lnm1.append((1, m.p1_rank))
    else:
      lnm1.append((0, m.p2_rank))
  lnm2 = []
  for m in p2_matches[:LAST_N_MATCHES]:
    if m.p1_name == p2_name:
      lnm2.append((1, m.p1_rank))
    else:
      lnm2.append((0, m.p2_rank))

  # get features
  f = get_match_features(p1_rank, p2_rank, t_level, surface, vs_p2, lnm1, lnm2, p1fr, p2fr)
  return f


# NOTE NOTE
# --> need to proceed serially through ALL the matches keeping everything in queue
# --> need custom function...

# vectorize all matches for training / testing
# >> proceed serially through matches by date for speed
def vectorize_data():
  err_count = 0
  session = mysql_session()

  # get matches
  matches = session.query(Match).order_by(Match.date).all()

  # pfr- player faceted record of (w,l)
  pfr = defaultdict(lambda : [[0,0] for i in range(8)])
  
  # last N matches as (win/loss, rank at time)
  lnm = defaultdict(list)
  
  # indexed by e.g. 'Bobby Reynolds::Rafael Nadal' w/ alphabetical order
  vs_win = defaultdict(int)
  vs_count = defaultdict(int)

  # get tournament level map
  tl_map = create_tournament_level_map(matches, session)

  # go through all the matches by date
  X = []
  Y = []
  for i,m in enumerate(matches):

    # try to get tournament level
    tl = tl_map[m.tournament]

    # vectorize based on current data
    if m.date and m.p1_rank and m.p2_rank:
      vs_key = '::'.join(sorted([m.p1_name, m.p2_name]))
      vs_p2 = vs_win[vs_key] / float(vs_count[vs_key]) if vs_count[vs_key] > 0 else 0.5

      # >> we want to predict whether p1 wins or not.  Since in the data, p1 is by definition
      # >> the winner, we must randomize the ordering here
      if random.random() > 0.5:
        vs_p2 = vs_p2 if ord(m.p1_name[0]) < ord(m.p2_name[0]) else 1.0 - vs_p2
        f = get_match_features(m.p1_rank, m.p2_rank, tl, m.surface, vs_p2, lnm[m.p1_name], lnm[m.p2_name], pfr[m.p1_name], pfr[m.p2_name])
        X.append(f)
        Y.append(1.0)
      else:
        vs_p2 = vs_p2 if ord(m.p1_name[0]) > ord(m.p2_name[0]) else 1.0 - vs_p2
        f = get_match_features(m.p2_rank, m.p1_rank, tl, m.surface, vs_p2, lnm[m.p2_name], lnm[m.p1_name], pfr[m.p2_name], pfr[m.p1_name])
        X.append(f)
        Y.append(-1.0)

      # update pfr dict
      
      # >> overall
      pfr[m.p1_name][0][0] += 1
      pfr[m.p2_name][0][1] += 1
      
      # >> surfaces
      try:
        s = SURFACE_MAP[m.surface]
        pfr[m.p1_name][s+1][0] += 1
        pfr[m.p2_name][s+1][1] += 1
      except KeyError:
        pass

      # >> tournament levels
      pfr[m.p1_name][tl+4][0] += 1
      pfr[m.p2_name][tl+4][1] += 1

      # update lnm dict
      lnm[m.p1_name].append((1, m.p1_rank))
      if len(lnm[m.p1_name]) > LAST_N_MATCHES:
        lnm[m.p1_name] = lnm[m.p1_name][1:]
      lnm[m.p2_name].append((0, m.p2_rank))
      if len(lnm[m.p2_name]) > LAST_N_MATCHES:
        lnm[m.p2_name] = lnm[m.p2_name][1:]
      
      # update vs dicts
      vs_win[vs_key] += 1 if ord(m.p1_name[0]) < ord(m.p2_name[0]) else 0
      vs_count[vs_key] += 1

    # >> track how many matches didn't have minimum required data
    else:
      err_count += 1

  print 'Number of matches with insuficient data = %s' % (err_count,)
  return csr_matrix(X), np.array(Y)



N_FOLDS = 5
def model1(X, Y):
  
  # create stratified K-fold splits
  skf = StratifiedKFold(y=Y, n_folds=N_FOLDS)

  # define the classifier & which hyper-parameters to x-validate over
  clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
  C_range = 2.0 ** np.arange(-3,10)
  pg = {'C':C_range}
  
  # grid search for best C values in parallel on multiple cores
  grid_clf = GridSearchCV(clf, param_grid=pg, cv=skf, n_jobs=8)
  grid_clf.fit(X,Y)

  # get precision (accuracy)
  clf = grid_clf.best_estimator_
  print clf.coef_
  Y_p = clf.predict(X)
  correct = 0
  for i in range(X.shape[0]):
    if Y[i] == Y_p[i]:
      correct += 1
  return correct / float(X.shape[0])

def model2(X, Y, K=5):
  print X.shape
  print Y.shape

  # >> random forest requires dense arrays...
  X = X.todense()

  # pick classifier
  clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
  #clf = SVC()
  #clf = RandomForestClassifier(n_estimators=100)

  # randomly pick K folds
  idx = range(X.shape[0])
  random.shuffle(idx)

  # store info on hyperplane distance / accuracy
  tc_scores = []
  tc_dists = []

  # do K-fold testing
  scores = []
  ks = int(len(idx)/float(K))
  for ki in range(K):

    # get the training / testing splits
    idx_kf = idx[ki*ks:(ki+1)*ks]
    idx_train = [0 if i in idx_kf else 1 for i in range(len(idx))]
    idx_test = [1 if i in idx_kf else 0 for i in range(len(idx))]
    X_train = np.compress(idx_train, X, axis=0)
    Y_train = np.compress(idx_train, Y, axis=0)
    X_test = np.compress(idx_test, X, axis=0)
    Y_test = np.compress(idx_test, Y, axis=0)
  
    # train the classifier
    clf.fit(X_train, Y_train)

    # predict and get simple accuracy (precision)
    Y_p = clf.predict(X_test)
    correct = 0
    for i in range(X_test.shape[0]):
      if Y_test[i] == Y_p[i]:
        correct += 1
        tc_scores.append(1)
      else:
        tc_scores.append(0)
    scores.append(correct / float(X_test.shape[0]))

    tc_dists += list(clf.decision_function(X_test))

  # re-train on all data & save to disk
  clf.fit(X, Y)
  cPickle.dump(clf, open('saved_model/atp_genius_trained.pkl', 'wb'))

  # save the confidence scores
  cPickle.dump(tc_scores, open('saved_model/tc_scores.pkl', 'wb'))
  cPickle.dump(tc_dists, open('saved_model/tc_dists.pkl', 'wb'))
  
  # average the results of the k-fold tests
  return np.mean(scores)





if __name__ == '__main__':
  
  if int(sys.argv[1]) == 1:
    print baseline1()

  elif int(sys.argv[1]) == 2:
    print baseline2(i=10)

  elif int(sys.argv[1]) == 3:
    t0 = time()

    # vectorize all the rows
    X,Y = vectorize_data()
    t1 = time(); print 'Vectorized in %s seconds.' % (t1-t0,); t0 = t1

    # train the model and print out accuracy
    print model2(X,Y)
    t1 = time(); print 'Trained in %s seconds.' % (t1-t0,); t0 = t1
