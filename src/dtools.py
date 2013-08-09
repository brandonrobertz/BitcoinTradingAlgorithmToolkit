import numpy as np
import pandas as pd
import processlogs2 as pl2
import indicators as ind
import data
import csv
import cPickle
import time

#################
#
#  MINMAX NORMALIZE
#
#################
def minmax( dataset, crt=False, std=False):
  """ Remap each column to -1, 1 based on min/max. We can choose to
      skip CRT and STD. By default we only normalize presumably the TIs.
      
      TODO: crt and std inclusion not implemented
  """
  columns = dataset.columns
  for column in columns:
    #if ("STD" not in column) and ("CRT" not in column):
    A = dataset[column]
    imin = -1
    imax =  1
    dmin = np.min(A)
    dmax = np.max(A)
    B = imin + (imax - imin)*(A - dmin)/(dmax - dmin)
    dataset = dataset.drop( [column], axis=1)
    dataset = dataset.join( pd.DataFrame( {column:B}, index=[B.index]), how="outer")
    #elif crt or std:
    #  raise NotImplementedError
  return dataset

#################
#
#  GEN DS
#
#################
def gen_ds( dataset, forward, opts, type='CRT'):
  # trim leading NaN vals ... takes time to warm up the indicators
  dataset = cut_init_nans( dataset)
  
  # targets as compound returns
  if type == 'CRT':
    # generate ohlc so we can calculate n-forward CRTs
    ohlc = pd.DataFrame( {"open":dataset["LTC_open_%s"%opts["time_str"]],
                          "high":dataset["LTC_high_%s"%opts["time_str"]],
                          "low":dataset["LTC_low_%s"%opts["time_str"]],
                          "close": dataset["LTC_close_%s"%opts["time_str"]]},
                            index=dataset["LTC_open_%s"%opts["time_str"]].index)
    CRT_1 = ind.CRT( ohlc, 1)
    
    # move those forward CRTs back, so we can use them as target, correct predictions
    tgt = CRT_1.shift(-1)
  elif type == 'PRICE':
    # bring next price back so we can use it as target
    tgt = dataset["LTC_close_%s"%opts["time_str"]].shift(-1)
    tgt.name = 'tgt'
  
  # drop off OHLC data
  dataset = dataset.drop( [ "LTC_open_%s"%opts["time_str"],
                            "LTC_high_%s"%opts["time_str"],
                            "LTC_low_%s"%opts["time_str"],
                            "LTC_close_%s"%opts["time_str"] ], axis=1)
  # trim nans off end
  tgt = tgt.ix[:-1*forward]; dataset = dataset.ix[:-1*forward]
  
  # return
  return dataset, tgt

###########################
#  CRT SINGLE
###########################
def crt_single( initial, final, n = 1):
  """ Calculate compound return of two prices. For use in
      generating target values from and to match our input.

      initial : initial investment
      final : final price
      n : (default 5) periods this happened over. Since its relative to
          the compound returns in our LTC Coin class, which is calculated
          from OHLC data every 1min, this needs to match that breakdown
          (mins to mins, secs to secs, etc).
  """
  crt = np.log( final / initial) / n
  return crt

###########################
# TARGETS
###########################
def gen_target( dataset, N=1):
  """ Generate N-period target predictions based on close column
      in dataset. We assume this is already in pct change. The close
      column shouldn't be normalized or anything.
  """
  tgt = dataset.close.shift(-1*N)
  return tgt

###########################
# TARGETS
###########################
def gen_target_crt( dataset, N=1):
  """ Generate N-period target predictions based on close column
      in dataset. The close column shouldn't be normalized or anything.
  """
  if N < 0:
    # what was the change over the last period? (btwn t-N and t)
    tgt = crt_single( dataset.close.shift(-1*N), dataset.close )
  else:
    # 
    tgt = crt_single( dataset.close, dataset.close.shift(-1*N))
  return tgt

############################
# CUT_INIT_NANS
###########################
def cut_init_nans( dataset):
  """ Because of initialization of things like rolling means, etc,
      the first vals of a lot of the columns produced by Data are
      NaN. Well, this fucks backprop, so we need to trim it.
  
      dataset : a pandas DataFrame
      
      returns: the dataset minus the NaNs at the start
  """
  # find first non-NaN val, cut there (it needs time to warm up)
  iii = 0
  for iii in xrange( len(dataset)):
    nn = False
    for iiii in dataset.ix[iii]:
      if np.isnan(iiii):
       nn = True
    if nn == False:
      dataset = dataset.ix[iii:]
      break
  return dataset

############################
# SAVE_CSV
###########################
def save_csv( dataset_t, tgt, name=""):
  """ Save a dataset, both the inputs (dataset) and targets (tgt)
      as two CSV files for input to a matlab neural network
      
      dataset_t : the dataset used as input to neural net
      tgt : the targets, or correct values, to train/evaluate the NN
      name : name to label dataset ... e.g. "train", "test", etc
  """
  if not name:
    name = str(time.time())
  with open('%s.dataset.csv'%name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    writer.writerows(dataset_t.values)
  
  with open('%s.tgt.csv'%name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    if type(tgt.values[0]) == np.ndarray:
      writer.writerows([ v for v in tgt.values])
    else:
      writer.writerows([ [v] for v in tgt.values])

###########################
#  STORE_PICK
###########################
def store_pick( d):
  """ Store a fully-loaded Data class as a pickle
  
      d : a Data class
  """
  name = str(time.time())+".d."+d.filename+".pickle"
  filename = os.path.realpath( os.path.join( "pickles", name))
  f = open( filename, "w")
  cPickle.dump( d, f)
  f.close()

###########################
#  LOAD_PICK
###########################
def load_pick( filename):
  """ Load a fully-loaded Data class as a pickle
  
      filename : filename of pickle
  """
  filename = os.path.realpath( os.path.join( "pickles", filename))
  f = open( filename, "r")
  d = cPickle.load( f)
  f.close()
  return d
