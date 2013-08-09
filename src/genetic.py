import data
import dtools as dts
import indicators as ind
import pandas as pd
import numpy as np
import random

def gene_type():
  """ Return the gene's structure, just the types. We will use this to know
      how the gene should be randomized and mutated.
  """
  n_codes = 40
  # create mutation gene
  g = [0]*n_codes
  g[0]  = bool   # calc rolling?
  g[1]  = int    # rolling n 1
  g[2]  = int    # rolling n 2
  g[3]  = int    # rolling n 3
  g[4]  = bool   # RSI?
  g[5]  = int    # RSI n
  g[6]  = bool   # ROC?
  g[7]  = int    # ROC n
  g[8]  = bool   # AMA?
  g[9]  = int    # AMA n
  g[10] = int    # AMA fn
  g[11] = int    # AMA sn
  g[12] = bool   # CCI?
  g[13] = int    # CCI n
  g[14] = bool   # FRAMA?
  g[15] = int    # FRAMA n
  g[16] = bool   # RVI2?
  g[17] = int    # RVI2 n
  g[18] = int    # RVI2 s
  g[29] = bool   # MACD?
  g[20] = int    # MACD f
  g[21] = int    # MACD s
  g[22] = int    # MACD m
  g[23] = bool   # ADX?
  g[24] = int    # ADX n
  g[25] = bool   # ELI?
  g[26] = int    # ELI n
  g[27] = bool   # TMI?
  g[28] = int    # TMI nb
  g[29] = int    # TMI nf
  g[30] = bool   # calc STD?
  g[31] = int    # STD n 1
  g[32] = int    # STD n 2
  g[33] = int    # STD n 3
  g[34] = bool   # calc CRT?
  g[35] = int    # CRT n 1
  g[36] = int    # CRT n 2
  g[37] = int    # CRT n 3
  g[38] = int    # CRT n 4
  g[39] = int    # CRT n 5
  return g

def different( x):
  for i in xrange( len(x)):
    for ii in xrange( len(x)):
      if i != ii:
        if x[i] == x[ii]:
          return False
  return True

def rand_indiv( indivClass):
  g = rand_gene()
  ind = indivClass( g)
  return ind

def rand_gene():
  ri = np.random.random_integers
  g = gene_type()
  ind = [0]*len(g)
  
  for i in xrange( len(g)):
    if g[i] == bool:
      ind[i] = ri(0,1)
    elif g[i] == int:
      ind[i] = ri(1,50)
  return ind

def decode_gene( g, mins):
  # build genetic list
  # g          = [0] * 42
  rolling    = g[0]  #= True  # calc rolling?
  r_1        = g[1]  #= 12    # rolling n 1
  r_2        = g[2]  #= 24    # rolling n 2
  r_3        = g[3]  #= 50    # rolling n 3
  rsi        = g[4]  #= True  # RSI?
  rsi_n      = g[5]  #= 14    # RSI n
  roc        = g[6]  #= True  # ROC?
  roc_n      = g[7]  #= 20    # ROC n
  ama        = g[8]  #= True  # AMA?
  ama_n      = g[9]  #= 10    # AMA n
  ama_fn     = g[10] #= 1     # AMA fn
  ama_sn     = g[11] #= 30    # AMA sn
  cci        = g[12] #= True  # CCI?
  cci_n      = g[13] #= 20    # CCI n
  frama      = g[14] #= True  # FRAMA?
  frama_n    = g[15] #= 10    # FRAMA n
  rvi2       = g[16] #= True  # RVI2?
  rvi2_n     = g[17] #= 14    # RVI2 n
  rvi2_s     = g[18] #= 10    # RVI2 s
  macd       = g[19] #= True  # MACD?
  macd_f     = g[20] #= 12    # MACD f
  macd_s     = g[21] #= 26    # MACD s
  macd_m     = g[22] #= 9     # MACD m
  adx        = g[23] #= True  # ADX?
  adx_n      = g[24] #= 14    # ADX n
  eli        = g[25] #= True  # ELI?
  eli_n      = g[26] #= 14    # ELI n
  tmi        = g[27] #= True  # TMI?
  tmi_nb     = g[28] #= 10    # TMI nb
  tmi_nf     = g[29] #= 5     # TMI nf
  std        = g[30] #= True  # calc STD?
  std_1      = g[31] #= 13    # STD n 1
  std_2      = g[32] #= 21    # STD n 2
  std_3      = g[33] #= 34    # STD n 3
  crt        = g[34] #= True  # calc CRT?
  crt_1      = g[35] #= 1     # CRT n 1
  crt_2      = g[36] #= 2     # CRT n 2
  crt_3      = g[37] #= 3     # CRT n 3
  crt_4      = g[38] #= 4     # CRT n 4
  crt_5      = g[39] #= 5     # CRT n 5

  # FIXING TIME ... it gravitates toward the smallest
  # sampling period so the peaks and valleys are smaller,
  # then it can fit a flat line with lower error
  time_str = "%smin"%int( mins) #mins)

  # rolling
  r_dict = False
  if rolling:
    r_dict = { time_str : {  int(r_1): pd.DataFrame(),
                             int(r_2): pd.DataFrame(),
                             int(r_3): pd.DataFrame() } }
  ind_dict = {}
  if rsi:
    ind_dict["RSI"] = { "data": pd.DataFrame(), "n":int(rsi_n) }
  if roc:
    ind_dict["ROC"] = { "data": pd.DataFrame(), "n":int(roc_n) }
  if ama:
    ind_dict["AMA"] = { "data": pd.DataFrame(), "n":int(ama_n), "fn":int(ama_fn), "sn":int(ama_sn )}
  if cci:
    ind_dict["CCI"] = { "data": pd.DataFrame(), "n":int(cci_n) }
  if frama:
    ind_dict["FRAMA"] = { "data": pd.DataFrame(), "n":int(frama_n) }
  if rvi2:
    ind_dict["RVI2"] = { "data": pd.DataFrame(), "n":int(rvi2_n), "s":int(rvi2_s )}
  if macd:
    ind_dict["MACD"] = { "data": pd.DataFrame(), "f":int(macd_f), "s":int(macd_s), "m":int(macd_m )}
  if adx:
    ind_dict["ADX"] = { "data": pd.DataFrame(), "n":int(adx_n) }
  if eli:
    ind_dict["ELI"] = { "data": pd.DataFrame(), "n":int(eli_n) }
  if tmi:
    ind_dict["TMI"] = { "data": pd.DataFrame(), "nb":int(tmi_nb), "nf":int(tmi_nf) }

  # STD DEV
  std_dict = False
  if std:
    std_dict = { int(std_1) : pd.DataFrame(), 
                 int(std_2) : pd.DataFrame(),
                 int(std_3) : pd.DataFrame() }

  # CRT
  crt_dict = False
  if crt:
    crt_dict = { int(crt_1) : pd.DataFrame(),
                 int(crt_2) : pd.DataFrame(),
                 int(crt_3) : pd.DataFrame(),
                 int(crt_4) : pd.DataFrame(),
                 int(crt_5) : pd.DataFrame() }

  # put it all together
  ltc_opts = \
  {  "debug": False,
     "relative": False,
     "calc_rolling": rolling,
     "rolling": r_dict,
     "calc_mid": True,
     "calc_ohlc": True,
     "ohlc": { time_str : pd.DataFrame()  },
     "calc_indicators": True,
     "indicators": ind_dict,
     "calc_std": std,
     "std": std_dict,
     "calc_crt": crt,
     "crt": crt_dict,
     "instant": True,
     "time_str": time_str }
  return ltc_opts

def mutate_gene( ind, mu=0, sigma=4, chance_mutation=0.4):
  """ Here, we create a mutation gene that will be
      iterativley added / subtracted from the ind
      gene.

      Params:
        ind   : our individual (gene)
        mu    : gaussian function mean
        sigma : standard deviation
        chance_mutation : chances that we're going to mutate a part of
                          the code

      Return:
        A "mutated," randomized, version of our input gene.
  """
  g = gene_type()
  
  for i in xrange( len(ind)):
    # if we're supposed to mutate, randomly
    if random.random() < chance_mutation:
      if g[i] == bool:
        ind[i] = int( not ind[i])
      # an int!
      elif g[i] == int:
        ind[i] += int(random.gauss(mu, sigma))
        # nothing can go below 1
        if ind[i] < 1:
          ind[i] = 1
  return ind

def verify_gene( ind):
  """ In some cases we need to make sure that our genes stay the right
      types and sane values.

      Params:
        ind : our input gene

      Returns:
        Our input gene, but with types and minor bounds enforced.
  """
  g = gene_type()
  
  for i in xrange( len(ind)):
    if g[i] == bool:
      # convert back to bool and then int
      ind[i] = int(bool(ind[i]))
    # an int!
    elif g[i] == int:
      # make sure its an int, no decimal places
      ind[i] = int(ind[i])
      # nothing can go below 1
      if ind[i] < 1:
        ind[i] = 1
  return ind