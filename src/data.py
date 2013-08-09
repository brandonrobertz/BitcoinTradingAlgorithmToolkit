import processlogs2 as pl2
import indicators as ind
import dtools as dts
import datetime
import pandas as pd
import numpy as np
import cPickle
import re
import os

######################################################################
#                                                                    #
#                                                                    #
#                                                                    #
#                             C O I N                                #
#                                                                    #
#                                                                    #
#                                                                    #
######################################################################
class Coin:
  """ This holds prices for each and any coin we want. It automatically
      updates a series of averages and other statistical manipulations
      of price movements.
  """

  def __init__( self, **kwargs):
    """ setup our basic dataframes ... raw prices, 5min rolling avg, etc
        Paramz:
          debug : (default False) print debugging output?
          time_str : (default "1min") period to analyze ... follows
                     pandas notation "10s", "5min", "1h", "1d" etc
          calc_rolling : (default False) calculate our rolling data?
          calc_mid : (default False) standard behavior is to record
                     the lastprice as last tick. Changing this to True
                     makes the lastprice become the average of bid
                     vs. ask
          calc_ohlc : (default False) calculate our OHLC data?
          calc_indicators : (default False) controls whether or not
                            to create our indicators. Most rely on
                            OHLC and last price, so those are necessary
          calc_crt : (default False) controls whether we wanna
                     calculate our compound returns based on
                     past price actionz
          rolling : a custom dict specifying what averages to
                    compute (optional) ... should be in the following
                    format:
                      { "time" : { N : pd.DataFrame(), ... } }
                      i.e., 
                      { "30s" : {  12: pd.DataFrame(), 
                            24: pd.DataFrame(), 
                            50: pd.DataFrame() } }
          std : similar to rolling. a custom dict telling Coin how
                to compute our n-period averages.
                  i.e., a 13 and 21 period std-dev
                  { 13: pd.DataFrame(), 21: pd.DataFrame() }
          crt : similar to above ... but for n-period compount return
                { 1: pd.DataFrame(), 5: pd.DataFrame() }
          indicators : a dict containing the indicators we want to calculate
                       with their parameters. i.e.,
                      { "RSI"  : { "data": pd.DataFrame(), "n": 14 },
                        "ROC"  : { "data": pd.DataFrame(), "n": 20 },
                        "AMA"  : { "data": pd.DataFrame(), "n": 10, "fn": 2.5, 
                                                           "sn":30 },
                        "CCI"  : { "data": pd.DataFrame(), "n": 20 },
                        "FRAMA": { "data": pd.DataFrame(), "n": 10 },
                        "RVI2" : { "data": pd.DataFrame(), "n": 14, "s": 10 },
                        "MACD" : { "data": pd.DataFrame(), "f": 12, "s": 26, 
                                                           "m": 9 },
                        "ADX"  : { "data": pd.DataFrame(), "n": 14 } }
          instant : (default False) setting this to a dataframe of last
                    prices will trigger it to automatically calculate all
                    indicators and averages for the whole set in one pass
    """
    self._debug = kwargs.get( "debug", False)
    self.relative = kwargs.get( "relative", False)
    self._calc_rolling = kwargs.get("calc_rolling", False)
    self.rolling = kwargs.get( "rolling", False)
    self._calc_mid = kwargs.get("calc_mid", False)
    self._calc_ohlc = kwargs.get("calc_ohlc", False)
    self.ohlc = kwargs.get("ohlc", False)
    self._calc_indicators = kwargs.get("calc_indicators", False)
    self.ti = kwargs.get( "indicators", False)
    self._calc_std = kwargs.get("calc_std", False)
    self.std = kwargs.get( "std", False)
    self._calc_crt = kwargs.get( "calc_crt", False)
    self.crt = kwargs.get( "crt", False)
    self.instant = kwargs.get( "instant", False)
    self.time_str = kwargs.get( "time_str", "5min")
    self.verbose = kwargs.get( "verbose", False)
      
    # this will hold moving averages, first by frequency, then by window
    if self._calc_rolling:
      # did we get rolling parameters sent to us?
      if type( self.rolling) != dict:
        # no, so set defaults
        self.rolling =  { self.time_str :  { 12: pd.DataFrame(), 
                                      24: pd.DataFrame(), 
                                      50: pd.DataFrame() } }

    # this will hold OHLC data
    if self._calc_ohlc:
      # some defaults if nothing provided
      if type( self.ohlc) != dict:
        self.ohlc = { self.time_str : pd.DataFrame() }

    # here's where our tecnical indicators go
    if self._calc_indicators:
      if type( self.ti) != dict:
        self.ti = { "RSI"  : { "data": pd.DataFrame(), "n":14 },
                    "ROC"  : { "data": pd.DataFrame(), "n":20 },
                    "AMA"  : { "data": pd.DataFrame(), "n":10, "fn":2.5, 
                                                      "sn":30 },
                    "CCI"  : { "data": pd.DataFrame(), "n":20 },
                    "FRAMA": { "data": pd.DataFrame(), "n":10 },
                    "RVI2" : { "data": pd.DataFrame(), "n":14, "s":10 },
                    "MACD" : { "data": pd.DataFrame(), "f":12, "s":26, 
                                                       "m":9 },
                    "ADX"  : { "data": pd.DataFrame(), "n":14 },
                    "ELI"  : { "data": pd.DataFrame(), "n":14 },
                    "TMI"  : { "data": pd.DataFrame(), "nb":10, "nf":5 }
                  }

    # running standard deviations
    if self._calc_std:
      # some defaults if nothing provided
      if type( self.std) != dict:
        self.std = { 13: pd.DataFrame(),
                     21: pd.DataFrame(),
                     34: pd.DataFrame() }
                     
    # get our n-period compound returns
    if self._calc_crt:
      # some defaults if nothing provided
      if type( self.crt) != dict:
        self.crt = { 1: pd.DataFrame() }

    # iterative move ... start blank
    if type(self.instant) != pd.DataFrame:
      # this will hold our last prices
      self.lastprice = pd.DataFrame()
    # INSTANT MODE
    else:
      if self.verbose: print "[*] Entering one-pass 'instant' mode"

      if type( self.instant) == pd.DataFrame:
        # set lastprices as given price DF .. make sure its called lastprice
        self.lastprice = self.instant.rename( columns={self.instant.columns[0]:"lastprice"})

        # OHLC first
        if self.ohlc:
          for time_str in self.ohlc:
            self.ohlc[time_str] = self.lastprice.lastprice.resample( time_str, 
                                    how="ohlc").ffill()
          
        # run through all necessary rolling averages and compute them
        if self._calc_rolling:
          for time_str in self.rolling:
            for window in self.rolling[time_str]:
              # default EMA ... TODO: allow users to change this
              self.rolling[time_str][window] = pd.ewma( self.lastprice.resample( 
                                                 time_str, fill_method="ffill"),
                                                 span=window ,freq=time_str)

        # calculate our technical indicators
        if self._calc_indicators:
          self._indicators()
        
        # running standard deviations
        if self._calc_std:
          self._std()
        
        # compound returns
        if self._calc_crt:
          self._compound()

      else:
        print "[!]","Error! Didn't pass instant a dataframe!"
        

  ###########################
  #      ADD
  ###########################
  def add( self, price, t, **kwargs):
    """ this is our main interface. w/ price & time it does the rest
        PARAMZ:
          price : last price from ticker
          t : time of price
          ba : bid/ask spread as tuple [bid, ask]
               (optional if not in midprice mode)
    """
    # make sure our t is a datetime
    if type( t ) != datetime.datetime:
      t = pd.to_datetime( t)
    
    # get new lastprice
    # if self._calc_mid = True then we're calculating the last price
    # as the avg between bid/ask ... this can be a better estimate thn last
    if self._calc_mid:
      bid, ask = kwargs.get( "ba", [np.NaN, np.NaN])
      self.lastprice = self._mid_df( bid, ask, t, "lastprice", self.lastprice)
    # otherwise, we're just using lastprice
    else:
      self.lastprice = self._lastprice_df( price, t)

    # calculate our OHLC data if needed
    if self._calc_ohlc:
      for time_str in self.ohlc:
        self.ohlc[time_str] = self._ohlc_df( t, self.ohlc[time_str], time_str)
    
    # run through all necessary rolling averages and compute them
    if self._calc_rolling:
      for time_str in self.rolling:
        for window in self.rolling[time_str]:
          self.rolling[time_str][window] = self._rolling( price,
                                             t, self.rolling[time_str][window],
                                             time_str, window)

    # calculate our technical indicators
    if self._calc_indicators:
      # TODO: update this if we ever want to add multiple OHLC frames
      self._indicators()

    # running standard deviations
    if self._calc_std:
      self._std()
    
    # compound returns
    if self._calc_crt:
      self._compound()

  ###########################
  #  COMBINE
  ###########################
  def combine( self, name):
    """ This will combine all statistical breakdowns in a coin into a
        single DataFrame
        
        name : a name to prepend all columns with, i.e., "LTC"
    """
    all = pd.DataFrame()
    # sort time_strs
    if self._calc_rolling:
      for time_str in self.rolling.keys():
        for N in self.rolling[time_str]:
          all = all.join( self.rolling[time_str][N].rename( 
                  columns={"lastprice":name+"_"+"EMA_"+time_str+"_"+str(N)}),
                  how="outer")
    # standard deviations
    if self._calc_std:
      for N in self.std.keys():
        all = all.join( self.std[N].rename( 
                columns={self.std[N].columns[0]:name+"_"+self.std[N].columns[0]+"_"+str(N)}), 
                how="outer")
    # technical indicators
    if self._calc_indicators:
      if type(self.ti) == dict:
        for ind in self.ti.keys():
          all = all.join( self.ti[ind]["data"], how="outer")
    # compound returns
    if self._calc_crt:
      for N in self.crt.keys():
        all = all.join( self.crt[N].rename( 
                columns={self.crt[N].columns[0]:name+"_"+self.crt[N].columns[0]}), 
                how="outer")
    # OHLC
    if self.ohlc:
      for time_str in self.ohlc:
        for col in self.ohlc[time_str]:
          all = all.join( pd.DataFrame( { "%s_%s_%s"%(name, 
                  self.ohlc[time_str][col].name, time_str): self.ohlc[time_str][col]},
                  index=[self.ohlc[time_str].index]), how="outer")
    return all

  ###########################
  #      _COMPOUND (RETURN)
  ###########################
  # TODO: update this if we ever want to add multiple OHLC frames
  def _compound( self):
    """ Once again, ugly ass hack, but fuck it. We're calculating the
        compound returns over the past N periods as defined in our crt
        dict.
    """
    for time_str in self.ohlc:
      for N in self.crt:
        # define reutrn as return over open and close
        self.crt[N] = ind.CRT( self.ohlc[time_str], N)
        #self.crt[N] = ind.CRT( self.ohlc[time_str].close, N)

  ###########################
  #      _INDICATORS
  ###########################
  def _indicators( self ):
    """ This will calculate our technical indicators based on the
        parameters in our ti dict ... this can be ran in one bang OR
        iteratively. It goes directly to the indicator structs. Not pretty,
        but what the fuck.
    """
    # TODO: update this if we ever want to add multiple OHLC frames
    for time_str in self.ohlc:
      for indicator in self.ti:
        if indicator == "RSI":
          self.ti[indicator]["data"] = ind.RSI( self.ohlc[time_str],
                                                self.ti[indicator]["n"] )
        elif indicator == "ROC":
          self.ti[indicator]["data"] = ind.ROC( self.ohlc[time_str],
                                                self.ti[indicator]["n"] )
        elif indicator == "AMA":
          self.ti[indicator]["data"] = ind.AMA( self.ohlc[time_str].close,
                                                self.ti[indicator]["n"],
                                                self.ti[indicator]["fn"],
                                                self.ti[indicator]["sn"] )
        elif indicator == "CCI":
          self.ti[indicator]["data"] = ind.CCI( self.ohlc[time_str],
                                                self.ti[indicator]["n"] )
        elif indicator == "FRAMA":
          self.ti[indicator]["data"] = ind.FRAMA( self.ohlc[time_str],
                                                  self.ti[indicator]["n"] )
        elif indicator == "RVI2":
          self.ti[indicator]["data"] = ind.RVI2( self.ohlc[time_str],
                                                 self.ti[indicator]["n"],
                                                 self.ti[indicator]["s"] )
        elif indicator == "MACD":
          self.ti[indicator]["data"] = ind.MACD( self.ohlc[time_str],
                                                 self.ti[indicator]["f"],
                                                 self.ti[indicator]["s"],
                                                 self.ti[indicator]["m"] )
        elif indicator == "ADX":
          self.ti[indicator]["data"] = ind.ADX( self.ohlc[time_str],
                                                 self.ti[indicator]["n"] )
        elif indicator == "ELI":
          self.ti[indicator]["data"] = ind.ELI( self.ohlc[time_str],
                                                self.ti[indicator]["n"] )
        elif indicator == "TMI":
          self.ti[indicator]["data"] = ind.TMI( self.ohlc[time_str],
                                                self.ti[indicator]["nb"],
                                                self.ti[indicator]["nf"])
  
  ###########################
  #      _LASTPRICE_DF
  ###########################
  def _lastprice_df( self, price, t):
    """ This will create a new DF with a price if our global lastprice
        dataframe is empty, or it will append a new price to existing.
        Returns: dataframe to replace global lastprice.
    """
    # get our new data in a dataframe
    new = pd.DataFrame( {"lastprice":price}, index=[t])
    # do we have any data in our DF?
    if len( self.lastprice) == 0:
      # no. so start us off
      return pd.DataFrame( new, columns=["lastprice"])
    else:
      # is our price the same as the old? and have we gone 30s without a value?
      if ( ( price == self.lastprice.ix[-1][0]) and
          (( t - self.lastprice.ix[-1].name).seconds < 30) ):
        # same. return same ol' bullshit
        return self.lastprice
      else:
        # no ... we got new shit, return updated DF
        return self.lastprice.append( new)

  ###########################
  #      _LOOKBACK
  ###########################
  def _lookback( self, t, time_str, window, buffer):
    """  Calculate correct lookback for each time unit D, min, s, etc
         lookback is the furthest date that we need to grab backwards to
         supply rolling_mean with data to calculate everything.
    """

    # get number supplied in time string
    ns = re.sub("[^0-9]", "", time_str)
    n = int(ns)

    # get time unit as str
    scale = time_str[ time_str.index( ns)+len(ns):]
    
    if self._debug:
      #print "scale, ns, n:", scale, ns, n
      self.scale = scale; self.ns = ns

    # figure out which scale we're in and calculare lookback properly
    if scale == "D":
      lookback = t - datetime.timedelta( days=(( n * window))*buffer)
    elif scale == "min":
      lookback = t - datetime.timedelta( minutes=(( n * window))*buffer)
    elif scale == "s":
      lookback = t - datetime.timedelta( seconds=(( n * window))*buffer)
    if self._debug:
      #print "lookback:", lookback
      self.l = lookback
    return lookback

  ###########################
  #      _MID_DF
  ###########################
  def _mid_df( self, bid, ask, t, name, old):
    """ Calculate price as average of bid and ask. We use this to
        give us a more realistic expectation of what prices we could
        actually get in the market than just last. I can't really decide
        which price quote is better in BTC-e ... last or mid, since there
        ain't no market makers/orders and ask can go up while bid stays
        the same, which would give you negative profit if you sold. Use
        both? Mid tend to be a little smoother with slightly diff highs and
        lows.

        Paramz:
          bid : bid price
          ask : ask price
          t : time of price
          name : name of column to append
          old : old global dataframe
    """
    # calc avg between bid and ask
    price = (bid + ask) / 2.0
    # get our new data in a datafr(bid + ask) / 2.0ame
    new = pd.DataFrame( {name:price}, index=[t])
    # do we have any data in our DF?
    if len( old) == 0:
      # no. so start us off
      return pd.DataFrame( new, columns=[name])
    else:
      # is our price the same as the old? and have
      # we gone less than 30s without a value?
      if ( ( price == old.ix[-1][0]) and
          (( t - old.ix[-1].name).seconds < 30) ):
        # same. return same ol' bullshit
        return old
      else:
        # no ... we got new shit, return updated DF
        return old.append( new)

  ###########################
  #      _NEW_DF
  ###########################
  def _new_df( self, lookback, t, window, time_str):
    """ Return a new, trimmed set of recent prices for use
        in rolling means.
    """
    lookback2 = pl2.nearest_by_date( self.lastprice, lookback, True)
    return pd.rolling_mean( self.lastprice.ix[lookback2.name:].resample( time_str,
                                                          fill_method="ffill"),
                            window,
                            freq=time_str)

  ###########################
  #      _NEW_OHLC_DF
  ###########################
  def _new_ohlc_df( self, lookback, time_str):
    """ Return a new, trimmed set of OHLC based on last prices
    """
    # get nearest index behind lookback
    lookback2 = pl2.nearest_by_date( self.lastprice, lookback, True)
    return self.lastprice.lastprice.ix[lookback2.name:].resample( time_str,
                                                                  how="ohlc")

  ###########################
  #      _OHLC_DF
  ###########################
  def _ohlc_df( self, t, old, time_str):
    lookback = self._lookback( t, time_str, 1, 3)
    if self._debug:
      print "OLD", old
      #self.o = old
    new = self._new_ohlc_df( lookback, time_str)
    if self._debug:
      print "new OHLC:", new
      #self.new = new
    # have we started it?
    if len(old) == 0:
      # no, so return started
      return new
    else:
      # add extra values in new that are not in old
      updated = old.combine_first( new)
      # update values from new into old
      updated.update( new)
      if self._debug:
        print "updated OHLC:", updated
        #self.u = updated
      return updated

  ###########################
  #      _ROLLING
  ###########################
  def _rolling( self, price, t, old, time_str="5min", window=3, type="EMA"):
    """ This will create an initial rolling average
        dataframe or it will generate a new dataframe with an
        updated last N min rolling value. The objective here
        is to not be recalculating the *entire* rolling average
        when we know we're only getting a few new values tacked
        onto the end of our prices.

        price : latest price to add
        t : time of latest price
        old : our old moving average to compare / append to
        time_str : minutes to average, defaults to 5min
        window   : rolling window size (in multiples of time_str chunks),
                   defaults to 3x
        type : do a Simple Moving Average (SMA) or Exponential
               Moving Average (EMA), defaults to SMA
    """
    if self._debug:
      #print "\n_rolling"
      print "old ROLLING", old
      self.o = old
      #print "price:", price
      self.p = price
      #print "t:", t
      self.t = t

    # buffer (extra time to look back just to compare un/changed vals
    # we will multiply this times our window size to make sure we have
    # everything we need in the case of a missing val, etc
    buffer = 3
    
    # get our lookback
    lookback = self._lookback( t, time_str, window, buffer)

    # choose an averaging scheme, then ...
    # calculate our rolling average from the most recent data
    # fill in any holes in our prices to the smallest we might nee
    if type == "SMA":
      new = self._new_df(  lookback, t, window, time_str)
    elif type == "EMA":
      new = pd.ewma( self.lastprice.ix[lookback:].resample( time_str,
                                                         fill_method="ffill"),
                     span=window,
                     freq=time_str)
    if self._debug:
      print "new ROLLING", new
      self.n = new
    # do we have anything in our rolling avg global?
    if len( old) < window:
      # return this as new
      return new
    # if we do, then we need to find where differences start,
    # shave off those from old, append new, return as new global rolling
    else:
      # add extra values in new that are not in old
      updated = old.combine_first( new)
      # update values from new into old
      updated.update( new)
      if self._debug:
        #print "updated:", updated
        self.u = updated
      return updated

  ###########################
  #      _STD
  ###########################
  def _std( self):
    """ Get our n-period standard deviations
    """
    # TODO: update this if we ever want to add multiple OHLC frames
    for time_str in self.ohlc:
      for N in self.std:
        self.std[N] = ind.STD( self.ohlc[time_str].close, N)

######################################################################
#                                                                    #
#                                                                    #
#                                                                    #
#                             D A T A                                #
#                                                                    #
#                                                                    #
#                                                                    #
######################################################################
class Data:
  """ Transparently loads new data from exchanges, either live or from
      disk so that we can simulate and trade IRL using the same
      framework.
  """

  def __init__( self, **kwargs):
    """ Set up our data structures and determine whether we're in
        live or simulated mode.
        
        time_str : (default "5min") time-frame to analyze on ... this
                   controls the length of each "bar" or period, can be
                   any pandas-recognized string, (10s, 10min, 1h, 1d, etc)
        live : live or simulated mode (whether or not to read from
               filename or from the web), defaults to False (simulated)
        filename : name of log file to read in simulated mode ... interpreted
                   as ./logs/filename ... file must be in this dir
        warp : whether or not to use our timedelta or just next value
               for each update() ... so we can do all calculations
               as fast as possible, defaults to False ("realtime")
        debug : whether or not to spit out debugging info
        sample_secs : if in warp-mode, N-seconds to sample on (the shorter
                      N, the more often we are "checking" the price and
                      the more iterations it will take to complete a series)
        instant : (default False) Setting this to true will make Data
                  send the lastprice series to the Coins to calculate all
                  in faster, one-pass mode
        ltc_opts : dict structure on what to do with LTC data ... see coin for
              options from kwargs (default is same as GOX...)
              Here's an example of a fully loaded options dict
              {  "debug": False,
                 "relative": False,
                 "calc_rolling": False,
                 "rolling": { self.time_str : {  5: pd.DataFrame(), 
                                                25: pd.DataFrame(), 
                                                50: pd.DataFrame() } },
                 "calc_mid": False,
                 "calc_ohlc": True,
                 "ohlc": { self.time_str : pd.DataFrame()  },
                 "calc_indicators": True,
                 "indicators":{ "RSI"  : { "data": pd.DataFrame(), "n":14 },
                                "ROC"  : { "data": pd.DataFrame(), "n":20 },
                                "AMA"  : { "data": pd.DataFrame(), "n":10, "fn":2.5, "sn":30 },
                                "CCI"  : { "data": pd.DataFrame(), "n":20 },
                                "FRAMA": { "data": pd.DataFrame(), "n":10 },
                                "RVI2" : { "data": pd.DataFrame(), "n":14, "s":10 },
                                "MACD" : { "data": pd.DataFrame(), "f":12, "s":26, "m":9 },
                                "ADX"  : { "data": pd.DataFrame(), "n":14 },
                                "ELI"  : { "data": pd.DataFrame(), "n":14 },
                                "TMI"  : { "data": pd.DataFrame(), "nb":10, "nf":5} }
                 "calc_std": True,
                 "std": { 10: pd.DataFrame(), 50: pd.DataFrame(), 100: pd.DataFrame() },
                 "calc_crt": True,
                 "crt": { 1: pd.DataFrame(), 2: pd.DataFrame(),
                          3: pd.DataFrame(), 5: pd.DataFrame(),
                          8: pd.DataFrame() },
                 "instant": False,
                 "time_str": self.time_str }
        gox_opts : dict structure on what to do with GOX BTC data ... see coin for
              options from kwargs (default: everything disabled but OHLC ... )
               { "debug": False,
                 "relative": False,
                 "calc_rolling": False,
                 "rolling": False,
                 "calc_mid": False,
                 "calc_ohlc": True,
                 "ohlc": { self.time_str : pd.DataFrame() },
                 "calc_indicators": False,
                 "calc_std": False,
                 "std": False,
                 "calc_crt": False,
                 "crt": False,
                 "instant": False,
                 "time_str": self.time_str }
        pickled_data : (default False) if this is set to a data structure,
                       from pickle'd pandas csv data structure, it'll take
                       it from here instead of from disk. Faster on multiple
                       iterations.
        verbose : (default False) whether or not to print out shit
    """
    self.live = kwargs.get("live", False)
    self.filename = kwargs.get("filename", "test.csv")
    self.warp = kwargs.get( "warp", True)
    self._debug = kwargs.get( "debug", False)
    self.sample_secs = kwargs.get( "sample_secs", 5)
    self.instant = kwargs.get( "instant", False)
    self.time_str = kwargs.get( "time_str", "5min")
    self.verbose = kwargs.get( "verbose", False)
    # default LTC options
    def_ltc =  { "debug": False,
                 "relative": False,
                 "calc_rolling": False,
                 "rolling": False,
                 "calc_mid": False,
                 "calc_ohlc": True,
                 "ohlc": { self.time_str : pd.DataFrame() },
                 "calc_indicators": False,
                 "indicators": False,
                 "calc_std": False,
                 "std": False,
                 "calc_crt": False,
                 "crt": False,
                 "instant": False,
                 "time_str": self.time_str }
    self.ltc_opts = kwargs.get( "ltc_opts", def_ltc)
    # default gox options
    def_gox = { "debug": False,
                 "relative": False,
                 "calc_rolling": False,
                 "rolling": False,
                 "calc_mid": False,
                 "calc_ohlc": True,
                 "ohlc": { self.time_str : pd.DataFrame() },
                 "calc_indicators": False,
                 "indicators": False,
                 "calc_std": False,
                 "std": False,
                 "calc_crt": False,
                 "crt": False,
                 "instant": False,
                 "time_str": self.time_str }
    self.gox_opts = kwargs.get( "gox_opts", def_gox)
    self.pickled_data = kwargs.get( "pickled_data", False)
    
    if self.verbose:
      print "[*]", "Online" if self.live else "Offline", "mode initiated"
      print "[*]", "Simulated" if not self.warp else "Speed", "mode initiated"
    
    # if we're running simulated, set up price logs so we can query them
    # in realtime as if they were actual price changes
    if self.live == False:
      # did we supply a pre-parsed pandas CSV data struct?
      if self.pickled_data != False:
        if self.verbose:
          print "[*]", "Loading supplied pickle!"
        data = self.pickled_data
      # nope ... load from disk!
      else:
        # loading from CSV takes a long time, lets prepare a pickle of the
        # loaded CSV if we haven't already done so, if we have then load it
        filename_pick = os.path.realpath( os.path.join( "logs", self.filename+".pickle"))
        if os.path.exists( filename_pick):
          if self.verbose:
            print "[*]", "Loading csv pickle from %s" % filename_pick
          f = open( filename_pick, "rb")
          data = cPickle.load( f)
          f.close()
        else:
          filename_csv = os.path.realpath( os.path.join( "logs", self.filename))
          if self.verbose: print "[*] Loading %s" % filename_csv
          data = pl2.load2( filename_csv)
          if self.verbose: print "[*] Generating pickle for next time to %s" % filename_pick
          f = open( filename_pick, "wb")
          cPickle.dump( data, f)
          f.close()

      # load our time-series dataframe from csv using pandas library
      self._gox_offline = data["gox"]
      self._ltc_offline = data["ltc"]
      self._ltc_depth_offline = data["ltc_depth"]
      
      # if we're running in non-simulated offline mode, where we just
      # want to run through our historical price data as quickly as
      # possible, then we build a range of dates that we will walk through
      if self.warp == True:
        # get our start and end points in our timerange
        start = max( [ self._gox_offline.index[0], self._ltc_offline.index[0]])
        end = max( [ self._gox_offline.index[-1], self._ltc_offline.index[-1]])
        
        # our list of total dates to run through
        # jump to N-seconds intervals (self.sample_secs)
        if self.verbose:
          print "[*]","Building daterange"
        self.logrange = self._daterange( start, end, self.sample_secs)
        
        # we're going to need to iterate through this one at a time ...
        # get new values, calculate indicators, train, repeat, so we'll
        # need to keep track of where we are
        self.logrange_n = 0
        if self.verbose:
          print "[*] Dates from", start, "to", end

      # otherwise we pretend we're live (slow so we can watch it IRT)
      else:
        # find out which has the earliest starting date. We will use
        # this to calculate our timedelta. In the future when we want
        # to check the price, we will use this delta compared to current
        # time to grab the proper simulated price
        # (we use max here so we don't get any initial NaN prices if possible)
        self.delta = datetime.datetime.today() - max( [ self._gox_offline.index[0], 
                                                        self._ltc_offline.index[0]])

        if self.verbose: print "[*] Timedelta: %s" % self.delta
        
    #####################################
    #                                   #
    #            C O I N S              #
    #                                   #
    ##################################### 

    # prepare instant if necessary
    if self.instant:
      # seed prices with midprice
      if self.ltc_opts["calc_mid"]:
        filename = os.path.realpath( os.path.join( "logs", 
                     self.filename+".midprices.pickle"))
        # if midprices pickle doesn't exist, we need to generate it ... this is slow as fuck
        # so we really want to have this preloaded
        if os.path.exists( filename):
          if self.verbose: print "[*]", "Loading midprices from %s" % filename
          f = open( filename, "rb")
          bas = cPickle.load( f)
        else:
          if self.verbose: print "[*]","Calculating midprices ..."
          bas = [ pl2.bid_ask(self._ltc_depth_offline.ix[i][0], 
                  avg=True) for i in xrange( len( self._ltc_depth_offline))]
          f = open( filename, "wb")
          if self.verbose: print "[*]", "Saving midprices to %s" % filename
          cPickle.dump( bas, f)
        self.ltc_opts["instant"] = pd.DataFrame( {"lastprice":bas}, 
                                     index=[self._ltc_depth_offline.index])
      # otherwise hand it lastprice
      else:
        self.ltc_opts["instant"] = self._ltc_offline

    self.ltc = Coin( debug=self.ltc_opts["debug"],
                     relative=self.ltc_opts["relative"],
                     calc_rolling=self.ltc_opts["calc_rolling"],
                     rolling=self.ltc_opts["rolling"],
                     calc_mid=self.ltc_opts["calc_mid"], 
                     calc_ohlc=self.ltc_opts["calc_ohlc"],
                     ohlc=self.ltc_opts["ohlc"], 
                     calc_indicators=self.ltc_opts["calc_indicators"],
                     indicators=self.ltc_opts["indicators"],
                     calc_std=self.ltc_opts["calc_std"], 
                     std=self.ltc_opts["std"],
                     calc_crt=self.ltc_opts["calc_crt"], 
                     crt=self.ltc_opts["crt"],
                     instant=self.ltc_opts["instant"], 
                     time_str=self.ltc_opts["time_str"],
                     verbose=self.verbose)

    # for gox, all I want to calculate is the EMA of the last prices ...
    # I chose last price, not mid, because I think that a lot of people
    # are trading based on the last price ticker, not where the market
    # really is.
    # prepare instant if necessary
    # prepare instant if necessary
    if self.instant:
      # seed prices with midprice
      if self.gox_opts["calc_mid"]:
        if self.verbose: print "[*]","Calculating midprices ..."
        bas = [ pl2.bid_ask(self._gox_depth_offline.ix[i][0], avg=True) for i in xrange( len( self._gox_depth_offline))]
        self.gox_opts["instant"] = pd.DataFrame( {"lastprice":bas}, index=[self._gox_depth_offline.index])
      # otherwise hand it lastprice
      else:
        self.gox_opts["instant"] = self._gox_offline

    self.gox = Coin( debug=self.gox_opts["debug"], 
                     relative=self.gox_opts["relative"],
                     calc_rolling=self.gox_opts["calc_rolling"], 
                     rolling=self.gox_opts["rolling"],
                     calc_mid=self.gox_opts["calc_mid"], 
                     calc_ohlc=self.gox_opts["calc_ohlc"],
                     ohlc=self.gox_opts["ohlc"], 
                     calc_indicators=self.gox_opts["calc_indicators"],
                     indicators=self.gox_opts["indicators"],
                     calc_std=self.gox_opts["calc_std"], 
                     std=self.gox_opts["std"],
                     calc_crt=self.gox_opts["calc_crt"], 
                     crt=self.gox_opts["crt"],
                     instant=self.gox_opts["instant"], 
                     time_str=self.gox_opts["time_str"],
                     verbose=self.verbose)

  def update( self):
    """ Grab most recent prices from on/offline and append them to
        our exchange data structures.
    """
    #######################################################
    #               -- SIMULATION MODE --                 #
    #######################################################
    # simulation mode. pull most recent price from our logs and
    # append if different
    if self.live == False:
      #######################################################
      #          -- REAL TIME SIMULATION MODE --            #
      #######################################################
      # if warp is false, we will pretend this is realtime and
      # grab prices from our logs using our timedelta
      if self.warp == False:
        # calculate our timedelta from NOW!!
        adjusted_t = datetime.datetime.today() - self.delta

        # Get our last prices from the logs
        last_gox , last_ltc , last_ltc_depth = self._offline_prices( adjusted_t)

        # make sure we got a timeseries object back, otherwise we
        # hit the end of the log
        if( type(last_gox) != pd.Series or
            type(last_ltc) != pd.Series or
            type(last_ltc_depth) != pd.Series):
          if self.verbose: print "[!]", "End of log."
          return False
        # we have values, so add them to each coin
        else:
          # give coins new price changes ... them bitches'll do the rest
          self.gox.add( last_gox[0], last_gox.name)
          # bid-ask avg for LTC only
          ba = pl2.bid_ask( last_ltc_depth[0])
          self.ltc.add( last_ltc[0], last_ltc.name, ba=ba)
          return True
      #######################################################
      #               -- FAST MODE --                       #
      #######################################################
      # otherwise, we'll grab our next price from the index
      else:
        # r we about to do something stupid? (hit end of the fucking log)
        if self.logrange_n >= len(self.logrange):
          if self.verbose: print "[!]", "End of log."
          return False
        # NO!
        else:
          # get our next date in our time index & grab the prices
          t = self.logrange[self.logrange_n]
          if self._debug:
            print "\n_update"
            print "t:", t
            print "logrange:", self.logrange_n

          last_gox, last_ltc, last_ltc_depth = self._offline_prices( t)
          # get LTC market data (bid ask)
          ba = pl2.bid_ask( last_ltc_depth[0])
          
          # upd8 fuk'n coinz
          if self._debug:
            print "\n_update"
            print "\nltc"
            print "last_ltc:", last_ltc[0], last_ltc.name
            print "ba:", ba
          
          self.ltc.add( last_ltc[0], last_ltc.name, ba=ba)
          if self._debug:
            print "\ngox"
            print "last_gox:", last_gox[0], last_gox.name
          self.gox.add( last_gox[0], last_gox.name)

          # increment for the next fucking time
          self.logrange_n += 1
          return True

  def _daterange(self, start_date, end_date, step=5):
    """ Give us a list of dates and times to run through in non-sim,
        offline mode.
        step : write a date every N seconds
    """
    total_seconds = ((end_date - start_date).days) * 86400
    total_seconds += ((end_date - start_date).seconds)
    return [ (start_date + datetime.timedelta(seconds=int(n))) for n in np.arange(0, total_seconds, step)]

  def _offline_prices( self, dt):
    """ Return last offline prices
    """
    # Get our last prices from the logs
    last_gox = pl2.nearest_by_date( self._gox_offline,
                                    dt, True)
    last_ltc = pl2.nearest_by_date( self._ltc_offline,
                                            dt, True)
    last_ltc_depth = pl2.nearest_by_date( self._ltc_depth_offline,
                                              dt, True)
    return last_gox, last_ltc, last_ltc_depth


'''
## -- SAMPLE BOX -- ##
import data
import datetime
import indicators as ind
from matplotlib import pylab as plt

data = reload(data)
ind = reload(ind)
d = data.Data( warp=True)
start = datetime.datetime.now()
while d.update():
  if d.logrange_n % 100 == 0:
    print d.logrange_n, ((datetime.datetime.now() - start).seconds)/100.0, "per iteration"
    start = datetime.datetime.now()
  if d.logrange_n == 1000:
    d.ltc.lastprice.join( [ d.ltc.rolling["30s"][12].rename( columns={"lastprice":"ema8"}), d.ltc.rolling["30s"][24].rename( columns={"lastprice":"ema12"}), d.ltc.rolling["30s"][50].rename( columns={"lastprice":"ema50"}), d.ltc.midprice ], how="outer").ffill().plot(); plt.show()
    d.ltc.lastprice.join( [ d.ltc.ti["AMA"] ], how="outer").ffill().plot(); plt.show()
    break

## -- PLOT -- ##
# prices & rollings
d.ltc.lastprice.join( [ d.ltc.rolling["1min"][8].rename( columns={"lastprice":"ema8"}), d.ltc.rolling["1min"][12].rename( columns={"lastprice":"ema12"}), d.ltc.rolling["1min"][50].rename( columns={"lastprice":"ema50"}), d.ltc.midprice ], how="outer").ffill().plot(); plt.show()
# all indicators
d.ltc.lastprice.join( [d.ltc.ti["AMA"], d.ltc.ti["RSI"], d.ltc.ti["ROC"], d.ltc.ti["CCI"]], how="outer").ffill().plot(subplots=True); plt.show()

# test AMA
ama = pd.DataFrame()
for i in range( len( d._ltc_offline)): ama = ind.AMA( ama, pd.DataFrame(), d._ltc_offline.rename( columns={"ltc_last":"lastprice"}).ix[0:i+1])

d._ltc_offline.join( [ ama ], how="outer").ffill().plot(); plt.show()


# test AMA
ama = pd.DataFrame()
for i in range( len( d._ltc_offline)): ama = ind.AMA( ama, pd.DataFrame(), d._ltc_offline.rename( columns={"ltc_last":"lastprice"}).ix[0:i+1])

d._ltc_offline.join( [ ama ], how="outer").ffill().plot(); plt.show()

# test CCI
ind = reload(ind)
cci = pd.DataFrame()
for i in range(len(ohlc)):
  cci = ind.CCI( ohlc.ix[0:i+1], cci)


cci.join( [ pd.DataFrame( ohlc.close) ], how="outer").ffill().plot(subplots=True); plt.show()
'''
