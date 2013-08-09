import processlogs2 as pl2
import datetime
import pandas as pd
import numpy as np


###########################
#  AVERAGE DIRECTIONAL MOVEMENT INDEX
###########################
def ADX( ohlc, n=14 ):
  """ The ADX measures the strength of a trend, not the direction,
      from OHLC data over n-periods. Readings below 20 show little
      to no trend and values over 40 show strong trends.

      PARAMETERS:
      ohlc : pandas OHLC-formatted dataframe
      n : n-periods to calculate ADX over
  """
  # get our true range
  TR = pd.DataFrame( {"h-l":ohlc.high - ohlc.low,
                      "h-cp":np.abs( ohlc.high - ohlc.close.shift(1)),
                      "l-cp":np.abs( ohlc.low  - ohlc.close.shift(1))}, index=[ohlc.index])

  TR = pd.Series( np.max( TR.values, axis=1 ), index=[ohlc.index])

  # n-period EMA makes this average true range (ATR)
  ATR = pd.ewma( TR, span=n, min_periods=n )

  # get our UDM (+ Directional Movement) and DDM (-DM)
  UM = ohlc.high - ohlc.high.shift(1)
  DM = ohlc.low.shift(1) - ohlc.low

  # get +DM and -DM ... all positive or zero vals
  for i in xrange( len(ohlc)):
    if ( UM.ix[i] > DM.ix[i]) and ( UM.ix[i] > 0):
      #UDM.ix[i] = UDM.ix[i]
      pass
    else:
      UM.ix[i] = 0
    if ( DM.ix[i] > UM.ix[i]) and ( DM.ix[i] > 0):
      pass
    else:
      DM.ix[i] = 0

  # N-period EMA of both
  UDI = ( 100 * pd.ewma( UM, span=n, min_periods=n)) / ATR
  DDI = ( 100 * pd.ewma( DM, span=n, min_periods=n)) / ATR

  _ADX = 100 * pd.ewma( np.abs( (UDI - DDI) / ( UDI + DDI)), span=n, min_periods=n)

  return pd.DataFrame( {"ADX_%s"%n:_ADX}, index=[ohlc.index])

###########################
#  ADAPTIVE MOVING AVERAGE
###########################
def AMA( df, n=10, fn=2.5, sn=30):
  """ Get an adaptive moving average 

      NOTE: ohlc & lastprice must represent the most up-to-date
              prices, as we will be using the last index as the
              last price. This works one-pass.

      df : the dataframe we want to run the adapive MA on ...
           YO ... THIS NEEDS TO BE PROPERLY SAMPLED!
           
      old_ama : the old AMA DF ... technically AMA(t-1)
      ohlc : an updated to current t, OHLC DF
      price : our lastprice DF, also must be up-to-date
      n = (default=10) n periods to calculate volitility sum
      fn = (default 2.5) fast moving average periods
      sn = (default 30) slow moving average periods
  """
  #if we got a TimeSeries, convert it to a DF
  if type( df) == pd.TimeSeries:
    df = pd.DataFrame( {df.name:df.values}, index=[df.index])
    
  # don't fuck wit it if we aint got the info to work
  if len( df) < n:
    return pd.DataFrame()
  else:
    # get the difference between our last price and last price t - 1
    directions = df - df.shift(n) # correct
    
    # get the volitility ... calulated as the sum of the absolute
    # value of the last n periods
    abs_diff = np.abs(df - df.shift(1))
    sums = np.zeros( len(abs_diff))
    sums[:n] = np.NaN

    for ii in xrange( n, len(abs_diff)):
      sums[ii] = abs_diff.ix[(ii-n)+1:ii+1].values.sum()
    
    # n-perod volitility for each tick in our DF
    volatilities = pd.DataFrame( {df.columns[0]:sums}, index=[directions.index])
    # division by 0 REALLY fucks this up, so sub a ridiculously small val
    volatilities = volatilities.replace( 0, 1e-100)
    
    # efficiency ratios
    ERs = np.abs( directions / volatilities)
    
    # fast and slow constants (based on fast n-periods)
    FC = 2.0 / (fn + 1)
    SC = 2.0 / (sn + 1)
    
    # calculate our smoothing constant, alpha squared
    alphas = np.power( ( ( ERs * ( FC - SC)) + SC ), 2 )
    
    ama = np.zeros( len(df))
    
    # get first index with a non-NaN value
    iii = 0
    for iii in xrange( len(ERs)):
      if not np.isnan( ERs.ix[iii][0]):
        break
    
    # get an average to start our AMA with
    ama[iii-1] = df.ix[:n].values.mean()
    ama[:iii-1] = np.NaN
    
    for i in xrange( iii, len( ama)):
      ama[i] = ( ERs.ix[i][0] * ( df.ix[i][0] - ama[i-1] ) ) + ama[i-1]
    
    ama_df = pd.DataFrame( {"AMA": ama}, index=[df.index])
    assert False == np.isnan(ama_df[n+1:]).any()
  return ama_df

###########################
#  COMMODITY CHANNEL INDEX
###########################
def CCI( ohlc, n=20):
  """ Calculates the CCI on our OHLC data.
      NOTE: ohlc must represent the most up-to-date
          prices, as we will be using the last index as the
          last price. This can be run in one pass.

      ohlc : current ohlc data. columns must include "high",
             "low", and "close"
      n : (default 20) number of periods to calculate over.
          the longer the n, the less volitile the oscillator
          and vice-a-versa
  """
  n_ohlc = len( ohlc)
  cci_tmp = np.zeros( n_ohlc)
  # do we have enough data?
  if n_ohlc < n:
    return pd.DataFrame()
  # we have enough, lets ROLL!!
  else:
    # vectorized
    # our typical prices
    typical_price = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
    # SMA of our typical prices
    SMAtp = pd.rolling_mean( typical_price, window=n)
    # mean absolute deviation

    # get each period's mean
    means = np.zeros( len(typical_price))
    means[:n-1] = np.NaN
    
    for i in xrange( n-1, len(typical_price)):
      means[i] = typical_price.ix[(i-n)+1:i+1].values.mean()

    # MAD = absolute difference between each point in
    # the period from the period's mean, averaged
    MADs = np.zeros( len(typical_price))
    MADs[:n-1] = np.NaN
    for i in xrange( n-1, len(typical_price)):
      MADs[i] = np.abs(typical_price.ix[(i-n)+1:i+1] - means[i]).mean()

    cci = pd.DataFrame( {"CCI_%s"%n: ((1 / 0.015) * ( ( typical_price - SMAtp) / MADs ))},
                        index=[typical_price.index] )

    # if we have any NaN values in the body of our CCI then we have problemz
    if np.isnan(cci[n+1:]).any():
      cci = cci[n+1:].replace( np.NaN, 0)
    assert False == np.isnan(cci[n+1:]).any()
    # say hello 2 our little friend
    return cci

###########################
#  COMPOUND RETURN OHLC
###########################
def CRT( ohlc, n):
  """ Logarithmic compound return over N-periods. Reworks entire DF 
      in one-pass. Calculates return as btwn open and close like so
      log( close ) - log( open t-n) ... this is based off the RNN FX
      article and they aren't dividing by N periods, so I won't here.

      df : DataFrame or TimeSeries
      n : number of periods to calculate over ... this gets subtracted
          by 1 so that when you input n=1 (last period) you don't get
          the crt of the open before the current (most recent) period.

          For example, assume :50 is the most recent ohlc. An n of 1 will
          calculate this where [] is the initial and {} is the final

          2013-05-29 05:30:00  2.977990   2.977990  2.976050   2.976050
          2013-05-29 05:40:00  2.976050   2.984300  2.976050   2.984299
          2013-05-29 05:50:00 [2.984299]  2.996333  2.980167  {2.991051}

          An n of 2 would put the start one above where the bracket is,
          resulting in the return if you'd held for the entire two periods.
  """
  # we need n+1 values to compute this
  if len(ohlc) > n+1:
    # get latest price and date
    crt = np.log(ohlc.close) - np.log(ohlc.open.shift(n-1))
    crt_df = pd.DataFrame( {"CRT_%s"%n:crt}, index=[ohlc.index])
    # this is a bad hack ... but apparently there are some oddities
    # making the final value appear to be inf or nan, caused by a
    # zero-val OHLC, we're going to make it 0 crt
    if np.isnan( crt_df.ix[-1]) or np.isinf( crt_df.ix[-1]):
      crt_df.ix[-1] = 0
    #assert False == np.isnan(crt_df[n+1:]).any()
    return crt_df
  else:
    return pd.DataFrame()

###########################
#  EHLER'S LEADING INDICATOR
###########################
def ELI( ohlc, n=14):
  """ This is the Ehler's Leading Indicator ... it predicts cyclical changes
      in price using a tripple EMA scheme. It first calculates two EMAs on the
      original price: one that's 1/2 n and the other 1/4 n. Then those are
      subtracted, giving us a synthetic price. This is smoothed with an EMA
      of 1/4 n. The ELI is calculated by subtracting the syn from the syn's EMA.
      This leads cyclical changes, assuming they exist in the data.

      Parameters:
      ohlc : OHLC dataframe to take our prices from. Right now we're using close,
             but theoretically this could be anything.
      n : n-period cycles in our data. Tuning this parameter totally depends
          on the market/dataset, but I found 14 to work well for now.
  """
  a = n / 4.0
  _EMA1 = pd.ewma( ohlc.close, span=a)

  a = n / 2.0
  _EMA2 = pd.ewma( ohlc.close, span=a)

  syn = _EMA1 - _EMA2
  _EMAsyn = pd.ewma( syn, span=a)
  _ELI = syn - _EMAsyn

  return pd.DataFrame( { "ELI_%s"%n: _ELI }, index=[ohlc.index])

###########################
#  FRACTAL ADAPTIVE MOVING AVERAGE
###########################
def FRAMA( ohlc, n=10):
  """ Get an adaptive moving average 

      NOTE: ohlc must represent the most up-to-date
            prices, as we will be using the last index as the
            last price. This works one-pass.

      ohlc : an updated to current t, OHLC DF
      n = (default=10) n periods to calculate over
  """
  # n must be even
  if n % 2 == 1:
    print "[!]", "FRAMA n must be even. Adding one"
    n += 1
  # don't fuck wit it if we aint got the info to work
  if len( ohlc) < n:
    return pd.DataFrame()
  else:
    # iterate through n-period chunks, calculate alpha values
    alphas = np.zeros( len(ohlc))
    alphas[:n] = np.NaN
    for i in xrange( n, len( ohlc)):
      per = ohlc.ix[i-n:i]
      N1 = ( np.max( per.ix[0:n/2].high) - np.min( per.ix[0:n/2].low)) / ( n / 2)
      N2 = ( np.max( per.ix[n/2:].high) - np.min( per.ix[n/2:].low)) / ( n / 2)
      N3 = ( np.max( per.high) - np.min( per.low)) / ( n)
      D = ( np.log(N1 + N2) - np.log(N3)) / np.log( 2) if (N1 > 0) and (N2 > 0) and (N3 > 0) else 0
      alphas[i] = min( max( np.exp( -4.6*(D-1)), 0.01), 1) # keep btwn 1 & 0.01

    # get an average to start our AMA with
    frama = np.zeros( len(ohlc))
    frama[n-1] = ohlc.ix[:n].values.mean()
    frama[:n-1] = np.NaN

    for i in xrange( n, len( frama)):
      frama[i] = ( alphas[i] * ( ohlc.ix[i]["close"] - frama[i-1] ) ) + frama[i-1]

    frama_df = pd.DataFrame( {"FRAMA": frama}, index=[ohlc.index])
    assert False == np.isnan(frama_df[n+1:]).any()

    return frama_df

###########################
#     MACD
###########################
def MACD( ohlc, f=12, s=26, m=9):
  fast_ema = pd.ewma( ohlc.close, span=f)
  slow_ema = pd.ewma( ohlc.close, span=s)
  macd = fast_ema - slow_ema
  macd_sig = pd.ewma( macd, span=9)
  hist = macd - macd_sig
  return pd.DataFrame( {"MACD_hist":hist, "MACD":macd_sig}, index=[ohlc.index])

###########################
#     NORMALIZE
###########################
def normalize( df, std_devs=1):
  """ normalization to zero mean and arbitary standard deviations
      df : the dataframe to normalize
      std_devs : (default 1) Number of std devs to normalize to
  """
  # multi-column: df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
  return (df - df.mean()) / (df.std() * std_devs)

###########################
#  PRICE CHANNEL INDEX
###########################
def PCI( ohlc, n=20):
  """ This will look for "price channel breakouts" ...
      where the current price / close is higher than
      the high from the last 20 periods.

      df : a dataframe with OHLC data
      n : (default 20) n-periods to look back
  """
  pass

###########################
#  RATE OF CHANGE
###########################
def ROC( ohlc, n=20):
  """ This gets the rate of change (aka momentum)
      indicator from OHLC data. Basically we're ghetto
      caculating the derrivative of the closing price
      slope. NOTE: ohlc must represent the most up-to-date
      prices, as we will be using the last index as the
      last price. This is one-pass safe.

      ohlc : an ohlc DataFrame
      n : (defaults to 20) n-periods to calculate over
  """
  c = ohlc.close
  c_n_ago = ohlc.close.shift(n)
  # get momentum
  momentum = c - c_n_ago
  roc = momentum / c_n_ago
  # if we have any NaN values in the body of our ROC then we have problemz
  assert False == np.isnan(roc[n+1:]).any()
  return pd.DataFrame( roc, index=ohlc.index, columns=["ROC_%s"%n])

###########################
#  RELATIVE STRENGTH INDEX
###########################
def RSI( ohlc, n=14):
  """ This calculates our Relative Strength
      Index. This can be ran in one-pass.

      ohlc : a DF containing our pandas OHLC data,
             this must contain a column named "close"
             and must represent the most up-to-date prices, as
             we will be using the last index as the last price
      n : (default 14) n-periods to calculate RSI's EMA on
  """
  # set up our vars so we can build our U/D arrays
  n_ohlc = len(ohlc)
  U = np.zeros(n_ohlc)
  D = np.zeros(n_ohlc)
  # build our Up/Dwn arrays
  for i in xrange( len( ohlc)):
    # close(now) and close(prev)
    now = ohlc.close[i]
    prev = ohlc.close.shift(1)[i]
    # if we're in am upward movement
    if now > prev:
      U[i] = now - prev
      D[i] = 0
    # downward move
    elif now < prev:
      U[i] = 0
      D[i] = prev - now
    # equal? (no change)
    else:
      U[i] = 0
      D[i] = 0
    # calulate our RS
  # calculate RS as ratio of the SMA(U,n) and SMA(D,n)
  # here we're using cutler's RSI because the EMA(D|U,n)
  # version is sensitive to where in the data the RSI begins
  RS = ( pd.rolling_mean( U, window=n, min_periods=n-1) /
         pd.rolling_mean( D, window=n, min_periods=n-1) )
  # convert to 0-100 indicator
  _RSI = 100 - ( 100 / (1 + RS))
  # n-period smoothing
  _RSI = pd.ewma( _RSI, span=n)
  # if we have any NaN values in the body of our ROC then we have problemz
  assert False == np.isnan(_RSI[n+1:]).any()
  # convert back into DF
  return pd.DataFrame( _RSI, index=ohlc.close.index, columns=["RSI_%s"%n] )

###########################
#  RELATIVE VOLUME INDEX (w/ Inertia)
###########################
def RVI2( ohlc, n=14, s=10):
  """ This calculates the RVI w/ inertia, which is a complimentary indicator
      to the RSI, which only measures price movements. Instead of price
      action, this measures the strength of the trend using standard devs.

      ohlc : pandas dataframe w/ ohlc data
      n : (default 14) N-period smoothing
      s : (default 10) S-period std deviation
  """
  #S = 10 # periods to get stddev over
  #N = 14 # smoothing period
  # RVIH
  USD = np.zeros( len(ohlc)) # up std dev
  USD[:s] = np.NaN
  DSD = np.zeros( len(ohlc)) # dwn std dev
  DSD[:s] = np.NaN
  for i in np.arange( s, len( ohlc)):
    USD[i] = ohlc.high.ix[i-s:i].std() if ohlc.high.ix[i] > ohlc.high.shift(1).ix[i] else 0
    DSD[i] = ohlc.high.ix[i-s:i].std() if ohlc.high.ix[i] < ohlc.high.shift(1).ix[i] else 0

  U = pd.ewma( USD, span=(n*2)-1, min_periods=n)
  D = pd.ewma( DSD, span=(n*2)-1, min_periods=n)
  RVIH = 100 * ( U / (U + D) )

  # RVIL
  USD = np.zeros( len(ohlc)) # up std dev
  USD[:s] = np.NaN
  DSD = np.zeros( len(ohlc)) # dwn std dev
  DSD[:s] = np.NaN
  for i in np.arange( s, len( ohlc)):
    USD[i] = ohlc.low.ix[i-s:i].std() if ohlc.low.ix[i] > ohlc.low.shift(1).ix[i] else 0
    DSD[i] = ohlc.low.ix[i-s:i].std() if ohlc.low.ix[i] < ohlc.low.shift(1).ix[i] else 0

  U = pd.ewma( USD, span=(n*2)-1, min_periods=n)
  D = pd.ewma( DSD, span=(n*2)-1, min_periods=n)
  RVIL = 100 * ( U / (U + D) )

  # new RVI (w/ inertia)
  _RVI2 = (RVIH + RVIL) / 2

  return pd.DataFrame( {"RVI2_%s"%n:_RVI2}, index=[ohlc.index])

###########################
#  STANDARD DEVIATIONS
###########################
def STD( df, n):
  """ Send a DataFrame or TimeSeries and we'll
      spit an N-period standard deviation back
      in yo face, appended to old STD DF. Calcs
      all in one pass.

      df : DataFrame or TimeSeries
      n : number of periods to calculate stddev over
  """
  std = pd.rolling_std( df, window=n, min_periods=n)
  return pd.DataFrame( {"STD": std}, index=[std.index])

###########################
#  TREND MOVEMENT INDEX
###########################
def TMI( ohlc, nb=10, nf=5):
  n = nb+nf
  PTMz = np.zeros( len(ohlc)); PTMz[:n] = np.NaN
  NTMz = np.zeros( len(ohlc)); NTMz[:n] = np.NaN
  TMIz = np.zeros( len(ohlc)); TMIz[:n*2] = np.NaN

  for i in np.arange( n, len(ohlc)):
    # get the lookback period ... we use this to calculate
    # bandwidth (from lookback midpoint)
    ohlc_lb = ohlc.ix[i-nf-nb:i-nf]
    midpoint = np.max( (ohlc_lb.high + ohlc_lb.low) / 2 )
    # user-set bandwidth
    #bw_h = midpoint + (bw / 2.0)
    #bw_l = midpoint - (bw / 2.0)
    # auto bendwidth based on past high/low
    bw_h = np.mean(ohlc_lb.high)
    bw_l = np.mean(ohlc_lb.low)
    # now we want the lookforward period to determine how
    # much change has happened compared to the lb period
    ohlc_lf = ohlc.ix[i-nf:i]
    #
    # trend movement defined as largest price movement outside
    # of the bandwidth in the lookforward period
    PTM = ( np.max(ohlc_lf.high) - bw_h)
    NTM = ( bw_l - np.min(ohlc_lf.low))
    TM = np.max( [ PTM, NTM, 0])
    #
    if TM == PTM:
      PTMz[i-1] = TM; NTMz[i-1] = 0
    elif TM == NTM:
      NTMz[i-1] = TM; PTMz[i-1] = 0
    elif TM == 0:
      NTMz[i-1] = 0; PTMz[i-1] = 0

  # TMI is calculated by taking the sums of the NTMs and
  # PTMs over n and subtracting them, this difference is divided
  # by the sum of both NTM and PTM over n days
  for i in np.arange( n, len(ohlc)):
    tmitmp = ( np.nansum(PTMz[i-n:i]) - np.nansum(NTMz[i-n:i])) / np.nansum(NTMz[i-n:i] + PTMz[i-n:i])
    if np.isnan( tmitmp):
      TMIz[i-1] = 0
    else:
      TMIz[i-1] = tmitmp

  return pd.DataFrame( {"TMI":TMIz}, index=[ohlc.index])