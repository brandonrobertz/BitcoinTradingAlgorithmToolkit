#/usr/bin/python
import numpy as np
import pandas as pd

def gain( iamt, icost, fcost):
  """ Simulate gain from trade.

      Parameters:
        iamt  : initial amount of coin purchased
        icost : initial price per coin
        fcost : final price per coin

      Returns:
        Amount of profit or loss to expect from the trade
  """
  spent = iamt * icost
  btcminusfee = iamt - (iamt * 0.006)
  sold = btcminusfee * fcost
  ret = sold - spent
  return ret

def profit( prices, signals):
  """ For a given series of ticker prices, simulate how much one would
      gain or lose based on trading signals. The prices array must align
      with the signals array. The simulation will only allow one trade
      at a time.

      Parameters:
        prices  : a dataframe of coin ticker prices
        signals : a dataframe of trading signals where -1 is sell, 0 is do
                  nothing, and 1 is buy.

      Returns:
        Amount of profit or loss generated using the strategy.
  """
  stop_loss   = -0.10
  periods     =  0.0
  max_hold    = False
  holding     = False
  returns     =  0.0
  prof_trades =  0.0
  trades      =  0.0
  down        =  0.0
  if type( signals) == pd.Series:
    if signals.size > 1:
      print "Signals must be a single column"
      return
  elif type(signals) == pd.DataFrame:
    if len(signals.columns) > 1:
      print "Signals must be a single column"
      return
  for i in xrange( len(prices)):
    if type(signals) == pd.DataFrame:
      signal = signals.ix[i][0]
    else:
      signal = signals.ix[i]
    # if we're not holding, consider buying
    if not holding:
      if signal == 1:
        bought = prices.ix[i]
        if type(bought) == pd.Series:
          bought = bought[0]
        holding = bought
        periods = 0
    # if we're holding, consider selling
    else:
      current = prices.ix[i]
      if type(current) == pd.Series:
        current = current[0]
      if signal == -1:
        sold = current
        # add to returns
        ret = gain( 1, holding, sold)
        returns += ret
        holding = False
        trades += 1
        if np.sign(ret) > 0:
          # increment trade counter
          prof_trades += 1
        elif ret < down:
          down = ret
      else:
        periods += 1
  
  
  if trades:
    prof_trades = (prof_trades / trades) * 100
  return returns, prof_trades, trades, down