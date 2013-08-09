#!/usr/bin/python
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import csv
import pandas as pd

# loads our raw csv file and returns separated
# individual exch data frames
def load2( filename):
  gox = pd.read_csv( filename, parse_dates=True, usecols=[0,2])
  gox.index = pd.to_datetime( gox.pop("gox_time"))
  btc_usd = pd.read_csv( filename, parse_dates=True, usecols=[3,5])
  btc_usd.index = pd.to_datetime( btc_usd.pop("btc_usd_time"))
  ltc_btc = pd.read_csv( filename, parse_dates=True, usecols=[6,8])
  ltc_btc.index = pd.to_datetime( ltc_btc.pop("ltc_btc_time"))
  ltc = pd.read_csv( filename, parse_dates=True, usecols=[9,11])
  ltc.index = pd.to_datetime( ltc.pop("ltc_time"))
  ltc_depth = pd.read_csv( filename, parse_dates=True, usecols=[12,13])
  ltc_depth.index = pd.to_datetime( ltc_depth.pop("ltc_depth_time"))
  return {"gox":gox, "btc_usd":btc_usd,
          "ltc_btc":ltc_btc, "ltc":ltc,
          "ltc_depth":ltc_depth}

# combine exchange charts into one
def merge_all( data):
  all = [ data[ddd] for ddd in data]
  return all[0].join( all[1:])

# undo a Base64'd BZ2 string
def decompress(encoded_str):
  return encoded_str.decode("base64").decode("bz2")

# load our depth chart to JSON
def load_depth( depth_str):
  return json.loads( decompress(depth_str))

# difference between buy and sell from encoded depth
def depth_diff( depth_json):
  ask = 0.0
  bid = 0.0
  for val, vol in depth_json["bids"]: bid += val * vol
  for val, vol in depth_json["asks"]: ask += val * vol
  return ask - bid

# difference between buy and sell from encoded depth
def shallow_depth_diff( depth_json):
  ask = 0.0
  bid = 0.0
  for val, vol in depth_json["bids"]: bid += val * vol
  for val, vol in depth_json["asks"]: ask += val * vol
  return ask - bid

# return buy-ask from encrypted market depth
def bid_ask( depth_str, **kwargs):
  avg = kwargs.get("avg", False)
  undone = load_depth( depth_str)
  if avg:
    return (undone["bids"][0][0] + undone["asks"][0][0])/2
  else:
    return undone["bids"][0][0], undone["asks"][0][0]

# convert our depth dataframe to a
# depth breakdown of our choosing
def convert_depth_df( depth_df, type):
  if type == "ddiff":
    ret = [ depth_diff(load_depth(ld))
              for ld in depth_df.ltc_depth.values]
  elif type == "bid-ask":
    ret = [ bid_ask(load_depth(ld))
            for ld in depth_df.ltc_depth.values]
  depth_index = [ ld for ld in depth_df.ltc_depth.index]
  return pd.DataFrame( ret,
                       index=depth_index, 
                       columns=[type])

def plot_indiv( x, y, c, l):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylabel(l)
  ax.plot( x, y, color=c, label=l)

# change to difference in price
def changes( data):
  return data.diff()

# this will replace 0 values with the prior value
def clean_missing_avgs( data):
  return data.ffill()

# this will take our individual timestamped
# data and group it by arbitrary timespans
# i.e., 10s, 1min, 1H, 1D, 1M
def average_by_time( data, time):
  return data.resample(time, how=['mean'])

# turn pandas data into a rolling mean
def rolling_averages( data):
  return pd.rolling_mean( data)

# plot a few charts for our viewing pleasure
def plot_charts( data):
  # first possible indicator:
  #   5min average gox price -> litecoin changes soon after
  #data["gox"].join(data["ltc"]).pct_change().resample( "5min", how=["mean"]).ffill().plot( secondary_y=True) # original!
  data["gox"].join(data["ltc"]).pct_change().resample( "5min", how=["mean"]).ffill().plot( secondary_y=True) # original!
  #data["ltc"].join( convert_depth_df( data["ltc_depth"], "bid-ask")).ffill().plot()
  plt.show()

# find the nearest value to a given date TODO!
def nearest_by_date( data, date="2013-04-18 00:01:01", return_date=False):
  try:
    i = data.index.searchsorted( date)
  except KeyError:
    print "Keyerror, resampling & retrying"
    data = data.resample("1s").ffill()
    i = data.index.searchsorted( date)
  # this returns a timeseries with one variable
  try:
    ts = data.ix[i]
    if ts.name > pd.to_datetime( date) and i-1 > 0:
      ts = data.ix[i-1]
  except IndexError:
    print "Index error"
    return np.NaN
  if return_date:
    # return the timeseries object (date & value)
    return ts
  else:
    # return just the value
    return ts[0]

#data = load2( "test.csv")
#plot_charts( data)