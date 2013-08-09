#!/usr/bin/python

import json
import re
import time
import sys
import eventlet
from eventlet.green import urllib2  
import csv
import datetime

# function to fetch our URL
def fetch(url):
  while(True):
    try:
      r = urllib2.urlopen(url, timeout=10)
      code = r.getcode()
      if( code == 200):
        return url, r.read()
      else:
        print "Error", code
    except Exception as e:
      print "WOAH!!!", url, e

# verify gox json API response
# DEFINE PRICE AS last price
def gox_tick_last( body):
  try:
    j = json.loads( body)
  except ValueError, e:
    print "BTC-e error", e, url
  if j["result"] == "success":
    last = float( j["data"]["last"]["value"])
    buy  = float( j["data"]["buy"]["value"])
    sell = float( j["data"]["sell"]["value"])
    # convert gox timestamp to regular
    t    = float( j["data"]["now"])/1000000
    bs = get_buy_sell( last, buy, sell)
    return { "price":last, "type":bs, "time":t}
  else:
    print "Error", j

# verify/process btc-e ticker json API response
# this defines price as last
def btce_tick_last(body):
  try:
    j = json.loads( body)
  except ValueError, e:
    print "BTC-e error", e, url
  buy  = float( j["ticker"]["buy"])
  buy  = float( j["ticker"]["buy"])
  sell = float( j["ticker"]["sell"])
  last = float( j["ticker"]["last"])
  vol  = float( j["ticker"]["vol"])
  bs = get_buy_sell( last, buy, sell)
  t    = float( j["ticker"]["server_time"])
  bs = get_buy_sell( last,
                     float(j["ticker"]["buy"]),
                     float(j["ticker"]["sell"]))
  return { "price":last, "type":bs, "time":t, "vol":vol}

# figure out whether it's a buy or sell based on three prices
def get_buy_sell(last, buy, sell):
  if( last == buy):
    return "buy"
  elif( last == sell):
    return "sell"
  else:
    return "?"

# compress a string using BZ2 and Base64 ... remove newlines
def compress(json_str):
  return re.sub("\n", "", json_str.encode("bz2").encode("base64"))

# undo a Base64'd BZ2 string
def decompress(encoded_str):
  return encoded_str.decode("base64").decode("bz2")

# get BTC-e litecoin market depth
def ltc_depth( body):
  return { "info":compress(body), "time":time.time()}

# get Mt. Gox last Bitcoin trade value
def get_gox_btc_last( body):
  j = gox_tick( body)

def ts2dt( timestamp):
  return datetime.datetime.fromtimestamp( timestamp).strftime("%Y-%m-%d %H:%M:%S")

# get current market activity as a csv line
def process_responses( url_body_list):
  # run through responses and parse accordingly
  processed = {}
  # THIS IS AN UGLY HACK ... but we're using a global to store the volume
  # and then tack it onto the end of our log to maintain compatability
  ltc_vol = 0
  for resp in url_body_list:
    u = resp["url"]
    b = resp["body"]
    # gox ticker
    if( u == "http://data.mtgox.com/api/2/BTCUSD/money/ticker"):
      g = gox_tick_last( b)
      processed["gox_price"] = g["price"]
      processed["gox_type"] = g["type"]
      processed["gox_time"] = ts2dt( g["time"])
    # btc-e ltc depth
    elif( u == "http://btc-e.com/api/2/ltc_usd/depth"):
      g = ltc_depth( b)
      processed["ltc_depth"] = g["info"]
      processed["ltc_depth_time"] = ts2dt( g["time"])
    # btc-e LAST (not avg)
    elif( u == "http://btc-e.com/api/2/ltc_usd/ticker"):
      g = btce_tick_last( b)
      processed["ltc_usd_price"] = g["price"]
      processed["ltc_type"] = g["type"]
      processed["ltc_time"] = ts2dt( g["time"])
      processed["ltc_24h_vol"] = g["vol"]
    elif( u == "http://btc-e.com/api/2/btc_usd/ticker"):
      g = btce_tick_last( b)
      processed["ltc_btc_price"] = g["price"]
      processed["ltc_btc_type"] = g["type"]
      processed["ltc_btc_time"] = ts2dt( g["time"])
    elif( u == "http://btc-e.com/api/2/ltc_btc/ticker"):
      g = btce_tick_last( b)
      processed["btc_usd_price"] = g["price"]
      processed["btc_type"] = g["type"]
      processed["btc_time"] = ts2dt( g["time"])
  # return our list
  return processed

def usage():
  print "USAGE: %s FILENAME\n" % sys.argv[0]
  sys.exit()

def process_args():
  if len(sys.argv) != 2:
    usage()
  return sys.argv[1]

if __name__ == "__main__":
  # process argv or die
  logname = str(time.time())+".csv" #process_args()

  # urls and set up
  urls = ["http://data.mtgox.com/api/2/BTCUSD/money/ticker",
          "http://btc-e.com/api/2/btc_usd/ticker",
          "http://btc-e.com/api/2/ltc_btc/ticker",
          "http://btc-e.com/api/2/ltc_usd/ticker",
          "http://btc-e.com/api/2/ltc_usd/depth"]
  pool = eventlet.GreenPool()

  # prepare log
  f = open(logname, "wb")
  w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
  w.writerow( [ "gox_last", "gox_buysell", "gox_time",
                "btc_usd_last", "btc_usd_buysell", "btc_usd_time",
                "ltc_btc_last", "ltc_btc_buysell", "ltc_btc_time",
                "ltc_last", "ltc_buysell", "ltc_time",
                "ltc_depth", "ltc_depth_time", "ltc_24h_volume" ])

  try:
    while(True):
      # get URLs in parallel
      responses = []
      for url, body in pool.imap(fetch, urls):
        responses.append({"url":url, "body":body})

      #process our responses
      processed = process_responses( responses)
      p = processed

      # insert timestamp and write to log
      print datetime.datetime.fromtimestamp( 
          time.time()).strftime('%Y-%m-%d %H:%M:%S')
      w.writerow( [ p["gox_price"], p["gox_type"], p["gox_time"],
                    p["btc_usd_price"], p["btc_type"], p["btc_time"],
                    p["ltc_btc_price"], p["ltc_btc_type"], p["ltc_btc_time"],
                    p["ltc_usd_price"], p["ltc_type"], p["ltc_time"],
                    p["ltc_depth"], p["ltc_depth_time"],
                    p["ltc_24h_vol"] ])

      f.flush()
      time.sleep(3)
  except KeyboardInterrupt:
    f.close()
    print "Caught Ctl-Z/C exiting!\nL8r."