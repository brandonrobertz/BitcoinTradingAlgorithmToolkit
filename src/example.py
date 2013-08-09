import pandas as pd
import data
import dtools
from pybrain.datasets            import SupervisedDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure           import RecurrentNetwork, FullConnection
from pybrain.structure.modules   import LinearLayer, TanhLayer
import numpy as np
from matplotlib import pylab as plt

# 10-minute timeframe
time_str = "10min"

# our litecoin, statistical analysis options
ltc_opts = \
{  "debug": False,
   "relative": False,
   "calc_rolling": True,
   "rolling": { time_str : {  12: pd.DataFrame(),
                              24: pd.DataFrame(),
                              50: pd.DataFrame() } },
   "calc_mid": True,
   "calc_ohlc": True,
   "ohlc": { time_str : pd.DataFrame()  },
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
                  "TMI"  : { "data": pd.DataFrame(), "nb":10, "nf":5 }
                },
   "calc_std": True,
   "std": { 13 : pd.DataFrame(), 21 : pd.DataFrame(), 34 : pd.DataFrame() },
   "calc_crt": True,
   "crt": { 1: pd.DataFrame(),
            2: pd.DataFrame(),
            3: pd.DataFrame(),
            5: pd.DataFrame(),
            8: pd.DataFrame() },
   "instant": True,
   "time_str": time_str }

# warp and instant = True forces Data to calculate everything all in one
# pass. Other options would make it run in simulated real time. Load 
# filename called "test.csv" from ./logs/ ... Data
# assumes all logs are in the local directory called logs
d = data.Data( warp=True, instant=True, ltc_opts=ltc_opts,
               time_str=time_str, filename="test.csv")

# Pull out all indicators into a single pandas dataframe. Prefix all rows
# with "LTC"
ltc = d.ltc.combine( "LTC")

# take our ltc dataframe, and get targets (prices in next 10 minutes)
# in the form of compound return prices (other options are "PRICES", 
# which are raw price movements)
dataset, tgt = dtools.gen_ds( ltc, 1, ltc_opts, "CRT")

# initialize a pybrain dataset
DS = SupervisedDataSet( len(dataset.values[0]), np.size(tgt.values[0]) )

# fill it
for i in xrange( len( dataset)):
  DS.appendLinked( dataset.values[i], [ tgt.values[i]] )

# split 70% for training, 30% for testing
train_set, test_set = DS.splitWithProportion( .7)

# build our recurrent network with 10 hidden neurodes, one recurrent
# connection, using tanh activation functions
net = RecurrentNetwork()
hidden_neurodes = 10
net.addInputModule( LinearLayer(len( train_set["input"][0]), name="in"))
net.addModule( TanhLayer( hidden_neurodes, name="hidden1"))
net.addOutputModule( LinearLayer( len( train_set["target"][0]), name="out"))
net.addConnection(FullConnection(net["in"], net["hidden1"], name="c1"))
net.addConnection(FullConnection(net["hidden1"], net["out"], name="c2"))
net.addRecurrentConnection(FullConnection(net["out"], net["hidden1"], name="cout"))
net.sortModules()
net.randomize()

# train for 30 epochs using the rprop- training algorithm
trainer = RPropMinusTrainer( net, dataset=train_set, verbose=True )
trainer.trainOnDataset( train_set, 30)

# test on training set
predictions_train = np.array( [ net.activate( train_set["input"][i])[0] for i in xrange( len(train_set)) ])
plt.plot( train_set["target"], c="k"); plt.plot( predictions_train, c="r"); plt.show()

# and on test set
predictions_test = np.array( [ net.activate( test_set["input"][i])[0] for i in xrange( len(test_set)) ])
plt.plot( test_set["target"], c="k"); plt.plot( predictions_test, c="r"); plt.show()