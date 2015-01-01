BitcoinTradingAlgorithmToolkit
==============================

A framework for logging, simulating, and analyzing prices of crypto currencies on various exchanges using technical analysis, fuzzy logic, and neural networks.

If you like this ... BTC: 1NejrUgQDm34CFyMHuaff9PNsd8zhd7SgR 

## What is this?

Can machine learning be used to trade Bitcoin profitably? Can ML algorithms predict the prices based on technical indicators? Can data mining crypto currency prices take me from fry-cook to bitcoin billionaire? This was an experiment in seeing how well a neural network or fuzzy logic could learn the "rules" of the various Bitcoin markets' price movements. (If they exist at all.) To do that I needed a framework for logging and then replaying price movements and a way to show them to the algorithms. I dug up some of the most common and more obscure indicators referenced in some classic NN-trading literature from old books, mathematical formulas, and other languages' implementations.

Don't bet with this code. I tested it a lot, but it's hack-y and it could easily have hidden bugs. (Especially when it comes to oddities with numpy's NaN and the fact that barely any of the indicators' original formulas accounted for division by zero.) I didn't go into this with the idea that anyone else would ever use it.

DISCLAIMER: Trading is gambling and technical indicators are just another way to fool humans into thinking they can see the future.

### Logging Exchange Data

`logtick.py` is what I used to log ticker, market depth, and volume information from BTC-e and Mt. Gox to a CSV file. There are better logging tools and scripts out there, but find them yourself. Mine uses multiprocessing and compresses the entire depth information moment-by-moment. If you want to train a machine learning algorithm, especially a neural network, you need tons of data. The more you get, the better at generalizing the AI gets. I set `logtick` to run day-and-night for about a month. Since my intent was to trade LTC fully-auto, I was interested in LTC volume, not BTC-e BTC or GOX volume, so that was all I logged. A tiny sample log, `logs/test.csv` is included here.

`logtick.py` logs the following fields to CSV:

    gox_last, gox_buysell, gox_time,
    btc_usd_last, btc_usd_buysell, btc_usd_time,
    ltc_btc_last, ltc_btc_buysell, ltc_btc_time,
    ltc_last, ltc_buysell, ltc_time,
    ltc_depth, ltc_depth_time, ltc_24h_volume

`gox_*` comes from Mt. Gox BTC, `btc_usd_*` is BTC<->USD on BTC-e, `ltc_btc_*` is BTC<->LTC on BTC-e, and `ltc_*` is LTC<->USD on BTC-e.

### Analyzing and Simulating

`data.py` holds most of the code used in processing the logged price data, playing it back to us (used in simulating what would happen if we traded using X-rules or Y-strategy), and statistically analyzing the logged data using technical indicators, averages, standard deviations, etc.

The two classes in `data.py`, `Data` and `Coin`, make up the core of the Bitcoin analysis functionality. The `Coin` class holds and generates ticker prices and averages, std devs, and technical indicators based on them. The `Data` class wraps the `Coin` class(es) and loads ticker prices into them (either from the actual exchange over the web, or log files from disk). It allows for mutiple exchanges to be loaded and ultimately compared (i.e., GOX Bitcoin and BTC-e Litecoin).

The project evolved over time and can be used several ways. Lets say that we want to load some logged Litecoin price data from disk and then get some rolling means, all available technical indicators, compound returns, and some standard deviations. We also want to look at the data in 10-minute OHLC (open, high, low, close) chunks:

    import pandas as pd
    import data

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

From there, `ltc` would be a dataframe filled with all the values we'd need to start training our neural network or use with any other algorithm.

### Technical indicators

I tried a lot of different inputs to the machine learning algorithms. Technical indicators seemed to be the most common strategy in the literature. `indicators.py` implements the following technical indicators and analysis functions:

- AVERAGE DIRECTIONAL MOVEMENT INDEX (ADI)
- ADAPTIVE MOVING AVERAGE (AMA)
- COMMODITY CHANNEL INDEX (CCI)
- COMPOUND RETURN
- EHLER'S LEADING INDICATOR (ELI)
- FRACTAL ADAPTIVE MOVING AVERAGE (FRAMA)
- MACD
- NORMALIZATION
- RATE OF CHANGE (ROC)
- RELATIVE STRENGTH INDEX (RSI)
- RELATIVE VOLUME INDEX (w/ Inertia) (RVI)
- STANDARD DEVIATIONS
- TREND MOVEMENT INDEX (TMI)

I chose these specific indicators because they were featured in a lot of "classic" algorithmic trading articles, particularly "Forecasting Foreign Exchange Rates Using Recurrent Neural Networks" by Paolo Tenti.

### Converting the Data for Training

`dtools.py` has a lot of convenience functions for turning our dataframes into input/target matrices. The `gen_ds` function in particular will perform this transformation (continuing from the above example):

    import dtools

    # take our ltc dataframe, and get targets (prices in next 10 minutes)
    # in the form of compound return prices (other options are "PRICES", 
    # which are raw price movements)
    dataset, tgt = dtools.gen_ds( ltc, 1, ltc_opts, "CRT")

That should have stripped any starting Nans (some of the technical indicators need time to "warm up") and shifted our prices forward into `tgt`. We will use `dataset` as our inputs and `tgt` as the training examples. We're ready to send this to a machine learning algorithm.

### Using with a Neural Network

There are a few python Neural Network packages. Only two had even close to enough functionality to use: [pybrain](http://pybrain.org/) and [neurolab](https://code.google.com/p/neurolab/). I also experimented with using [Python-Matlab-Wormholes](http://code.google.com/p/python-matlab-wormholes/) and using the Matlab Neural Network Toolkit (which has some great implementations of recurrent/time-delayed neural networks). Here's an example with pybrain:

    from pybrain.datasets            import SupervisedDataSet
    from pybrain.tools.shortcuts     import buildNetwork
    from pybrain.supervised.trainers import RPropMinusTrainer
    from pybrain.structure           import RecurrentNetwork, FullConnection
    from pybrain.structure.modules   import LinearLayer, TanhLayer
    import numpy as np
    from matplotlib import pylab as plt

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

And there you have it.

### And beyond ... genetic codes & fuzzy logic

If you did all this using the `test.csv` dataset, only about a day's worth of price data, your model isn't going to be very good. Also, the simple recurrent neural network with generic mean-squared error function turned out to not be a very good way to represent "error" in predicting price changes. When all you're trying to do it get as "close as possible" to the price change value, the NN seemed to hover around zero, without regard to positive or minus. This is a huge problem if you're trading based on whether or not the network says the price will go up or down. There are alternative error functions that could work better for our ultimate goal. One idea I read about was to use an error function that penalizes more heavily if our model predicts the wrong direction of price movement.

Trying to hack this into pybrain turned out to be a nightmare because the error function seems to be obscurely hardcoded into the NN code. Neurolab makes it easier to implement custom error functions and I had more success using a "weighted directional symmetry" (WDS) error function.

There was also the problem of optimizing the parameters to each technical indicator. I approached this problem by using genetic algorithms to find the most accurate combinations of which technical indicators and statistical analysis methods to use and which parameters to use with them. I used [DEAP](https://code.google.com/p/deap/) for this. `genetic.py` has some functions that help generate, encode, decode, and mutate random genes:

    import genetic

    # create a random individual
    individual = genetic.rand_gene()

    # mutate with 0-mean, 4-std dev gaussian random fcn & 40% chance of mutation
    individual = genetic.mutate_gene( individual, 0, 4, 0.4)

    # turn the genetic code into a Data options struct, with a 5 min timeframe
    btc_opts = genetic.decode_gene( individual, 5) # this can be used in Data()

The end result still wasn't anywhere near perfect at predicting the price movements, though, so I tried another strategy ...

Turning to fuzzy-logic, I had even more success (theoretically profitable models) using [pyfuzzy](http://pyfuzzy.sourceforge.net/) (which took a lot of hacking to import for some reason), a minimalistic set of indicators, and DEAP-based genetic algorithms for technical indicator parameter optimization. These models didn't stay profitable for very long at all, though.

### Conclusions

My dreams of quitting the fry-cook business were crushed. The best model I developed used a genetic algorithm to find:

1. Which technical indicators helped predict price movements best, and
2. What parameters worked best for each indicator.

It sent the inputs and targets (generated with the `data.py`) to Matlab using Python-Matlab Wormhole. It used a Matlab Neural Network to train and evaluate the result. The most successful NN had only three hidden neurodes, one recurrent layer, about five technical indicators (mostly standard deviations and trend-following indicators), and a short timespan (~5 mins). This model mostly hovered around zero (do nothing) and caught some of the large price spikes in the test and validation sets. *But* the model got wildly inaccurate quickly after the period it was trained and validated on, which suggests the Bitcoin market changes "modes" a lot or that my model wasn't generalizing as well as it needs to.

A lot of bad news came out in the period of my building and training the networks, which sent the markets into random panics. All it takes is one miner to do a market-order sell of 1,000 BTC to send the market into total chaos. And no technical analysis-based machine learning system was good at dealing with this.

## Dependencies

I tried to keep it minimal.

- pandas: http://pandas.pydata.org/
- numpy: http://www.numpy.org/
- matplotlib (for plots): http://matplotlib.org/
- eventlet (for ticker logging): http://eventlet.net/
