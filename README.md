BitcoinTradingAlgorithmToolkit
==============================

A framework for logging, simulating, and analyzing prices of crypto currencies on various exchanges using technical analysis, fuzzy logic, and neural networks.

Throw me a (fraction of a bit)coin if you find this handy. 1GqES1gCLdHd2gE2P6L1gqyYjf87bjn8mA

## What is this?

Can machine learning be used to trade Bitcoin profitably? Can ML algorithms predict the prices based on technical indicators? This was an experiment in seeing how well a neural network or fuzzy logic could learn the "rules" of the various Bitcoin markets' price movements. (If they exist at all.) To do that I needed a framework for logging and then replaying price movements and a way to show them to the algorithms. I dug up some of the most common and more obscure indicators referenced in some classic NN-trading literature from old books, mathematical formulas, and other languages' implementations.

Careful using this code. Especially if you're betting actual money. (Insane!) I tested this code a lot, but it could easily still have a good amount of bugs. (Especially when it comes to oddities with numpy's NaN and the fact that barely any of the indicators accounted for division by zero.)

Remember that trading is gambling and that technical indicators are just a way to fool us into thinking we can see the future.

### Logging Exchange Data

`logtick.py` is what I used to log ticker, market depth, and volume information from BTC-e and Mt. Gox to a CSV file. There are better logging tools and scripts out there, but you can find them yourself. Mine uses multiprocessing and compresses the entire depth information for that moment. If you want to train a machine learning algorithm, especially a neural network, you need tons of data. The more you get, the better at generalizing your NN can be. I set it to log day and night for about a month. Since my intent was to trade LTC automatically, I was interested in LTC volume, not BTC-e BTC or GOX volume, so that was all I logged.

`logtick.py` logs the following fields to CSV:

    gox_last, gox_buysell, gox_time ,
    btc_usd_last, btc_usd_buysell, btc_usd_time,
    ltc_btc_last, ltc_btc_buysell, ltc_btc_time,
    ltc_last, ltc_buysell, ltc_time,
    ltc_depth, ltc_depth_time, ltc_24h_volume

### Analyzing and Simulating

`data.py` has most of the code used in processing the logged price data, playing it back to us (used in simulating what would happen if we traded using X-rules or Y-strategy), and statistically analyzing the logged data using technical indicators, averages, standard deviations, etc.

The two classes in `data.py` (`Data` and `Coin`) make up the core of the Bitcoin analysis functionality. The Coin class holds and generates ticker prices and averages, std devs, and technical indicators based on them. The Data class wraps the Coin class(es) and loads ticker prices into them (either from the actual exchange over the web, or from log files from disk). It allows for mutiple exchanges to be loaded and ultimately compared (i.e., GOX Bitcoin and BTC-e Litecoin).

The usage of it evolved over time and can be used several ways. Lets say that we want to load some logged Litecoin price data from disk and then run some rolling means, all available technical indicators, compound returns, and some standard deviations on it. We also want to look at the data in 10-minute OHLC (open, high, low, close) chunks:

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

### Converting and Training with Data

`dtools.py` has a lot of convenience functions for turning our dataframes into input and target matrices. The `gen_ds` function in particular will perform this transformation (continuing from the above example):

    import dtools

    # take our ltc dataframe, and get targets (prices in next 10 minutes)
    # in the form of compound return prices (other options are "PRICES", 
    # which are raw price movements)
    dataset, tgt = dtools.gen_ds( ltc, 1, ltc_opts, "CRT")

That should have stripped any starting Nans (some of the technical indicators need time to "warm up") and shifted our prices forward into `tgt`. We will use `dataset` as our inputs and `tgt` as the training examples.

## Dependencies

- Pandas
- Numpy
