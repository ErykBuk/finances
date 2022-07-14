#
#   Simple MACD calculation
#
#   MIT License
#   Copyright (c) 2022 Eryk Buk
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.backends.backend_pdf  # to save multiple figures in one pdf
from datetime import datetime   # to get today's date
import yfinance as yf

pdf = matplotlib.backends.backend_pdf.PdfPages('plots.pdf') # We will be saving all figures to .pdf file

#   Getting Tesla's stocks form Yahoo
stocks = 'TSLA'
start_date = datetime(2020, 1, 1)
end_date   =  datetime.today() # today
raw = yf.download(
    stocks,
    start=start_date,
    end=end_date,
    progress=False  # This option shows progress bar of downloading
)
data = raw[['Close']] # We are only interested in close prices

#   We calculate appropriate size for our plots. Neat, but irrelevant to calculating MACD.
plot_size_x = round(((end_date - start_date).days/365)*6)
plot_size_y = round(max([6*(max(data['Close'])-min(data['Close']))/1000, 1/3*plot_size_x]))

#   Plotting stock prices
plot.figure(figsize=(plot_size_x,plot_size_y))  # For Tesla stocks 2020-2022 size (12,6) would suffice.
data['Close'].plot(grid=True)
plot.xlabel('Date')
plot.ylabel('Close price (USD)')
plot.title(stocks+' price (USD)')
pdf.savefig()

#   Exponentially Moving Average
def EMA(data, time):
    #   EMA is moving average, whose weights fall off exponentially, depending on smoothing factor.
    #   We can provide value of smoothing factor explicite or implicite (check pandas.DataFrame.ewm documentation).
    #   Here: smoothing factor = 2/(span+1)
    #
    #   We are using pandas.DataFrame.ewm for exponentially weighted (EW) averages. Function ewm() only
    #   returns dataframe with exponentially weighted prices. We then need to average it with mean().
    #   Parameter 'adjust' describes if we want to calculate our data as exponentialy weighted moving averages (True)
    #   or if we prefer to calculate it recursively (False). We choose the latter.

    return data.ewm(span=time, adjust=False).mean()

    #   Here we calculate EMA by hand
    '''
    results = []
    results.append(data[0])
    alpha = 2/(time+1)
    for i in range(1, len(data)):
        results.append(alpha*data.Close[i] + (1-alpha)*results[i-1])
    '''

    return results

#   Moving Average Convergence/Divergence
def MACD(data, a, b, c):
    #   In general we can define MACD as difference of two EMAs of some characteristic times a and b respectfully, and
    #   define signal line as EMA of MACD of yet another characteristic time c.
    #   FOr MACD to make sense we need to subtract longer moving average from shorter moving average.
    if a > b:
        return 0

    macd = EMA(data, a) - EMA(data, b)
    signal = EMA(macd,c)

    return(macd,signal)


#   Averages over 9, 12 and 26 days are most commonly used. While only the last two will be nedded,
#   we will be able to see how averaging smooths data.
ema9   = EMA(data.Close, 9  )
ema12  = EMA(data.Close, 12 )
ema26  = EMA(data.Close, 26 )
ema69  = EMA(data.Close, 69 )
ema100 = EMA(data.Close, 100)
#   Here we choose 12, 26 and 9 days respectively.
macd   = MACD(data.Close, 12, 26, 9)[0]
signal = MACD(data.Close, 12, 26, 9)[1]
#   MACD with another characteristic times
macdx = MACD(data.Close, 69,100,26)[0]
signalx = MACD(data.Close, 69,100,26)[1]

#   Plotting EMAs
plot.figure(figsize=(plot_size_x,plot_size_y))
plot.plot(data.index, ema9,  label='EMA9',  color='red'   )
plot.plot(data.index, ema12, label='EMA12', color='yellow')
plot.plot(data.index, ema26, label='EMA26', color='green' )
plot.plot(data.index, ema69, label='EMA69', color='blue'  )
plot.plot(data.index, ema100,label='EMA100',color='black' )
plot.xlabel('Date')
plot.ylabel('Value')
plot.legend(loc='upper left')
pdf.savefig()
#   Plotting MACDs and their signal lines.
plot.figure(figsize=(plot_size_x,plot_size_y))
plot.plot(data.index, macd,   label=stocks+' MACD',   color='red'  )
plot.plot(data.index, signal, label='Signal Line', color='green')
plot.plot(data.index, macdx,   color='orange'  )
plot.plot(data.index, signalx, color='blue')
plot.xlabel('Date')
plot.ylabel('Value')
plot.legend(loc='upper left')
pdf.savefig()

#   Crossing of MACD and singnal line is the point where we should (potentially) buy or sell stocks.
#   If after crossing MACD is bigger then signal, then we buy. If it's the opposite, then we sell.
#   Function bellow iterates over all data, and decides if its time to buy or sell at every moment.
#   By default we order funcion to buy, by setting flag=-1. Then each time MACD and signal line cross,
#   we will buy/sell depending on flag value.
#   If we don't buy or sell, then we append NaN into buy and sell lists.
def buy_sell(data):
    buy = []
    sell = []
    flag = -1

    for i in range(0, len(data)):
        if data['MACD'][i] > data['Signal Line'][i]:
            sell.append(np.nan)
            if flag != 1:   # Will always buy at the beginning
                buy.append(data['Close'][i])
                flag = 1    # Won't buy as long, as MACD > signal line
            else:
                buy.append(np.nan)
        elif data['MACD'][i] < data['Signal Line'][i]:
            buy.append(np.nan)
            if flag != 0:   # Will sell when MACD and signal line cross i.e when MACD-(signal line) changes sign
                sell.append(data['Close'][i])
                flag = 0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return(buy,sell)

#   We are appending appropriated data to 'data', and using our function
data = data.assign(MACD=macd)
data['Signal Line'] = signal
result = buy_sell(data)
data['Buy'] = result[0]
data['Sell'] = result[1]

#   Buy/sell plot
plot.figure(figsize=(plot_size_x,plot_size_y))
plot.scatter(data.index, data['Buy'], color='green', label='Buy', marker='^', alpha=1)
plot.scatter(data.index, data['Sell'], color='red', label='Sell', marker='v', alpha=1)
plot.plot(data['Close'], label='Close price', alpha=0.9)
plot.xlabel('Date')
plot.ylabel('Close price (USD)')
plot.title(stocks+' price; buy and sell signal')
plot.grid()
pdf.savefig()

#   We repeat analysis for additional MACD
data['MACD'] = macdx
data['Signal Line'] = signalx
result = buy_sell(data)
data['Buy'] = result[0]
data['Sell'] = result[1]
plot.figure(figsize=(plot_size_x,plot_size_y))
plot.scatter(data.index, data['Buy'], color='green', label='Buy', marker='^', alpha=1)
plot.scatter(data.index, data['Sell'], color='red', label='Sell', marker='v', alpha=1)
plot.plot(data['Close'], label='Close price', alpha=0.9)
plot.xlabel('Date')
plot.ylabel('Close price (USD)')
plot.title(stocks+' stocks price; buy and sell signal')
plot.grid()
pdf.savefig()


pdf.close()
