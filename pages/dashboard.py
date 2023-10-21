# importing libraries
import yfinance as yf
import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from ta import add_all_ta_features
from ta.utils import dropna

import scipy

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from pmdarima import auto_arima                              # for determining ARIMA orders

from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.title("STOCK DATA VISUALIZATION")

data = yf.download('ZOMATO.NS')

st.markdown("Take a look on closing stock prices of Zomato")

#fig = data['Close'].plot()
#plt.title('Closing Stock Prices of Zomato')
#st.plotly_chart(fig)

st.markdown("We are not interested in the absolute prices for these companies but want to understand how these stock fluctuate with time. We can also use candlestick charts to visulize opening, closing, high and low prices to stocks altogether.")

fig_1 = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
fig_1.update_layout(
    title='Zomato Stock Chart',
    yaxis_title='ZOMATO Stock',
)
st.plotly_chart(fig_1)

st.markdown("Now let's check the volume of stocks traded")
# fig_2=data['Volume'].plot()
# plt.title('Volume of stock traded')
# st.plotly_chart(fig_2)

st.markdown("Volume of stock traded of apple is highest on its IPO listing date. Second highest amount of stock traded was on 26 july and 2022. Shares of zomato hits all time low that day.")

st.markdown("https://www.livemint.com/market/stock-market-news/why-zomato-shares-plunged-to-all-time-low-today-11658721600204.html")

st.markdown("Let's see the total amount of stocks traded")

# data['TotalTraded'] = data['Open'] * data['Volume']
# fig_3=data['TotalTraded'].plot()
# plt.title('Total Traded Amount')
# st.plotly_chart(fig_3)
st.markdown("There is a similar pattern with volume of stock traded.")

st.markdown("Daily Percentage Gain")

st.markdown("$$r_t = {p_t \over p_{t-1}}-1$$")

# data['returns'] = data['Close'].pct_change()
# fig_4=data['returns'].plot(marker='o')
# st.plotly_chart("fig_4")

# fig_5=sns.distplot(data['returns'], bins=50)

# fig_6=sns.boxplot(data['returns'])
st.markdown("Negative daily returns seem to be slightly more frequent than negative returns for Apple.")

# fig_7=data['returns'].rolling(window=30).std().plot(figsize=(20, 10), title="30 Day Rolling Standard Deviation")

st.markdown("Cumulative Return")

st.markdown("$$i_t = (1+r_t)i_{t-1}$$")

# data['Cumulative Return'] = (1+data['returns']).cumprod()-1
# fig_8=data['Cumulative Return'].plot()
# plt.title('Zomato Stock Cumulative Return')
# st.plotly_chart('fig_8')

st.header("Average directional movement (ADX)")
st.markdown("- ADX ≤ 25: No trend")
st.markdown("- 25 < ADX ≤ 50: Trending")
st.markdown("- ADX > 50: Strong Trending")

# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.set_ylabel('Price')
# ax1.plot(data['Close'])
# ax2.set_ylabel('ADX')
# ax2.plot(ti_data['trend_adx'])
# ax1.set_title('Daily Close Price and ADX')
# ax2.axhline(y = 50, color = 'r', linestyle = '--')
# ax2.axhline(y = 25, color = 'r', linestyle = '--')

st.header("Relative Strength Index (RSI)")
st.markdown("- RSI > 70: Overbought")
st.markdown("- RSI < 30: Oversold")

# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.set_ylabel('Price')
# ax1.plot(data['Close'])
# ax2.set_ylabel('RSI')
# ax2.plot(ti_data['momentum_rsi'])
# ax1.set_title('Daily Close Price and RSI')
# ax2.axhline(y = 70, color = 'r', linestyle = '--')
# ax2.axhline(y = 30, color = 'r', linestyle = '--')

st.markdown("Zomato stock was overbought and hit all time high at 18 nov 2021. Zomato up on news of $500 million investment in Grofers on that day.")

st.markdown(r"https://economictimes.indiatimes.com/markets/stocks/news/zomato-gains-3-on-report-of-500-million-investment-in-grofers/articleshow/87774172.cms")

st.markdown("- From 18 jan to 22 jan there was a freefall in zomato stocks.")

st.markdown(r" https://www.livemint.com/market/stock-market-news/paytm-zomato-nykaa-shares-drop-to-record-low-is-the-unicorn-craze-over-11643345855087.html")

st.markdown("- There was a similar trend from 27 apr to 7 may.")

st.markdown(r"  https://www.livemint.com/market/stock-market-news/zomato-share-price-hits-new-low-stock-down-65-from-highs-11651818720666.html")


st.header("Time-Series Forecasting")
# result = seasonal_decompose(data['Close'], model='mul', period=5)
# result.plot()

st.header("Using ARIMA Model")

# train_data, test_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

# model_autoARIMA = auto_arima(train_data['Close'], start_p=0, start_q=0,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=3, # maximum p and q
#                       m=5,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       start_P=0, 
#                       D=0, 
#                       trace=True,
#                       stepwise=True)
# print(model_autoARIMA.summary())
# model_autoARIMA.plot_diagnostics(figsize=(15,8))


st.header("Stock Prediction")


# start=len(train_data)
# end=len(train_data)+len(test_data)-1
# predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMAX(0, 1, 0) Predictions')

# predictions.index = list(predictions.index)
# predictions.index  = test_data.index

# title = 'Zomato Closing Stock Price'
# ylabel='Stock Price'

# ax = data['Close'].plot(legend=True,figsize=(12,6),title=title, label='Closing Price')

# predictions.plot(legend=True)
# ax.autoscale(axis='x',tight=True)
# ax.set(xlabel=' ', ylabel=ylabel)