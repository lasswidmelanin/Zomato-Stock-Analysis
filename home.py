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

# set title of the page 
st.title("ZOMATO STOCK ANALYSIS")
st.markdown("Launched in 2010, Zomato's technology platform connects customers, restaurant partners and delivery partners, serving their multiple needs.")

from PIL import Image
image = Image.open(r'C:\Users\Naincy\OneDrive\Desktop\Zomato Stock Analysis\assets\l17420220805125400.webp')
st.image(image)

st.markdown('<h2 style="font-weight: Calibri; color: brown;">About Dataset</h2>', unsafe_allow_html=True)
st.markdown("The data is the price history and trading volumes of the fifty stocks in the index NIFTY 50 from NSE (National Stock Exchange) India. All datasets are at a day-level with pricing and trading values split across .cvs files for each stock along with a metadata file with some macro-information about the stocks itself. The data spans from 1st January, 2020 to 30th April, 2021.")
