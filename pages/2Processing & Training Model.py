import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="Processing & Training Model", page_icon="📈")

st.header("Processing & Training Model")

st.markdown("""
## Importing Library
""")

st.code("""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    import ta
""")

st.markdown("""  
Kode ini mengimpor berbagai pustaka yang diperlukan untuk analisis data, pelatihan model, pengunduhan data saham, visualisasi, dan perhitungan indikator teknikal.

## Pre-Processing
### 1. Data Loading and Preprocessing   
""")

st.code("""
    def load_data(ticker="BBCA.JK", start="2022-01-01", end="2023-12-31", period=5):       
    \"""
    Mengunduh dan memproses data saham dari Yahoo Finance.

    Parameters:
    ticker (str): Kode saham
    start (str): Tanggal mulai
    end (str): Tanggal akhir
    period (int): Periode untuk indikator teknis

    Returns:
    pd.DataFrame: Data saham yang telah diproses
    list: Fitur yang digunakan untuk pelatihan model
    \"""
    data = yf.download(ticker, start=start, end=end)
    data['return'] = data['Adj Close'].pct_change(period)
    data['sma'] = ta.trend.sma_indicator(data['Adj Close'], window=period)
    data['ema'] = ta.trend.ema_indicator(data['Adj Close'], window=period)
    data['rsi'] = ta.momentum.rsi(data['Adj Close'], window=period)

    # Drop NA after all calculations
    data = data.dropna()

    scaler = MinMaxScaler()
    features = ['sma', 'ema', 'rsi']
    data[features] = scaler.fit_transform(data[features])
    return data, features
""")

st.markdown("""  
Fungsi load_data mengunduh data saham dari Yahoo Finance dan menghitung beberapa indikator teknikal seperti Simple Moving Average (SMA), 
Exponential Moving Average (EMA), dan Relative Strength Index (RSI). 
Data kemudian di-normalisasi menggunakan MinMaxScaler.

### 2. Training Model
""")

st.code("""
    def train_model(data, features, step):
    \"""
    Melatih model regresi untuk memprediksi drift.

    Parameters:
    data (pd.DataFrame): Data saham yang telah diproses
    features (list): Daftar fitur yang digunakan
    step (int): Jumlah data yang digunakan untuk pelatihan

    Returns:
    model: Model yang telah dilatih
    \"""
    X = data[features].iloc[:step].values
    y = data["return"].iloc[:step].values
    model = RandomForestRegressor()
    model.fit(X, y)
    return model
""")

st.markdown("""
Fungsi train_model melatih model RandomForestRegressor menggunakan fitur yang telah dihitung 
dan mengembalikan model yang telah dilatih.
""")

st.code("""
    def calculate_volatility(data):
    \"""
    Menghitung volatilitas tahunan berdasarkan return harian.

    Parameters:
    data (pd.DataFrame): Data saham yang telah diproses

    Returns:
    float: Volatilitas tahunan
    \"""
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility
""")

st.markdown("""
Fungsi calculate_volatility menghitung volatilitas tahunan dari return harian saham.
####
### Next Visit Simulation & Visualization Page on the Left Navigation Bar
""")