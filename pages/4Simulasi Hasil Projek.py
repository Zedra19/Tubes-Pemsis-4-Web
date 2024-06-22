import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="Hasil Projek", page_icon="📈")

st.header("Prediksi Harga Saham dengan Kombinasi Machine Learning dan Gerakan Brown Geometrik (GBM)")

# 1. Data Loading and Preprocessing
def load_data(ticker="BBCA.JK", start="2022-01-01", end="2023-12-31", period=5):
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

# 2. Machine Learning Model Training
def train_model(data, features, step):
    X = data[features].iloc[:step].values
    y = data["return"].iloc[:step].values
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# 3. Function to calculate annualized standard deviation (volatility)
def calculate_volatility(data):
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility

# 4. GBM Simulation with Machine Learning Predicted Drift
def gbm_sim(spot_price, volatility, time_horizon, model, features, data):
    dt = 1
    actual = [spot_price] + list(data['Adj Close'].values)
    drift = model.predict(data[features])
    paths = [spot_price]
    for i in range(len(data)):
        paths.append(actual[i] * np.exp((drift[i] - 0.5 * (volatility/252)**2) * dt + (volatility/252) * np.random.normal(scale=np.sqrt(1/252))))
    return paths, drift

# 5. Main Function with Streamlit Interface
if __name__ == "__main__":
    ticker = st.text_input("Masukkan Kode Saham", "BBCA.JK")
    st.write("Kode Saham Default: BBCA.JK")
    start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("Tanggal Akhir", pd.to_datetime("2023-12-31"))
    period = st.number_input("Periode Indikator Teknis (1 - 5)", min_value=1, value=5)
    time_horizon = st.number_input("Jangka Waktu Simulasi (1 - 252 hari)", min_value=1, value=252)

    if st.button("Mulai Simulasi"):
        data, features = load_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), period=period)
        steps = int(len(data) / 2)
        model = train_model(data, features, steps)
        spot_price = data["Adj Close"].iloc[steps-1]
        volatility = calculate_volatility(data.iloc[:steps])
        simulated_paths, drifts = gbm_sim(spot_price, volatility, time_horizon, model, features, data.iloc[steps:])
        
        harga_pred = simulated_paths
        harga_act = data['Adj Close'][steps-1:].values
        index = data.index[steps-1:]

        # Plotting results
        st.subheader("Hasil Simulasi")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
        ax[0].plot(drifts)
        ax[1].plot(data['return'].iloc[steps:].values)
        ax[2].plot([abs(i-j) for i, j in zip(drifts, data['return'].iloc[steps:].values)])
        for i in range(len(labels)):
            ax[i].set_title(labels[i])
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(index, harga_pred, label='Predicted')
        ax.plot(index, harga_act, label='Actual')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Stock Price")
        ax.set_title("Simulated Stock Price Paths")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(index, [abs(i-j) for i, j in zip(harga_pred, harga_act)], '.-')
        ax[0].set_title('Absolute Error of Prediction Price')
        ax[1].plot(index, [abs(i-j)/j*100 for i, j in zip(harga_pred, harga_act)], '.-')
        ax[1].set_title('Relative Absolute Error of Prediction Price (in %)')
        for i in range(2):
            ax[i].tick_params(axis='x', labelrotation=40)
        st.pyplot(fig)