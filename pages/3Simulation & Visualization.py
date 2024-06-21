import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="Simulation & Visualization", page_icon="📈")

st.header("Simulation & Visualization")

st.markdown("""       
### Simulasi GBM dengan Drift yang Diprediksi
""")

st.code("""
    def gbm_sim(spot_price, volatility, time_horizon, model, features, data):
    \"""
    Melakukan simulasi GBM dengan drift yang diprediksi oleh model.

    Parameters:
    spot_price (float): Harga awal
    volatility (float): Volatilitas tahunan
    time_horizon (int): Jangka waktu simulasi
    model (model): Model yang telah dilatih
    features (list): Daftar fitur yang digunakan
    data (pd.DataFrame): Data saham yang telah diproses

    Returns:
    list: Simulasi harga saham
    np.ndarray: Drift yang diprediksi
    \"""
    dt = 1
    actual = [spot_price] + list(data['Adj Close'].values)
    drift = model.predict(data[features])
    paths = [spot_price]
    for i in range(len(data)):
        paths.append(actual[i] * np.exp((drift[i] - 0.5 * (volatility/252)**2) * dt + (volatility/252) * np.random.normal(scale=np.sqrt(1/252))))
    return paths, drift
""")

st.markdown("""
Fungsi gbm_sim melakukan simulasi harga saham menggunakan model GBM dengan drift yang diprediksi oleh model 
machine learning yang telah dilatih.
""")

st.code("""
if __name__ == "__main__":
    period = 5  # Menentukan periode untuk indikator teknis
    data, features = load_data(period=period)  # Memuat data saham dan menghitung fitur teknis
    steps = int(len(data) / 2)  # Menentukan jumlah data untuk pelatihan (setengah dari dataset)
    model = train_model(data, features, steps)  # Melatih model dengan data dan fitur yang telah diproses
    spot_price = data["Adj Close"].iloc[steps-1]  # Mengambil harga saham terakhir dari data pelatihan
    volatility = calculate_volatility(data.iloc[:steps])  # Menghitung volatilitas tahunan berdasarkan data pelatihan
    time_horizon = len(data) - steps  # Menentukan jangka waktu simulasi (sisa dari dataset)
    simulated_paths, drifts = gbm_sim(spot_price, volatility, time_horizon, model, features, data.iloc[steps:])  # Melakukan simulasi GBM
""")

st.write("Bagian if name == \"main\" merupakan fungsi utama yang menjalankan keseluruhan proses, mulai dari pemuatan dan pemrosesan data, pelatihan model, simulasi harga saham, hingga visualisasi hasil simulasi.")
st.write("***")
st.write("#### Visualisasi hasil simulasi")

st.code("""
# Plotting results
    labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
    fig, ax = plt.subplots(1, 3, figsize=(10, 2))
    ax[0].plot(drifts)
    ax[1].plot(data['return'].iloc[steps:].values)
    ax[2].plot([abs(i-j) for i, j in zip(drifts, data['return'].iloc[steps:].values)])
    [ax[i].set_title(labels[i]) for i in range(len(labels))]
""")

st.image("./Resources/Hasil Simulasi 1.png")

st.code("""
n = int(len(data)/2)
histo = data.iloc[:n]
histo['return'] = [None]+[(i-j)/j for (i,j) in zip(histo['Close'].iloc[1:],histo['Close'].iloc[0:-1])]
histo = histo.dropna()
histo)
""")

st.image("./Resources/Hasil Simulasi 2.png")

st.code("""
    harga_pred = simulated_paths
    harga_act = data['Adj Close'][steps-1:].values
    plt.figure(figsize=(10, 6))
    index = data.index[steps-1:]
    plt.plot(index, harga_pred, label='Predicted')
    plt.plot(index, harga_act, label='Actual')
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    plt.grid(True)
    plt.legend()
    plt.show()
""")

st.image("./Resources/Hasil Simulasi 3.png")

st.code("""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(index, [abs(i-j) for i, j in zip(harga_pred, harga_act)], '.-')
    ax[0].set_title('Absolute Error of Prediction Price')
    ax[1].plot(index, [abs(i-j)/j*100 for i, j in zip(harga_pred, harga_act)], '.-')
    ax[1].set_title('Relative Absolute Error of Prediction Price (in %)')
    [ax[i].tick_params(axis='x', labelrotation=40) for i in range(2)]
""")

st.image("./Resources/Hasil Simulasi 4.png")

st.markdown("""
Simulasi hasilnya kemudian ditampilkan dalam bentuk grafik untuk analisis lebih lanjut.
####
### Next Visit Analysis & Conclusion Page on the Left Navigation Bar
""")
