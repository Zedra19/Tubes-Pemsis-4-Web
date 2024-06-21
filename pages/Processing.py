import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="Processing", page_icon="📈")

st.header("Processing & Training Model")
