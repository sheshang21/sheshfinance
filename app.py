import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings, time, re, requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Ultimate Stock Analysis", page_icon="📊", layout="wide")

st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.metric-card{background:#f8f9fb;padding:0.6rem 0.9rem;border-radius:8px;border-left:4px solid #1f77b4}
.rec-buy{background:#d4edda;border-left:4px solid #28a745;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-hold{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-sell{background:#f8d7da;border-left:4px solid #dc3545;padding:1rem;border-radius:8px;margin:0.5rem 0}
.formula-box{background:#f0f8ff;padding:1.5rem;border-radius:8px;border-left:4px solid #4682b4;margin:1rem 0}
.interp-box{background:#fff9e6;padding:1rem;border-radius:8px;border-left:4px solid #ffa500;margin:0.8rem 0}
.example-box{background:#f0fff0;padding:1rem;border-radius:8px;border-left:4px solid #32cd32;margin:0.8rem 0}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end, freq='1d'):
    for i in range(3):
        try:
            if i > 0: time.sleep(2**i)
            s = yf.download(f"{ticker}.NS", start, end, interval=freq, progress=False)
            time.sleep(0
