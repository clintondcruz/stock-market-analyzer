import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

ticker = st.text_input('Which stock do you want to search for?')
if(ticker!=''):
    data = yf.download(ticker, start='2021-01-01', end='2021-11-01', progress=False)
    #data.reset_index(inplace=True)
    st.bar_chart(data['Volume'].tail(100))
else:
    pass
