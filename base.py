# Importing the required packages
import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt


def plot_main():
    fig, ax =  plt.subplots()



def main():
    st.write("""
        # Welcome to the stock analyzer
        In this app, you can search for a particular stock/option and specify the required timeline to
        get relevant information which you can use for your analyses.
    """)

    # Feeding in the required Ticker symbol
    ticker = st.text_input('Which stock do you want to search for?')
    date_col1,date_col2 = st.columns(2)
    with date_col1:
        from_date = st.date_input('From', date(2018,1,1), min_value=date(2000,1,1))
    with date_col2:
        to_date = st.date_input('To', datetime.today() - timedelta(days=1), max_value=datetime.today() - timedelta(days=1))
    if(ticker!='' and to_date!=''):

        data = yf.download(ticker, start=from_date, end=to_date, progress=False)
        company = yf.Ticker(ticker)
        #data.reset_index(inplace=True)
        data.describe()
        #st.line_chart(data['Close'])
        plot_main()

        financial_mini_col, desc_col = st.columns(2)
        with financial_mini_col:
            st.metric(  label="Net Close Price",
                        value=data['Close'][-1].round(2),
                        delta=(data['Close'][-1] - data['Close'][-2]).round(2),
                        delta_color="normal"
            )
            st.write("Demo")
        with desc_col:
            #sns.lineplot(x=data.index, y=data['Close'])
            pass
        fig, ax = plt.subplots()
        ax.plot(data['Close'])
            
        #st.bokeh_chart(data['Close'])
    else:
        pass

if __name__ == "__main__":
    main()