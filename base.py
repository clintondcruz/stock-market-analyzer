# Importing the required packages
import matplotlib
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.gridspec as gridspec
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
import plotly.graph_objects as go


#@st.cache(suppress_st_warning=True)
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
        from_date = st.date_input('From', date(2018,1,1), min_value=date(2000,1,1), max_value=datetime.today() - timedelta(days=2))
    with date_col2:
        to_date = st.date_input('To', datetime.today() - timedelta(days=1), max_value=datetime.today() - timedelta(days=1))
    if(ticker!='' and to_date!=''):

        data = yf.download(ticker, start=from_date, end=to_date, progress=False)
        company = yf.Ticker(ticker)
        data.describe()

        one_day_growth_col, description_col = st.columns([1,2])
        with one_day_growth_col:
            try:
                st.metric(  label="1Day Growth",
                value=data['Close'][-1].round(2),
                delta=(data['Close'][-1] - data['Close'][-2]).round(2),
                delta_color="normal"
                )
            except:
                st.write("#### Please enter a wider date range")
            
        with description_col:
            st.write(f"{ticker.upper()}, on {to_date} opened at *${data['Open'][-1].round(2)}* and closed at ${data['Close'][-1].round(2)}.\
                        The total volume of stocks traded that day was {(data['Volume'][-1]/10**6).round(2)}M and the highest close price in the selected timeframe has been ${data['Close'].round(2).max()}.")

        plot_main(data)
        with st.expander('Read more info about the company'):
            st.write(company.info['longBusinessSummary'])
        
        st.table(descriptive(data))
        desc_graphs(data)

    else:
        pass



def descriptive(data):
    info = data.describe()[['Open', 'Close', 'Adj Close']].round(2).transpose()

    info['cov'] = info['std']/info['mean']
    info['range'] = info['max'] - info['min']
    info['iqr'] = info['75%'] - info['25%']

    info.drop(columns = ['count', 'min', 'max'], axis=1, inplace = True)
    return info



def plot_main(data):
    line_bar_graph = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=line_bar_graph)
    ax1 = line_bar_graph.add_subplot(gs[0,:],ylabel="Closing Price")
    data['Close'].plot(ax=ax1,color='r', lw=1.5, legend= True)

    line_bar_graph.tight_layout(pad=2.0)

    # bar graph for volume
    gs = gridspec.GridSpec(2, 2, figure=line_bar_graph)
    ax2 = line_bar_graph.add_subplot(gs[1,:],ylabel="Volume")
    bar_graph = ax2.bar(data.index, data["Volume"], color = "b", width = 1)
    st.pyplot(line_bar_graph)


def numpy_ewma_vectorized(data):
    import numpy as np
    window = 10
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def desc_graphs(data):
    from matplotlib.pyplot import figure
    import numpy as np
    import plotly.express as px

    close = data['Close'].to_numpy()

    window_size = 50
    n_sma = np.convolve(close, np.ones(window_size)/window_size, mode='valid')
    twelve_sma = np.convolve(close, np.ones(12)/12, mode='valid')
    tsix_sma = np.convolve(close, np.ones(26)/26, mode='valid')
    
    x = np.arange(data['Close'].size)
    fit = np.polyfit(x, data['Close'], deg=3)
    fit_function = np.poly1d(fit)
    trend_op = fit_function(x)

    n_sma = np.append(np.zeros(close.shape[0]-n_sma.shape[0]), n_sma)
    twelve_sma = np.append(np.zeros(close.shape[0]-twelve_sma.shape[0]), twelve_sma)
    tsix_sma = np.append(np.zeros(close.shape[0]-tsix_sma.shape[0]), tsix_sma)

    data.reset_index(inplace=True) 
    data['n_sma'] = pd.Series(n_sma)
    data['12_sma'] = pd.Series(twelve_sma)
    data['26_sma'] = pd.Series(tsix_sma)
    data['trend'] = pd.Series(trend_op)


    figure(figsize=(9, 6), dpi=80)
    fig = px.line(data, x='Date', y=['Close', 'n_sma', 'trend'])
    plt.legend()
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()