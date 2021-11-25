# Importing the required packages
import matplotlib
from pandas._config.config import options
from pkg_resources import require
import streamlit as st
from streamlit.state.session_state import SessionState
import yfinance as yf
import pandas as pd
import matplotlib.gridspec as gridspec
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
import predictor as predictor
import numpy as np

st.set_page_config(page_title='Stocks Analyzer', layout="wide")


# def download_data(ticker, from_date, to_date):
#     data = yf.download(ticker, start=from_date, end=to_date)
#     return data

#@st.cache(suppress_st_warning=True)
def main():
    if "base_plot" not in st.session_state:
        st.session_state['base_plot'] = "Close"

    if "plot_this" not in st.session_state:
        st.session_state['plot_this'] = "n_sma"

    if "window_size" not in st.session_state:
        st.session_state['window_size'] = 10
    
    if "n_predictor" not in st.session_state:
        st.session_state['n_predictor'] = 10
    
    if "trend_degree" not in st.session_state:
        st.session_state['trend_degree'] = 1

    if "training_period" not in st.session_state:
        st.session_state['training_period'] = 30

    if "ticker" not in st.session_state:
        st.session_state['ticker'] = ''

    if "data" not in st.session_state:
        st.session_state['data'] = None

    if "company" not in st.session_state:
        st.session_state.company = None

    st.write("""
        # Welcome to the stock analyzer
        In this app, you can search for a particular stock/option and specify the required timeline to
        get relevant information which you can use for your analyses.
    """)

    # Feeding in the required Ticker symbol
    #ticker = st.text_input('Which stock do you want to search for?')
    #ticker = st.text_input('Which stock do you want to search for?', value="AAPL")
    ticker_list = pd.read_csv('nasdaq_ticker_list.csv')['Symbol'][:300].to_list()
    ticker = st.selectbox('Which stock do you want to search for?', options = ticker_list, index=0)
    date_col1,date_col2 = st.columns(2)
    with date_col1:
        from_date = st.date_input('From', date(2018,1,1), min_value=date(2000,1,1), max_value=datetime.today() - timedelta(days=32))
    with date_col2:
        to_date = st.date_input('To', datetime.today()-timedelta(days=1), max_value=datetime.today()-timedelta(days=1))
    #if(ticker!='' and to_date!=''):
    # try:
    if st.session_state.ticker != ticker:
        st.session_state.data = yf.download(ticker, period="max", progress=False)
        st.session_state.company = yf.Ticker(ticker)
        print(f"downloaded new dataset for {ticker}")
        st.session_state.ticker = ticker
    # except:
    #     st.write('Please enter a valid ticker symbol')
    #data = st.session_state.data[[(st.session_state.data.index>=from_date) & (st.session_state.data.index<=to_date)]]
    data = st.session_state.data.reset_index()
    data = data[(data['Date'].dt.date>=from_date) & (pd.to_datetime(data['Date']).dt.date<=to_date)]
    data.set_index('Date', inplace=True)
    company = st.session_state.company
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
        st.write(f"{ticker.upper()}, on {data.index[-1].date()} opened at *${data['Open'][-1].round(2)}* and closed at ${data['Close'][-1].round(2)}.\
                    The total volume of stocks traded that day was {(data['Volume'][-1]/10**6).round(2)}M and the highest close price in the selected timeframe has been ${data['Close'].round(2).max()}.")

    #plot_main(data)
    with st.expander('Read more info about the company'):
        st.write(company.info['longBusinessSummary'])
    
    st.table(descriptive(data))
    data = desc_graphs(data)
    graph_pane, radio_buttons_pane = st.columns([4,1])
    with graph_pane:
        plot_graph(data)
    with radio_buttons_pane:
        st.number_input(value=10, label="Enter the value of (n)", on_change=handle_change, key="window")
        st.number_input(value=1, label="Enter the degree of trendline", on_change=handle_change, key="degree", min_value=1, max_value=3)
        st.radio("Base plot-", ['Close', 'Adj Close'], on_change=handle_change, key='base')
        st.radio("Plot Close against-", ['n_sma', 'ema', 'trend'], on_change=handle_change, key='versus')

    plot_candlestick(data)

    st.area_chart(data.Volume/10**6)

    training_period_index = -1*int(st.session_state['training_period'])
    model, mae, rme, rmse, score = predictor.train_model(y = data['Close'][training_period_index:],
                                                            x = data[['Open', 'High', 'Low']][training_period_index:]
                                                    )

    pred_plot_pane, pred_config_pane = st.columns([4,1])

    fig, temp = predict(model, data[-1*int(st.session_state['training_period']):], st.session_state['n_predictor'])

    with pred_plot_pane:
        st.write("### Predicting using Random Forest model")
        st.plotly_chart(fig, use_container_width=True)
    with pred_config_pane:
        st.selectbox(label="Select training period (historic days)", options=[7, 15, 30, 90, 180, 365, 730], index=2, on_change=handle_change, key="training_select")
        st.number_input(label="Predict for days (n)?", min_value=1, max_value=15, value=10, on_change=handle_change, key="predict_for_n")
        st.table(pd.DataFrame({'Values': [mae,rme,rmse,score]
        }, index=['Mean Absolute Err', 'Root Mean Err', 'RMSE', 'R Squared']))

    

    st.table(temp[-1*int(st.session_state['n_predictor']):])

    

    #MACD
    ShortEMA = data['Adj Close'].ewm(span=12, adjust=False).mean()
    LongEMA = data['Adj Close'].ewm(span=26, adjust=False).mean()
    MACD = LongEMA - ShortEMA
    signal = MACD.ewm(span = 9, adjust=False).mean()

    MACD_graph = plt.figure(figsize=(12.2,4.5))
    plt.plot(data.index,MACD, label = "{} MACD".format(ticker), color ="r")
    plt.plot(data.index,signal, label = "Signal Line", color ="b")
    plt.legend(loc='upper left')
    plt.xticks(rotation = 45)


    data['MACD'] =MACD
    data["Signal Line"]= signal

    # buy_sell graph
    def buy_sell(signal):
        Buy = []
        Sell = []
        flag =-1

        for i in range(0,len(signal)):
            if signal['MACD'][i] > signal['Signal Line'][i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal['Close'][i])
                    flag =1
                else:
                    Buy.append(np.nan)
            elif signal['MACD'][i] < signal['Signal Line'][i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal['Close'][i])
                    flag =0
                else:
                    Sell.append(np.nan)
            else:
                Sell.append(np.nan)
                Buy.append(np.nan)
        return(Buy, Sell)

    a = buy_sell(data)
    data['Buy_Signal_Price'] = a[0]
    data['Sell_Signal_Price'] = a[1]

    
    buy_sell_graph = plt.figure(figsize = (12.2, 4.5))
    plt.scatter(data.index, data['Buy_Signal_Price'], label = 'Buy', color ="green",marker = '^', alpha =1)
    plt.scatter(data.index, data['Sell_Signal_Price'], label = 'Sell', color ="red",marker = 'v', alpha =1)
    plt.plot(data['Close'], label = 'Close Price', alpha =0.36)
    plt.title("Close Price Buy & Sell Signals")
    plt.xlabel('Date')
    plt.xticks(rotation =45)
    plt.ylabel("Close Price USD ($)")
    plt.legend(loc='upper left')

    st.pyplot(MACD_graph)
    st.pyplot(buy_sell_graph)

    

def predict(model, data, n):
    import plotly.express as xp
    sma_window = 9
    required = data['Close']
    num_rows = max(sma_window, n)

    a1 = data['Open'].tail(num_rows).to_list()
    a1.append(data['Close'][-1])
    for i in range(num_rows-1):
        a1.append(sum(a1[-sma_window:])/(sma_window))

    a2 = data['High'].tail(num_rows).to_list()
    for i in range(num_rows):
        a2.append(sum(a2[-sma_window:])/sma_window)

    a3 = data['Low'].tail(num_rows).to_list()
    for i in range(num_rows):
        a3.append(sum(a3[-sma_window:])/sma_window)


    base = data.index[-1]
    dates = (data.index[-num_rows:]).to_list() + [base + timedelta(days=x) for x in range(1,num_rows+1)]

    temp = pd.DataFrame({
        'Date': dates,
        'Open': a1,
        'High': a2,
        'Low': a3
    })
    temp.set_index('Date', inplace=True)
    temp['Close'] = model.predict(temp[['Open', 'High', 'Low']])
    
    temp = temp[temp.index>base]
    required = required.append(temp['Close'][:n])

    fig = xp.line(required, x=required.index, y='Close', markers=True)
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'paper', y0 = 0, y1 = 1,
        xref= 'x', x0= required.index[-n], x1= required.index[-n],
        line = dict(
            color='red',
            width=1.5,
            dash='dashdot'
        )
        )
    ])

    return (fig,temp)





def handle_change():
    if st.session_state.versus:
        st.session_state.plot_this = st.session_state.versus
    if st.session_state.base:
        st.session_state.base_plot = st.session_state.base
    if st.session_state.window_size:
        st.session_state.window_size = st.session_state.window
    if st.session_state.degree:
        st.session_state.trend_degree = st.session_state.degree
    if st.session_state.predict_for_n:
        st.session_state.n_predictor = st.session_state.predict_for_n
    if st.session_state.training_select:
        st.session_state.training_period = st.session_state.training_select
        


def descriptive(data):
    info = data.describe()[['Open', 'Close', 'Adj Close']].round(2).transpose()

    info['cov'] = info['std']/info['mean']
    info['range'] = info['max'] - info['min']
    info['iqr'] = info['75%'] - info['25%']

    info.drop(columns = ['count', 'min', 'max'], axis=1, inplace = True)
    return info


def plot_graph(data):
    import plotly.express as px
    fig = px.line(data, x=data.index, y=[st.session_state.base_plot, st.session_state.plot_this])
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick(data):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data.Open,
                high=data.High,
                low=data.Low,
                close=data.Close)])
    st.plotly_chart(fig, use_container_width=True)



def desc_graphs(data):
    from matplotlib.pyplot import figure
    import numpy as np
    import plotly.express as px

    close = data[st.session_state.base_plot].to_numpy()

    window_size = st.session_state.window_size
    n_sma = np.convolve(close, np.ones(window_size)/window_size, mode='valid')
    
    x = np.arange(data['Close'].size)
    fit = np.polyfit(x, data['Close'], deg=st.session_state.trend_degree)
    fit_function = np.poly1d(fit)
    trend_op = fit_function(x)

    n_sma = np.append(np.zeros(close.shape[0]-n_sma.shape[0]), n_sma)

    data.reset_index(inplace=True) 
    data['n_sma'] = pd.Series(n_sma)
    data['trend'] = pd.Series(trend_op)
    data['ema'] = data['Adj Close'].ewm(span = window_size).mean()
    data.set_index('Date', inplace=True)

    return data



if __name__ == "__main__":
    main()