# Importing the required packages
import matplotlib
from pkg_resources import require
import streamlit as st
from streamlit.state.session_state import SessionState
import yfinance as yf
import pandas as pd
import matplotlib.gridspec as gridspec
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
import predictor as predictor

st.set_page_config(page_title='Stocks Analyzer', layout="wide")


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
        to_date = st.date_input('To', datetime.today()-timedelta(days=1), max_value=datetime.today()-timedelta(days=1))
    if(ticker!='' and to_date!=''):
        try:
            data = yf.download(ticker, start=from_date, end=to_date+timedelta(days=1), progress=False)
            company = yf.Ticker(ticker)
        except:
            st.write('Please enter a valid ticker symbol')
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

        model, mae, rme, rmse, score = predictor.train_model(y = data['Close'],
                                                             x = data[['Open', 'High', 'Low']]
                                                        )
        
        pred_plot_pane, pred_config_pane = st.columns([4,1])

        fig, temp = predict(model, data[-1*int(st.session_state['training_period']):], st.session_state['n_predictor'])
        with pred_plot_pane:
            st.plotly_chart(fig, use_container_width=True)
        with pred_config_pane:
            st.selectbox(label="Select training period (days)", options=[7, 15, 30, 90, 180, 365, 730], index=2, on_change=handle_change, key="training_select")
            st.number_input(label="Predict for (n)?", min_value=1, max_value=60, value=10, on_change=handle_change, key="predict_for_n")
        st.table(temp)

        st.write(mae, rme, rmse, score)

    else:
        pass

    

def predict(model, data, n):
    import plotly.express as xp
    required = data['Close']
    a1 = data['Open'].tail(n).to_list()
    for i in range(n):
        a1.append(sum(a1[-n:])/n)

    a2 = data['High'].tail(n).to_list()
    for i in range(n):
        a2.append(sum(a2[-n:])/n)

    a3 = data['Low'].tail(n).to_list()
    for i in range(n):
        a3.append(sum(a3[-n:])/n)

    base = data.index[-1]
    dates = (data.index[-n:]).to_list() + [base + timedelta(days=x) for x in range(1,n+1)]

    temp = pd.DataFrame({
        'Date': dates,
        'Open': a1,
        'High': a2,
        'Low': a3
    })
    temp.set_index('Date', inplace=True)
    temp['Close'] = model.predict(temp[['Open', 'High', 'Low']])
    
    required = required.append(temp['Close'][-n:])
    fig = xp.line(required, x=required.index, y='Close')
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'paper', y0= 0, y1= 1,
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



# def plot_main(data):
#     line_bar_graph = plt.figure(constrained_layout=True)
#     gs = gridspec.GridSpec(2, 2, figure=line_bar_graph)
#     ax1 = line_bar_graph.add_subplot(gs[0,:],ylabel="Closing Price")
#     data['Close'].plot(ax=ax1,color='r', lw=1.5, legend= True)

#     line_bar_graph.tight_layout(pad=2.0)

#     # bar graph for volume
#     gs = gridspec.GridSpec(2, 2, figure=line_bar_graph)
#     ax2 = line_bar_graph.add_subplot(gs[1,:],ylabel="Volume")
#     bar_graph = ax2.bar(data.index, data["Volume"], color = "b", width = 1)
#     st.pyplot(line_bar_graph)


def desc_graphs(data):
    from matplotlib.pyplot import figure
    import numpy as np
    import plotly.express as px

    close = data[st.session_state.base_plot].to_numpy()

    window_size = st.session_state.window_size
    n_sma = np.convolve(close, np.ones(window_size)/window_size, mode='valid')
    twelve_sma = np.convolve(close, np.ones(12)/12, mode='valid')
    tsix_sma = np.convolve(close, np.ones(26)/26, mode='valid')
    
    x = np.arange(data['Close'].size)
    fit = np.polyfit(x, data['Close'], deg=st.session_state.trend_degree)
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
    data['ema'] = data['Adj Close'].ewm(span = window_size).mean()
    data = data.set_index('Date')

    return data



if __name__ == "__main__":
    main()