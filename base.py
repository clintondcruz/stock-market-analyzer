# Importing the required packages
from typing import Container
import streamlit as st, yfinance as yf
import pandas as pd, numpy as np
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
from predictive import inferential              # imported a user-created package
import plotly.express as xp


st.set_page_config(page_title='Stocks Analyzer', layout="wide")

@st.cache
def read_ticker_list(file_name):
    return pd.read_csv(file_name)['Symbol - Name'][:8000]


def main():
    # DEFINING SESSION VARIABLES TO STORE DATA IN RUN-TIME
    if "base_plot" not in st.session_state:     # USED TO STORE OUR BASE PLOT FOR THE N_SMA & TREND PLOT
        st.session_state['base_plot'] = "Close"

    if "plot_this" not in st.session_state:     # USED TO STORE OUR SECONDARY DATAPOINT TO PLOT OVER OUR BASE
        st.session_state['plot_this'] = "n_sma"

    if "window_size" not in st.session_state:   # USED TO STORE THE WINDOW SIZE FOR N_SMA
        st.session_state['window_size'] = 1
    
    if "n_predictor" not in st.session_state:   # USED TO STORE THE NUMBER OF DAYS FOR WHICH PREDICTION IS REQ.
        st.session_state['n_predictor'] = 10
    
    if "trend_degree" not in st.session_state:  # USED TO STORE THE DEGREE OF TREND LINE
        st.session_state['trend_degree'] = 1

    if "training_period" not in st.session_state:   # USED TO STORE THE NUM DAYS (LAST N ROWS) OF TRAINING DATA
        st.session_state['training_period'] = 30

    if "ticker" not in st.session_state:        # USED TO STORE OUR TICKER SYMBOL FOR COMPARING OLD AND NEW (ON CHANGE)
        st.session_state['ticker'] = ''

    if "data" not in st.session_state:          # USED TO STORE OUR DOWNLOADED DATAFRAME
        st.session_state['data'] = None

    if "company" not in st.session_state:       # USED TO STORE INFO ABOUT THE COMPANY
        st.session_state.company = None


    # INTRODUCTION TO THE APPLICATION
    st.write("""
        # Welcome to the Stock Analyzer
        In this app, you can search for a particular stock and specify the required timeline to
        get relevant information which you can use for your analyses.
    """)


    # SIDEBAR FOR TICKER SYMBOL AND DATE RANGE SELECTOR
    with st.sidebar:
        ticker_list = read_ticker_list('nasdaq_ticker_list.csv').to_list()  # READS THE LIST OF TICKER SYMBOL FROM THE CSV FILE
        ticker = st.selectbox('Which stock do you want to search for?', options = ticker_list, index=0).split(" - ")[0]     # USED TO FETCH THE TICKER SYMBOL FROM USER

        # CHECKING IF THE TICKER SYMBOL HAS CHANGED
        # IF CHANGE IN TICKER SYMBOL => DOWNLOAD NEW DATASET AND SAVE THE INFO IN SESSION STATE
        # ELSE CONTINUE
        if st.session_state.ticker != ticker:
            st.session_state.data = yf.download(ticker, period="max", progress=False)   # DOWNLOADS THE DATASET FROM YFINANCE
            st.session_state.company = yf.Ticker(ticker)                                # FETCHES INFO ABOUT THE COMPANY
            print(f"downloaded new dataset for {ticker}")
            st.session_state.ticker = ticker                                            # UPDATES TICKER SYMBOL IN THE SESSION VARIABLE

        dummy = st.session_state.data.reset_index()
        min_date = dummy['Date'].iloc[0]                                                # MIN DATE FOR VALIDATION (FIRST DATE FOR WHICH DATA IS AVAILABLE)
        max_date = dummy['Date'].iloc[-1]                                               # MAX DATE FOR VALIDATION (LAST DATE FOR WHICH DATA IS AVAILABLE)

        COL_50_PERCENT_LEFT, COL_50_PERCENT_RIGHT = st.columns(2)
        with COL_50_PERCENT_LEFT:
            # ENSURES THAT THE FROM DATE MEETS REQ. VALIDATION => SHOULD BE BETWEEN 2000/01/01 AND MAX(TODAY-32 DAYS, MIN DATE)
            from_date = st.date_input(  'From',
                                        value=max(date(2021,9,11), min_date),
                                        min_value=date(2000,1,1),
                                        max_value=max(datetime.today() - timedelta(days=32), min_date))
            
        with COL_50_PERCENT_RIGHT:
            # ENSURES THAT THE TO DATE MEETS REQ. VALIDATION => SHOULD BE BETWEEN (FROM DATE + 3 DAYS) TO MAX DATE
            to_date = st.date_input(    'To',
                                        value=max_date,
                                        min_value=from_date + timedelta(days=3),
                                        max_value=max_date)
        

        st.markdown("""------------------------""")                                     # DISPLAYS A HORIZONTAL LINE
        st.text_input(label="ADD A NEW SYMBOL", placeholder='New Ticker Symbol')        # TEXT INPUT FOR ADDING THE NEW SYMBOL
        st.button('Add')                                                                # BUTTON TO TRIGGER SYMBOL ADDITION LOGIC =>
                                                                                        # IF SYMBOL PRESENT THEN DO NOTHING
                                                                                        # ELSE CHECK IF DATA FOR SPECIFIED SYMBOL EXISTS
                                                                                            # IF EXISTS THEN ADD TO THE LIST
                                                                                            # ELSE DISPLAY ERROR MESSAGE


    # ONCE THE DATA HAS BEEN DOWNLAODED AND FILTERED, PROCEED AHEAD
    data = st.session_state.data.reset_index()
    data = data[(data['Date'].dt.date>=from_date) & (pd.to_datetime(data['Date']).dt.date<=to_date)]    # FILTERING THE DATA AS PER THE DATE RANGE
    data.set_index('Date', inplace=True)
    company = st.session_state.company
    #data.describe()

    
    basic_information(data, company)


    # MORE INFO EXPANDER
    with st.expander('Read more info about the company'):
        try:
            try: website = f" For more info, visit: {company.info['website']}"
            except: pass
            st.write(company.info['longBusinessSummary'] + website)   # DISPLAYS LONG BUSINESS SUMMARY OF THE COMPANY
        except:
            st.write('No business info found.')             # DEFAULT MESSAGE IF THE INFO IS NOT PRESENT
    

    # DISPLAYS THE DESCRIPTIVE TABLE
    with st.container():
        """## FUNDAMENTAL ANALYSIS"""
        st.table(descriptive(data))                         # SHOWS MEAN, MEDIAN, INTER QUARTILE RANGE (IQR) FOR VARIOUS METRICS
        fundamental_analysis(company)                       # DISPLAYS FUNDAMENTAL (ANALYSIS) INFORMATION ABOUT THE COMPANY

    st.markdown("-----")

    with st.container():
        """## DESCRIPTIVE ANALYSIS"""
        with st.container():
            """### Moving averages and trend lines"""
            COL_80_PERCENT, COL_20_PERCENT = st.columns([4,1])          # DEFINED 2 COLUMNS OF WIDTH 80% AND 20% RESP.
            # PLOTTING SMA, TRENDLINE GRAPHS AGAINST CLOSE AND ADJ CLOSE
            data = base_graphs(data)                                    # COMPUTES VALUES FOR VARIOUS METRICS TO BE PLOTTED
            with COL_80_PERCENT:
                    plot_graph(data)                                    # PLOTS THE GRAPH
            with COL_20_PERCENT:
                st.number_input(value=1, label="Enter the value of (n)", on_change=handle_change, key="window", min_value=1, max_value=max(len(data)//4,1))     # WINDOW SIZE INPUT FOR N_SMA => RANGES FROM 1 TO 25% OF LENGTH OF DATA
                st.number_input(value=1, label="Enter the degree of trendline", on_change=handle_change, key="degree", min_value=1, max_value=3)                # INPUT FOR DEGREE OF TREND LINE => RANGES FROM 1 TO 3
                st.radio("Base plot-", ['Close', 'Open'], on_change=handle_change, key='base')
                st.radio("Plot base against-", ['n_sma', 'ema', 'trend'], on_change=handle_change, key='versus')

        with st.container():
            """### Candle stick graph"""
            try:
                # CANDLE-STICK PLOT
                plot_candlestick(data)
            except:
                st.error('Not enough data to display the candle stick graph')

            """### Volume area graph (Volume in millions)"""
            try:
                # VOLUME AREA GRAPH
                st.area_chart(data.Volume/10**6, use_container_width=True)
            except:
                st.error('Not enough data to display the area graph')

    st.markdown("---------")
    with st.container():
        """### PREDICTIVE ANALYSIS"""
        inferential()

    try:
        """### MACD & BUY-SELL Recommendation Graphs"""
        macd_bs(data)
    except:
        st.error('Not enough data to display the MACD or buy/sell graph')


def get_info(company, reqs: list):
    _ = {}
    for req in reqs:
        try:
            _[req] = company.info[req]
        except:
            continue
    st.table(pd.DataFrame(_, index=[0]).transpose())


#st.cache(allow_output_mutation=True)
def fundamental_analysis(company):
    # THIS SECTION DEALS WITH THE FUNDAMENTAL ANALYSIS OF THE STOCK FOR THE SELECTED TIME-PERIOD
    # DISPLAYS VARIOUS METRICS WHICH ARE USED TO EVALUATE THE PRICE OF A STOCK

    COL_50_PERCENT_LEFT, COL_50_PERCENT_RIGHT = st.columns(2) # DEFINED 2 COLUMNS OF WIDTH 50% EACH
    info = company.info # FETCHES STOCK RELATED INFORMATION FROM THE API
    with COL_50_PERCENT_LEFT:
        """### Business details"""
        get_info(company, ['sector', 'industry'])
        """### Margins"""
        get_info(company, ['ebitdaMargins', 'profitMargins', 'revenueGrowth'])

    
    with COL_50_PERCENT_RIGHT:
        """### Metrics"""
        get_info(company, [ 'beta', 'pegRatio', 'trailingEPS', 'forwardEPS', 'revenuePerShare',
                            'currentRatio', 'payoutRatio', 'trailingPE', 'forwardPE', 'marketCap',
                            'enterpriseValue'])


def basic_information(data, company):
    COL_LEFT, COL_CENTER, COL_RIGHT = st.columns([2,4,2]) # DEFINED 2 COLUMNS OF WIDTH 33% AND 67% RESP.
    with COL_LEFT:
        try:
            st.metric(  label="1Day Growth",
            value=st.session_state.data['Close'][-1].round(2),
            delta=(st.session_state.data['Close'][-1] - st.session_state.data['Close'][-2]).round(2),
            delta_color="normal"
            )
        except:
            st.write("#### Please enter a wider date range")
        
    with COL_CENTER:
        st.write(f"{company.info['shortName']}, on {data.index[-1].date()} opened at ${data['Open'][-1].round(2)} and closed at ${data['Close'][-1].round(2)}.\
                    The highest close price in the selected timeframe has been ${data['High'].round(2).max()}.")

    with COL_RIGHT:
        try: website = company.info['website']
        except: website = '#'
        try: logo = company.info['logo_url']
        except: logo = '#'
        st.markdown(f"<a href='{website}'><img src='{logo}'></a>", unsafe_allow_html=True)

def macd_bs(data): 
    ShortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    LongEMA = data['Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span = 9, adjust=False).mean()

    data['MACD'] = MACD
    data["Signal Line"]= signal
    fig = xp.line(data[['MACD', 'Signal Line']], x=data.index, y=['MACD', 'Signal Line'])
    st.plotly_chart(fig, use_container_width=True)


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

    marker = buy_sell(data)
    data['Buy_Signal_Price'] = marker[0]
    data['Sell_Signal_Price'] = marker[1]

    
    buy_sell_graph = plt.figure(figsize = (12.2, 4.5))
    plt.scatter(data.index, data['Buy_Signal_Price'], label = 'Buy', color ="green",marker = '^', alpha =1)
    plt.scatter(data.index, data['Sell_Signal_Price'], label = 'Sell', color ="red",marker = 'v', alpha =1)
    plt.plot(data['Close'], label = 'Close Price', alpha =0.7)
    plt.title("Close Price Buy & Sell Signals")
    plt.xlabel('Date')
    plt.xticks(rotation =45)
    plt.ylabel("Close Price USD ($)")
    plt.legend(loc='upper center')

    # st.pyplot(MACD_graph)
    st.pyplot(buy_sell_graph)


def handle_change():
    if st.session_state.versus:
        st.session_state.plot_this = st.session_state.versus
    if st.session_state.base:
        st.session_state.base_plot = st.session_state.base
    if st.session_state.window_size:
        st.session_state.window_size = st.session_state.window
    if st.session_state.degree:
        st.session_state.trend_degree = st.session_state.degree

        
@st.cache
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


def base_graphs(data):
    close = data[st.session_state.base_plot].to_numpy()
    close = close.astype(np.float)

    window_size = st.session_state.window_size
    n_sma = np.convolve(close, np.ones(window_size)/window_size, mode='valid')
    
    x = np.arange(data[st.session_state.base_plot].size)
    fit = np.polyfit(x, data[st.session_state.base_plot], deg=st.session_state.trend_degree)
    fit_function = np.poly1d(fit)
    trend_op = fit_function(x)


    temp = np.zeros(close.size - n_sma.size) + np.nan
    n_sma = np.append(temp, n_sma)

    data.reset_index(inplace=True) 
    data['n_sma'] = pd.Series(n_sma)
    data['trend'] = pd.Series(trend_op)
    data['ema'] = data[st.session_state.base_plot].ewm(span = window_size).mean()
    data.set_index('Date', inplace=True)

    return data


if __name__ == "__main__":
    main()