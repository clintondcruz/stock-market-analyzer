# Importing the required packages
import streamlit as st, yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
from predictive import inferential              # imported a user-created package
from descriptive import descriptive
from macd import macd_bs


st.set_page_config(page_title='Stocks Analyzer', layout="wide")

@st.cache
def read_ticker_list(file_name):
    return pd.read_csv(file_name)['Symbol - Name']


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
        ticker_list = read_ticker_list('filtered_tickers.csv').to_list()  # READS THE LIST OF TICKER SYMBOL FROM THE CSV FILE
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

        st.write(' ')
        st.markdown('Enter date range for descriptive analysis')
        COL_50_PERCENT_LEFT, COL_50_PERCENT_RIGHT = st.columns(2)
        with COL_50_PERCENT_LEFT:
            # ENSURES THAT THE FROM DATE MEETS REQ. VALIDATION => SHOULD BE BETWEEN 2000/01/01 AND MAX(TODAY-32 DAYS, MIN DATE)
            from_date = st.date_input(  'From',
                                        value=max(date(2021,9,11), min_date),
                                        min_value=max(date(2000,1,1), min_date),
                                        max_value=max(datetime.today() - timedelta(days=3), min_date))
            
        with COL_50_PERCENT_RIGHT:
            # ENSURES THAT THE TO DATE MEETS REQ. VALIDATION => SHOULD BE BETWEEN (FROM DATE + 3 DAYS) TO MAX DATE
            to_date = st.date_input(    'To',
                                        value=max_date,
                                        min_value=from_date + timedelta(days=3),
                                        max_value=max_date)


    # ONCE THE DATA HAS BEEN DOWNLAODED AND FILTERED, PROCEED AHEAD
    try:
        data = st.session_state.data.reset_index()
        data = data[(data['Date'].dt.date>=from_date) & (pd.to_datetime(data['Date']).dt.date<=to_date)]    # FILTERING THE DATA AS PER THE DATE RANGE
        data.set_index('Date', inplace=True)
        company = st.session_state.company
    
        #data.describe()

        
        basic_information(data, company)                        # FETCHES & DISPLAYS BASIC DETAILS OF THE COMPANY LIKE LOGO, WEBSITE


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
            st.table(overview(data))                            # SHOWS MEAN, MEDIAN, INTER QUARTILE RANGE (IQR) FOR VARIOUS METRICS
            fundamental_analysis(company)                       # DISPLAYS FUNDAMENTAL (ANALYSIS) INFORMATION ABOUT THE COMPANY

        st.markdown("-----")
        descriptive(data)                                       # PERFORMS DESCRIPTIVE ANALYSIS
        
        st.markdown("-----")
        with st.container():
            """### PREDICTIVE ANALYSIS"""
            inferential()                                       # PERFORMS INFERENTIAL ANALYSIS

        """### MACD & BUY-SELL Recommendation Graphs"""
        macd_bs(data)

    except:
        st.error('Not enough data to do anything. Please select a wider date range.')


def get_info(company, reqs: list):
    _ = {}                                              # TEMP VARIABLE
    # FETCHING DATA IN A LOOP
    for req in reqs:
        try:
            _[req] = company.info[req]
        except:
            continue
    st.table(pd.DataFrame(_, index=[0]).transpose())    # CONVERTING FETCHED DATA INTO A DATAFRAME


def fundamental_analysis(company):
    # THIS SECTION DEALS WITH THE FUNDAMENTAL ANALYSIS OF THE STOCK FOR THE SELECTED TIME-PERIOD
    # DISPLAYS VARIOUS METRICS WHICH ARE USED TO EVALUATE THE PRICE OF A STOCK
    # THE LOGIC IMPLEMENTED HERE IS THAT THE FUNCTION CALLS GET_INFO TO FETCH THE REQUIRED DETAILS
    # GET_INFO FETCHES THE DATA AND PRESENTS INTO A TABLE DYNAMICALLY

    COL_50_PERCENT_LEFT, COL_50_PERCENT_RIGHT = st.columns(2) # DEFINED 2 COLUMNS OF WIDTH 50% EACH
    with COL_50_PERCENT_LEFT:
        """### Business details"""
        try:    get_info(company, ['sector', 'industry'])
        except: st.warning('No company business details found')

        """### Margins"""
        try:    get_info(company, ['ebitdaMargins', 'profitMargins', 'revenueGrowth'])
        except: st.warning('No margins data found')
    
    with COL_50_PERCENT_RIGHT:
        """### Metrics"""
        try:
            get_info(company, [ 'beta', 'pegRatio', 'trailingEPS', 'forwardEPS', 'revenuePerShare',
                            'currentRatio', 'payoutRatio', 'trailingPE', 'forwardPE', 'marketCap',
                            'enterpriseValue'])
        except:
            st.warning('No metrics data found')


def basic_information(data, company):
    COL_LEFT, COL_CENTER, COL_RIGHT = st.columns([2,4,2]) # DEFINED 2 COLUMNS OF WIDTH 33% AND 67% RESP.
    with COL_LEFT:
        try:                                        # Compute 1 Day growth (Close Price) from latest and Day-1 values
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
        except: logo = "#"
        st.markdown(f"<a href='{website}'><img src='{logo}' alt='No logo found' style='width:75px, height:75px'></a>", unsafe_allow_html=True)
 
@st.cache
def overview(data):
    info = data.describe()[['Open', 'Close', 'Adj Close']].round(2).transpose()

    info['cov'] = info['std']/info['mean']
    info['range'] = info['max'] - info['min']
    info['iqr'] = info['75%'] - info['25%']

    info.drop(columns = ['count', 'min', 'max'], axis=1, inplace = True)
    return info


if __name__ == "__main__":
    main()
