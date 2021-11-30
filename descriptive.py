import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go


def descriptive(data):
    with st.container():
        st.write("## DESCRIPTIVE ANALYSIS")
        with st.container():
            st.write("### Moving averages and trend lines")
            COL_80_PERCENT, COL_20_PERCENT = st.columns([4,1])          # DEFINED 2 COLUMNS OF WIDTH 80% AND 20% RESP.
            # PLOTTING SMA, TRENDLINE GRAPHS AGAINST CLOSE AND ADJ CLOSE
            data = base_graph_data(data)                                    # COMPUTES VALUES FOR VARIOUS METRICS TO BE PLOTTED
            with COL_80_PERCENT:
                    plot_base_graph(data)                               # PLOTS THE GRAPH
            with COL_20_PERCENT:
                st.number_input(value=1, label="Enter the value of (n) to compute n_sma", on_change=handle_change, key="window", min_value=1, max_value=max(len(data)//4,1))     # WINDOW SIZE INPUT FOR N_SMA => RANGES FROM 1 TO 25% OF LENGTH OF DATA
                st.number_input(value=1, label="Enter the degree of trendline", on_change=handle_change, key="degree", min_value=1, max_value=3)                # INPUT FOR DEGREE OF TREND LINE => RANGES FROM 1 TO 3
                st.radio("Base plot-", ['Close', 'Open'], on_change=handle_change, key='base')                      # RADIO BUTTON TO ENABLE THE USER TO CHOOSE THE BASE PLOT
                st.radio("Plot base against-", ['n_sma', 'ema', 'trend'], on_change=handle_change, key='versus')    # RADIO BUTTON TO ENABLE THE USER TO CHOOSE THE VERSUS (DEPENDENT/SECONDARY) PLOT

        with st.container():
            st.write("### Candle stick graph")
            try:
                # CANDLE-STICK PLOT
                plot_candlestick(data)          # FUNCTION CALL TO PLOT CANDLE STICK GRAPH BASIS THE DATE RANGE SELECTED BY THE USER
            except:
                st.error('Not enough data to display the candle stick graph')

            st.write("### Volume area graph (Volume in millions)")      # BASIC ERROR HANDLING MECHANISM
            try:
                # VOLUME AREA GRAPH
                st.area_chart(data.Volume/10**6, use_container_width=True)      # PLOTS AN AREA GRAPH ON VOLUME COLUMN. VOLUME IS FIRST CONVERTED INTO MILLIONS BEFORE PLOTTING
            except:
                st.error('Not enough data to display the area graph')   # BASIC ERROR HANDLING MECHANISM


# TO HANDLE SESSION CALLBACKS
def handle_change():
    if st.session_state.versus:
        st.session_state.plot_this = st.session_state.versus    # SESSION VARIABLE TO STORE SECONDARY PLOT
    if st.session_state.base:
        st.session_state.base_plot = st.session_state.base      # SESSION VARIABLE TO STORE BASE PLOT
    if st.session_state.window_size:
        st.session_state.window_size = st.session_state.window  # SESSION VARIABLE TO STORE WINDOW SIZE (FOR N-SMA & EMA COMPUTATION)
    if st.session_state.degree:
        st.session_state.trend_degree = st.session_state.degree # SESSION VARIABLE TO STORE DEGREE OF TREND LINE



# COMPUTES VALUES FOR BASE (MAIN) GRAPH
def base_graph_data(data):
    close = data[st.session_state.base_plot].to_numpy()     # CONVERTS PANDAS SERIES TO NUMPY ARRAY FOR FASTER COMPUTATION
    close = close.astype(np.float)

    window_size = st.session_state.window_size              # WINDOW SIZE TO COMPUTE N-SIMPLE MOVING AVERAGE AND EXPONENTIAL MOVING AVERAGE
    n_sma = np.convolve(close, np.ones(window_size)/window_size, mode='valid')  # USING CONVOLVE FUNCTION TO COMPUTE MOVING AVERAGE
    
    x = np.arange(data[st.session_state.base_plot].size)    # INITIALIZING AN NP ARRAY WITH ELEMENTS OF RANGE 1 TO BASE PLOT COLUMN SIZE (CLOSE OR OPEN)
    fit = np.polyfit(x, data[st.session_state.base_plot], deg=st.session_state.trend_degree)    # INSTANTIATING A POLYFIT MODEL
    fit_function = np.poly1d(fit)                                                               # FITTING THE DATA INTO LINEAR/NON-LINEAR POLY FIT FUNCTION TO PREDICT TREND
    trend_op = fit_function(x)                                                                  # COMPUTING TREND VALUES


    temp = np.zeros(close.size - n_sma.size) + np.nan       # INITIALIZING A NUMPY ARRAY WITH NA VALUES OF LENGTH (CLOSE SIZE - SMA SIZE)
    n_sma = np.append(temp, n_sma)                          # APPENDING N-SMA ARRAY TO TEMP ARRAY

    data.reset_index(inplace=True)
    data['n_sma'] = pd.Series(n_sma)                        # CONVERTING ARRAY TO SERIES
    data['trend'] = pd.Series(trend_op)
    data['ema'] = data[st.session_state.base_plot].ewm(span = window_size).mean()   # COMPUTING EMA VALUES OF SPAN SIZE = WINDOW SIZE
    data.set_index('Date', inplace=True)                                            # SETTING INDEX OF THE NEW DATAFRAME TO BE 'DATE'

    return data


# A MODULAR FUNCTION TO PLOT LINE GRAPHS
def plot_base_graph(data):
    fig = px.line(data, x=data.index, y=[st.session_state.base_plot, st.session_state.plot_this])   # CREATING A PLOT/FIGURE TO PLOT
    st.plotly_chart(fig, use_container_width=True)                                                  # PLOTTING THE FIGURE


# FUNCTION TO PLOT A CANDLE STICK GRAPH
def plot_candlestick(data):
    # USES AN IN-BUILT FUNCTION FROM PLOTLY
    # WHICH TAKES 4 PARAMETERS -- OPEN, HIGH, LOW, CLOSE TO PLOT CANDLE STICK
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data.Open,
                high=data.High,
                low=data.Low,
                close=data.Close)])
    st.plotly_chart(fig, use_container_width=True)      # PLOTS THE CHART
