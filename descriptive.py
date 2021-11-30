import streamlit as st
import pandas as pd, numpy as np


def descriptive(data):
    with st.container():
        st.write("## DESCRIPTIVE ANALYSIS")
        with st.container():
            st.write("### Moving averages and trend lines")
            COL_80_PERCENT, COL_20_PERCENT = st.columns([4,1])          # DEFINED 2 COLUMNS OF WIDTH 80% AND 20% RESP.
            # PLOTTING SMA, TRENDLINE GRAPHS AGAINST CLOSE AND ADJ CLOSE
            data = base_graphs(data)                                    # COMPUTES VALUES FOR VARIOUS METRICS TO BE PLOTTED
            with COL_80_PERCENT:
                    plot_base_graph(data)                                    # PLOTS THE GRAPH
            with COL_20_PERCENT:
                st.number_input(value=1, label="Enter the value of (n)", on_change=handle_change, key="window", min_value=1, max_value=max(len(data)//4,1))     # WINDOW SIZE INPUT FOR N_SMA => RANGES FROM 1 TO 25% OF LENGTH OF DATA
                st.number_input(value=1, label="Enter the degree of trendline", on_change=handle_change, key="degree", min_value=1, max_value=3)                # INPUT FOR DEGREE OF TREND LINE => RANGES FROM 1 TO 3
                st.radio("Base plot-", ['Close', 'Open'], on_change=handle_change, key='base')
                st.radio("Plot base against-", ['n_sma', 'ema', 'trend'], on_change=handle_change, key='versus')

        with st.container():
            st.write("### Candle stick graph")
            try:
                # CANDLE-STICK PLOT
                plot_candlestick(data)
            except:
                st.error('Not enough data to display the candle stick graph')

            st.write("### Volume area graph (Volume in millions)")
            try:
                # VOLUME AREA GRAPH
                st.area_chart(data.Volume/10**6, use_container_width=True)
            except:
                st.error('Not enough data to display the area graph')


def handle_change():
    if st.session_state.versus:
        st.session_state.plot_this = st.session_state.versus
    if st.session_state.base:
        st.session_state.base_plot = st.session_state.base
    if st.session_state.window_size:
        st.session_state.window_size = st.session_state.window
    if st.session_state.degree:
        st.session_state.trend_degree = st.session_state.degree



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


def plot_base_graph(data):
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
