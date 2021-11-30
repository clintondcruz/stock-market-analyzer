# Importing the required packages
import matplotlib.pyplot as plt
import numpy as np, plotly.express as xp, streamlit as st

def macd_bs(data): 
    try:
        ShortEMA = data['Close'].ewm(span=12, adjust=False).mean()  # Calculate 12 day Exponentoal moving average
        LongEMA = data['Close'].ewm(span=26, adjust=False).mean()   # Calculate 26 day Exponentoal moving average
        MACD = ShortEMA - LongEMA                                   # Calculate Moving day convergence Divergence
        signal = MACD.ewm(span = 9, adjust=False).mean()            # Calculate signal line i.e. 9 day EMA of MACD

        data['MACD'] = MACD
        data["Signal Line"]= signal
        fig = xp.line(data[['MACD', 'Signal Line']], x=data.index, y=['MACD', 'Signal Line'])
        st.plotly_chart(fig, use_container_width=True)
    except: st.error('Could not load MACD graph. Not enough data.')


    # buy_sell graph
    def buy_sell(signal):
        Buy = []
        Sell = []
        flag =-1

        #Calculate the Buy & Sell recomendations
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

    
    # Configure the graph giving buy sell recommendations
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
    try:
        st.pyplot(buy_sell_graph)
    except: st.error('Could not load Buy/Sell graph. Not enought data.')