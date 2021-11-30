from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np, pandas as pd
import streamlit as st
from datetime import timedelta


def inferential():
    COL_80_PERCENT, COL_20_PERCENT = st.columns([4,1])                          # DEFINED 2 COLUMNS OF WIDTH 80% AND 20% RESP.
    training_period_index = int(st.session_state['training_period'])            # 
    model, mae, rme, rmse, score = create_model(st.session_state.data, training_period_index)
    fig, temp = predict(model, st.session_state.data.tail(training_period_index), st.session_state['n_predictor'])

    with COL_80_PERCENT:
        st.write("### Predicting using Random Forest model")
        st.plotly_chart(fig, use_container_width=True)
    with COL_20_PERCENT:
        st.selectbox(label="Select training period (historic days)", options=[15, 30, 90, 180, 365, 730], index=0, on_change=handle_change, key="training_select")
        st.number_input(label="Predict for days (n)?", min_value=1, max_value=15, value=10, on_change=handle_change, key="predict_for_n")
        st.table(pd.DataFrame({'Values': [mae,rme,rmse,score]
        }, index=['Mean Absolute Err', 'Root Mean Err', 'RMSE', 'R Squared']))

    st.markdown("""### Future Projections""")
    # st.table(temp[-1*int(st.session_state['n_predictor']):])
    st.table(temp)



def handle_change():
    if st.session_state.predict_for_n:
        st.session_state.n_predictor = st.session_state.predict_for_n
    if st.session_state.training_select:
        st.session_state.training_period = st.session_state.training_select



def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
    model.fit(x_train, y_train)
    pred_values = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, pred_values)
    mse = metrics.mean_squared_error(y_test, pred_values)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_values))
    return (model, mae, mse, rmse, model.score(x_train, y_train))



def predict(model, data, n):
    import plotly.express as xp
    sma_window = 7
    required = data['Close']
    num_rows = max(sma_window, n)

    open_temp = data['Open'].tail(num_rows).to_list()
    for i in range(num_rows):
        open_temp.append(sum(open_temp[-sma_window:])/(sma_window))

    high_temp = data['High'].tail(num_rows).to_list()
    for i in range(num_rows):
        high_temp.append(sum(high_temp[-sma_window:])/sma_window)

    low_temp = data['Low'].tail(num_rows).to_list()
    for i in range(num_rows):
        low_temp.append(sum(low_temp[-sma_window:])/sma_window)


    base = data.index[-1]
    dates = (data.index[-num_rows:]).to_list() + [base + timedelta(days=x) for x in range(1,num_rows+1)]

    temp = pd.DataFrame({
        'Date': dates,
        'Open': open_temp,
        'High': high_temp,
        'Low': low_temp
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

    return (fig,temp[:n])



@st.experimental_singleton   # CACHING THE MODEL UNTIL THE INPUT PARAMEMTERS ARE CHANGED
def create_model(data, n):
    return train_model( y = data['Close'].tail(n),
                        x = data[['Open', 'High', 'Low']].tail(n)
    )
