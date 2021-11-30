# THIS FILE CONTAINS THE PREDICTIVE PART OF THE APPLICATION
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np, pandas as pd
import streamlit as st
from datetime import timedelta


# THIS FUNCTION ACTS AS AN INITIALIZER AND COMMUNICATES DIRECTLY WITH BASE.PY
def inferential():
    COL_80_PERCENT, COL_20_PERCENT = st.columns([4,1])                          # DEFINED 2 COLUMNS OF WIDTH 80% AND 20% RESP.
    training_period_index = int(st.session_state['training_period'])            # FETCHING TRAINING PERIOD (LATST N-NUMBER OF ROWS FROM STOCK'S HISTORICAL DATA) FOR TRAINING THE MODEL
    model, mae, rme, rmse, r_score = create_model(st.session_state.data, training_period_index)       # INITIALIZING A MODEL OBJECT AND SAVING THE EVALUATION METRICS
    fig, predicted_values = predict(model, st.session_state.data.tail(training_period_index), st.session_state['n_predictor'])  # PREDICTING FUTURE VALUES AND SAVING PLOT

    with COL_80_PERCENT:
        st.write("### Predicting using Random Forest model")
        st.plotly_chart(fig, use_container_width=True)
    with COL_20_PERCENT:
        st.selectbox(label="Select training period (historic days)", options=[15, 30, 90, 180, 365, 730], index=0, on_change=handle_change, key="training_select")  # SELECTBOX FOR TRAINING PERIOD
        st.number_input(label="Predict for days (n)?", min_value=1, max_value=15, value=10, on_change=handle_change, key="predict_for_n")                           # VALUE FOR N (DAYS) FOR WHICH THE PREDICTION IS REQUIRED
        st.table(pd.DataFrame({'Values': [mae,rme,rmse,r_score]
        }, index=['Mean Absolute Err', 'Root Mean Err', 'RMSE', 'R Squared']))

    st.markdown("""### Future Projections""")
    st.table(predicted_values)


# HANDLING ON_CALL CHANGES
def handle_change():
    if st.session_state.predict_for_n:
        st.session_state.n_predictor = st.session_state.predict_for_n
    if st.session_state.training_select:
        st.session_state.training_period = st.session_state.training_select



# CREATING AN INSTANCE (MODEL) BASIS RANDOM FOREST REGRESSION
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)    # SPLITTING THE DATASET INTO TRAINING AND TESTING
    model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)           # INSTANTIATING RANDOM FOREST REGRESSOR AND PASSING REQ VALUES TO THE CONSTRUCTOR
    model.fit(x_train, y_train)                                                                 # TRAINING/FITTING THE MODEL WITH TRAINING DATA
    pred_values = model.predict(x_test)                                                         # PREDICTING Y_TEST BASIS X_TEST -- FOR EVALUATION
    # COMPUTING NECESSARY METRICS TO EVALUATE THE MODEL CREATED
    mae = metrics.mean_absolute_error(y_test, pred_values)
    mse = metrics.mean_squared_error(y_test, pred_values)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_values))
    r_score = model.score(x_train, y_train)
    return (model, mae, mse, rmse, r_score)


# PREDICTING FUTURE VALUES BASIS OPEN, HIGH, LOW AND THE MODEL CREATED
def predict(model, data, n):
    import plotly.express as xp
    sma_window = 7                              # WINDOW TO COMPUTE SIMPLE MOVING AVERAGES. THIS IS KEPT CONSTANT TO ENSURE SAME VALUES IRRESPECTIVE OF N
    required = data['Close']                    # FETCHING THE TARGET COLUMN
    num_rows = max(sma_window, n)               # MAX(N, SMA_WINDOW) TO ENSURE CORRECT SMA COMPUTATION FOR N<SMA_WINDOW (7)

    # COMPUTING SIMPLE MOVING AVERAGES OF WINDOW SIZE 7 FOR OPEN, HIGH AND LOW
    open_temp = data['Open'].tail(num_rows).to_list()
    for i in range(num_rows):
        open_temp.append(sum(open_temp[-sma_window:])/(sma_window))

    high_temp = data['High'].tail(num_rows).to_list()
    for i in range(num_rows):
        high_temp.append(sum(high_temp[-sma_window:])/sma_window)

    low_temp = data['Low'].tail(num_rows).to_list()
    for i in range(num_rows):
        low_temp.append(sum(low_temp[-sma_window:])/sma_window)


    # FETCHING BASE DATE (LAST DATE ENTRY OF THE DATA) AND INCREMENTING IT FOR N-NUMBER OF DAYS
    base = data.index[-1]
    dates = (data.index[-num_rows:]).to_list() + [base + timedelta(days=x) for x in range(1,num_rows+1)]

    # CREATING A TEMPORARY DATAFRAME
    temp = pd.DataFrame({
        'Date': dates,
        'Open': open_temp,
        'High': high_temp,
        'Low': low_temp
    })
    temp.set_index('Date', inplace=True)
    temp['Close'] = model.predict(temp[['Open', 'High', 'Low']])
    
    # FILTERING REQUIRED DATA
    temp = temp[temp.index>base]
    required = required.append(temp['Close'][:n])

    # PLOTTING THE GRAPH WITH A RED VERTICAL LINE TO DISTINGUISH BETWEEN ACTUAL AND PREDICTED VALUES
    fig = xp.line(required, x=required.index, y='Close', markers=True)
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'paper', y0 = 0, y1 = 1,                                  # (Y1,Y2) FOR RED LINE
        xref= 'x', x0= required.index[-n], x1= required.index[-n],      # (X1,X2) FOR RED LINE
        line = dict(
            color='red',
            width=1.5,
            dash='dashdot'
        )
        )
    ])

    return (fig,temp[:n])



@st.experimental_singleton   # CACHING THE MODEL UNTIL THE INPUT PARAMEMTERS ARE CHANGED
def create_model(data, n):   # PASSING THE REQUIRED INDEPENDENT (X) VARIABLE AND DEPENDENT (Y) VARIABLE FOR MODEL CREATION
    return train_model( y = data['Close'].tail(n),                  # PASSING LAST N ROWS OF DEPENDENT VARIABLE
                        x = data[['Open', 'High', 'Low']].tail(n)   # PASSING LAST N ROWS OF INDEPENDENT VARIABLES
    )
