from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
    model.fit(x_train, y_train)
    pred_values = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, pred_values)
    mse = metrics.mean_squared_error(y_test, pred_values)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_values))
    return (model, mae, mse, rmse, model.score(x_train, y_train))
