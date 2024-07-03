import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from xgboost import XGBRegressor
import torch
import torch.nn as nn
def build_sarima_model(data: pd.Series, order: tuple, seasonal_order: tuple):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    print("SARIMA Model Summary:\n", model_fit.summary())
    return model_fit

def build_auto_arima(data: pd.Series, max_p: int, max_q: int, seasonal: bool = False):
    start_p, start_q = 0, 0
    model = pm.auto_arima(data, start_p=start_p, start_q=start_q, max_p=max_p, max_q=max_q, seasonal=seasonal)
    return model


class XgboostTimeSeries:
    def __init__(self, data: pd.Series) -> None:
        self.data = data.to_frame()
        self.X = None
        self.y = None
        self.model = XGBRegressor()

    def __create_features(self, label=None):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("The data index must be a DatetimeIndex")

        self.data['hour'] = self.data.index.hour
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['quarter'] = self.data.index.quarter
        self.data['month'] = self.data.index.month
        self.data['year'] = self.data.index.year
        self.data['dayofyear'] = self.data.index.dayofyear
        self.data['dayofmonth'] = self.data.index.day
        self.data['weekofyear'] = self.data.index.isocalendar().week

        X = self.data[['hour', 'dayofweek', 'quarter', 'month', 'year',
                       'dayofyear', 'dayofmonth', 'weekofyear']].copy()
        if label:
            y = self.data[label].copy()
            return X, y
        return X

    def fit(self, label):
        self.X, self.y = self.__create_features(label)
        self.model.fit(self.X, self.y)
    def tune_hyperparameters(self, param_grid):
        self.X, self.y = self.__create_features(label='exchange_rate')
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X, self.y)
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_}")
    
    def predict(self, start_date, end_date):
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame(index=future_dates)
        future_df.index.name = 'timestamp'
        X_future = self.__create_features().reindex(future_df.index)
        forecast = self.model.predict(X_future)
        return pd.Series(forecast, index=future_df.index)


class LSTM(nn.Module):
    
    def __init__(self,input_size = 1, hidden_size = 50, out_size = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
    
    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1]