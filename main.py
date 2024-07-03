from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessing import check_outlier, cleaning_outlier, check_stationary
from utils.constants import ERROR
from EDA import VisualizationACF_PACF, VisualizationSeasonality
from model import build_sarima_model,build_auto_arima,LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import torch
import torch.nn as nn
def run_checkoutlier(data: pd.Series):
    if not isinstance(data, pd.Series):
        print(f"{ERROR} WARNING")
        raise ValueError("Tipe data harus Series")
    data = cleaning_outlier(data)
    print(data)

def run_sarima_model():
    # Split data into train and test sets
    train_ratio = 0.8
    train_size = int(len(df) * train_ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # Build SARIMA model
    model_fit = build_sarima_model(train["exchange_rate"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    
    # Forecast
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_index = test.index
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train['exchange_rate'], label='Train')
    plt.plot(test.index, test['exchange_rate'], label='Test')
    plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.show()
    
    # Build and print auto ARIMA model
    auto_model = build_auto_arima(train["exchange_rate"], max_p=5, max_q=5, seasonal=False)
    print(auto_model.summary())



if __name__ == "__main__":
    df = pd.read_csv('data/output.csv')
    
    # Check and clean outliers
    # run_checkoutlier(df["exchange_rate"])
    
    # Visualize ACF and PACF
    VisualizationACF_PACF(df["exchange_rate"])
    
    # Convert timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index('timestamp', inplace=True)
    scaler = MinMaxScaler()
    df['exchange_rate'] = scaler.fit_transform(df['exchange_rate'].values.reshape(-1, 1))
    def create_dataset(series, time_step=1):
        X, y = [], []
        for i in range(len(series) - time_step - 1):
            a = series[i:(i + time_step)]
            X.append(a)
            y.append(series[i + time_step])
        return np.array(X), np.array(y)

    time_step = 10
    X, y = create_dataset(df['exchange_rate'].values, time_step)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the LSTM model
    model = LSTM(hidden_size=100)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        for seq, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            model.hidden = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {single_loss.item()}')

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_predict = [model(seq) for seq in X_train]
        test_predict = [model(seq) for seq in X_test]

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(np.array([tp.item() for tp in train_predict]).reshape(-1, 1))
    test_predict = scaler.inverse_transform(np.array([tp.item() for tp in test_predict]).reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot the results
    plt.figure(figsize=(15, 8))
    plt.plot(scaler.inverse_transform(df['exchange_rate'].values.reshape(-1, 1)), label='Original Time Series')
    plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Training Predictions')
    plt.plot(np.arange(len(train_predict) + 2 * time_step, len(train_predict) + 2 * time_step + len(test_predict)), test_predict, label='Testing Predictions')
    plt.legend()
    plt.show()