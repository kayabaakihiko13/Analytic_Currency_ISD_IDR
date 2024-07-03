import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Union

def check_outlier(data:Union[pd.DataFrame,
                             pd.Series]):
    median_val = data.median()
    deviation = np.abs(data-median_val)
    mad = np.median(deviation)
    threshold_lower = median_val - mad
    threshold_upper = median_val + mad
    outliers_upper = data > threshold_upper
    outliers_lower = data < threshold_lower
    return data[outliers_lower|outliers_upper].index

def cleaning_outlier(data:Union[pd.DataFrame,pd.Series]):
    median_val = data.median()
    deviation = np.abs(data-median_val)
    mad = np.median(deviation)
    idx = check_outlier(data)
    data[idx] = mad
    return data

def check_stationary(series: pd.Series):
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    result = adfuller(series.values)
    
    print(f"ADF Statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    print("Critical values:")
    for key, val in result[4].items():
        print(f"\t{key}: {val:.5f}")
    
    if result[1] < 0.05 and result[0] < result[4]['5%']:
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")
if __name__=="__main__":
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=100)
    data = np.random.randn(100)  # Normally distributed data
    time_series = pd.Series(data, index=dates)
    mad = check_outlier(time_series)
