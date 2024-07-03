import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plot 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def VisualizationACF_PACF(data: pd.Series):
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    plot_acf(data, lags=20, ax=ax[0])
    plot_pacf(data, lags=20, ax=ax[1])
    plt.tight_layout()
    plt.show()

def Visualizationmoving_averange(sample:pd.Series,windows:int=7):
    plt.figure(figsize=(12, 6))
    movingAV_value = sample.rolling(window=windows).mean()
    plt.plot(sample, label='Original High', color='blue')
    plt.plot(movingAV_value, label=f'Moving Average (Window={windows})', linestyle='--', color='orange')
    plt.xlabel('Date')
    plt.ylabel('High')
    plt.title('Original vs Moving Average')
    plt.legend()
    plt.show()

def VisualizationSeasonality(sample:pd.Series,periods:int=7):
    decomposition = seasonal_decompose(sample, model='additive', period=periods)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')

    plt.tight_layout()
    plt.show()
    