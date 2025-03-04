import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate synthetic time series data
np.random.seed(42)
n = 100
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
# Create a cumulative sum of random values to simulate a time series
time_series = pd.Series(np.random.randn(n).cumsum(), index=dates)

# Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Original Series')
plt.title('Synthetic Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Define and fit the ARIMA model
# Here, order=(p,d,q) = (1, 1, 1) is used as an example
model = ARIMA(time_series, order=(1, 1, 1))
model_fit = model.fit()

# Print the model summary to inspect the model parameters
print(model_fit.summary())

# Forecast future values (e.g., next 10 time steps)
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the forecast along with the historical data
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
