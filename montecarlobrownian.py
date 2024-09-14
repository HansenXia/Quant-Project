#my first quantitative trading model, using brownian motion as premise, and monte carlo as method,
#program limited to personal PC computing power, 100000 runs max for effective output


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import certifi
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def fetch_historical_data(ticker, days=14):
    """
    Fetch historical stock data for the given ticker over the specified number of days.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        days (int): Number of past calendar days to fetch data for.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch data with daily intervals
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')

    # Drop any non-trading days (weekends, holidays)
    data = data.dropna()

    return data


# Define the ticker symbol
ticker_symbol = 'AAPL'

# 1. Fetch Historical Stock Data
historical_data = fetch_historical_data(ticker_symbol, days=14)

# Check if data is fetched
if historical_data.empty:
    print(f"No data fetched for {ticker_symbol}. Please check the ticker symbol or your internet connection.")
    exit()

print("Historical Data:\n", historical_data)

# 2. Calculate Daily Returns and Estimate Parameters
historical_data['Daily Return'] = historical_data['Close'].pct_change()
daily_returns = historical_data['Daily Return'].dropna()

if daily_returns.empty:
    print("Not enough data to calculate returns.")
    exit()

mu = daily_returns.mean()
sigma = daily_returns.std()

print(f"\nEstimated Mean Daily Return (mu): {mu}")
print(f"Estimated Daily Volatility (sigma): {sigma}")

# 3. Define the Monte Carlo Simulation Parameters
num_simulations = 1000000  # Number of simulation runs
num_steps = 60  # Number of time steps (e.g., minutes in a trading hour)
time_horizon = 1  # Total time horizon (1 day)

# Time increment
dt = time_horizon / num_steps

# Adjusted drift and volatility per time step
mu_step = mu * dt
sigma_step = sigma * np.sqrt(dt)

# Current stock price (last closing price)
S0 = historical_data['Close'][-1]

# 4. Run the Monte Carlo Simulation
# Initialize the simulation matrix
simulation_matrix = np.zeros((num_simulations, num_steps + 1))
simulation_matrix[:, 0] = S0  # Set the initial price for all simulations

start_time = time.time()

# Generate random variables for all simulations and time steps
Z = np.random.standard_normal((num_simulations, num_steps))

# Iterate over each time step to simulate price changes
for step in range(1, num_steps + 1):
    # Calculate the price change using Geometric Brownian Motion
    simulation_matrix[:, step] = simulation_matrix[:, step - 1] * np.exp(
        mu_step - 0.5 * sigma_step ** 2 + sigma_step * Z[:, step - 1])

end_time = time.time()

simulation_time = end_time - start_time
print(f"Monte Carlo simulation for {num_simulations} runs took {simulation_time:.2f} seconds.")

# 5. Calculate the Average Path
average_path = simulation_matrix.mean(axis=0)
time_array = np.linspace(0, time_horizon, num_steps + 1)

# 6. Visualize the Results
plt.figure(figsize=(12, 6))

# Plot a subset of simulation paths for visualization (e.g., 100 paths)
for i in range(100):
    plt.plot(time_array, simulation_matrix[i], color='lightgray', linewidth=0.5)

# Plot the average path
plt.plot(time_array, average_path, color='blue', linewidth=2, label='Average Path')

# Plot the initial price
plt.axhline(y=S0, color='red', linestyle='--', linewidth=2, label=f'Initial Price: {S0:.2f}')

# Add titles and labels
plt.title(f'Monte Carlo Simulation of {ticker_symbol} Intraday Stock Price')
plt.xlabel('Time (Days)')
plt.ylabel('Price ($)')
plt.legend()

plt.show()
predicted_closing_price = average_paths[-1]
print(predicted_closing_price)

