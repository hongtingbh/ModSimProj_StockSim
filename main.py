import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import tempfile
import os


# LOAD DATA
def load_data(ticker="NVDA", start="2010-01-05", end="2020-01-06"):
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(proxy_var, None)

    cache_dir = Path(tempfile.gettempdir()) / "py-yfinance-cache"
    cache_dir.mkdir(exist_ok=True)
    yf.cache.set_cache_location(str(cache_dir))

    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty or 'Close' not in data:
        raise ValueError(f"No price data returned for {ticker}.")

    prices = data['Close']
    
    if isinstance(prices, pd.DataFrame):
        if ticker in prices.columns:
            prices = prices[ticker]
        else:
            prices = prices.iloc[:, 0]

    prices = prices.dropna()

    if prices.empty:
        raise ValueError(f"Close price series is empty for {ticker}.")

    return prices

# PARAMETER ESTIMATION
def estimate_parameters(prices):
    # Returns: percent change between start and end price
    log_returns = np.log(prices / prices.shift(1)).dropna() # Log of ratio between consecutive prices

    if log_returns.empty:
        raise ValueError("Not enough price history to estimate parameters.")
    
    mu = log_returns.mean() # Drift, the average log return
    sigma = log_returns.std() # Volatility, the standard deviation of log returns
    
    return mu, sigma, log_returns

# GBM SIMULATION
def simulate_gbm(S0, mu, sigma, T=30, dt=1, n_simulations=1000):
    steps = int(T / dt)
    
    simulations = np.zeros((steps, n_simulations))
    simulations[0] = S0
    
    for t in range(1, steps):
        Z = np.random.standard_normal(n_simulations)
        simulations[t] = simulations[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    return simulations

# PLOTTING
def plot_simulations(simulations, real_prices):
    plt.figure()
    plt.plot(simulations[:, :50])  # show 50 paths
    plt.title("Monte Carlo Simulated Paths")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

def plot_distribution(simulations):
    final_prices = simulations[-1]
    
    plt.figure()
    plt.hist(final_prices, bins=50)
    plt.title("Distribution of Final Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

# EVALUATION (RMSE)
def compute_rmse(simulations, real_prices):
    # Compare mean simulation to real prices
    sim_mean = simulations.mean(axis=1)
    
    # Align lengths
    min_len = min(len(sim_mean), len(real_prices))
    
    rmse = np.sqrt(np.mean((sim_mean[:min_len] - real_prices[:min_len])**2))
    return rmse

# MAIN PIPELINE
def main():
    ticker = "NVDA"
    
    try:
        # Load data
        prices = load_data(ticker)
        print("HELLO")
        # Estimate parameters
        mu, sigma, log_returns = estimate_parameters(prices)

        print(f"Drift (mu): {mu}")
        print(f"Volatility (sigma): {sigma}")

        # Last known price
        S0 = prices.iloc[-1]

        # Simulate future
        simulations = simulate_gbm(S0, mu, sigma, T=60, n_simulations=1000)

        # Plot results
        plot_simulations(simulations, prices)
        plot_distribution(simulations)

        # Fake "real future" for evaluation (just reuse last segment)
        real_future = prices[-60:].values

        rmse = compute_rmse(simulations, real_future)
        print(f"RMSE: {rmse}")
    except Exception as exc:
        print(f"Error: {exc}")

if __name__ == "__main__":
    main()
