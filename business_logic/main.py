import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import efficient_frontier
import param_estimator


tickers = ["MSFT", "AMZN", "KO", "MA", "COST",
           "LUV", "XOM", "PFE", "JPM", "UNH",
           "ACN", "DIS", "GILD", "F", "TSLA"]
ohlc = yf.download(tickers, period="max")
prices = ohlc["Adj Close"].dropna(how="all")
prices.tail()

sample_cov = param_estimator.sample_cov(prices, frequency=252)
mu = param_estimator.capm_return(prices)

ef = efficient_frontier.EfficientFrontier([mu, mu], [sample_cov, sample_cov], trade_horizon=2)
ef.min_volatility(target_return=0.30)
weights = ef.clean_weights()
print(weights)
print(ef._constraints)
