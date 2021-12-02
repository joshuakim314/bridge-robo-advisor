import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cp

import efficient_frontier
import param_estimator


tickers = ["MSFT", "AMZN", "KO", "MA", "COST",
           "LUV", "XOM", "PFE", "JPM", "UNH",
           "ACN", "DIS", "GILD", "F", "TSLA"]
# tickers = ["MSFT", "AMZN", "KO"]
ohlc = yf.download(tickers, period="max")
prices = ohlc["Adj Close"].dropna(how="all")
prices.tail()

sample_cov = param_estimator.sample_cov(prices, frequency=252)
mu = param_estimator.capm_return(prices)

np.random.seed(0)
ef = efficient_frontier.EfficientFrontier([mu, mu+np.random.normal(0, 0.05, mu.shape)],
                                          [sample_cov, sample_cov],
                                          trade_horizon=2, solver=cp.GUROBI)
# ef.min_volatility(target_return=0.30)
# ef.max_sharpe()
# ef.max_quadratic_utility()
# ef.efficient_return(target_return=0.20)
ef.robust_efficient_frontier(target_return=0.20, uncertainty='box')
weights = ef.clean_weights()
print(weights)
