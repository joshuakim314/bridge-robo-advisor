import pandas_datareader as pdr
import numpy as np
import pandas as pd
import yfinance as yf
import scipy


#Get Data
df = pdr.data.DataReader('North_America_5_Factors', 'famafrench')[0]
print(df)

tickers = ["MSFT", "AMZN", "KO", "MA", "COST",
           "LUV", "XOM", "PFE", "JPM", "UNH",
           "ACN", "DIS", "GILD", "F", "TSLA"]

ohlc = yf.download(tickers, start="2017-01-01", end="2021-10-31", interval='1mo')
prices = ohlc["Adj Close"].dropna(how="all")
prices.reset_index()

#Format data
p_prices = pd.DataFrame()

for col in prices.columns.to_list():
    new_df = prices[[col]].reset_index().rename(columns={col: 'price'})
    new_df['stock'] = col
    new_df = new_df[['Date', 'stock', 'price']]
    new_df['momReturns'] = new_df.price.pct_change()
    new_df['momReturns'] = new_df['momReturns'].shift(-1)
    p_prices = p_prices.append(new_df)

prices = p_prices.reset_index().drop(columns='index')
print(prices)








