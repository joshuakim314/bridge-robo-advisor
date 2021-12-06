import warnings
import csv
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import sklearn
import pmdarima as pm
import arch
from arch.__future__ import reindexing

from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

import param_estimator

import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

conn = psycopg2.connect(
        host='database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
        port=5432,
        user='postgres',
        password='capstone',
        database='can2_etfs'
    )
conn.autocommit = True
cursor = conn.cursor()

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
    table = 'americanetfs'
    tickers = param_estimator.get_all_tickers(table)
    start, end = '2016-12-01', '2021-11-30'
    returns_tick = []

    for tick in tickers:
        returns = param_estimator.get_returns(tick, table, start, end, freq='monthly')
        if returns.empty:
            print(f'{tick} is empty')
            continue
        returns[tick] = returns['adj_close']
        returns_tick += [returns[[tick]]]
    returns = pd.concat(returns_tick, axis=1).T.dropna()
    print(returns.shape)

    pca = PCA(n_components='mle')
    principal_comp = pca.fit_transform(returns)
    print(principal_comp.shape)

    n_clusters = 10
    X = np.asarray(principal_comp)
    k_medoids = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++').fit(X)
    print([tickers[np.where(X == k)[0][0]] for k in k_medoids.cluster_centers_])
    print(k_medoids.inertia_)

    cursor.close()