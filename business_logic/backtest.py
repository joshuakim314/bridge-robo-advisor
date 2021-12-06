import warnings
import csv
import collections
from functools import reduce
import itertools
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
import objective_functions
import efficient_frontier

import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

warnings.filterwarnings("ignore")

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


def historical_avg(returns, freq, trade_horizon=12):
    mu = returns.agg(lambda data: reduce(lambda x, y: x*y, [x+1 for x in data])**(1/freq) - 1)
    mu.rename("mu", inplace=True)
    # mu = pd.DataFrame({'ticker': mu.index, 'mu': mu.values})
    # mu.set_index('ticker', inplace=True)
    cov = param_estimator.sample_cov(returns, returns_data=True, frequency=freq)
    return [mu for _ in range(trade_horizon)], [cov for _ in range(trade_horizon)]


def get_factor_mu(model_name, model, factors, month, factor_columns, ag):
    data = [ag[model_name[:3]][factor_name][1][int(month)-1] for factor_name in factor_columns]
    mu = model.predict(np.array(data).reshape(1, -1))
    # diag_data = [ag[model_name[:3]][factor_name][3].residual_variance[f'h.{month}'] for factor_name in factor_columns]
    # print(np.array(model.coef_).shape)
    # print((factors[factor_columns].cov()*60).shape)
    # cov = np.matmul(np.matmul(np.array(model.coef_), np.array(factors[factor_columns])).cov()*60, np.transpose(np.array(model.coef_))) + np.diag(diag_data)
    return mu

# TODO
# def get_factor_cov(model_name, factor_loading, factors, month, factor_columns, ag):
#     # diag_data = [ag[model_name[:3]][factor_name][3].residual_variance[f'h.{month}'].iloc[-1:] for factor_name in factor_columns]
#     print("C")
#     print(np.array(factor_loading).shape)
#     print(np.cov(np.array(factors[factor_columns])).shape)
#     cov = np.matmul(np.matmul(np.array(factor_loading), np.cov(np.array(factors[factor_columns])), np.transpose(np.array(factor_loading))))
#     # TODO: + np.diag(diag_data)
#     return cov


if __name__ == '__main__':
    stock_picks = ['AAPL', 'DIS', 'JPM', 'KO', 'WMT']
    val_periods = [(f'{str(year)}-01-01', f'{str(year+4)}-12-31', f'{str(year+5)}-01-01', f'{str(year+5)}-12-31')
                   for year in range(2001, 2011)]
    for val in val_periods:
        print(val)
        train_start, train_end, test_start, test_end = val

        etf_table = 'americanetfs'
        etf_tickers = param_estimator.get_all_tickers(etf_table)
        etf_returns_by_tick = []
        for tick in etf_tickers:
            returns = param_estimator.get_returns(tick, etf_table, train_start, test_end, freq='monthly')
            if returns.empty:
                continue
            returns[tick] = returns['adj_close']
            etf_returns_by_tick += [returns[[tick]]]
        etf_returns = pd.concat(etf_returns_by_tick, axis=1).T.dropna()
        train_etf_returns = etf_returns.T.head(12*5)
        test_etf_returns = etf_returns.T.tail(12*1)
        valid_etf_tickers = train_etf_returns.columns
        print(f'number of etfs: {train_etf_returns.shape[1]}, number of months: {train_etf_returns.shape[0]}')

        pca = PCA(n_components='mle')
        principal_comp = pca.fit_transform(train_etf_returns.T)
        print(f'number of principal components: {principal_comp.shape[1]}')

        n_clusters = 10
        X = np.asarray(principal_comp)
        k_medoids = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++').fit(X)
        selected_etfs = [valid_etf_tickers[np.where(X == k)[0][0]] for k in k_medoids.cluster_centers_]
        inertia = k_medoids.inertia_
        print(f'selected etfs: {selected_etfs}, cluster inertia: {inertia}')

        etf_table = 'spy'
        stock_returns_by_tick = []
        for tick in stock_picks:
            returns = param_estimator.get_returns(tick, etf_table, train_start, test_end, freq='monthly')
            if returns.empty:
                continue
            returns[tick] = returns['adj_close']
            stock_returns_by_tick += [returns[[tick]]]
        stock_returns = pd.concat(stock_returns_by_tick, axis=1).T.dropna()
        train_stock_returns = stock_returns.T.head(12*5)
        test_stock_returns = stock_returns.T.tail(12*1)

        # Fama-French factors
        train_factors = param_estimator.get_factors(start=int(train_start[0:4] + train_start[5:7]), end=int(train_end[0:4] + train_end[5:7]), freq='monthly')
        test_factors = param_estimator.get_factors(start=int(test_start[0:4] + test_start[5:7]), end=int(test_end[0:4] + test_end[5:7]), freq='monthly')

        asset_universe = stock_picks + selected_etfs
        train_returns = pd.concat([train_stock_returns, train_etf_returns], axis=1)
        test_returns = pd.concat([test_stock_returns, test_etf_returns], axis=1)

        # historical average param. estimation
        mu, cov = historical_avg(train_returns, 12*5, 12)
        print(mu)
        print(cov)

        factor_models = collections.OrderedDict()

        for tick in asset_universe:
            merged = pd.merge(train_factors, train_returns[[tick]], left_on='date', right_on='date', how="inner", sort=False)
            ff3 = merged[['excess', 'smb', 'hml']]
            ff5 = merged[['excess', 'smb', 'hml', 'rmw', 'cma']]
            merged[tick] = merged[tick] - merged['riskfree'].astype('float')
            adj_returns = merged[[tick]]

            alphas = [1e-2, 1e-1, 0.0]
            l1_ratios = list(np.arange(0, 0.1, 0.05))

            mlr3, r_sq3 = param_estimator.MLR(ff3, adj_returns[tick])
            factor_models[(tick, 'ff3_mlr')] = (mlr3, r_sq3)
            mlr5, r_sq5 = param_estimator.MLR(ff5, adj_returns[tick])
            factor_models[(tick, 'ff5_mlr')] = (mlr5, r_sq5)
            for alpha, l1_ratio in list(itertools.product(alphas, l1_ratios)):
                en3, en_r_sq3 = param_estimator.EN(ff3, adj_returns[tick], alpha=alpha, l1_ratio=l1_ratio)
                factor_models[(tick, f'ff3_en_{alpha}_{l1_ratio}')] = (en3, en_r_sq3)
                en5, en_r_sq5 = param_estimator.EN(ff5, adj_returns[tick], alpha=alpha, l1_ratio=l1_ratio)
                factor_models[(tick, f'ff5_en_{alpha}_{l1_ratio}')] = (en5, en_r_sq5)

        # arima-garch
        ag = dict()
        ag['ff3'] = param_estimator.arima_garch(train_factors[['excess', 'smb', 'hml']], trade_horizon=12, columns=['excess', 'smb', 'hml'])
        ag['ff5'] = param_estimator.arima_garch(train_factors[['excess', 'smb', 'hml', 'rmw', 'cma']], trade_horizon=12, columns=['excess', 'smb', 'hml', 'rmw', 'cma'])

        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        model_names = []
        mu_factor = [[] for _ in range(12)]
        for key, value in factor_models.items():
            tick, model_name = key
            model, r_sq = value
            model_names += [model_name]
            factor_columns = ['excess', 'smb', 'hml'] if model_name[:3] == 'ff3' else ['excess', 'smb', 'hml', 'rmw', 'cma']
            for month in months:
                mu_month = get_factor_mu(model_name, model, train_factors, month, factor_columns, ag)
                print(mu_month)
                mu_factor[int(month)-1] += [mu_month[0]]
        print(len(mu_factor))
        print(len(mu_factor[0]))
        print(len(mu_factor[1]))
        mu_factor = [pd.DataFrame(mu_factor[i], columns=asset_universe) for i in range(12)]

        # factor_loadings = collections.OrderedDict()
        # for model_name in model_names:
        #     factor_loadings[model_name] = []
        # for month in months:
        #     for key, value in factor_models.items():
        #         tick, model_name = key
        #         model, r_sq = value
        #         factor_loadings[model_name] += [list(model.coef_)]
        # for model_name in model_names:
        #     factor_columns = ['excess', 'smb', 'hml'] if model_name[:3] == 'ff3' else ['excess', 'smb', 'hml', 'rmw', 'cma']
        #     cov_factor = []
        #     for month in months:
        #         cov_month = get_factor_cov(model_name, factor_loadings[model_name], train_factors, month, factor_columns, ag)
        #         cov_factor += [cov_month]
        #     print(cov_factor)

        ef = efficient_frontier.EfficientFrontier(mu_factor, [cov for _ in range(12)], trade_horizon=12)
        ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
        ef.max_sharpe()
        weights = ef.clean_weights()
        print(weights)

        ef = efficient_frontier.EfficientFrontier(mu_factor, [cov for _ in range(12)], trade_horizon=12)
        ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
        ef.efficient_return(target_return=0.10)
        weights = ef.clean_weights()
        print(weights)

        ef = efficient_frontier.EfficientFrontier(mu_factor, [cov for _ in range(12)], trade_horizon=12)
        ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
        ef.robust_efficient_frontier(target_return=0.10, uncertainty='box')
        weights = ef.clean_weights()
        print(weights)

        ef = efficient_frontier.EfficientFrontier(mu_factor, [cov for _ in range(12)], trade_horizon=12)
        ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
        ef.risk_parity()
        weights = ef.clean_weights()
        print(weights)


    cursor.close()