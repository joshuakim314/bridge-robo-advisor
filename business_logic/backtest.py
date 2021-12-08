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
    cov = param_estimator.sample_cov(returns, returns_data=True, frequency=freq)
    return [mu for _ in range(trade_horizon)], [cov for _ in range(trade_horizon)]


def get_factor_mu(model_name, model, asset_list, factor_columns, ag):
    mu_factor = []
    for month in range(12):
        mu_month = []
        for tick in asset_list:
            data = [ag[model_name[:3]][factor_name][1][month - 1] for factor_name in factor_columns]
            mu = model[(tick, model_name)][0].predict(np.array(data).reshape(1, -1))
            mu_month.append(mu[0])
        mu_factor.append(mu_month)
    mu_factor = [pd.Series(mu_factor[i]) for i in range(12)]
    return mu_factor


def get_scenarios(model_name, model, asset_list, factor_columns, ag, res, num_s):
    scenarios = []
    for s in range(num_s):
        mu_factor = []
        for month in range(12):
            mu_month = []
            for idx, tick in enumerate(asset_list):
                data = [ag[model_name[:3]][factor_name][1][month - 1] for factor_name in factor_columns]
                mu = model[(tick, model_name)][0].predict(np.array(data).reshape(1, -1))
                mu_month.append(mu[0] + np.random.normal(0.0, res[idx][month]**0.5))
            mu_factor.append(mu_month)
        mu_factor = [pd.Series(mu_factor[i]) for i in range(12)]
        scenarios.append(mu_factor)
    return scenarios


if __name__ == '__main__':
    stock_picks = ['DIS', 'IBM', 'JPM', 'KO', 'WMT']
    weights_dict = collections.OrderedDict()
    returns_dict = collections.OrderedDict()
    assets_dict = collections.OrderedDict()
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

        # market return
        market_return = historical_avg(train_etf_returns[['SPY']], 60)[0][0].tolist()[0]
        print(f'market return: {market_return}')

        pca = PCA(n_components='mle')
        principal_comp = pca.fit_transform(train_etf_returns.T)
        print(f'number of principal components: {principal_comp.shape[1]}')

        n_clusters = 10
        X = np.asarray(principal_comp)
        k_medoids = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++').fit(X)
        selected_etfs = [valid_etf_tickers[np.where(X == k)[0][0]] for k in k_medoids.cluster_centers_]
        inertia = k_medoids.inertia_
        print(f'selected etfs: {selected_etfs}, cluster inertia: {inertia}')

        stock_table = 'spy'
        stock_returns_by_tick = []
        for tick in stock_picks:
            returns = param_estimator.get_returns(tick, stock_table, train_start, test_end, freq='monthly')
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

        assets_dict[test_start] = asset_universe
        returns_dict[test_start] = test_returns

        # historical average param. estimation
        mu, cov = historical_avg(train_returns, 12*5, 12)
        # print(mu[0])
        # print(cov[0] / 60.0)

        factor_models = collections.OrderedDict()
        for tick in asset_universe:
            merged = pd.merge(train_factors, train_returns[[tick]], left_on='date', right_on='date', how="inner", sort=False)
            ff3 = merged[['excess', 'smb', 'hml']]
            ff5 = merged[['excess', 'smb', 'hml', 'rmw', 'cma']]
            merged[tick] = merged[tick] - merged['riskfree'].astype('float')/100.0
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
        mu_factor = collections.OrderedDict()
        cov_factor = collections.OrderedDict()
        scenarios_factor = collections.OrderedDict()
        for key, value in factor_models.items():
            tick, model_name = key
            model, r_sq = value
            model_names += [model_name]
        model_names = list(set(model_names))
        for model_name in model_names:
            factor_columns = ['excess', 'smb', 'hml', 'rmw', 'cma'] if model_name[:3] == 'ff5' else ['excess', 'smb', 'hml']
            mu_factor[model_name] = get_factor_mu(model_name, factor_models, asset_universe, factor_columns, ag)
            factor_cov = train_factors[factor_columns].cov()
            factor_loadings = []
            r = []
            for tick in asset_universe:
                factor_loadings += [factor_models[(tick, model_name)][0].coef_]
                res_var = [ag[model_name[:3]][factor][3].residual_variance.values.tolist()[0] for factor in
                           factor_columns]
                r.append([sum(i[0] * i[1] for i in zip([k ** 2 for k in factor_models[(tick, model_name)][0].coef_], [j[month] for j in res_var])) for month in range(12)])
            factor_loadings = pd.DataFrame(factor_loadings, columns=factor_columns)
            res_diag = [np.diag([res[month] for res in r]) for month in range(12)]
            cov_factor[model_name] = [pd.DataFrame(np.dot(factor_loadings, np.dot(factor_cov, factor_loadings.T)) + res_diag[month], columns=asset_universe) for month in range(12)]
            scenarios = get_scenarios(model_name, factor_models, asset_universe, factor_columns, ag, r, 1000)
            scenarios_factor[model_name] = scenarios

        # TODO: transaction cost, ellipsoidal uncertainty, multi-period leverage
        weight_constraints = {'DIS':(0.05, 1.0), 'IBM':(0.05, 1.0), 'JPM':(0.05, 1.0), 'KO':(0.05, 1.0), 'WMT':(0.05, 1.0)}
        for model_name in model_names:

            # robust MVO
            for conf in [1.645, 1.960, 2.576]:
                ef = efficient_frontier.EfficientFrontier(mu_factor[model_name], cov_factor[model_name], trade_horizon=12)
                # ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
                card = np.zeros(shape=len(asset_universe))
                control = 0.50
                for i in range(len(stock_picks)):
                    card[i] = 1
                ef.add_constraint(lambda w: card @ w >= control, broadcast=False, var_list=[0])
                for i in range(len(stock_picks)):
                    min = np.zeros(shape=len(asset_universe))
                    max = np.ones(shape=len(asset_universe))
                    min[i] = weight_constraints[asset_universe[i]][0]
                    max[i] = weight_constraints[asset_universe[i]][1]
                    ef.add_constraint(lambda w: w >= min, broadcast=False, var_list=[0])
                    ef.add_constraint(lambda w: w <= max, broadcast=False, var_list=[0])
                ef.robust_efficient_frontier(target_return=market_return, conf=conf)
                weights = ef.clean_weights()
                weights_dict[(test_start, model_name, conf)] = weights

            # risk parity
            ef = efficient_frontier.EfficientFrontier(mu_factor[model_name], cov_factor[model_name], trade_horizon=12)
            # ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
            card = np.zeros(shape=len(asset_universe))
            control = 0.50
            for i in range(len(stock_picks)):
                card[i] = 1
            ef.add_constraint(lambda w: card @ w >= control, broadcast=False, var_list=[0])
            for i in range(len(stock_picks)):
                min = np.zeros(shape=len(asset_universe))
                max = np.ones(shape=len(asset_universe))
                min[i] = weight_constraints[asset_universe[i]][0]
                max[i] = weight_constraints[asset_universe[i]][1]
                ef.add_constraint(lambda w: w >= min, broadcast=False, var_list=[0])
                ef.add_constraint(lambda w: w <= max, broadcast=False, var_list=[0])
            ef.risk_parity()
            weights = ef.clean_weights()
            weights_dict[(test_start, model_name, None)] = weights

            # max sharpe ratio
            ef = efficient_frontier.EfficientFrontier([mu_factor[model_name][0] for _ in range(2)], [cov_factor[model_name][0] for _ in range(2)], trade_horizon=2)
            # ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
            card = np.zeros(shape=len(asset_universe))
            control = 0.50
            for i in range(len(stock_picks)):
                card[i] = 1
            ef.add_constraint(lambda w: card @ w >= control, broadcast=False, var_list=[0])
            for i in range(len(stock_picks)):
                min = np.zeros(shape=len(asset_universe))
                max = np.ones(shape=len(asset_universe))
                min[i] = weight_constraints[asset_universe[i]][0]
                max[i] = weight_constraints[asset_universe[i]][1]
                ef.add_constraint(lambda w: w >= min, broadcast=False, var_list=[0])
                ef.add_constraint(lambda w: w <= max, broadcast=False, var_list=[0])
            ef.max_sharpe()
            weights = ef.clean_weights()
            weights_dict[(test_start, model_name, None)] = weights

            # cvar opt.
            for alpha in [0.90, 0.95, 0.99]:
                ef = efficient_frontier.EfficientFrontier(mu_factor[model_name], cov_factor[model_name], trade_horizon=12)
                # ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
                card = np.zeros(shape=len(asset_universe))
                control = 0.50
                for i in range(len(stock_picks)):
                    card[i] = 1
                ef.add_constraint(lambda w: card @ w >= control, broadcast=False, var_list=[0])
                for i in range(len(stock_picks)):
                    min = np.zeros(shape=len(asset_universe))
                    max = np.ones(shape=len(asset_universe))
                    min[i] = weight_constraints[asset_universe[i]][0]
                    max[i] = weight_constraints[asset_universe[i]][1]
                    ef.add_constraint(lambda w: w >= min, broadcast=False, var_list=[0])
                    ef.add_constraint(lambda w: w <= max, broadcast=False, var_list=[0])
                ef.min_cvar(target_return=market_return, scenarios=scenarios_factor[model_name], alpha=alpha)
                weights = ef.clean_weights()
                weights_dict[(test_start, model_name, alpha)] = weights

    cursor.close()