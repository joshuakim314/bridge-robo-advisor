import yfinance as yf
import matplotlib.pyplot as plt
import collections
import pandas as pd
import numpy as np
import cvxpy as cp
import static.efficient_frontier as efficient_frontier
import static.param_estimator as param_estimator
import static.backtest as backtest
import static.objective_functions as objective_functions


def port_opt(stock_picks, weight_constraints, control, trade_horizon, cardinality, target_return, risk_aversion):
    selected_etfs = ['IWD', 'IYH', 'IYW', 'MDY', 'EWT', 'XLE', 'EWZ', 'EWY', 'IWB', 'EZU']
    if cardinality >= 20:
        selected_etfs = ['IWD', 'IYH', 'IYW', 'MDY', 'EWT', 'XLE', 'EWZ', 'EWY', 'IWB', 'EZU']
    num_stocks = len(stock_picks)
    train_start, train_end = '2016-12-01', '2021-11-30'

    etf_table = 'americanetfs'
    etf_tickers = selected_etfs
    etf_returns_by_tick = []
    for tick in etf_tickers:
        returns = param_estimator.get_returns(tick, etf_table, train_start, train_end, freq='monthly')
        if returns.empty:
            continue
        returns[tick] = returns['adj_close']
        etf_returns_by_tick += [returns[[tick]]]
    etf_returns = pd.concat(etf_returns_by_tick, axis=1).T.dropna()
    train_etf_returns = etf_returns.T

    etf_table = 'spy'
    print(stock_picks)
    stock_returns_by_tick = []
    for tick in stock_picks:
        returns = param_estimator.get_returns(tick, etf_table, train_start, train_end, freq='monthly')
        if returns.empty:
            continue
        returns[tick] = returns['adj_close']
        stock_returns_by_tick += [returns[[tick]]]
    stock_returns = pd.concat(stock_returns_by_tick, axis=1).T.dropna()
    train_stock_returns = stock_returns.T

    # Fama-French factors
    train_factors = param_estimator.get_factors(start=int(train_start[0:4] + train_start[5:7]),
                                                end=int(train_end[0:4] + train_end[5:7]), freq='monthly')

    asset_universe = stock_picks + selected_etfs
    train_returns = pd.concat([train_stock_returns, train_etf_returns], axis=1)

    # historical average param. estimation
    mu, sample_cov = backtest.historical_avg(train_returns, 12 * 5, 12)
    print(sample_cov)
    factor_model = dict()

    for tick in asset_universe:
        merged = pd.merge(train_factors, train_returns[[tick]], left_on='date', right_on='date', how="inner",
                          sort=False)
        ff5 = merged[['excess', 'smb', 'hml', 'rmw', 'cma']]
        merged[tick] = merged[tick] - merged['riskfree'].astype('float')/100.0
        adj_returns = merged[[tick]]

        alpha = 1e-1
        l1_ratio = 0.05

        en5, en_r_sq5 = param_estimator.EN(ff5, adj_returns[tick], alpha=alpha, l1_ratio=l1_ratio)
        factor_model[tick] = en5

    # arima-garch
    ag = param_estimator.arima_garch(train_factors[['excess', 'smb', 'hml', 'rmw', 'cma']], trade_horizon=trade_horizon,
                                            columns=['excess', 'smb', 'hml', 'rmw', 'cma'])

    mu_factor = []
    for month in range(trade_horizon):
        mu_month = []
        for tick in asset_universe:
            data = [ag[factor_name][1][month-1] for factor_name in ['excess', 'smb', 'hml', 'rmw', 'cma']]
            mu = factor_model[tick].predict(np.array(data).reshape(1, -1))
            mu_month.append(mu[0])
        mu_factor.append(mu_month)
        print(mu_month)
    mu_factor = [pd.Series(mu_factor[i]) for i in range(trade_horizon)]
    print(mu_factor)

    print(sample_cov)

    ef = efficient_frontier.EfficientFrontier(mu_factor, sample_cov, trade_horizon=trade_horizon)
    # ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(len(asset_universe)), k=0.001)
    for i in range(num_stocks):
        min = np.zeros(shape=len(asset_universe))
        max = np.ones(shape=len(asset_universe))
        min[i] = weight_constraints[asset_universe[i]][0]/100.0
        max[i] = weight_constraints[asset_universe[i]][1]/100.0
        ef.add_constraint(lambda w: w >= min, broadcast=False, var_list=[0])
        ef.add_constraint(lambda w: w <= max, broadcast=False, var_list=[0])
    card = np.zeros(shape=len(asset_universe))
    for i in range(num_stocks):
        card[i] = 1
    ef.add_constraint(lambda w: card @ w >= control, broadcast=False, var_list=[0])
    print(ef.n_assets)
    print(ef.trade_horizon)
    print(ef.cov_matrices)
    print(ef.expected_returns)
    ef.efficient_return(target_return=target_return)
    weights = ef.clean_weights()
    print(weights)

    new_weights = dict(weights)
    proper_weights = {}
    for key in new_weights.keys():
        proper_weights[asset_universe[key]] = weights[key]

    print(proper_weights)


    weights = pd.DataFrame.from_dict(new_weights, orient='index')
    exp_returns = {month: np.dot(mu_factor[month-1], weights) for month in range(trade_horizon)}
    ret_exp = {}
    for key in exp_returns.keys():
        ret_exp[key+1] = (1 + exp_returns[key][0])

    for key in ret_exp.keys():
        if key != 1:
            ret_exp[key] = ret_exp[key]*ret_exp[key-1]

    return proper_weights, ret_exp