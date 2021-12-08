"""
The ``expected_returns`` module provides functions for estimating the expected returns of
the assets, which is a required input in mean-variance optimization.
By convention, the output of these methods is expected *annual* returns. It is assumed that
*daily* prices are provided, though in reality the functions are agnostic
to the time period (just change the ``frequency`` parameter). Asset prices must be given as
a pandas dataframe, as per the format described in the :ref:`user-guide`.
All of the functions process the price data into percentage returns data, before
calculating their respective estimates of expected returns.
Currently implemented:
    - general return model function, allowing you to run any return model from one function.
    - mean historical return
    - exponentially weighted mean historical return
    - CAPM estimate of returns
Additionally, we provide utility functions to convert from returns to prices and vice-versa.
"""

import warnings
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pmdarima as pm
import arch
from arch.__future__ import reindexing

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


def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.
    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        return np.log(1 + prices.pct_change()).dropna(how="all")
    else:
        return prices.pct_change().dropna(how="all")


def prices_from_returns(returns, log_returns=False):
    """
    Calculate the pseudo-prices given returns. These are not true prices because
    the initial prices are all set to 1, but it behaves as intended when passed
    to any PyPortfolioOpt method.
    :param returns: (daily) percentage returns of the assets
    :type returns: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) pseudo-prices.
    :rtype: pd.DataFrame
    """
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = 1 + returns
    ret.iloc[0] = 1  # set first day pseudo-price
    return ret.cumprod()


def return_model(prices, method="mean_historical_return", **kwargs):
    """
    Compute an estimate of future returns, using the return model specified in ``method``.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param method: the return model to use. Should be one of:
        - ``mean_historical_return``
        - ``ema_historical_return``
        - ``capm_return``
    :type method: str, optional
    :raises NotImplementedError: if the supplied method is not recognised
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if method == "mean_historical_return":
        return mean_historical_return(prices, **kwargs)
    elif method == "ema_historical_return":
        return ema_historical_return(prices, **kwargs)
    elif method == "capm_return":
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError("Return model {} not implemented".format(method))


def mean_historical_return(
    prices, returns_data=False, compounding=True, frequency=252, log_returns=False
):
    """
    Calculate annualised mean (daily) historical return from input (daily) asset prices.
    Use ``compounding`` to toggle between the default geometric mean (CAGR) and the
    arithmetic mean.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
                         These **should not** be log returns.
    :type returns_data: bool, defaults to False.
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised mean (daily) return for each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency


def ema_historical_return(
    prices,
    returns_data=False,
    compounding=True,
    span=500,
    frequency=252,
    log_returns=False,
):
    """
    Calculate the exponentially-weighted mean of (daily) historical returns, giving
    higher weight to more recent data.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
                         These **should not** be log returns.
    :type returns_data: bool, defaults to False.
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param span: the time-span for the EMA, defaults to 500-day EMA.
    :type span: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised exponentially-weighted mean (daily) return of each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency


def capm_return(
    prices,
    market_prices=None,
    returns_data=False,
    risk_free_rate=0.02,
    compounding=True,
    frequency=252,
    log_returns=False,
):
    """
    Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM,
    asset returns are equal to market returns plus a :math:`\beta` term encoding
    the relative risk of the asset.
    .. math::
        R_i = R_f + \\beta_i (E(R_m) - R_f)
    :param prices: adjusted closing prices of the asset, each row is a date
                    and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param market_prices: adjusted closing prices of the benchmark, defaults to None
    :type market_prices: pd.DataFrame, optional
    :param returns_data: if true, the first arguments are returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                           You should use the appropriate time period, corresponding
                           to the frequency parameter.
    :type risk_free_rate: float, optional
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                        of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised return estimate
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    market_returns = None

    if returns_data:
        returns = prices.copy()
        if market_prices is not None:
            market_returns = market_prices
    else:
        returns = returns_from_prices(prices, log_returns)
        if market_prices is not None:
            market_returns = returns_from_prices(market_prices, log_returns)
    # Use the equally-weighted dataset as a proxy for the market
    if market_returns is None:
        # Append market return to right and compute sample covariance matrix
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")

    # Compute covariance matrix for the new dataframe (including markets)
    cov = returns.cov()
    # The far-right column of the cov matrix is covariances to market
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")
    # Find mean market return on a given time period
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (
            frequency / returns["mkt"].count()
        ) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency

    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)


"""
The ``risk_models`` module provides functions for estimating the covariance matrix given
historical returns.
The format of the data input is the same as that in :ref:`expected-returns`.
**Currently implemented:**
- fix non-positive semidefinite matrices
- general risk matrix function, allowing you to run any risk model from one function.
- sample covariance
- semicovariance
- exponentially weighted covariance
- minimum covariance determinant
- shrunk covariance matrices:
    - manual shrinkage
    - Ledoit Wolf shrinkage
    - Oracle Approximating shrinkage
- covariance to correlation matrix
"""


def _is_positive_semidefinite(matrix):
    """
    Helper function to check if a given matrix is positive semidefinite.
    Any method that requires inverting the covariance matrix will struggle
    with a non-positive semidefinite matrix
    :param matrix: (covariance) matrix to test
    :type matrix: np.ndarray, pd.DataFrame
    :return: whether matrix is positive semidefinite
    :rtype: bool
    """
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """
    Check if a covariance matrix is positive semidefinite, and if not, fix it
    with the chosen method.
    The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,
    while the ``diag`` method adds a small positive value to the diagonal.
    :param matrix: raw covariance matrix (may not be PSD)
    :type matrix: pd.DataFrame
    :param fix_method: {"spectral", "diag"}, defaults to "spectral"
    :type fix_method: str, optional
    :raises NotImplementedError: if a method is passed that isn't implemented
    :return: positive semidefinite covariance matrix
    :rtype: pd.DataFrame
    """
    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn(
        "The covariance matrix is non positive semidefinite. Amending eigenvalues."
    )

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    if fix_method == "spectral":
        # Remove negative eigenvalues
        q = np.where(q > 0, q, 0)
        # Reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError("Method {} not implemented".format(fix_method))

    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        warnings.warn(
            "Could not fix matrix. Please try a different risk model.", UserWarning
        )

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix


def risk_matrix(prices, method="sample_cov", **kwargs):
    """
    Compute a covariance matrix, using the risk model supplied in the ``method``
    parameter.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param method: the risk model to use. Should be one of:
        - ``sample_cov``
        - ``semicovariance``
        - ``exp_cov``
        - ``ledoit_wolf``
        - ``ledoit_wolf_constant_variance``
        - ``ledoit_wolf_single_factor``
        - ``ledoit_wolf_constant_correlation``
        - ``oracle_approximating``
    :type method: str, optional
    :raises NotImplementedError: if the supplied method is not recognised
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if method == "sample_cov":
        return sample_cov(prices, **kwargs)
    elif method == "semicovariance" or method == "semivariance":
        return semicovariance(prices, **kwargs)
    elif method == "exp_cov":
        return exp_cov(prices, **kwargs)
    elif method == "ledoit_wolf" or method == "ledoit_wolf_constant_variance":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf()
    elif method == "ledoit_wolf_single_factor":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
            shrinkage_target="single_factor"
        )
    elif method == "ledoit_wolf_constant_correlation":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
            shrinkage_target="constant_correlation"
        )
    elif method == "oracle_approximating":
        return CovarianceShrinkage(prices, **kwargs).oracle_approximating()
    else:
        raise NotImplementedError("Risk model {} not implemented".format(method))


def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, **kwargs):
    """
    Calculate the annualised sample covariance matrix of (daily) asset returns.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    return fix_nonpositive_semidefinite(
        returns.cov() * frequency, kwargs.get("fix_method", "spectral")
    )


def semicovariance(
    prices,
    returns_data=False,
    benchmark=0.000079,
    frequency=252,
    log_returns=False,
    **kwargs
):
    """
    Estimate the semicovariance matrix, i.e the covariance given that
    the returns are less than the benchmark.
    .. semicov = E([min(r_i - B, 0)] . [min(r_j - B, 0)])
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param benchmark: the benchmark return, defaults to the daily risk-free rate, i.e
                      :math:`1.02^{(1/252)} -1`.
    :type benchmark: float
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year). Ensure that you use the appropriate
                      benchmark, e.g if ``frequency=12`` use the monthly risk-free rate.
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: semicovariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    drops = np.fmin(returns - benchmark, 0)
    T = drops.shape[0]
    return fix_nonpositive_semidefinite(
        (drops.T @ drops) / T * frequency, kwargs.get("fix_method", "spectral")
    )


def _pair_exp_cov(X, Y, span=180):
    """
    Calculate the exponential covariance between two timeseries of returns.
    :param X: first time series of returns
    :type X: pd.Series
    :param Y: second time series of returns
    :type Y: pd.Series
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :return: the exponential covariance between X and Y
    :rtype: float
    """
    covariation = (X - X.mean()) * (Y - Y.mean())
    # Exponentially weight the covariation and take the mean
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    return covariation.ewm(span=span).mean().iloc[-1]


def exp_cov(
    prices, returns_data=False, span=180, frequency=252, log_returns=False, **kwargs
):
    """
    Estimate the exponentially-weighted covariance matrix, which gives
    greater weight to more recent data.
    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised estimate of exponential covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    N = len(assets)

    # Loop over matrix, filling entries with the pairwise exp cov
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(
                returns.iloc[:, i], returns.iloc[:, j], span
            )
    cov = pd.DataFrame(S * frequency, columns=assets, index=assets)

    return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))


def min_cov_determinant(
    prices,
    returns_data=False,
    frequency=252,
    random_state=None,
    log_returns=False,
    **kwargs
):  # pragma: no cover
    warnings.warn("min_cov_determinant is deprecated and will be removed in v1.5")

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    # Extra dependency
    try:
        import sklearn.covariance
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Please install scikit-learn via pip or poetry")

    assets = prices.columns

    if returns_data:
        X = prices
    else:
        X = returns_from_prices(prices, log_returns)
    # X = np.nan_to_num(X.values)
    X = X.dropna().values
    raw_cov_array = sklearn.covariance.fast_mcd(X, random_state=random_state)[1]
    cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * frequency
    return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))


def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame
    :return: correlation matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn("cov_matrix is not a dataframe", RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)


def corr_to_cov(corr_matrix, stdevs):
    """
    Convert a correlation matrix to a covariance matrix
    :param corr_matrix: correlation matrix
    :type corr_matrix: pd.DataFrame
    :param stdevs: vector of standard deviations
    :type stdevs: array-like
    :return: covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(corr_matrix, pd.DataFrame):
        warnings.warn("corr_matrix is not a dataframe", RuntimeWarning)
        corr_matrix = pd.DataFrame(corr_matrix)

    return corr_matrix * np.outer(stdevs, stdevs)


class CovarianceShrinkage:
    """
    Provide methods for computing shrinkage estimates of the covariance matrix, using the
    sample covariance matrix and choosing the structured estimator to be an identity matrix
    multiplied by the average sample variance. The shrinkage constant can be input manually,
    though there exist methods (notably Ledoit Wolf) to estimate the optimal value.
    Instance variables:
    - ``X`` - pd.DataFrame (returns)
    - ``S`` - np.ndarray (sample covariance matrix)
    - ``delta`` - float (shrinkage constant)
    - ``frequency`` - int
    """

    def __init__(self, prices, returns_data=False, frequency=252, log_returns=False):
        """
        :param prices: adjusted closing prices of the asset, each row is a date and each column is a ticker/id.
        :type prices: pd.DataFrame
        :param returns_data: if true, the first argument is returns instead of prices.
        :type returns_data: bool, defaults to False.
        :param frequency: number of time periods in a year, defaults to 252 (the number of trading days in a year)
        :type frequency: int, optional
        :param log_returns: whether to compute using log returns
        :type log_returns: bool, defaults to False
        """
        # Optional import
        try:
            from sklearn import covariance

            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):  # pragma: no cover
            raise ImportError("Please install scikit-learn via pip or poetry")

        if not isinstance(prices, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            prices = pd.DataFrame(prices)

        self.frequency = frequency

        if returns_data:
            self.X = prices.dropna(how="all")
        else:
            self.X = returns_from_prices(prices, log_returns).dropna(how="all")

        self.S = self.X.cov().values
        self.delta = None  # shrinkage constant

    def _format_and_annualize(self, raw_cov_array):
        """
        Helper method which annualises the output of shrinkage calculations,
        and formats the result into a dataframe
        :param raw_cov_array: raw covariance matrix of daily returns
        :type raw_cov_array: np.ndarray
        :return: annualised covariance matrix
        :rtype: pd.DataFrame
        """
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        return fix_nonpositive_semidefinite(cov, fix_method="spectral")

    def shrunk_covariance(self, delta=0.2):
        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.
        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        self.delta = delta
        N = self.S.shape[1]
        # Shrinkage target
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)

    def ledoit_wolf(self, shrinkage_target="constant_variance"):
        """
        Calculate the Ledoit-Wolf shrinkage estimate for a particular
        shrinkage target.
        :param shrinkage_target: choice of shrinkage target, either ``constant_variance``,
                                 ``single_factor`` or ``constant_correlation``. Defaults to
                                 ``constant_variance``.
        :type shrinkage_target: str, optional
        :raises NotImplementedError: if the shrinkage_target is unrecognised
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        if shrinkage_target == "constant_variance":
            X = np.nan_to_num(self.X.values)
            shrunk_cov, self.delta = self.covariance.ledoit_wolf(X)
        elif shrinkage_target == "single_factor":
            shrunk_cov, self.delta = self._ledoit_wolf_single_factor()
        elif shrinkage_target == "constant_correlation":
            shrunk_cov, self.delta = self._ledoit_wolf_constant_correlation()
        else:
            raise NotImplementedError(
                "Shrinkage target {} not recognised".format(shrinkage_target)
            )

        return self._format_and_annualize(shrunk_cov)

    def _ledoit_wolf_single_factor(self):
        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the Sharpe single-factor matrix as the shrinkage target.
        See Ledoit and Wolf (2001).
        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        """
        X = np.nan_to_num(self.X.values)

        # De-mean returns
        t, n = np.shape(X)
        Xm = X - X.mean(axis=0)
        xmkt = Xm.mean(axis=1).reshape(t, 1)

        # compute sample covariance matrix
        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        F = np.dot(betas, betas.T) / varmkt
        F[np.eye(n) == 1] = np.diag(sample)

        # compute shrinkage parameters
        c = np.linalg.norm(sample - F, "fro") ** 2
        y = Xm ** 2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample ** 2)

        # r is divided into diagonal
        # and off-diagonal terms, and the off-diagonal term
        # is itself divided into smaller terms
        rdiag = 1 / t * np.sum(y ** 2) - sum(np.diag(sample) ** 2)
        z = Xm * np.tile(xmkt, (n,))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(betas, (n,)) * sample
        roff1 = (
            np.sum(v1 * np.tile(betas, (n,)).T) / varmkt
            - np.sum(np.diag(v1) * betas.T) / varmkt
        )
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = (
            np.sum(v3 * np.dot(betas, betas.T)) / varmkt ** 2
            - np.sum(np.diag(v3).reshape(-1, 1) * betas ** 2) / varmkt ** 2
        )
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        # compute shrinkage constant
        k = (p - r) / c
        delta = max(0, min(1, k / t))

        # compute the estimator
        shrunk_cov = delta * F + (1 - delta) * sample
        return shrunk_cov, delta

    def _ledoit_wolf_constant_correlation(self):
        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the constant correlation matrix as the shrinkage target.
        See Ledoit and Wolf (2003)
        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        """
        X = np.nan_to_num(self.X.values)
        t, n = np.shape(X)

        S = self.S  # sample cov matrix

        # Constant correlation target
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        # Estimate pi
        Xm = X - X.mean(axis=0)
        y = Xm ** 2
        pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
        pi_hat = np.sum(pi_mat)

        # Theta matrix, expanded term by term
        term1 = np.dot((Xm ** 3).T, Xm) / t
        help_ = np.dot(Xm.T, Xm) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(
            np.dot((1 / std), std.T) * theta_mat
        )

        # Estimate gamma
        gamma_hat = np.linalg.norm(S - F, "fro") ** 2

        # Compute shrinkage constant
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

        # Compute shrunk covariance matrix
        shrunk_cov = delta * F + (1 - delta) * S
        return shrunk_cov, delta

    def oracle_approximating(self):
        """
        Calculate the Oracle Approximating Shrinkage estimate
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.oas(X)
        return self._format_and_annualize(shrunk_cov)


def execute_sql(sql, columns=None, suppress=True):
    if not suppress:
        print(f'executing the command: {sql}')
    cursor.execute(sql)
    selected = cursor.fetchall()
    selected = convert_db_fetch_to_df(selected, columns)
    return selected


def convert_db_fetch_to_df(fetched, column_names=None):
    """
    This method converts the cursor.fetchall() output of SELECT query into a Pandas dataframe.
    :param fetched: the output of SELECT query
    :type fetched: list of row tuples
    :param column_names: column names to use for the dataframe
    :type column_names: list of column names
    :return: converted dataframe
    :rtype: Pandas dataframe
    """
    return pd.DataFrame(fetched, columns=column_names)


def get_factors(start, end, freq='monthly'):
    table = 'fama' if freq == 'monthly' else 'fama_daily' if freq == 'daily' else None
    if not table:
        raise ValueError('frequency must be daily or monthly')
    sql = f'''SELECT * FROM {table} WHERE date BETWEEN '{start}' AND '{end}';''' if freq == 'daily' \
        else f'''SELECT * FROM {table} WHERE date BETWEEN {start} AND {end};'''
    factor_columns = ['date', 'excess', 'smb', 'hml', 'rmw', 'cma', 'riskfree']
    selected = execute_sql(sql, factor_columns)
    selected.set_index('date', inplace=True)
    return selected


def get_returns(ticker, table, start, end, freq='daily'):
    sql = f'''SELECT * FROM {table} WHERE ticker = '{ticker}' AND date BETWEEN '{start}' AND '{end}';'''
    stock_columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']
    selected = execute_sql(sql, stock_columns)
    selected.set_index('date', inplace=True)
    selected.drop(columns=['ticker'], inplace=True)
    return returns_from_prices(selected) if freq == 'daily' else get_monthly_returns(returns_from_prices(selected))


def MLR(X_train, y_train):
    mlr = sklearn.linear_model.LinearRegression()
    mlr.fit(X_train, y_train)
    r_sq = mlr.score(X_train, y_train)
    # print(r_sq)
    # print('intercept:', mlr.intercept_)
    # print('slope:', mlr.coef_)
    return mlr, r_sq


def EN(X_train, y_train, alpha, l1_ratio):
    regr = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    regr.fit(X_train, y_train)
    r_sq = regr.score(X_train, y_train)
    return regr, r_sq


# def factor_forecast(factors, trade_horizon, r=2, s=2):
#     # fit ARIMA on returns
#     arima_model_fitted = pm.auto_arima(factors, start_p=1, start_q=1, d=0, max_p=5, max_q=5,
#                                              out_of_sample_size=trade_horizon, suppress_warnings=True,
#                                              stepwise=True, error_action='ignore')
#     p, d, q = arima_model_fitted.order
#     print(f'arima model: p={p}, d={d}, q={q}')
#     print(arima_model_fitted.summary())
#
#     arima_residuals = arima_model_fitted.arima_res_.resid
#     # fit a GARCH(r,s) model on the residuals of the ARIMA model
#     garch = arch.arch_model(arima_residuals, p=r, q=s)
#     garch_fitted = garch.fit()
#     print(garch_fitted.summary())
#
#     # Use ARIMA to predict mu
#     predicted_mu = arima_model_fitted.predict(n_periods=trade_horizon)[0]
#     # Use GARCH to predict the residual
#     garch_forecast = garch_fitted.forecast(horizon=trade_horizon)
#     # predicted_et = garch_forecast.mean['h.1'].iloc[-1]
#     # Combine both models' output: yt = mu + et
#     # prediction = predicted_mu + predicted_et
#     print(arima_model_fitted.predict(n_periods=trade_horizon))
#     print(garch_forecast.mean)
#     print(garch_forecast.residual_variance)
#     print(garch_forecast.variance)
#     print(predicted_mu)
#     # print(predicted_et)
#     # return prediction


def get_monthly_returns(returns):
    returns.reset_index(level=0, inplace=True)
    returns['date'] = returns['date'].astype(str).str[:7]
    returns['date'] = returns['date'].str.replace('-', '').astype('int')
    returns[[col for col in returns.columns if col != 'date']] += 1
    monthly_returns = returns.groupby('date').prod()
    monthly_returns[[col for col in monthly_returns.columns if col != 'date']] -= 1
    return monthly_returns


def get_all_tickers(table):
    sql = f'''SELECT DISTINCT ticker FROM {table};'''
    stock_columns = ['ticker']
    selected = execute_sql(sql, stock_columns)
    return sorted(list(selected['ticker']))


def arima_garch(factors, trade_horizon, columns):
    fitted_models = {}
    for factor in columns:
        data = factors[factor]
        arima = pm.auto_arima(data, start_p=1, start_q=1, start_P=1, start_Q=1, start_D=1, test='adf',
                              max_p=5, max_q=5, max_P=5, max_Q=5, max_D=5, m=12, seasonal=True,
                              d=None, D=1, trace=False,
                              error_action='ignore', suppress_warnings=True, stepwise=True)
        # print(arima.summary())
        arima_preds, arima_conf_int = arima.predict(n_periods=trade_horizon, return_conf_int=True)

        arima_residuals = arima.arima_res_.resid
        # fit a GARCH(r,s) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=2, q=2).fit(disp="off")
        # print(garch.summary())
        garch_preds = garch.forecast(horizon=trade_horizon)
        fitted_models[factor] = (arima, arima_preds, garch, garch_preds)
    return fitted_models


if __name__ == '__main__':
    freq = 'monthly'
    start_date = '2006-01-01' if freq == 'daily' else '200601'
    end_date = '2015-12-31' if freq == 'daily' else '201512'
    factors = get_factors(start=start_date, end=end_date, freq=freq)
    print(factors)

    start_date = '2006-01-01'
    end_date = '2015-12-31'

    tables = ['canadianetfs', 'americanetfs']
    header = ['ticker', 'alpha', 'excess_beta', 'smb_beta', 'hml_beta', 'rmw_beta', 'cma_beta', 'size', 'r_sq']
    for table in tables:
        with open(f'factor_loadings_{table}.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            tickers = get_all_tickers(table)
            for tick in tickers:
                returns = None
                try:
                    returns = get_returns(tick, table, start=start_date, end=end_date, freq=freq)
                    has_null = returns[['adj_close']].isnull().values.any()
                    if has_null: print(tick, 'NULL')
                    else:
                        merged = pd.merge(factors, returns, left_on='date', right_on='date', how="inner", sort=False)
                        merged.dropna(inplace=True)
                        factors_only = merged[['excess', 'smb', 'hml', 'rmw', 'cma']]
                        merged['adj_close_rf'] = merged['adj_close'] - merged['riskfree'].astype('float')
                        adj_returns = merged[['adj_close_rf']]
                        adj_returns = get_monthly_returns(adj_returns)
                        mlr, r_sq = MLR(factors_only, adj_returns['adj_close_rf'])
                        wdata = [tick, mlr.intercept_] + mlr.coef_.tolist() + [adj_returns.shape[0], r_sq]
                        writer.writerow(wdata)
                        print(wdata)
                except KeyError:
                    print(tick, 'KEYERROR')

    cursor.close()

    # factor_columns = ['excess', 'smb', 'hml', 'rmw', 'cma']
    # for factor in factor_columns:
    #     data = factors[factor]
    #     trade_horizon = 12
    #     train, test = pm.model_selection.train_test_split(data, test_size=trade_horizon)
    #
    #     # #############################################################################
    #     # Fit with some validation (cv) samples
    #     # arima = pm.auto_arima(train, start_p=1, start_q=1, d=0, max_p=5, max_q=5,
    #     #                       out_of_sample_size=trade_horizon, suppress_warnings=True,
    #     #                       stepwise=True, error_action='ignore')
    #     # arima = pm.ARIMA(order=(2, 0, 2), seasonal_order=(0, 1, 1, 12))
    #     # arima.fit(train)
    #     arima = pm.auto_arima(data, start_p=1, start_q=1, start_P=1, start_Q=1, start_D=1, test='adf',
    #                           max_p=5, max_q=5, max_P=5, max_Q=5, max_D=5, m=12, seasonal=True,
    #                           d=None, D=1, trace=True,
    #                           error_action='ignore', suppress_warnings=True, stepwise=True)
    #
    #     p, d, q = arima.order
    #     print(f'arima model: p={p}, d={d}, q={q}')
    #     print(arima.summary())
    #
    #     # Now plot the results and the forecast for the test set
    #     preds, conf_int = arima.predict(n_periods=test.shape[0],
    #                                     return_conf_int=True)
    #
    #     # Print the error:
    #     print("Test RMSE: %.3f" % np.sqrt(sklearn.metrics.mean_squared_error(test, preds)))
    #
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    #     x_axis = np.arange(train.shape[0] + preds.shape[0])
    #     axes[0].plot(x_axis[:train.shape[0]], train, alpha=0.75)
    #     axes[0].scatter(x_axis[train.shape[0]:], preds, alpha=0.4, marker='o')
    #     axes[0].scatter(x_axis[train.shape[0]:], test, alpha=0.4, marker='x')
    #     axes[0].fill_between(x_axis[-preds.shape[0]:], conf_int[:, 0], conf_int[:, 1],
    #                          alpha=0.1, color='b')
    #
    #     # fill the section where we "held out" samples in our model fit
    #
    #     axes[0].set_title("Train samples & forecasted test samples")
    #
    #     # Now add the actual samples to the model and create NEW forecasts
    #     arima.update(test)
    #     new_preds, new_conf_int = arima.predict(n_periods=trade_horizon, return_conf_int=True)
    #     new_x_axis = np.arange(data.shape[0] + trade_horizon)
    #
    #     axes[1].plot(new_x_axis[:data.shape[0]], data, alpha=0.75)
    #     axes[1].scatter(new_x_axis[data.shape[0]:], new_preds, alpha=0.4, marker='o')
    #     axes[1].fill_between(new_x_axis[-new_preds.shape[0]:],
    #                          new_conf_int[:, 0],
    #                          new_conf_int[:, 1],
    #                          alpha=0.1, color='g')
    #     axes[1].set_title("Added new observed values with new forecasts")
    #     plt.show()
