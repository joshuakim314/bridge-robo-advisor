"""
``BaseMPO`` is the base class for all multi-period optimization
inherits the parent class ``BaseConvexOptimizer``.
"""
import collections
import json
import warnings
from collections.abc import Iterable
from typing import List

import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.optimize as sco

import objective_functions
import exceptions
import base_optimizer

# TODO: modify non_convex_objective name and code in base_optimizer.py
# TODO: allow _map_bounds_to_constraints and add_sector_constraints to have different bounds at different time steps

class BaseMPO(base_optimizer.BaseConvexOptimizer):

    """
    The major difference between BaseConvexOptimizer and BaseMPO is that
    self._w can be a list of cp.Variable().
    Instance variables:
    - ``n_assets`` - int
    - ``tickers`` - str list
    - ``weights`` - np.ndarray
    - ``_opt`` - cp.Problem
    - ``_solver`` - str
    - ``_solver_options`` - {str: str} dict
    Public methods:
    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints
    - ``nonconvex_objective()`` solves for a generic nonconvex objective using the scipy backend.
      This is prone to getting stuck in local minima and is generally *not* recommended.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        n_assets,
        trade_horizon=None,
        tickers=None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param trade_horizon: number of periods to look ahead, default to None (identical to SPO).
        :type trade_horizon: None OR int
        :param solver: name of solver. list available solvers with: ``cvxpy.installed_solvers()``
        :type solver: str, optional. Defaults to "ECOS"
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        """
        super().__init__(n_assets, tickers, weight_bounds, solver, verbose, solver_options)

        # Override the variable as a list of variables
        if not ((isinstance(trade_horizon, int) and trade_horizon >= 1) or trade_horizon is None):
            raise TypeError("trade_horizon must be None or a positive integer")
        self.trade_horizon = trade_horizon
        self._w = [cp.Variable(n_assets) for _ in range(trade_horizon)] if not (trade_horizon is (None or 1)) \
            else cp.Variable(n_assets)
        # self._map_bounds_to_constraints(weight_bounds)

    def _map_bounds_to_constraints(self, test_bounds):
        """
        Convert input bounds into a form acceptable by cvxpy and add to the constraints list.
        :param test_bounds: minimum and maximum weight of each asset OR single min/max pair
                            if all identical OR pair of arrays corresponding to lower/upper bounds. defaults to (0, 1).
        :type test_bounds: tuple OR list/tuple of tuples OR pair of np arrays
        :raises TypeError: if ``test_bounds`` is not of the right type
        :return: bounds suitable for cvxpy
        :rtype: tuple pair or list of tuple pairs of np.ndarray
        """
        if self.trade_horizon is (None or 1):
            return base_optimizer.BaseConvexOptimizer._map_bounds_to_constraints(self, test_bounds)
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) "
                    "OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else:
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds, broadcast=True)
        self.add_constraint(lambda w: w <= self._upper_bounds, broadcast=True)

    def _solve_cvxpy_opt_problem(self):
        """
        Helper method to solve the cvxpy problem and check output,
        once objectives and constraints have been defined
        :raises exceptions.OptimizationError: if problem is not solvable by cvxpy
        """
        if self.trade_horizon is (None or 1):
            return base_optimizer.BaseConvexOptimizer._solve_cvxpy_opt_problem(self)
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._objective.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            else:
                if not self._objective.id == self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization. "
                        "Please create a new instance instead."
                    )

                constr_ids = {const.id for const in self._constraints}
                if not constr_ids == self._initial_constraint_ids:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization. "
                        "Please create a new instance instead."
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        self.weights = self._w[0].value.round(16) + 0.0  # +0.0 removes signed zero
        return self._make_output_weights()

    def add_objective(self, new_objective, broadcast=True, var_list=None, **kwargs):
        """
        Add a new term into the objective function. This term must be convex,
        and built from cvxpy atomic functions.
        Example::
            def L1_norm(w, k=1):
                return k * cp.norm(w, 1)
            ef.add_objective(L1_norm, k=2)
        :param new_objective: the objective to be added
        :type new_objective: cp.Expression (i.e function of cp.Variable)
        :param broadcast: whether the objective is broadcasted to every variable
        :type broadcast: bool, optional
        :param var_list: the list of variable indices to apply the objective
        :type var_list: list or tuple of variable indices (int)
        """
        if self.trade_horizon is (None or 1):
            return base_optimizer.BaseConvexOptimizer.add_objective(self, new_objective, **kwargs)
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding objectives to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of objectives."
            )
        if broadcast:
            if var_list is not None:
                warnings.warn("var_list is not used if broadcast is true")
            for _w_ in self._w:
                self._additional_objectives.append(new_objective(_w_, **kwargs))
        else:
            if not isinstance(var_list, (list, tuple)):
                raise TypeError("var_list must be a list or tuple of variable indices")
            for i in var_list:
                self._additional_objectives.append(new_objective(self._w[i], **kwargs))

    def add_constraint(self, new_constraint, broadcast=True, var_list=None, pairwise=False, block=False):
        """
        Add a new constraint to the optimization problem. This constraint must satisfy DCP rules,
        i.e be either a linear equality constraint or convex inequality constraint.
        Examples::
            ef.add_constraint(lambda x : x[0] == 0.02)
            ef.add_constraint(lambda x : x >= 0.01)
            ef.add_constraint(lambda x: x <= np.array([0.01, 0.08, ..., 0.5]))
        :param new_constraint: the constraint to be added
        :type new_constraint: callable (e.g lambda function)
        :param broadcast: whether the constraint is broadcasted to every variable
        :type broadcast: bool, optional
        :param var_list: the list of variable indices to apply the objective
        :type var_list: list or tuple of variable indices (int)
        :param pairwise: whether the constraint is broadcasted in a pairwise manner
        :type pairwise: bool, optional
        :param block: whether the constraint uses the entire variable list
        :type block: bool, optional
        """
        if self.trade_horizon is (None or 1):
            return base_optimizer.BaseConvexOptimizer.add_constraint(self, new_constraint)
        if not callable(new_constraint):
            raise TypeError(
                "New constraint must be provided as a callable (e.g lambda function)"
            )
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of constraints."
            )
        if broadcast:
            if var_list is not None:
                warnings.warn("var_list is not used if broadcast is true")
            if pairwise:
                for _w1, _w2 in zip(self._w, self._w[1:]):
                    self._constraints.append(new_constraint(_w1, _w2))
            for _w_ in self._w:
                self._constraints.append(new_constraint(_w_))
        else:
            if not (isinstance(var_list, (list, tuple)) or var_list is None):
                raise TypeError("var_list must be a list or tuple of variable indices")
            if block:
                self._constraints.append(new_constraint(self._w))
            else:
                for i in var_list:
                    self._constraints.append(new_constraint(self._w[i]))

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        """
        Adds constraints on the sum of weights of different groups of assets.
        Most commonly, these will be sector constraints e.g portfolio's exposure to
        tech must be less than x%::
            sector_mapper = {
                "GOOG": "tech",
                "FB": "tech",,
                "XOM": "Oil/Gas",
                "RRC": "Oil/Gas",
                "MA": "Financials",
                "JPM": "Financials",
            }
            sector_lower = {"tech": 0.1}  # at least 10% to tech
            sector_upper = {
                "tech": 0.4, # less than 40% tech
                "Oil/Gas": 0.1 # less than 10% oil and gas
            }
        :param sector_mapper: dict that maps tickers to sectors
        :type sector_mapper: {str: str} dict
        :param sector_lower: lower bounds for each sector
        :type sector_lower: {str: float} dict
        :param sector_upper: upper bounds for each sector
        :type sector_upper: {str:float} dict
        """
        if self.trade_horizon is (None or 1):
            return base_optimizer.BaseConvexOptimizer.add_sector_constraints(
                self, sector_mapper, sector_lower, sector_upper
            )
        if np.any(self._lower_bounds < 0):
            warnings.warn(
                "Sector constraints may not produce reasonable results if shorts are allowed."
            )
        for sector in sector_upper:
            is_sector = [sector_mapper[t] == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
        for sector in sector_lower:
            is_sector = [sector_mapper[t] == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    def weights_sum_to_one_constraints(self, broadcast=True, var_list=None):
        if broadcast:
            if var_list is not None:
                warnings.warn("var_list is not used if broadcast is true")
            self.add_constraint(lambda w: cp.sum(w) == 1, broadcast=True)
        else:
            if not isinstance(var_list, (list, tuple)):
                raise TypeError("var_list must be a list or tuple of variable indices")
            for i in var_list:
                self.add_constraint(lambda w: cp.sum(w) == 1, broadcast=False, var_list=var_list)


def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
    """
    Helper function to recursively get all arguments from a cvxpy expression
    :param expression: input cvxpy expression
    :type expression: cp.Expression
    :return: a list of cvxpy arguments
    :rtype: List[cp.Expression]
    """
    if expression.args == []:
        return [expression]
    else:
        return list(_flatten([_get_all_args(arg) for arg in expression.args]))


def _flatten(l: Iterable) -> Iterable:
    # Helper method to flatten an iterable
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
