import operator
from functools import reduce
from fractions import Fraction
from typing import NamedTuple, Dict, List

import numpy as np
from scipy.linalg import null_space

class Variable(NamedTuple):
    name: str
    exponents: Dict[str, Fraction]

    def __str__(self):
        return f"{self.name} = " + ' * '.join(f"{var}^{exp}" for var, exp in self.exponents.items())

class BuckinghamTransformer:
    def __init__(self, variables: List[Variable]):
        self._variables = variables
        self._fundamentals = list({d for var in variables for d in var.exponents})
        self._dimensional_matrix = self._get_dimensional_matrix()
        self._null_space = self._get_null_space()
        self._groups = self._get_groups()

    def _get_dimensional_matrix(self):
        m = np.zeros((len(self.fundamentals), len(self.variables)))
        for j, variable in enumerate(self.variables):
            for var, exp in variable.exponents.items():
                i = self.fundamentals.index(var)
                m[i, j] = exp
        return m

    def _get_null_space(self):
        ns = null_space(self.dimensional_matrix)
        return ns / np.abs(ns).min()

    def _get_groups(self):
        groups = np.zeros_like(self.null_space[0], dtype=Variable)
        for i, exponents in enumerate(self.null_space.T):
            groups[i] = Variable(f"pi_{i}", {var.name: exp for var, exp in zip(self.variables, exponents)})
        return groups

    @property
    def variables(self):
        return self._variables

    @property
    def fundamentals(self):
        return self._fundamentals

    @property
    def dimensional_matrix(self):
        return self._dimensional_matrix

    @property
    def null_space(self):
        return self._null_space

    @property
    def groups(self):
        return self._groups

    def transform(self, values):
        assert np.shape(values)[0] == np.shape(self.variables)[0], "Incorrect number of values"

        result = np.zeros_like(self.groups, dtype=float)
        for i, group in enumerate(self.groups):
            result[i] = reduce(operator.mul, (val ** exp for val, exp in zip(values, group.exponents.values())))
        return result

    def axis_transform(self, values, axis=1):
        result = np.apply_along_axis(self.transform, axis, values)

        assert result[0].shape == self.groups.shape, "The result is not the expected shape. Check the axis"
        return result
