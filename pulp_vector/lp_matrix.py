import pulp
import numpy as np
import pandas as pd
from typing import Generator, Union, Literal, Callable, Sequence, Any
from operator import add, sub, mul, eq, le, ge
from collections.abc import Generator
from collections import defaultdict
import itertools

from .lp_array import LpArray


class LpMatrix:
    """2-Dimensional `pandas.DataFrame`-like structure with support for `pulp.LpVariable` objects, and support for \
        applying element-wise PuLP constraints when associated with `pulp.LpProblem`.
    """

    def __init__(self, data: Sequence[Sequence | LpArray] | np.ndarray | 'LpMatrix' = None,
                 index: Sequence[float] = None, columns: Sequence[float | int | np.int64 | str] = None):
        """This class models an array with the following parameters:
        Args:
            data (Sequence[Sequence  |  LpArray] | np.ndarray | LpMatrix, optional): 1d or 2d data.\
                Defaults to `None`.
            index (Sequence[float], optional): Indices (usually `int`) paired to corrresponding rows.\
                Defaults to `None`.
            columns (Sequence[Any], optional): Names for corrresponding columns. Defaults to `None`.
        """
        if data is None:
            data = []
        self.values = np.array(data)

        if index is None:
            index = range(len(self.values))
        self.index = np.array(index)

        if columns is None:
            if self.values.ndim == 1:
                columns = []
            elif self.values.ndim == 2:
                columns = range(len(self.values[0]))
            else:
                raise ValueError(
                    f"{self.values.ndim} is not a valid # of dimensions for LpMatrix data")
        self.columns = np.array(columns)

        try:
            if len(self.values) != len(self.index):
                raise ValueError(f"data and index length are not compatible")
        except TypeError:
            raise TypeError("data and/or index are not sequential")
        try:
            if self.values.ndim == 2 and len(self.values[0]) != len(self.columns):
                raise ValueError("data and column size are not compatible")
        except TypeError:
            raise TypeError("columns are not sequential")

    @classmethod
    def from_dict(cls, data) -> 'LpMatrix':
        """Generate LpMatrix instance from dict with following structure:\
            `{col_1: {index_1: i_11, ..., index_n: i_n1}, ...col_m: {index_1: i_1m, ..., index_n: i_nm}`"""

        match data[0]:
            case dict():
                return cls(np.array([list(d.values()) for d in data.values()]).T,
                           list(list(data.values())[0].keys()), list(data.keys()))

            case LpArray():
                return cls(np.array([array.values for array in data.values()]).T, list(data.values())[0].index,
                           list(data.keys()))

    @ classmethod
    def variable(cls, name: str = "NoName", index: Sequence[float] = None,
                 columns: Sequence[Any] = None, lower: float = None, upper: float = None,
                 cat: type[bool | int | float] = None) -> 'LpMatrix[pulp.LpVariable]':
        """Initialise `LpMatrix` containing `pulp.LpVariable` objects, with the following parameters:
        Args:
            name (str, optional): Name for `pulp.LpVariable`. Defaults to `"NoName"`.
            index (Sequence[float], optional): Index of returned `LpMatrix`. Defaults to `None`.
            columns (Sequence[Any], optional): Column names of returned `LpMatrix`. Defaults to `None`.
            lower (float, optional): Lower bound for variables to be created. Defaults to `None`.
            upper (float, optional): Upper bound for variables to be created. Defaults to `None`.
            cat (type[bool | int | float], optional): Category of variables: `bool`, `int`, or `float`.\
                Defaults to `None`.
        Returns:
            LpMatrix[pulp.LpVariable]: Values named "{name}_{i}_{col}" for i, col in index, columns
        """
        # Create dict of dicts of pulp.LpVariable objects
        dct = {i: pulp.LpVariable.dict(f"{name}_{i}", columns, lower, upper, (
            'Binary', 'Integer', 'Continuous')[(bool, int, float).index(cat)]) for i in index}

        # Transpose layers of dict
        d = defaultdict(dict)
        for key1, inner in dct.items():
            for key2, value in inner.items():
                d[key2][key1] = value
        return cls.from_dict(d)

    def __str__(self) -> str:
        return str(pd.DataFrame(self.values, self.index, self.columns).astype(str))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Generator:
        for column in self.columns:
            yield column

    def __getitem__(self, item: float | str | Sequence[float | bool] | tuple, by: Literal['index', 'location'] = 'index'):
        try:
            return self.filter(item)    # Try item as binary filter
        except (ValueError, TypeError):  # Non-binary or zero length
            # Try item as sequence of indices
            return self.get_subset(item, by)

    def filter(self, item: Sequence[bool], inplace: bool = False) -> Union[None, 'LpMatrix']:
        """Filter `LpMatrix` using a binary sequence of the same length.
        Args:
            item (Sequence[bool]): Squence of `bool` values, indicating whether to include nth row
            inplace (bool): True => filter existing object. False => return new filtered object. Defaults to `False`
        Raises:
            ValueError: Attempt to filter with non-binary or differently-sized data
         Returns:
             LpMatrix | None: Filtered LpMatrix if not inplace
        """
        if len(item) != len(self):  # Filter has the wrong length
            raise ValueError(
                f"Invalid LpMatrix filter: {item} does not have the same length as LpMatrix \
                    ({len(item)} vs. {len(self)})")

        if not all([(i in (0, 1)) for i in item]):  # Filter is not binary
            raise ValueError(
                f"Invalid LpMatrix filter: {item} is not a binary sequence")

        if inplace:
            self.values, self.index = self.values[item], self.index[item]

        return LpMatrix(self.values[item], self.index[item], self.columns)

    def get_subset(self, item: float | str | Sequence[float | bool] | tuple,
                   by: Literal['index', 'location'] = 'index',
                   inplace: bool = False) -> Union[None, 'LpMatrix']:
        """_summary_
        Args:
            inplace: ONLY FOR MULTIPLE ROW AND COLUMN SELECTION
        Raises:
            ValueError: _description_
            ValueError: _description_
            TypeError: _description_
        Returns:
            _type_: _description_
        """

        match item:
            case float() | int() | np.int64() | str() | slice() as col:   # Single column wanted
                if type(item) == slice:
                    return LpMatrix(self.values, self.index, self.columns)
                if by == 'index':
                    try:
                        col = self.columns.tolist().index(col)
                    except ValueError:
                        raise ValueError(f"{col} is not in columns")
                return LpArray(self.values[:, col], self.index)

            case ([*indices], [*columns]):  # Non-trivial subset of index and columns
                if by == 'index':
                    try:
                        indices, columns = [self.index.tolist().index(i) for i in indices], [
                            self.columns.tolist().index(col) for col in columns]
                    except ValueError:
                        raise ValueError(
                            "Not all given indices and columns are in LpMatrix")
                if not inplace:
                    return LpMatrix(self.values[[[i] for i in indices], columns],
                                    self.index[indices], self.columns[columns])
                self.values = self.values[[[i] for i in indices], columns]
                self.index, self.columns = self.index[indices], self.columns[columns]

            case ([*indices], col):  # Non-trivial subset of index across one column
                if type(col) == slice:
                    return self.get_subset((indices, list(self.columns)), by=by)
                return self[col].get_subset(indices, by=by)

            case (i, [*columns]):  # Non-trivial subset of columns across one index
                if type(i) == slice:
                    return self.get_subset((list(self.index), columns), by=by)
                return self.T[i].get_subset(columns, by=by)

            case (i, col):  # Single entry in LpMatrix
                if type(i) == slice:
                    return self.get_subset((list(self.index), col), by=by)
                elif type(col) == slice:
                    return self.get_subset((i, list(self.columns)), by=by)

                return self[col].get_subset(i, by=by)

            case [*columns]:  # Non-trivial subset of columns
                return self[:, columns]

            case _:
                raise TypeError(f"{item} is not a valid index")

    def drop(self, *item:  float | Sequence[float], by: Literal['index', 'location'] = 'index',
             axis: Literal[0, 1] = 0) -> Union[None, 'LpMatrix']:
        """Remove element from LpArray by its index or location
       Args:
            item (float | Sequence[float]): Index/location or sequence of indices/locations to be dropped
            by (Literal['index', 'location'], optional): If elements are to be selected by index or location.\
                Defaults to `'index'`
            inplace (bool): True => drop from existing object. False => return copy with elements dropped.\
                Defaults to `False`
            axis (Literal[0, 1]): `0` => row-wise operation, `1` => column-wise operation. Defaults to `0`
        Returns:
            LpMatrix: with relevant rows/columns dropped
        """
        try:
            axis_obj = [self.index, self.columns][axis]
        except IndexError:
            raise ValueError(f"{axis} is not a valid axis. Must be 0 or 1")
        match item:
            case float() | int() | np.int64():  # Drop single element
                if by == 'location':
                    item = axis_obj[item]

            case [*items]:  # Drop sequence of elements
                if by == 'location':
                    item = axis_obj[items]

            case _:  # Unrecognized type of element index/location
                raise TypeError(
                    f"Type {type(item).__name__} cannot be dropped")

        item = np.setdiff1d(axis_obj, np.array(item))

        if axis == 1:
            return self[:, list(item)]

        else:
            return self[list(item), :]

    def remove(self) -> Union[None, 'LpMatrix']:
        pass  # TODO

    def operator(self, operation: Callable, other: Union['LpArray', pd.Series, np.ndarray, list, float],
                 drop: bool = True) -> 'LpMatrix':
        match other:
            case LpMatrix() | pd.DataFrame():
                if list(other.index) == list(self.index) and list(other.columns) == list(self.columns):
                    return LpMatrix(operation(self.values, other.values), self.index, self.columns)
                if not drop:
                    return
                common_indices = np.intersect1d(self.index, other.index)
                return LpMatrix.from_dict({
                    col: self[list(common_indices), col].operator(
                        operation, other[col][common_indices], drop) for col in self if col in other
                })

            case np.ndarray() | list() | LpArray() | pd.Series() | pulp.pulp.LpVariable() | pulp.pulp.LpAffineExpression():
                if len(other) != len(self):
                    raise ValueError(
                        f"cannot {operation.__name__} 'LpMatrix' to '{type(other).__name__}' of different size")

                return LpMatrix.from_dict({col: self[col].operator(operation, other, drop) for col in self})

            case float() | int() | np.int64():
                return LpMatrix(operation(self.values, other), self.index, self.columns)

    # Apply generic operation method to specific operations
    def __add__(self, other, drop=True):
        return self.operator(add, other, drop)
    __radd__ = __add__

    def __or__(self, other, drop=False):
        return self.__add__(other, drop=drop)

    def __sub__(self, other, drop=True):
        return self.operator(sub, other, drop)

    def __rsub__(self, other, drop=True):
        return -self.operator(sub, other, drop)

    def __mul__(self, other, drop=True):
        return self.operator(mul, other, drop)
    __rmul__ = __mul__

    def __neg__(self):

        return LpArray(-self.values, self.index, self.columns)

    def sum(self, axis: bool = 1) -> LpArray:
        if axis == 1:
            return LpArray([self[col].sum() for col in self], self.columns)
        elif axis == 0:
            return LpArray([self[i, :].sum() for i in self.index], self.index)
        raise ValueError(f"Invalid axis for summation: {axis}")

    def lag(self, lag_value: int = 1, axis: bool = 0) -> 'LpMatrix':
        try:
            axis = [self.index, self.columns][axis]
        except IndexError:
            raise ValueError(f"Invalid axis for lag: {axis}")
        axis += 1

    def transpose(self, inplace=False) -> 'LpMatrix':
        if inplace:
            self.index, self.columns = self.columns, self.index
            self.values = self.values.T
        return LpMatrix(self.values.T, self.columns, self.index)

    @property
    def T(self):
        return self.transpose()

    def add_constraint(self, other: Union[float, 'LpArray', pd.Series, Sequence, pd.DataFrame, 'LpMatrix'],
                       sense: type[pulp.const.LpConstraintEQ |
                                   pulp.const.LpConstraintLE | pulp.const.LpConstraintGE]
                       ) -> Generator[pulp.LpConstraint]:
        match other:
            case pd.DataFrame() | LpMatrix():
                columns = np.intersect1d(self.columns, other.columns)
                print(columns)
                constraints = itertools.chain()
                for col in columns:
                    constraints = itertools.chain(
                        constraints, self[col].add_constraint(other[col], sense))
                return (const for const in constraints)

            case _:
                constraints = itertools.chain()
                for col in self.columns:
                    constraints = itertools.chain(
                        constraints, self[col].add_constraint(other, sense))
                return (const for const in constraints)

    def __eq__(self, other):
        return self.add_constraint(other, eq)

    def __le__(self, other):
        return self.add_constraint(other, le)

    def __ge__(self, other):
        return self.add_constraint(other, ge)
