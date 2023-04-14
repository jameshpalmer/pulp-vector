from typing import Literal, Sequence, Union
import numpy as np


class LpIndexer:
    from .lp_array import LpArray
    from .lp_matrix import LpMatrix

    def __init__(self, object: Union[LpArray, LpMatrix],
                 index_type: Literal['index', 'location'] = 'index'):
        """Utility to assist indexwise referenceing in LpArray and LpMatrix object types
        Args:
            object (Union[LpArray, LpMatrix]): Instance of `LpArray` or `LpMatrix`.
            index_type (Literal['index', 'location'], optional): Locate by index or location. Defaults to 'index'.
        """
        self.object = object

        if index_type == 'location':
            self.object.index = np.array(range(len(object)))
        if object.dim > 2:
            self.levels = object.levels
        elif object.dim > 1:
            self.columns = object.columns

    def __getitem__(self, item:  float | Sequence[float | bool]):
        from .lp_array import LpArray
        from .lp_matrix import LpMatrix

        match self.object:
            case LpArray():
                subset = self.object[item]
                if len(subset) == 1:
                    return subset.values[0]
                elif len(subset) > 1:
                    return subset.values

            case LpMatrix():    # TODO
                pass

    def __setitem__(self, item, value):
        from .lp_array import LpArray

        match self.object:
            case LpArray():
                self.object[item] = value
