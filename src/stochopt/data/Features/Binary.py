from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

from ..Types import CategValue, OneDimData, FloatArray

from .Feature import Feature, Monotonicity, _check_dims_on_encode


class Binary(Feature):
    def __init__(
        self,
        training_vals: OneDimData,
        value_names: Optional[list[CategValue]] = None,
        name: Optional[str] = None,
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        super().__init__(training_vals, name, monotone, modifiable)
        if value_names is None:
            value_names = list(np.unique(training_vals))
        else:
            valid_vals = np.isin(training_vals, value_names)
            if not np.all(valid_vals):
                raise ValueError(
                    f"""Incorrect value in a binary feature {self.name}.
                    Values {np.unique(training_vals[~valid_vals])} are not one of {value_names}"""
                )
        self.__negative_val: CategValue
        self.__positive_val: CategValue
        self.__negative_val, self.__positive_val = value_names
        self._MAD = np.asarray([1.48 * np.nanstd(self.encode(training_vals, one_hot=False))])

    @_check_dims_on_encode
    def encode(self, vals: OneDimData, normalize: bool = True, one_hot: bool = True) -> FloatArray:
        positive = vals == self.__positive_val
        if np.any(vals[~positive] != self.__negative_val):
            unknown = vals[~positive] != self.__negative_val
            raise ValueError(
                f"""Incorrect value in a binary feature {self.name}.
                Values {vals[~positive][unknown]} are not one of [{self.__negative_val},{self.__positive_val}]"""
            )

        # if one_hot:
        #     return np.concatenate(
        #         [np.array(~positive).reshape(-1, 1), np.array(positive).reshape(-1, 1)],
        #         axis=1,
        #         dtype=np.float64,
        #     )
        return self._to_numpy(positive).astype(np.float64)

    def decode(
        self,
        vals: FloatArray,
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        if not np.isin(vals, [0, 1]).all():
            raise ValueError(
                f"""Incorrect value in an encoded feature {self.name}.
                All values must be either 0 or 1. Found values {np.unique(vals[~np.isin(vals, [0,1])])}."""
            )
        vals = vals.flatten()  # TODO put the shape handlings outside, similar to encode
        res = np.empty(vals.shape, dtype=object)
        res[vals == 0] = self.__negative_val
        res[vals == 1] = self.__positive_val

        if return_series:
            return pd.Series(res, name=self.name)
        return res

    def encoding_width(self, one_hot: bool) -> int:
        # return 2 if one_hot else 1
        return 1

    def allowed_change(self, pre_val: CategValue, post_val: CategValue, encoded=True) -> bool:
        if not encoded:
            pre_val = self.encode([pre_val], one_hot=False)[0]
            post_val = self.encode([post_val], one_hot=False)[0]
        if self.modifiable:
            if self.monotone == Monotonicity.INCREASING:
                return pre_val == self.__negative_val or post_val == self.__positive_val
            if self.monotone == Monotonicity.DECREASING:
                return pre_val == self.__positive_val or post_val == self.__negative_val
            return True
        return pre_val == post_val

    @property
    def value_mapping(self) -> dict[CategValue, int]:
        return {self.__positive_val: 1, self.__negative_val: 0}

    @property
    def orig_vals(self) -> List[CategValue]:
        return [self.__negative_val, self.__positive_val]

    @property
    def numeric_vals(self):
        return [0, 1]

    @property
    def n_values(self) -> int:
        return 2
