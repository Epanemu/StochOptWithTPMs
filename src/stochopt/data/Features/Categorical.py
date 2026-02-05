from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from ..Types import CategValue, FloatArray, OneDimData
from .Feature import Feature, Monotonicity, _check_dims_on_encode


class Categorical(Feature):
    def __init__(
        self,
        training_vals: OneDimData,
        value_names: Optional[list[CategValue]] = None,
        map_to: Optional[list[int]] = None,
        ordering: list[CategValue] | None = None,  # TODO separate into subclass?
        name: Optional[str] = None,
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        super().__init__(training_vals, name, monotone, modifiable)
        if value_names is None:
            value_names = list(np.unique(training_vals))
        if map_to is None:
            map_to = list(range(len(value_names)))
        self.__value_names = value_names
        self.__mapped_to = map_to
        self._MAD = np.asarray(
            1.48 * np.nanstd(self.encode(training_vals, one_hot=True), axis=0)
        )
        if ordering is not None and len(ordering) != len(value_names):
            raise ValueError("Ordering is not complete")
        self.__ordering: list[CategValue] | None = ordering

    @property
    def n_categorical_vals(self) -> int:
        return len(self.__value_names)

    @property
    def n_values(self) -> int:
        return self.n_categorical_vals

    @property
    def orig_vals(self) -> List[CategValue]:
        return self.__value_names

    @property
    def numeric_vals(self) -> List[int]:
        if self.__ordering is not None:
            return [self.value_mapping[i] for i in self.__ordering]
        else:
            return self.__mapped_to

    @_check_dims_on_encode
    def encode(
        self, vals: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> FloatArray:
        masks = np.zeros_like(vals, dtype=bool)
        if one_hot:
            res = np.zeros((vals.shape[0], self.n_categorical_vals), dtype=np.float64)
            for val, mapped in zip(self.__value_names, self.__mapped_to):
                mask = vals == val
                res[mask, mapped] = 1.0
                masks |= mask
        else:
            res = np.empty_like(vals, dtype=np.float64)
            for val, mapped in zip(self.__value_names, self.__mapped_to):
                mask = vals == val
                res[mask] = mapped
                masks |= mask
        if not np.all(masks):
            raise ValueError(
                f"""Incorrect value in a categorical feature {self.name}.
                Values {np.unique(vals[~masks])}
                    are not one of {self.__value_names}."""
            )

        return res

    def decode(
        self,
        vals: FloatArray,
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        is_one_hot = len(vals.shape) > 1 and vals.shape[1] > 1
        relevant_vals = [0, 1] if is_one_hot else self.__mapped_to
        if not np.isin(vals, relevant_vals).all():
            raise ValueError(
                f"""Incorrect value in an encoded feature {self.name}.
                All values must be in {relevant_vals}. Found values {np.unique(vals[~np.isin(vals, relevant_vals)])}."""
            )

        res = np.empty((vals.shape[0],), dtype=object)
        if is_one_hot:
            for i in range(vals.shape[1]):
                res[vals[:, i].astype(bool)] = self.__value_names[i]
        else:
            for val, mapped in zip(self.__value_names, self.__mapped_to):
                res[vals == mapped] = val
        if return_series:
            return pd.Series(res, name=self.name)
        return res

    def encoding_width(self, one_hot: bool) -> int:
        if one_hot:
            return self.n_categorical_vals
        return 1

    @property
    def value_mapping(self) -> dict[CategValue, int]:
        return {
            val: mapped for val, mapped in zip(self.__value_names, self.__mapped_to)
        }

    def lower_than(self, num_val: int) -> list[int]:
        if self.__ordering is None:
            return []
        lower: list[int] = []
        for v in self.__ordering:
            if self.value_mapping[v] == num_val:
                break
            lower.append(self.value_mapping[v])
        return lower

    def greater_than(self, num_val: int) -> list[int]:
        if self.__ordering is None:
            return []
        greater: list[int] = []
        adding = False
        for v in self.__ordering:
            if adding:
                greater.append(self.value_mapping[v])
            if self.value_mapping[v] == num_val:
                adding = True
        return greater

    def allowed_change(
        self, pre_val: CategValue, post_val: CategValue, encoded=True
    ) -> bool:
        if not encoded:
            pre_val = self.encode([pre_val], one_hot=False)[0]
            post_val = self.encode([post_val], one_hot=False)[0]
        if self.modifiable:
            if self.monotone == Monotonicity.INCREASING:
                return (
                    post_val in self.greater_than(int(pre_val)) or post_val == pre_val
                )
            if self.monotone == Monotonicity.DECREASING:
                return post_val in self.lower_than(int(pre_val)) or post_val == pre_val
            return True
        return pre_val == post_val

    # TODO fix the numeric/non-numeric value handling
