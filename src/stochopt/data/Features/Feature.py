from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

from ..Types import OneDimData, FloatArray


class Monotonicity(Enum):
    INCREASING = 1
    NONE = 0
    DECREASING = -1


def _check_dims_on_encode(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to ensure encode methods receive 1D data."""

    def dim_check(self: Any, vals: OneDimData, *args: Any, **kwargs: Any) -> Any:
        if isinstance(vals, (np.ndarray, pd.Series)):
            # can be squeezed to 0 dims if a single value is passed
            if len(vals.shape) > 1:
                if len(np.squeeze(vals).shape) > 1:
                    raise ValueError("Incorrect dimension of feature")
                return func(self, vals.flatten(), *args, **kwargs)
            return func(self, vals, *args, **kwargs)
        if isinstance(vals, list):
            return list(func(self, np.array(vals), *args, **kwargs))
        # we assume it is a single value
        return func(self, np.array([vals]), *args, **kwargs)[0]

    return dim_check


class Feature(ABC):
    """Abstract base class for all feature types."""

    def __init__(
        self,
        training_vals: OneDimData,
        name: Optional[str],
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        if name is None:
            if isinstance(training_vals, pd.Series):
                name = str(training_vals.name)
            else:
                raise ValueError("Name of the feature must be specified in pd.Series or directly")
        if training_vals.shape[0] == 0:
            raise ValueError(f"No data provided to feature {name}")
        self.__name = name
        self.__monotone = monotone
        self.__modifiable = modifiable
        self._MAD: FloatArray = np.array([1.0], dtype=np.float64)

    @property
    def monotone(self) -> Monotonicity:
        return self.__monotone

    @property
    def modifiable(self) -> bool:
        return self.__modifiable

    def _to_numpy(self, vals: OneDimData) -> FloatArray:
        if isinstance(vals, pd.Series):
            res: FloatArray = vals.to_numpy().astype(np.float64)
            return res
        return vals.astype(np.float64)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def MAD(self) -> FloatArray:
        return self._MAD

    @property
    @abstractmethod
    def n_values(self) -> Union[int, float]:
        """Returns the number of discrete values/bins for this feature."""

    @abstractmethod
    def encode(self, vals: OneDimData, normalize: bool = True, one_hot: bool = True) -> FloatArray:
        """Encodes the vals"""

    @abstractmethod
    def decode(
        self,
        vals: FloatArray,
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        """Decodes the vals into the original form"""

    @abstractmethod
    def encoding_width(self, one_hot: bool) -> int:
        """Returns the width of the encoded values, i.e., the size in the second dimension (axis 1)"""

    @abstractmethod
    def allowed_change(self, pre_val: Any, post_val: Any, encoded: bool = True) -> bool:
        """Checks whether value change from pre_val to post_val is allowed by mutability and similar properties"""

    def __str__(self) -> str:
        return str(self.__name)
