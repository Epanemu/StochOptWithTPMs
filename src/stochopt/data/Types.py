from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

OneDimData = Union[npt.NDArray[Any], pd.Series]
CategValue = Union[int, str]
DataLike = Union[npt.NDArray[Any], pd.DataFrame]
FeatureID = Union[int, str]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]
AnyArray = npt.NDArray[Any]
