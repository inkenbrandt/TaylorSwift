from dataclasses import dataclass
import pandas as pd

@dataclass
class FluxResult:
    Ta: float
    Td: float
    D: float
    Ustr: float
    zeta: float
    H: float
    StDevUz: float
    StDevTa: float
    direction: float
    exchange: float
    lambdaE: float
    ET: float
    Uxy: float

    def to_series(self) -> pd.Series:
        return pd.Series(self.__dict__)
