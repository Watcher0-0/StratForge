import numpy as np
import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class VolatilityScalingTrend(BaseStrategy):
    def __init__(self,
                 lookback: int = 20,
                 vol_window: int = 10,
                 target_vol: float = 0.20,
                 min_size: float = 0.2,
                 max_size: float = 2.0,
                 stop_vol_mult: float = 2.0,
                 name: Optional[str] = None):
        super().__init__(name)
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.min_size = min_size
        self.max_size = max_size
        self.stop_vol_mult = stop_vol_mult
        self.parameters = {
            'lookback': lookback,
            'vol_window': vol_window,
            'target_vol': target_vol,
            'min_size': min_size,
            'max_size': max_size,
            'stop_vol_mult': stop_vol_mult
        }

    def _realized_vol(self, series: pd.Series) -> pd.Series:
        ret = series.pct_change(fill_method='ffill')
        roll_std = ret.rolling(self.vol_window, min_periods=3).std()
        ann_vol = roll_std * np.sqrt(252.0)
        return ann_vol.clip(lower=0.05)

    def _momentum_signal(self, series: pd.Series) -> pd.Series:
        roc_short = series.pct_change(5)
        roc_med = series.pct_change(self.lookback)
        sma = series.rolling(window=self.lookback, min_periods=1).mean()
        sig = pd.Series(0, index=series.index, dtype=float)
        sig[(roc_short > 0) & (roc_med > 0) & (series > sma)] = 1.0
        sig[(roc_short < 0) & (roc_med < 0) & (series < sma)] = -1.0
        if len(sig) > 0:
            sig.iloc[0] = 1.0 if series.iloc[0] > series.iloc[0] * 0.9995 else -1.0
        return sig.fillna(method='ffill').fillna(0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        close = data['close']
        vol = self._realized_vol(close)
        signal = self._momentum_signal(close)
        position_size = self.target_vol / vol
        position_size = position_size.clip(lower=self.min_size, upper=self.max_size)
        final_signal = (signal * position_size).astype(float)
        min_threshold = 0.1
        final_signal = final_signal.where(final_signal.abs() >= min_threshold, 0.0)
        final_signal = final_signal.fillna(0.0).clip(-self.max_size, self.max_size)
        return final_signal
