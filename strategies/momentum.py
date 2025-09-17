import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    - uses short/long SMA crossover with multiple confirmation signals
    - combines trend and momentum for stronger signals...
    - outputs signals in -1,0 and 1 with enough magnitude to trigger trades
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        threshold: float = 0.005,
        filter_window: int = 2,
        name: Optional[str] = None
    ):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
        self.filter_window = filter_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window,
            'threshold': threshold,
            'filter_window': filter_window
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        close = data['close'].astype(float)

        short_sma = self.sma(close, self.short_window)
        long_sma = self.sma(close, self.long_window)
        sma_diff = (short_sma - long_sma) / long_sma.replace(0, np.nan)
        sma_diff = sma_diff.fillna(0.0)

        momentum = close.pct_change(self.filter_window)

        above_sma = (close > short_sma)

        signal = pd.Series(0.0, index=close.index, dtype=float)

        buy_mask = (sma_diff > self.threshold) & (momentum > 0) & (above_sma)
        sell_mask = (sma_diff < -self.threshold) & (momentum < 0) & (~above_sma)
        signal[buy_mask.fillna(False)] = 1.0
        signal[sell_mask.fillna(False)] = -1.0

        if len(signal) > 0 and 'open' in data.columns:
            signal.iloc[0] = 1.0 if close.iloc[0] > data['open'].iloc[0] else -1.0

        signal = signal.fillna(0.0).astype(float)
        signal = signal.clip(-1.0, 1.0)
        return signal