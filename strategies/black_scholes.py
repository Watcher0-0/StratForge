"""
    -It is basically a momentum strategy with volatility adjusted position sizing.
    -uses short/long lookback periods for trend confirmation
    -adjusts position size based on volatility
    -includes price momentum and volume confirmation
    """
import pandas as pd
import numpy as np
from typing import List, Optional
from .base_strategy import BaseStrategy



class BlackScholesStrategy(BaseStrategy):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "EnhancedMomentum")
        self.parameters = {
            'short_ma': 5,
            'long_ma': 20,
            'vol_window': 10,
            'min_change': 0.02
        }
        self.logger.info(f"Initialized {self.name} with {self.parameters}")

    def validate_stock_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate strong daily momentum signals with proper bounds."""
        if not self.validate_stock_data(data):
            raise ValueError("Invalid stock data provided")

        close = data['close'].astype(float)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        volume = data['volume'].astype(float)

        daily_returns = close.pct_change()

        mom1 = close.pct_change(1)
        mom5 = close.pct_change(5)
        mom10 = close.pct_change(10)

        sma5 = close.rolling(window=5, min_periods=1).mean()
        sma20 = close.rolling(window=20, min_periods=1).mean()
        trend = (close > sma5) & (sma5 > sma20)

        vol_sma = volume.rolling(window=10, min_periods=1).mean()
        strong_volume = volume > vol_sma * 1.2

        signals = pd.Series(0.0, index=close.index)

        long_conditions = (
            (mom1 > 0.001) &
            (mom5 > 0) &
            (mom10 > 0) &
            trend &
            strong_volume
        )
        signals[long_conditions] = 1.0

        short_conditions = (
            (mom1 < -0.001) &
            (mom5 < 0) &
            (mom10 < 0) &
            ~trend &
            strong_volume
        )
        signals[short_conditions] = -1.0

        amplification = ((abs(mom1) * 10).clip(0.5, 2.0))
        signals = signals * amplification

        if len(signals) > 0:
            signals.iloc[0] = np.sign(daily_returns.iloc[0]) if pd.notna(daily_returns.iloc[0]) else 0.0

        signals = signals.fillna(0.0)
        signals = signals.replace([np.inf, -np.inf], 0.0)
        signals = signals.clip(-1.0, 1.0)

        signals = signals.shift(1).fillna(0.0)

        return signals

    def get_trading_rules(self) -> List[str]:
        return [
            "Buy if today's close > yesterday's close",
            "Sell if today's close < yesterday's close",
            "Flat if unchanged"
        ]