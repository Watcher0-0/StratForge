import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging


class BaseStrategy(ABC):    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.parameters = {}  
        self.last_signals = None  
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        
        self.parameters.update(kwargs)
        self.logger.info(f"Updated parameters: {kwargs}")
    
    def get_description(self) -> str:
        
        return f"{self.name} - {self.__doc__ or 'No description available'}"
    
    
    def sma(self, data: pd.Series, window: int) -> pd.Series:
        
        return data.rolling(window=window, min_periods=1).mean()
    
    def ema(self, data: pd.Series, window: int) -> pd.Series:
        
        return data.ewm(span=window, adjust=False).mean()
    
    def bollinger_bands(
        self, 
        data: pd.Series, 
        window: int = 20, 
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        middle = self.sma(data, window)
        std = data.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self, 
        data: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        macd = ema_fast - ema_slow
        signal_line = self.ema(macd, signal)
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        k_window: int = 14, 
        d_window: int = 3
    ) -> Dict[str, pd.Series]:
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    def atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        window: int = 14
    ) -> pd.Series:
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    def williams_r(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        window: int = 14
    ) -> pd.Series:
        
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return wr
    
    def commodity_channel_index(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        window: int = 20
    ) -> pd.Series:
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window, min_periods=1).mean()
        mad = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
    
    
    def normalize_signals(self, signals: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        if isinstance(signals, pd.Series):
            return np.clip(signals, -1.0, 1.0)
        else:
            return np.clip(signals, -1.0, 1.0)
    
    def apply_signal_filter(
        self, 
        signals: Union[pd.Series, np.ndarray], 
        filter_type: str = 'none',
        **filter_params
    ) -> Union[pd.Series, np.ndarray]:
        if filter_type == 'none':
            return signals
        
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals)
        
        window = filter_params.get('window', 5)
        
        if filter_type == 'sma':
            return signals.rolling(window=window, min_periods=1).mean()
        elif filter_type == 'ema':
            return signals.ewm(span=window, adjust=False).mean()
        elif filter_type == 'median':
            return signals.rolling(window=window, min_periods=1).median()
        else:
            self.logger.warning(f"Unknown filter type: {filter_type}")
            return signals
    
    def calculate_signal_strength(self, signals: Union[pd.Series, np.ndarray]) -> float:
        if isinstance(signals, pd.Series):
            abs_signals = signals.abs()
        else:
            abs_signals = np.abs(signals)
        
        
        return float(abs_signals.mean())
    
    def backtest_strategy(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        signals = self.generate_signals(data)
        
        if isinstance(signals, pd.Series):
            signals = signals.values
        
        
        prices = data['close'].values
        capital = initial_capital
        position = 0
        
        returns = []
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1]
            
            
            period_return = position * price_change
            
            if i < len(signals) and abs(signals[i] - position) > 1e-8:
                period_return -= transaction_cost
                position = signals[i]
            
            returns.append(period_return)
            capital *= (1 + period_return)
        
        returns = np.array(returns)
        
        
        total_return = (capital - initial_capital) / initial_capital
        volatility = returns.std() * np.sqrt(252)  
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        max_dd = self._calculate_max_drawdown(returns)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_capital': capital
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        
        if len(data) < 10:  
            self.logger.error(f"Insufficient data: {len(data)} rows")
            return False
        
        
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains missing values")
        
        
        price_columns = ['open', 'high', 'low', 'close']
        if (data[price_columns] <= 0).any().any():
            self.logger.warning("Data contains non-positive prices")
        
        
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"OHLC inconsistencies found in {invalid_ohlc.sum()} rows")
        
        return True
    
    def __repr__(self) -> str:
        
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"
    
    def __str__(self) -> str:
        
        return self.__repr__()