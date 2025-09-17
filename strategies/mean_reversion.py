import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List
from scipy import stats
from .base_strategy import BaseStrategy

"""
    Basically this strategy uses statistical measures to identify overextended price movements
    and trades on the expectation of price returning to its mean...and the components used are for mean Reversion -
    Z-score based signals,Bollinger Bands,RSI mean reversion
    Support/Resistance levels and statistical significance testing
Logic:
    - Long when price is oversold and likely to revert upward and Short when price is overbought and likely to revert downward..
    - position sizing based on statistical confidence
    """
class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        lookback_window: int = 15,
        zscore_threshold: float = 1.5,
        bollinger_window: int = 15,
        bollinger_std: float = 1.8,
        rsi_window: int = 10,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        min_periods: int = 8,
        confidence_threshold: float = 0.6,
        position_scaling: bool = True,
        exit_zscore: float = 0.4
    ):
        super().__init__(name="MeanReversionStrategy")
        self.parameters = {
            'lookback_window': lookback_window,
            'zscore_threshold': zscore_threshold,
            'bollinger_window': bollinger_window,
            'bollinger_std': bollinger_std,
            'rsi_window': rsi_window,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'min_periods': min_periods,
            'confidence_threshold': confidence_threshold,
            'position_scaling': position_scaling,
            'exit_zscore': exit_zscore
        }
        self.logger.info(f"Initialized {self.name} with parameters: {self.parameters}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.validate_data(data):
            raise ValueError("Invalid data provided to mean reversion strategy")
        self.logger.info(f"Generating mean reversion signals for {len(data)} data points")
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        volume = data['volume']
        zscore_signal = self._calculate_zscore_signal(close_prices)
        bollinger_signal = self._calculate_bollinger_signal(close_prices)
        rsi_signal = self._calculate_rsi_reversion_signal(close_prices)
        support_resistance_signal = self._calculate_support_resistance_signal(
            high_prices, low_prices, close_prices
        )
        statistical_signal = self._calculate_statistical_significance_signal(close_prices)
        combined_signal = (
            zscore_signal * 0.30 +
            bollinger_signal * 0.25 +
            rsi_signal * 0.20 +
            support_resistance_signal * 0.15 +
            statistical_signal * 0.10
        )
        amplification = 2.0
        combined_signal = (combined_signal * amplification).clip(-1.0, 1.0)
        filtered_signal = self.apply_signal_filter(
            combined_signal,
            filter_type='ema',
            window=3
        )
        final_signals = self._apply_confidence_processing(filtered_signal, close_prices)
        self.last_signals = final_signals
        signal_strength = self.calculate_signal_strength(final_signals)
        self.logger.info(f"Generated signals with average strength: {signal_strength:.3f}")
        return final_signals

    def _calculate_zscore_signal(self, prices: pd.Series) -> pd.Series:
        window = self.parameters['lookback_window']
        min_periods = self.parameters['min_periods']
        rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prices.rolling(window=window, min_periods=min_periods).std()
        zscore = (prices - rolling_mean) / (rolling_std + 1e-8)
        threshold = self.parameters['zscore_threshold']
        zscore_signal = -np.tanh(zscore / threshold)
        return pd.Series(zscore_signal, index=prices.index).fillna(0)

    def _calculate_bollinger_signal(self, prices: pd.Series) -> pd.Series:
        bb_dict = self.bollinger_bands(
            prices,
            window=self.parameters['bollinger_window'],
            num_std=self.parameters['bollinger_std']
        )
        upper_band = bb_dict['upper']
        lower_band = bb_dict['lower']
        middle_band = bb_dict['middle']
        band_position = (prices - middle_band) / (upper_band - lower_band + 1e-8)
        bollinger_signal = -np.tanh(band_position * 2)
        bollinger_series = pd.Series(bollinger_signal, index=prices.index)
        near_upper = prices > (upper_band - (upper_band - middle_band) * 0.2)
        near_lower = prices < (lower_band + (middle_band - lower_band) * 0.2)
        bollinger_series.loc[near_upper] = bollinger_series.loc[near_upper] * 1.3
        bollinger_series.loc[near_lower] = bollinger_series.loc[near_lower] * 1.3
        return bollinger_series.fillna(0)

    def _calculate_rsi_reversion_signal(self, prices: pd.Series) -> pd.Series:
        rsi = self.rsi(prices, self.parameters['rsi_window'])
        oversold_threshold = self.parameters['rsi_oversold']
        overbought_threshold = self.parameters['rsi_overbought']
        rsi_signal = np.where(
            rsi < oversold_threshold,
            (oversold_threshold - rsi) / oversold_threshold,
            np.where(
                rsi > overbought_threshold,
                (overbought_threshold - rsi) / (100 - overbought_threshold),
                0
            )
        )
        rsi_signal = np.tanh(rsi_signal * 2)
        return pd.Series(rsi_signal, index=prices.index).fillna(0)

    def _calculate_support_resistance_signal(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series
    ) -> pd.Series:
        window = self.parameters['lookback_window']
        rolling_high = high_prices.rolling(window=window, min_periods=self.parameters['min_periods']).max()
        rolling_low = low_prices.rolling(window=window, min_periods=self.parameters['min_periods']).min()
        price_range = rolling_high - rolling_low
        position_in_range = (close_prices - rolling_low) / (price_range + 1e-8)
        sr_signal = -(position_in_range - 0.5) * 2
        sr_series = pd.Series(sr_signal, index=close_prices.index)
        near_support = position_in_range < 0.2
        near_resistance = position_in_range > 0.8
        sr_series.loc[near_support] = sr_series.loc[near_support] * 1.5
        sr_series.loc[near_resistance] = sr_series.loc[near_resistance] * 1.5
        return sr_series.fillna(0)

    def _calculate_statistical_significance_signal(self, prices: pd.Series) -> pd.Series:
        window = self.parameters['lookback_window']
        min_periods = self.parameters['min_periods']
        rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prices.rolling(window=window, min_periods=min_periods).std()
        deviation = prices - rolling_mean
        t_stat = deviation / (rolling_std / np.sqrt(window) + 1e-8)
        stat_signal = -np.tanh(t_stat / 3)
        return pd.Series(stat_signal, index=prices.index).fillna(0)

    def _apply_confidence_processing(
        self,
        raw_signals: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        if not isinstance(raw_signals, pd.Series):
            raw_signals = pd.Series(raw_signals, index=prices.index)
        confidence = self._calculate_signal_confidence(raw_signals, prices)
        confidence_threshold = self.parameters['confidence_threshold']
        signals_series = raw_signals.copy()
        signals_series = signals_series.where(confidence >= confidence_threshold, 0.0)
        signals_series = self.normalize_signals(signals_series)
        if isinstance(signals_series, np.ndarray):
            signals_series = pd.Series(signals_series, index=raw_signals.index)
        if self.parameters['position_scaling']:
            signals_series = signals_series * confidence
        else:
            threshold = 0.3
            signals_series = signals_series.where(signals_series > threshold, 0.0)
            signals_series = signals_series.where(signals_series < -threshold, signals_series)
            signals_series = signals_series.apply(lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))
        return signals_series.fillna(0)

    def _calculate_signal_confidence(
        self,
        signals: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        window = self.parameters['lookback_window']
        rolling_vol = prices.pct_change(fill_method=None).rolling(window=window).std()
        vol_percentile = rolling_vol.rolling(window=max(1, window*2)).rank(pct=True)
        vol_confidence = vol_percentile.fillna(0.5)
        signal_consistency = 1 - signals.rolling(window=5).std().fillna(1)
        zscore = self._calculate_price_zscore(prices)
        extremity_confidence = np.minimum(zscore.abs() / 3, 1.0)
        volume_confidence = pd.Series(0.5, index=prices.index)
        combined_confidence = (
            vol_confidence * 0.3 +
            signal_consistency * 0.3 +
            extremity_confidence * 0.3 +
            volume_confidence * 0.1
        )
        return combined_confidence.fillna(0.5)

    def _calculate_price_zscore(self, prices: pd.Series) -> pd.Series:
        window = self.parameters['lookback_window']
        min_periods = self.parameters['min_periods']
        rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prices.rolling(window=window, min_periods=min_periods).std()
        zscore = (prices - rolling_mean) / (rolling_std + 1e-8)
        return zscore.fillna(0)

    def get_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        return {
            'zscore': self._calculate_zscore_signal(close_prices),
            'bollinger': self._calculate_bollinger_signal(close_prices),
            'rsi_reversion': self._calculate_rsi_reversion_signal(close_prices),
            'support_resistance': self._calculate_support_resistance_signal(
                high_prices, low_prices, close_prices
            ),
            'statistical': self._calculate_statistical_significance_signal(close_prices)
        }

    def detect_mean_reversion_opportunities(self, data: pd.DataFrame) -> Dict[str, Any]:
        close_prices = data['close']
        signals = self.generate_signals(data)
        zscore = self._calculate_price_zscore(close_prices)
        strong_long_setups = (signals > 0.5) & (zscore < -self.parameters['zscore_threshold'])
        strong_short_setups = (signals < -0.5) & (zscore > self.parameters['zscore_threshold'])
        try:
            current_z = float(zscore.iloc[-1]) if len(zscore.dropna()) > 0 else 0.0
        except Exception:
            current_z = 0.0
        try:
            current_sig = float(signals.iloc[-1]) if len(signals.dropna()) > 0 else 0.0
        except Exception:
            current_sig = 0.0
        analysis = {
            'current_zscore': current_z,
            'current_signal': current_sig,
            'strong_long_opportunities': int(strong_long_setups.sum()),
            'strong_short_opportunities': int(strong_short_setups.sum()),
            'mean_reversion_frequency': {
                'total_reversals': self._count_mean_reversions(close_prices),
                'successful_reversals': self._count_successful_reversions(close_prices, signals),
            },
            'statistical_properties': {
                'price_mean': float(close_prices.mean()),
                'price_std': float(close_prices.std()),
                'stationarity_test': self._test_stationarity(close_prices),
                'half_life': self._calculate_half_life(close_prices)
            }
        }
        return analysis

    def _count_mean_reversions(self, prices: pd.Series) -> int:
        zscore = self._calculate_price_zscore(prices)
        threshold = self.parameters['zscore_threshold']
        extreme_points = (zscore.abs() > threshold).astype(int)
        reversions = (extreme_points.diff() == -1).sum()
        return int(reversions)

    def _count_successful_reversions(self, prices: pd.Series, signals: pd.Series) -> int:
        future_returns = prices.pct_change(fill_method=None).shift(-5)
        future_returns = future_returns.replace([np.inf, -np.inf], np.nan)
        successful = ((signals > 0) & (future_returns > 0)) | ((signals < 0) & (future_returns < 0))
        return int(successful.sum())

    def _test_stationarity(self, prices: pd.Series) -> Dict[str, float]:
        try:
            returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
            mean_return = float(returns.mean()) if not returns.empty else 0.0
            std_return = float(returns.std()) if not returns.empty else 0.0
            stationarity_score = 1 - abs(mean_return) / std_return if std_return > 0 else 0.0
            return {
                'stationarity_score': float(stationarity_score),
                'returns_mean': float(mean_return),
                'returns_std': float(std_return)
            }
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {e}")
            return {'stationarity_score': 0.5}

    def _calculate_half_life(self, prices: pd.Series) -> float:
        try:
            price_lag = prices.shift(1)
            delta_price = prices.diff()
            valid_data = ~(price_lag.isna() | delta_price.isna())
            if valid_data.sum() < 10:
                return float('inf')
            x = price_lag[valid_data].replace([np.inf, -np.inf], np.nan).dropna()
            y = delta_price[valid_data].replace([np.inf, -np.inf], np.nan).dropna()
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            if len(x) < 10 or len(y) < 10:
                return float('inf')
            x_mean = x.mean()
            y_mean = y.mean()
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            if denominator == 0:
                return float('inf')
            beta = numerator / denominator
            if beta >= 0:
                return float('inf')
            try:
                half_life = -np.log(2) / np.log1p(beta)
            except Exception:
                return float('inf')
            return float(max(1, half_life))
        except Exception as e:
            self.logger.warning(f"Half-life calculation failed: {e}")
            return float('inf')

    def get_exit_signals(self, data: pd.DataFrame, current_positions: pd.Series) -> pd.Series:
        close_prices = data['close']
        zscore = self._calculate_price_zscore(close_prices)
        exit_threshold = self.parameters['exit_zscore']
        exit_signals = pd.Series(0, index=current_positions.index)
        long_positions = current_positions > 0
        exit_long = long_positions & (zscore > -exit_threshold)
        exit_signals[exit_long] = -1
        short_positions = current_positions < 0
        exit_short = short_positions & (zscore < exit_threshold)
        exit_signals[exit_short] = 1
        return exit_signals

    def get_trading_rules(self) -> List[str]:
        return [
            f"Enter long when price is {self.parameters['zscore_threshold']:.1f}+ standard deviations below mean",
            f"Enter short when price is {self.parameters['zscore_threshold']:.1f}+ standard deviations above mean",
            f"Use {self.parameters['lookback_window']}-period rolling window for mean calculation",
            f"Exit positions when Z-score returns to ±{self.parameters['exit_zscore']:.1f}",
            f"RSI thresholds: oversold < {self.parameters['rsi_oversold']}, overbought > {self.parameters['rsi_overbought']}",
            f"Bollinger bands: {self.parameters['bollinger_window']}-period, {self.parameters['bollinger_std']}σ",
            f"Minimum confidence threshold: {self.parameters['confidence_threshold']:.2f}",
            "Combine multiple mean reversion indicators for robust signals",
            "Scale position size by statistical confidence" if self.parameters['position_scaling'] else "Use binary position signals"
        ]