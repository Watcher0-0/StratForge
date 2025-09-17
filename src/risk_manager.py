import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import warnings


class RiskManager:    
    def __init__(
        self,
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        lookback_periods: List[int] = [30, 60, 120, 252]
    ):
        self.confidence_levels = confidence_levels
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(__name__)
    
    def analyze_portfolio_risk(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        self.logger.info("Performing comprehensive risk analysis...")
        
        #I am explicitly setting the fill_method=None to avoid pandas FutureWarning about default filling...
        returns = equity_curve.pct_change(fill_method=None).dropna()
        
        risk_analysis = {}
        
        
        risk_analysis.update(self._calculate_basic_risk_metrics(returns))
        
        
        risk_analysis.update(self._calculate_var_metrics(returns))
        
        
        risk_analysis.update(self._calculate_drawdown_risk(equity_curve))
        
        
        risk_analysis.update(self._calculate_tail_risk(returns))
        
        
        risk_analysis.update(self._calculate_rolling_risk_metrics(returns))
        
        
        if not trades.empty:
            risk_analysis.update(self._calculate_concentration_risk(trades))
        
        
        if benchmark_returns is not None:
            risk_analysis.update(self._calculate_relative_risk(returns, benchmark_returns))
        
        
        risk_analysis.update(self._calculate_risk_adjusted_performance(returns))
        
        
        risk_analysis.update(self._perform_stress_tests(returns))
        
        self.logger.info(f" Completed risk analysis with {len(risk_analysis)} metrics")
        return risk_analysis
    
    def _calculate_basic_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        
        
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        
        non_na = returns.dropna()
        skewness = stats.skew(non_na) if len(non_na) > 1 else 0.0
        kurtosis = stats.kurtosis(non_na, fisher=True) if len(non_na) > 1 else 0.0
        
        
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        
        
        mean_return = returns.mean()
        semi_deviation = returns[returns < mean_return].std() * np.sqrt(252)
        
        
        return_range = returns.max() - returns.min()
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'downside_deviation': downside_deviation,
            'semi_deviation': semi_deviation,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'return_range': return_range,
            'coefficient_of_variation': annualized_vol / abs(returns.mean() * 252) if returns.mean() != 0 else np.inf
        }
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        
        
        var_metrics = {}
        
        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            
            
            var_historical = returns.quantile(alpha)
            
            
            mean_return = returns.mean()
            std_return = returns.std()
            var_parametric = mean_return + stats.norm.ppf(alpha) * std_return
            
            
            es_historical = returns[returns <= var_historical].mean()
            
            
            non_na = returns.dropna()
            skew = stats.skew(non_na) if len(non_na) > 1 else 0.0
            kurt = stats.kurtosis(non_na, fisher=True) if len(non_na) > 1 else 0.0
            z_score = stats.norm.ppf(alpha)
            
            
            z_cf = z_score + (z_score**2 - 1) * skew / 6 + \
                   (z_score**3 - 3*z_score) * kurt / 24 - \
                   (2*z_score**3 - 5*z_score) * skew**2 / 36
            
            var_modified = mean_return + z_cf * std_return
            
            conf_pct = int(confidence * 100)
            var_metrics.update({
                f'var_{conf_pct}_historical': var_historical,
                f'var_{conf_pct}_parametric': var_parametric,
                f'var_{conf_pct}_modified': var_modified,
                f'es_{conf_pct}_historical': es_historical
            })
        
        return var_metrics
    
    def _calculate_drawdown_risk(self, equity_curve: pd.Series) -> Dict[str, Any]:
        
        
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        
        drawdown_periods = self._get_drawdown_periods(drawdown)
        
        if drawdown_periods:
            durations = [p['duration'] for p in drawdown_periods]
            depths = [p['depth'] for p in drawdown_periods]
            
            avg_duration = np.mean(durations)
            max_duration = max(durations)
            avg_depth = np.mean(depths)
            
            
            recoveries = [p.get('recovery_time', 0) for p in drawdown_periods if 'recovery_time' in p and p['recovery_time'] is not None]
            avg_recovery = np.mean(recoveries) if recoveries else 0
        else:
            avg_duration = max_duration = avg_depth = avg_recovery = 0
        
        
        ulcer_index = np.sqrt(np.mean(drawdown**2))
        
        
        pain_index = abs(drawdown.mean())
        
        
        burke_ratio = self._calculate_burke_ratio(equity_curve.pct_change().dropna(), drawdown) if len(drawdown) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': avg_duration,
            'max_drawdown_duration': max_duration,
            'avg_drawdown_depth': avg_depth,
            'avg_recovery_time': avg_recovery,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'burke_ratio': burke_ratio
        }
    
    def _calculate_tail_risk(self, returns: pd.Series) -> Dict[str, Any]:
        
        
        
        extreme_threshold = 2  
        std = returns.std()
        
        extreme_positive = returns[returns > extreme_threshold * std]
        extreme_negative = returns[returns < -extreme_threshold * std]
        
        
        tail_ratio = len(extreme_positive) / len(extreme_negative) if len(extreme_negative) > 0 else np.inf
        
        
        expected_tail_loss = extreme_negative.mean() if len(extreme_negative) > 0 else 0
        
        
        loss_streaks = self._calculate_loss_streaks(returns)
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
        avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
        
        
        hit_rate = (returns > 0).mean()
        
        return {
            'extreme_positive_days': len(extreme_positive),
            'extreme_negative_days': len(extreme_negative),
            'tail_ratio': tail_ratio,
            'expected_tail_loss': expected_tail_loss,
            'max_loss_streak': max_loss_streak,
            'avg_loss_streak': avg_loss_streak,
            'hit_rate': hit_rate
        }
    
    def _calculate_rolling_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        
        
        rolling_metrics = {}
        
        for period in self.lookback_periods:
            if len(returns) <= period:
                continue
            
            
            rolling_vol = returns.rolling(period).std() * np.sqrt(252)
            
            
            rolling_var = returns.rolling(period).quantile(0.05)
            
            
            rolling_sharpe = (returns.rolling(period).mean() / returns.rolling(period).std()) * np.sqrt(252)
            
            
            rolling_max_dd = self._calculate_rolling_max_drawdown(returns, period)
            
            period_key = f'{period}d'
            rolling_metrics.update({
                f'avg_rolling_vol_{period_key}': rolling_vol.mean(),
                f'max_rolling_vol_{period_key}': rolling_vol.max(),
                f'avg_rolling_var_{period_key}': rolling_var.mean(),
                f'worst_rolling_var_{period_key}': rolling_var.min(),
                f'avg_rolling_sharpe_{period_key}': rolling_sharpe.mean(),
                f'worst_rolling_sharpe_{period_key}': rolling_sharpe.min(),
                f'avg_rolling_max_dd_{period_key}': rolling_max_dd.mean(),
                f'worst_rolling_max_dd_{period_key}': rolling_max_dd.min()
            })
        
        return rolling_metrics
    
    def _calculate_concentration_risk(self, trades: pd.DataFrame) -> Dict[str, Any]:
        
        
        if trades.empty:
            return {}
        
        
        trade_sizes = abs(trades['quantity'] * trades['price'])
        
        
        trade_concentration = {
            'largest_trade_pct': trade_sizes.max() / trade_sizes.sum() if trade_sizes.sum() > 0 else 0,
            'top_5_trades_pct': trade_sizes.nlargest(5).sum() / trade_sizes.sum() if len(trade_sizes) >= 5 else 1,
            'trade_size_std': trade_sizes.std(),
            'trade_size_skew': stats.skew(trade_sizes) if len(trade_sizes) > 2 else 0
        }
        
        
        if 'symbol' in trades.columns:
            symbol_exposure = trades.groupby('symbol')['quantity'].sum().abs()
            total_exposure = symbol_exposure.sum()
            
            if total_exposure > 0:
                asset_weights = symbol_exposure / total_exposure
                
                
                hhi = (asset_weights ** 2).sum()
                
                
                effective_assets = 1 / hhi if hhi > 0 else 0
                
                trade_concentration.update({
                    'asset_concentration_hhi': hhi,
                    'effective_number_assets': effective_assets,
                    'largest_asset_weight': asset_weights.max()
                })
        
        return trade_concentration
    
    def _calculate_relative_risk(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        
        
        
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        
        correlation = aligned_returns.corr(aligned_benchmark)
        r_squared = correlation ** 2 if not pd.isna(correlation) else 0
        
        
        relative_var_95 = excess_returns.quantile(0.05)
        
        
        up_market = aligned_benchmark > 0
        down_market = aligned_benchmark < 0
        
        up_capture = (aligned_returns[up_market].mean() / aligned_benchmark[up_market].mean()) if up_market.any() and aligned_benchmark[up_market].mean() != 0 else 0
        down_capture = (aligned_returns[down_market].mean() / aligned_benchmark[down_market].mean()) if down_market.any() and aligned_benchmark[down_market].mean() != 0 else 0
        
        return {
            'tracking_error': tracking_error,
            'beta': beta,
            'r_squared': r_squared,
            'relative_var_95': relative_var_95,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation_with_benchmark': correlation
        }
    
    def _calculate_risk_adjusted_performance(self, returns: pd.Series) -> Dict[str, float]:
        
        
        mean_return = returns.mean() * 252  
        volatility = returns.std() * np.sqrt(252)
        
        
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
        
        
        equity_curve = (1 + returns).cumprod()
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        calmar_ratio = mean_return / abs(max_dd) if max_dd != 0 else 0
        
        
        avg_dd = self._calculate_avg_drawdown_from_returns(returns)
        sterling_ratio = mean_return / abs(avg_dd) if avg_dd != 0 else 0
        
        
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        
        return {
            'risk_adjusted_sharpe': sharpe_ratio,
            'risk_adjusted_sortino': sortino_ratio,
            'risk_adjusted_calmar': calmar_ratio,
            'risk_adjusted_sterling': sterling_ratio,
            'risk_adjusted_omega': omega_ratio
        }
    
    def _perform_stress_tests(self, returns: pd.Series) -> Dict[str, Any]:
        
        
        stress_results = {}
        
        
        scenarios = [
            ('market_crash', -0.20, 0.50),  
            ('volatility_spike', 0.0, 0.80),   
            ('bear_market', -0.10, 0.30),      
            ('flash_crash', -0.10, 1.0),       
        ]
        
        current_vol = returns.std() * np.sqrt(252)
        current_return = returns.mean() * 252
        
        for scenario_name, stress_return, stress_vol in scenarios:
            
            vol_multiplier = stress_vol / current_vol if current_vol > 0 else 1
            return_shift = stress_return - current_return
            
            
            stressed_returns = returns * vol_multiplier + return_shift / 252
            
            
            stressed_sharpe = (stressed_returns.mean() * 252) / (stressed_returns.std() * np.sqrt(252)) if stressed_returns.std() > 0 else 0
            stressed_max_dd = self._calculate_max_drawdown_from_returns(stressed_returns)
            
            stress_results[f'stress_{scenario_name}_sharpe'] = stressed_sharpe
            stress_results[f'stress_{scenario_name}_max_dd'] = stressed_max_dd
        
        return stress_results
    
    def _get_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        
        periods = []
        in_drawdown = False
        start_date = None
        min_value = 0
        
        for i, (date, value) in enumerate(drawdown.items()):
            if value < 0 and not in_drawdown:
                
                in_drawdown = True
                start_date = date
                start_index = i
                min_value = value
            elif value < 0 and in_drawdown:
                
                min_value = min(min_value, value)
            elif value >= 0 and in_drawdown:
                
                in_drawdown = False
                duration = (date - start_date).days
                
                
                recovery_date = None
                for future_date, future_value in drawdown[date:].items():
                    if future_value >= 0:
                        recovery_date = future_date
                        break
                
                recovery_time = (recovery_date - date).days if recovery_date else None
                
                periods.append({
                    'start': start_date,
                    'end': date,
                    'duration': duration,
                    'depth': min_value,
                    'recovery_time': recovery_time
                })
        
        
        if in_drawdown:
            duration = (drawdown.index[-1] - start_date).days
            periods.append({
                'start': start_date,
                'end': drawdown.index[-1],
                'duration': duration,
                'depth': min_value,
                'recovery_time': None  
            })
        
        return periods
    
    def _calculate_loss_streaks(self, returns: pd.Series) -> List[int]:
        
        streaks = []
        current_streak = 0
        
        for ret in returns:
            if ret < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    def _calculate_rolling_max_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        
        rolling_max_dd = pd.Series(index=returns.index, dtype=float)
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            equity_segment = (1 + window_returns).cumprod()
            running_max = equity_segment.expanding().max()
            drawdown = (equity_segment - running_max) / running_max
            rolling_max_dd.iloc[i] = drawdown.min()
        
        return rolling_max_dd.dropna()
    
    def _calculate_burke_ratio(self, returns: pd.Series, drawdown: pd.Series) -> float:
        
        mean_return = returns.mean() * 252  
        squared_drawdowns = drawdown[drawdown < 0] ** 2
        sqrt_mean_squared_dd = np.sqrt(squared_drawdowns.mean()) if len(squared_drawdowns) > 0 else 0
        
        return mean_return / sqrt_mean_squared_dd if sqrt_mean_squared_dd > 0 else 0
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def _calculate_avg_drawdown_from_returns(self, returns: pd.Series) -> float:
        
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    
    def generate_risk_report(
        self,
        risk_metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        
        
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE RISK ANALYSIS REPORT",
            "=" * 80,
            "",
            "BASIC RISK METRICS:",
            f"  Annualized Volatility:    {risk_metrics.get('annualized_volatility', 0):.2%}",
            f"  Downside Deviation:       {risk_metrics.get('downside_deviation', 0):.2%}",
            f"  Skewness:                 {risk_metrics.get('skewness', 0):.3f}",
            f"  Excess Kurtosis:          {risk_metrics.get('excess_kurtosis', 0):.3f}",
            "",
            "VALUE AT RISK METRICS:",
            f"  VaR (95% Historical):     {risk_metrics.get('var_95_historical', 0):.2%}",
            f"  VaR (99% Historical):     {risk_metrics.get('var_99_historical', 0):.2%}",
            f"  Expected Shortfall (95%): {risk_metrics.get('es_95_historical', 0):.2%}",
            "",
            "DRAWDOWN ANALYSIS:",
            f"  Maximum Drawdown:         {risk_metrics.get('max_drawdown', 0):.2%}",
            f"  Average Drawdown:         {risk_metrics.get('avg_drawdown', 0):.2%}",
            f"  Ulcer Index:              {risk_metrics.get('ulcer_index', 0):.3f}",
            f"  Max Drawdown Duration:    {risk_metrics.get('max_drawdown_duration', 0):.0f} days",
            "",
            "TAIL RISK ANALYSIS:",
            f"  Max Loss Streak:          {risk_metrics.get('max_loss_streak', 0):.0f} days",
            f"  Hit Rate:                 {risk_metrics.get('hit_rate', 0):.2%}",
            f"  Expected Tail Loss:       {risk_metrics.get('expected_tail_loss', 0):.2%}",
            "",
            "RISK-ADJUSTED PERFORMANCE:",
            f"  Risk-Adjusted Sharpe:     {risk_metrics.get('risk_adjusted_sharpe', 0):.3f}",
            f"  Risk-Adjusted Sortino:    {risk_metrics.get('risk_adjusted_sortino', 0):.3f}",
            f"  Risk-Adjusted Calmar:     {risk_metrics.get('risk_adjusted_calmar', 0):.3f}",
            "=" * 80
        ]
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report