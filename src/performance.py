import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from scipy import stats
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float = 1000000.0,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        self.logger.info("Calculating comprehensive performance metrics...")
        
        if equity_curve is None or equity_curve.empty:
            self.logger.warning("Empty or invalid equity curve passed to PerformanceAnalyzer")
            return self._calculate_return_metrics(pd.Series(dtype=float), pd.Series(dtype=float), initial_capital)
        try:
            equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan)
            equity_curve = equity_curve.dropna()
            equity_curve = equity_curve.clip(lower=1e-6)
            pct_changes = equity_curve.pct_change().fillna(0)
            extreme_mask = (pct_changes > 5.0) | (pct_changes < -0.9)
            if extreme_mask.any():
                self.logger.warning("Detected extreme equity curve jumps; sanitizing by clipping and forward-filling")
                clipped_changes = pct_changes.clip(lower=-0.9, upper=5.0)
                equity_curve = (1 + clipped_changes).cumprod() * float(equity_curve.iloc[0])
                equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).ffill().fillna(float(equity_curve.iloc[0]))
        except Exception as e:
            self.logger.warning(f"Error cleaning equity curve: {str(e)}")
            equity_curve = pd.Series(dtype=float)
            
        if equity_curve.empty:
            return self._calculate_return_metrics(pd.Series(dtype=float), pd.Series(dtype=float), initial_capital)
        
        returns = pd.Series(dtype=float)
        try:
            returns = equity_curve.pct_change(fill_method=None)
            returns = returns.replace([np.inf, -np.inf], np.nan)
            returns = returns.dropna()
            returns = returns.clip(-0.5, 2.0)
        except Exception as e:
            self.logger.warning(f"Error calculating returns: {str(e)}")
            returns = pd.Series(dtype=float)
        
        # metrics calculations
        metrics = self._calculate_return_metrics(equity_curve, returns, initial_capital)
        metrics.update(self._calculate_risk_metrics(equity_curve, returns))
        metrics.update(self._calculate_drawdown_metrics(equity_curve))
        if trades is not None and not trades.empty:
            metrics.update(self._calculate_trade_metrics(trades, equity_curve))
        metrics.update(self._calculate_advanced_metrics(returns))
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics.update(self._calculate_relative_metrics(returns, benchmark_returns))
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        total_return = metrics.get('total_return', 0.0)
        ann_return = metrics.get('annualized_return', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        profit_factor = metrics.get('profit_factor', 0.0)
        
        #Checking here  for impossible combinations
        if total_return < -1.0:  
            metrics['total_return'] = -1.0
        
        if abs(ann_return) > 10.0:  
            metrics['annualized_return'] = np.sign(ann_return) * 10.0
            
        if max_dd < -1.0: 
            metrics['max_drawdown'] = -1.0
            
        if max_dd == -1.0 and sharpe > 0:  
            metrics['sharpe_ratio'] = -abs(sharpe)
            
        if win_rate == 0 and profit_factor > 0:  
            metrics['profit_factor'] = 0.0
        for k, v in metrics.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                try:
                    v = float(v)
                    if not np.isfinite(v) or pd.isna(v):
                        metrics[k] = 0.0
                    elif k != 'total_return' and abs(v) > 1e6: 
                        metrics[k] = np.sign(v) * 1e6
                except:
                    metrics[k] = 0.0
        
        self.logger.info(f" Calculated {len(metrics)} performance metrics")
        return metrics
    
    def _calculate_return_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        
        if equity_curve is None or equity_curve.empty:
            return {
                'final_value': initial_capital,
                'total_return': 0.0,
                'cagr': 0.0,
                'annualized_return': 0.0,
                'best_day': 0.0,
                'worst_day': 0.0,
                'positive_days': 0,
                'negative_days': 0,
                'avg_rolling_1m': 0.0,
                'avg_rolling_3m': 0.0,
                'avg_rolling_6m': 0.0,
                'avg_rolling_1y': 0.0,
            }

        # Use actual equity curve values with initial_capital as fallback
        start_value = float(equity_curve.iloc[0])  
        final_value = float(equity_curve.iloc[-1]) 

        # Validate and clean the values
        if not np.isfinite(start_value) or start_value <= 0:
            self.logger.warning(f"Invalid start value {start_value}, using initial capital")
            start_value = float(initial_capital)

        if not np.isfinite(final_value) or final_value <= 0:
            self.logger.warning(f"Invalid final value {final_value}, using last valid value")
            valid_values = equity_curve[np.isfinite(equity_curve) & (equity_curve > 0)]
            if not valid_values.empty:
                final_value = float(valid_values.iloc[-1])
            else:
                final_value = start_value

        # Debug log the values for verification
        self.logger.info(f"Starting portfolio value: ${start_value:,.2f}")
        self.logger.info(f"Final portfolio value: ${final_value:,.2f}")
        self.logger.info(f"Initial capital reference: ${float(initial_capital):,.2f}")
        
        # Calculate total return as percentage gain from actual start value
        total_return = (final_value - start_value) / start_value if start_value > 0 else 0.0
        
        if not np.isfinite(total_return):
            self.logger.warning(f"Invalid total return detected ({total_return})")
           
            if np.isfinite(final_value) and np.isfinite(start_value) and start_value > 0:
                total_return = (final_value - start_value) / start_value
            else:
                total_return = 0.0
        
        self.logger.info(f"Final portfolio value: ${final_value:,.2f}")
        self.logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")

        # Time period in years
        try:
            if isinstance(equity_curve.index[0], pd.Timestamp) and isinstance(equity_curve.index[-1], pd.Timestamp):
                days = (equity_curve.index[-1] - equity_curve.index[0]).days
                num_years = days / 365.25 if days > 0 else max(len(equity_curve) / 252.0, 1/252.0)
            else:
                num_years = max(len(equity_curve) / 252.0, 1/252.0)
        except Exception:
            num_years = max(len(equity_curve) / 252.0, 1/252.0)

        # Geometric CAGR using helper
        try:
            cagr = self._calculate_cagr(equity_curve)
        except Exception:
            cagr = 0.0
        cagr = float(np.clip(cagr, -1.0, 10.0))

        # Annualized return, align with CAGR(geometric)
        ann_return = cagr

        # Clean returns for rolling stats
        if returns is not None and not returns.empty:
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = returns.clip(-0.25, 0.25)
        else:
            returns = pd.Series(dtype=float)

        # Rolling returns
        rolling_1m = self._calculate_rolling_returns(returns, 21)
        rolling_3m = self._calculate_rolling_returns(returns, 63)
        rolling_6m = self._calculate_rolling_returns(returns, 126)
        rolling_1y = self._calculate_rolling_returns(returns, 252)

        return {
            'final_value': final_value,
            'total_return': float(total_return),
            'cagr': float(cagr),
            'annualized_return': float(ann_return),
            'best_day': float(returns.max()) if not returns.empty else 0.0,
            'worst_day': float(returns.min()) if not returns.empty else 0.0,
            'positive_days': int((returns > 0).sum()) if not returns.empty else 0,
            'negative_days': int((returns < 0).sum()) if not returns.empty else 0,
            'avg_rolling_1m': float(rolling_1m.mean()) if not rolling_1m.empty else 0.0,
            'avg_rolling_3m': float(rolling_3m.mean()) if not rolling_3m.empty else 0.0,
            'avg_rolling_6m': float(rolling_6m.mean()) if not rolling_6m.empty else 0.0,
            'avg_rolling_1y': float(rolling_1y.mean()) if not rolling_1y.empty else 0.0,
        }

    def _calculate_risk_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        if returns is None:
            returns = pd.Series(dtype=float)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty and equity_curve is not None and not equity_curve.empty:
            try:
                equity_curve = equity_curve.clip(lower=0.0)
                temp_ret = equity_curve.pct_change(fill_method=None)
                temp_ret = temp_ret.replace([np.inf, -np.inf], np.nan).dropna()
                temp_ret = temp_ret.clip(-1.0, 10.0)
                returns = temp_ret if not temp_ret.empty else returns
            except Exception:
                returns = pd.Series(dtype=float)

        if returns.empty:
            return {
                'volatility': 0.0,
                'daily_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
            }

        # Daily volatility(sample std)
        daily_vol = float(returns.std())
        daily_vol = min(1.0, daily_vol) if daily_vol and not pd.isna(daily_vol) else 0.0
        annualized_vol = float(daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0

        #Standard Sharpe i.e(mean_daily - rf/252) / std_daily * sqrt(252)
        if not returns.empty and returns.std() > 0:
            mean_daily = float(returns.mean())
            std_daily = float(returns.std())
            excess_daily = mean_daily - (self.risk_free_rate / 252.0)
            sharpe = (excess_daily / std_daily) * np.sqrt(252)
            sharpe = float(np.clip(sharpe, -5.0, 5.0))
        else:
            sharpe = 0.0

        #Sortino ratio using downside deviation (annualized)
        if not returns.empty:
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std_daily = float(negative_returns.std())
                if downside_std_daily > 0:
                    excess_daily = float(returns.mean()) - (self.risk_free_rate / 252.0)
                    sortino = (excess_daily / downside_std_daily) * np.sqrt(252)
                    sortino = float(np.clip(sortino, -5.0, 5.0))
                else:
                    sortino = 0.0
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # VaR and CVaR
        raw_var95 = float(returns.quantile(0.05))
        raw_var99 = float(returns.quantile(0.01))
        var_95 = min(1.0, float(-raw_var95)) if not np.isclose(raw_var95, 0.0) else 0.0
        var_99 = min(1.0, float(-raw_var99)) if not np.isclose(raw_var99, 0.0) else 0.0
        cvar_95 = min(1.0, float(-returns[returns <= raw_var95].mean())) if not returns[returns <= raw_var95].empty else var_95
        cvar_99 = min(1.0, float(-returns[returns <= raw_var99].mean())) if not returns[returns <= raw_var99].empty else var_99

        non_na = returns.dropna()
        skewness = float(stats.skew(non_na)) if len(non_na) > 30 else 0.0
        skewness = np.clip(skewness, -10.0, 10.0)
        kurtosis = float(stats.kurtosis(non_na)) if len(non_na) > 30 else 0.0
        kurtosis = np.clip(kurtosis, -10.0, 10.0)

        return {
            'volatility': annualized_vol,
            'daily_volatility': daily_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        
        if equity_curve is None or equity_curve.empty:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_periods': 0,
                'avg_drawdown_duration': 0.0,
                'max_drawdown_duration': 0,
                'recovery_time_days': None,
                'calmar_ratio': 0.0,
                'ulcer_index': 0.0,
            }

        # Ensuring non negative equity values and remove NaN/inf
        equity_curve = equity_curve.clip(lower=1e-6)
        equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).dropna()
        
        if equity_curve.empty:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_periods': 0,
                'avg_drawdown_duration': 0.0,
                'max_drawdown_duration': 0,
                'recovery_time_days': None,
                'calmar_ratio': 0.0,
                'ulcer_index': 0.0,
            }

        # Calculate running maximum with min value protection
        running_max = equity_curve.expanding().max()
        running_max = running_max.clip(lower=1e-6) 
        drawdown = (equity_curve - running_max) / running_max
        drawdown = drawdown.clip(-1.0, 0.0) 
        #calculate max drawdown with validation
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        max_drawdown = max(-1.0, max_drawdown) 
        is_drawdown = drawdown < -0.0001  
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = float(np.mean(drawdown_periods)) if drawdown_periods else 0.0
        max_drawdown_duration = int

        max_dd_idx = drawdown.idxmin() if not drawdown.empty else None
        recovery_time = 0
        
        if max_dd_idx is not None:
            try:
                max_dd_position = drawdown.index.get_loc(max_dd_idx)
                if max_dd_position < len(drawdown) - 1:
                    recovery_slice = drawdown[max_dd_idx:]
                    if (recovery_slice > -0.0001).any():  # Check if recovery occurred
                        recovery_idx = recovery_slice[recovery_slice > -0.0001].index[0]
                        if hasattr(recovery_idx - max_dd_idx, 'days'):
                            recovery_time = (recovery_idx - max_dd_idx).days
                        else:
                            recovery_time = 0
            except Exception:
                recovery_time = 0
        
        #calculating Calmar ratio with bounds
        cagr = self._calculate_cagr(equity_curve)
        calmar_ratio = min(100.0, abs(cagr / max_drawdown)) if max_drawdown < -0.0001 else 0.0
        # Calculate Ulcer Index with validation
        ulcer_index = float(np.sqrt(np.mean(drawdown ** 2))) if not drawdown.empty else 0.0
        ulcer_index = min(1.0, ulcer_index) 
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': float(drawdown.mean()) if not drawdown.empty else 0.0,
            'drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_time_days': recovery_time if recovery_time > 0 else None,
            'calmar_ratio': calmar_ratio,
            'ulcer_index': ulcer_index,
        }
    
    def _calculate_trade_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.Series
     ) -> Dict[str, Any]:
        if trades is None or trades.empty:
            return self._get_empty_trade_metrics()

        total_trades = len(trades)
        if total_trades == 0:
            return self._get_empty_trade_metrics()

        long_trades = len(trades[trades['quantity'] > 0])
        short_trades = len(trades[trades['quantity'] < 0])

        base_capital = float(equity_curve.iloc[0]) if equity_curve is not None and not equity_curve.empty else 1e6
        max_trade_pnl = base_capital * 0.10

        trade_pnls = []
        if 'pnl' in trades.columns:
            for pnl in trades['pnl'].dropna():
                try:
                    pnl_val = float(pnl)
                    if np.isfinite(pnl_val):
                        pnl_val = np.clip(pnl_val, -max_trade_pnl, max_trade_pnl)
                        trade_pnls.append(pnl_val)
                except:
                    continue

        if not trade_pnls:
            return self._get_empty_trade_metrics()

        trade_pnls = np.array(trade_pnls)
        min_profit_threshold = 1e-6
        winning_trades = trade_pnls[trade_pnls > min_profit_threshold]
        losing_trades = trade_pnls[trade_pnls < -min_profit_threshold]
        breakeven_trades = trade_pnls[~(trade_pnls > min_profit_threshold) & ~(trade_pnls < -min_profit_threshold)]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        breakeven_count = len(breakeven_trades)

        actual_trades = win_count + loss_count
        win_rate = win_count / actual_trades if actual_trades > 0 else 0.0
        loss_rate = loss_count / actual_trades if actual_trades > 0 else 0.0

        if breakeven_count > 0:
            self.logger.info(f"Found {breakeven_count} breakeven trades (±{min_profit_threshold:.6f})")

        total_profit = float(np.sum(winning_trades)) if len(winning_trades) > 0 else 0.0
        total_loss = float(abs(np.sum(losing_trades))) if len(losing_trades) > 0 else 0.0
        net_profit = total_profit - total_loss

        if total_loss > 0 and total_profit > 0:
            profit_factor = min(100.0, total_profit / total_loss)
        elif total_profit > 0:
            profit_factor = 100.0
        else:
            profit_factor = 0.0

        average_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
        average_loss = float(abs(np.mean(losing_trades))) if len(losing_trades) > 0 else 0.0
        # normalizing to % of starting capital and clamp
        average_win_pct = average_win / base_capital if base_capital > 0 else 0.0
        average_loss_pct = average_loss / base_capital if base_capital > 0 else 0.0
        average_win = float(np.clip(average_win, -base_capital, base_capital))
        average_loss = float(np.clip(average_loss, -base_capital, base_capital))

        largest_win = float(np.max(winning_trades)) if len(winning_trades) > 0 else 0.0
        largest_loss = float(abs(np.min(losing_trades))) if len(losing_trades) > 0 else 0.0

        expectancy = (win_rate * average_win) - (loss_rate * average_loss)

        trades_per_day = 0.0
        if 'timestamp' in trades.columns:
            try:
                start_date = pd.Timestamp(trades['timestamp'].min())
                end_date = pd.Timestamp(trades['timestamp'].max())
                days_trading = max(1, (end_date - start_date).days)
                trades_per_day = total_trades / days_trading
            except:
                trades_per_day = 0.0

        total_commissions = float(trades['commission'].sum()) if 'commission' in trades.columns else 0.0
        total_slippage = float(trades['slippage_cost'].sum()) if 'slippage_cost' in trades.columns else 0.0
        total_commissions = max(0.0, total_commissions)
        total_slippage = max(0.0, total_slippage)

        return {
            'total_trades': total_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'average_win_pct': average_win_pct,
            'average_loss_pct': average_loss_pct,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'expectancy': expectancy,
            'trades_per_day': trades_per_day,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
        }
        
    def _get_empty_trade_metrics(self) -> Dict[str, Any]:
        return {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'expectancy': 0.0,
            'trades_per_day': 0.0,
            'total_commissions': 0.0,
            'total_slippage': 0.0,
        }
    
    def _calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        
        rolling_ic = returns.rolling(30).apply(lambda x: x.corr(x.shift(1)), raw=False) if not returns.empty else pd.Series(dtype=float)
        avg_ic = rolling_ic.mean() if not rolling_ic.empty else 0.0
        tracking_error = returns.std() if not returns.empty else 0.0
        information_ratio = returns.mean() / tracking_error if tracking_error and tracking_error > 0 else 0.0
        avg_drawdown = self._calculate_avg_drawdown(returns) if not returns.empty else 0.0
        sterling_ratio = returns.mean() / abs(avg_drawdown) if avg_drawdown != 0 else 0.0
        burke_ratio = returns.mean() / np.sqrt(np.mean(returns[returns < 0] ** 2)) if not returns.empty else 0.0
        return {
            'information_coefficient': float(avg_ic),
            'information_ratio': float(information_ratio * np.sqrt(252)),
            'sterling_ratio': float(sterling_ratio),
            'burke_ratio': float(burke_ratio),
        }
    
    def _calculate_relative_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        if len(aligned_returns) == 0:
            return {}
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        information_ratio = excess_returns.mean() / tracking_error if tracking_error and tracking_error > 0 else 0.0
        return {
            'beta': float(beta),
            'alpha': float(alpha * 252),
            'tracking_error': float(tracking_error * np.sqrt(252)),
            'information_ratio_vs_benchmark': float(information_ratio * np.sqrt(252)),
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        
        if returns is None or returns.empty:
            return {'modified_sharpe': 0.0, 'omega_ratio': 0.0}
        skew = stats.skew(returns.dropna())
        kurt = stats.kurtosis(returns.dropna())
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() and returns.std() > 0 else 0.0
        modified_sharpe = sharpe * (1 + (skew / 6) - (kurt / 24))
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')
        return {
            'modified_sharpe': float(modified_sharpe),
            'omega_ratio': float(omega_ratio if np.isfinite(omega_ratio) else 0.0),
        }
    
    def _calculate_rolling_returns(self, returns: pd.Series, window: int) -> pd.Series:
        
        if returns is None or returns.empty:
            return pd.Series(dtype=float)
        return returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    
    def _calculate_cagr(self, equity_curve: pd.Series) -> float:
        # more robust CAGR that handles timestamps and avoids division by zero
        if equity_curve is None or equity_curve.empty:
            return 0.0
        try:
            #Determine number of years covered by the equity curve
            if isinstance(equity_curve.index[0], pd.Timestamp) and isinstance(equity_curve.index[-1], pd.Timestamp):
                days = (equity_curve.index[-1] - equity_curve.index[0]).days
                num_years = days / 365.25 if days > 0 else max(len(equity_curve) / 252.0, 1/252.0)
            else:
                num_years = max(len(equity_curve) / 252.0, 1/252.0)

            start = float(equity_curve.iloc[0])
            end = float(equity_curve.iloc[-1])
            #Protect against zero/negative start or end values
            if num_years <= 0 or start <= 0 or end <= 0:
                return 0.0
            #geometric CAGR
            cagr = (end / start) ** (1.0 / num_years) - 1
            return float(np.clip(cagr, -1.0, 10.0))
        except Exception:
            return 0.0
    
    def _calculate_avg_drawdown(self, returns: pd.Series) -> float:
        
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.mean()
    
    def generate_performance_report(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        report_lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            "",
            "RETURN METRICS:",
            f"  Total Return:     {metrics.get('total_return', 0):.4%}",  # Show more decimal places for accuracy
            f"  CAGR:            {metrics.get('cagr', 0):.2%}",
            f"  Best Day:        {metrics.get('best_day', 0):.2%}",
            f"  Worst Day:       {metrics.get('worst_day', 0):.2%}",
            "",
            "RISK METRICS:",
            f"  Volatility:      {metrics.get('volatility', 0):.2%}",
            f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.3f}",
            f"  Sortino Ratio:   {metrics.get('sortino_ratio', 0):.3f}",
            f"  Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}",
            f"  VaR (95%):       {metrics.get('var_95', 0):.2%}",
            "",
            "TRADE METRICS:",
            f"  Total Trades:    {metrics.get('total_trades', 0):,}",
            f"  Win Rate:        {metrics.get('win_rate', 0):.2%}",
            f"  Profit Factor:   {metrics.get('profit_factor', 0):.2f}",
            f"  Average Win:     {metrics.get('average_win', 0):.4f}",
            f"  Average Loss:    {metrics.get('average_loss', 0):.4f}",
            "",
            "ADVANCED METRICS:",
            f"  Calmar Ratio:    {metrics.get('calmar_ratio', 0):.3f}",
            f"  Ulcer Index:     {metrics.get('ulcer_index', 0):.3f}",
            f"  Information Ratio: {metrics.get('information_ratio', 0):.3f}",
            "=" * 60
        ]
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
