import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BacktestVisualizer:

    def __init__(self, figsize: tuple = (15, 10), dpi: int = 300):

        self.figsize = figsize
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        
        plt.rcParams.update({
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def _sanitize_series(self, s: Optional[pd.Series]) -> pd.Series:
        """Replace infinities with NaN and drop NA. Return empty Series if no valid data."""
        if s is None:
            return pd.Series(dtype=float)
        try:
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.dropna()
        except Exception:
            return pd.Series(dtype=float)
        return s

    def _safe_numeric(self, v, default: float = 0.0) -> float:
        """Try to coerce a value to finite float; return default for non-finite or invalid values."""
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else default
        except Exception:
            return default

    def create_full_report(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        performance_metrics: Dict[str, Any],
        benchmark_curve: Optional[pd.Series] = None,
        output_dir: str = "results"
    ) -> None:
        """Create a complete visualization report
        
        Args:
            equity_curve:portfolio equity over time
            trades: Trade execution records
            performance_metrics:Performance metrics dictionary
            benchmark_curve:Optional benchmark for comparison
            output_dir:Directory to save the plots
        """
        self.logger.info("Creating comprehensive visualization report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        
        self.plot_equity_curve(
            equity_curve, 
            benchmark_curve,
            save_path=output_path / "equity_curve.png"
        )
        
        
        self.plot_drawdown_analysis(
            equity_curve,
            save_path=output_path / "drawdown.png"
        )
        
        
        if not trades.empty:
            self.plot_trade_analysis(
                trades,
                save_path=output_path / "trade_analysis.png"
            )
        
        returns = equity_curve.pct_change(fill_method=None).dropna()
        returns = self._sanitize_series(returns)
        self.plot_return_distribution(
            returns,
            save_path=output_path / "return_distribution.png"
        )
        
        
        self.plot_rolling_performance(
            returns,
            save_path=output_path / "rolling_performance.png"
        )
        
        
        self.plot_risk_analysis(
            returns, 
            performance_metrics,
            save_path=output_path / "risk_analysis.png"
        )
        
        
        self.plot_performance_dashboard(
            equity_curve, 
            performance_metrics,
            save_path=output_path / "performance_dashboard.png"
        )
        
        self.logger.info(f" Visualization report saved to {output_path}")
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
        show_drawdowns: bool = True
    ) -> None:
        """Plot equity curve with optional benchmark comparison"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})

        #sanitize series to avoid inf/NaN propagating to plots
        equity_curve = self._sanitize_series(equity_curve)
        benchmark_curve = self._sanitize_series(benchmark_curve) if benchmark_curve is not None else None
        equity_curve.plot(ax=ax1, linewidth=2, label='Portfolio', color='blue')

        if benchmark_curve is not None and not benchmark_curve.empty:
            benchmark_curve.plot(ax=ax1, linewidth=2, label='Benchmark',
                               color='orange')

        ax1.set_title('Portfolio Equity Curve', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))


        if show_drawdowns:
            running_max = equity_curve.expanding().max()
            # avoid divide-by-zero producing infinite drawdowns: replace zero denom with NaN
            denom = running_max.replace(0, np.nan)
            drawdown = (equity_curve - running_max) / denom
            # clean any +/-inf and fill NaN with 0 (no drawdown where denom invalid)
            drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)

            ax2.fill_between(drawdown.index, drawdown, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)

            ax2.set_title('Drawdown', fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontweight='bold')
            ax2.set_xlabel('Date', fontweight='bold')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Equity curve saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown_analysis(
        self,
        equity_curve: pd.Series,
        save_path: Optional[str] = None
    ) -> None:
        """Create detailed drawdown analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        
        equity_curve = self._sanitize_series(equity_curve)
        if equity_curve.empty:
            self.logger.warning("No equity curve data for drawdown analysis")
            return
        running_max = equity_curve.expanding().max()
        #guard division to avoid infinities
        denom = running_max.replace(0, np.nan)
        drawdown = (equity_curve - running_max) / denom
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        
        ax1.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax1.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax1.set_title('Drawdown Over Time', fontweight='bold')
        ax1.set_ylabel('Drawdown (%)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1.grid(True, alpha=0.3)
        
        
        drawdown_only = drawdown[drawdown < 0]
        if not drawdown_only.empty:
            ax2.hist(drawdown_only, bins=30, color='red', alpha=0.7, edgecolor='black')
            ax2.set_title('Drawdown Distribution', fontweight='bold')
            ax2.set_xlabel('Drawdown (%)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(drawdown.min(), color='darkred', linestyle='--', 
                       label=f'Max DD: {drawdown.min():.2%}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        
        underwater = drawdown.copy()
        underwater[underwater >= 0] = 0
        ax3.fill_between(underwater.index, underwater, 0, color='blue', alpha=0.3)
        ax3.plot(underwater.index, underwater, color='blue', linewidth=1)
        ax3.set_title('Underwater Plot', fontweight='bold')
        ax3.set_ylabel('Underwater (%)')
        ax3.set_xlabel('Date')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax3.grid(True, alpha=0.3)
        
        
        drawdown_periods = self._get_drawdown_periods(drawdown)
        if drawdown_periods:
            durations = [period['duration'] for period in drawdown_periods]
            ax4.hist(durations, bins=min(20, len(durations)), color='orange', 
                    alpha=0.7, edgecolor='black')
            ax4.set_title('Drawdown Duration Distribution', fontweight='bold')
            ax4.set_xlabel('Duration (Days)')
            ax4.set_ylabel('Frequency')
            ax4.axvline(np.mean(durations), color='red', linestyle='--',
                       label=f'Avg: {np.mean(durations):.1f} days')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Drawdown Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Drawdown analysis saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_trade_analysis(
        self,
        trades: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Analyze and visualize trade execution"""
        
        if trades.empty:
            self.logger.warning("No trades to analyze")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        
        if 'timestamp' in trades.columns:
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        else:
            
            trades['timestamp'] = pd.date_range(start='2023-01-01', 
                                               periods=len(trades), freq='D')
        
        
        if 'pnl' in trades.columns and trades['pnl'].notna().sum() > 0:
            wins = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] < 0]
            
            if not wins.empty:
                ax1.scatter(wins['timestamp'], wins['pnl'], 
                          color='green', alpha=0.6, s=20, label='Wins')
            if not losses.empty:
                ax1.scatter(losses['timestamp'], losses['pnl'], 
                          color='red', alpha=0.6, s=20, label='Losses')
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_title('Trade PnL Over Time', fontweight='bold')
            ax1.set_ylabel('PnL')
            ax1.legend()
        else:
            
            buys = trades[trades['quantity'] > 0]
            sells = trades[trades['quantity'] < 0]
            
            if not buys.empty:
                ax1.scatter(buys['timestamp'], buys['quantity'], 
                          color='green', alpha=0.6, s=20, label='Buys')
            if not sells.empty:
                ax1.scatter(sells['timestamp'], sells['quantity'], 
                          color='red', alpha=0.6, s=20, label='Sells')
            
            ax1.set_title('Trade Quantities Over Time', fontweight='bold')
            ax1.set_ylabel('Quantity')
            ax1.legend()
        
        ax1.grid(True, alpha=0.3)
        
        
        trade_values = (trades['quantity'].abs() * trades['price']).replace([np.inf, -np.inf], np.nan).dropna()
        if trade_values.empty:
            ax2.text(0.5, 0.5, 'No trade value data', horizontalalignment='center', verticalalignment='center')
        else:
            ax2.hist(trade_values, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_title('Trade Value Distribution', fontweight='bold')
            ax2.set_xlabel('Trade Value ($)')
            ax2.set_ylabel('Frequency')
            try:
                mean_tv = float(trade_values.mean())
                ax2.axvline(mean_tv, color='red', linestyle='--',
                           label=f'Mean: ${mean_tv:,.0f}')
            except Exception:
                pass
        
        ax2.grid(True, alpha=0.3)
        
        
        trade_counts = trades.groupby(trades['timestamp'].dt.date).size()
        ax3.plot(trade_counts.index, trade_counts.values, color='purple', linewidth=2)
        ax3.set_title('Trading Frequency', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Trades per Day')
        ax3.grid(True, alpha=0.3)
        
        
        if 'commission' in trades.columns:
            total_costs = trades['commission']
            if 'slippage_cost' in trades.columns:
                total_costs = total_costs + trades['slippage_cost']
            
            cumulative_costs = total_costs.cumsum()
            ax4.plot(range(len(cumulative_costs)), cumulative_costs, 
                    color='red', linewidth=2, label='Cumulative Costs')
            ax4.set_title('Cumulative Transaction Costs', fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative Costs ($)')
            ax4.grid(True, alpha=0.3)
        else:
            
            buy_count = (trades['quantity'] > 0).sum()
            sell_count = (trades['quantity'] < 0).sum()
            
            ax4.pie([buy_count, sell_count], labels=['Buys', 'Sells'], 
                   colors=['green', 'red'], autopct='%1.1f%%')
            ax4.set_title('Buy/Sell Distribution', fontweight='bold')
        
        plt.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Trade analysis saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_return_distribution(
        self,
        returns: pd.Series,
        save_path: Optional[str] = None
    ) -> None:
        """Plot return distribution analysis"""
        
        # sanitize returns
        if returns is None:
            returns = pd.Series(dtype=float)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            self.logger.warning("No returns available for return distribution plot")
            #create empty plots with message
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.text(0.5, 0.5, 'No return data', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Actual Returns')
        
        mu, sigma = returns.mean(), returns.std()
        rmin = np.nanmin(returns)
        rmax = np.nanmax(returns)
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin == rmax:
            rmin = mu - 0.01
            rmax = mu + 0.01
        x = np.linspace(rmin, rmax, 100)
        if sigma == 0 or not np.isfinite(sigma):
            normal_dist = np.zeros_like(x)
        else:
            normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
        
        ax1.set_title('Return Distribution', fontweight='bold')
        ax1.set_xlabel('Daily Returns')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        from scipy import stats
        try:
            stats.probplot(returns, dist="norm", plot=ax2)
        except Exception:
            ax2.text(0.5, 0.5, 'Insufficient data for Q-Q plot', horizontalalignment='center', verticalalignment='center')
        ax2.set_title('Q-Q Plot (vs Normal)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3.boxplot(returns, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax3.set_title('Return Box Plot', fontweight='bold')
        ax3.set_ylabel('Daily Returns')
        ax3.grid(True, alpha=0.3)
        
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns = cumulative_returns.replace([np.inf, -np.inf], np.nan).dropna()
        ax4.plot(cumulative_returns.index, cumulative_returns, 
                linewidth=2, color='green')
        ax4.set_title('Cumulative Returns', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Return Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Return distribution saved: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_risk_analysis(
        self,
        returns: pd.Series,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Create comprehensive risk analysis visualization"""
        
        # sanitize returns
        if returns is None:
            returns = pd.Series(dtype=float)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            self.logger.warning("No returns available for risk analysis plot")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No return data', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        var_levels = [0.01, 0.05, 0.10]
        var_values = [returns.quantile(level) for level in var_levels]
        colors_var = ['red', 'orange', 'yellow']
        
        ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', density=True)
        for i, (level, value) in enumerate(zip(var_levels, var_values)):
            if not np.isfinite(value):
                continue
            ax1.axvline(value, color=colors_var[i], linestyle='--', label=f'VaR {int(level*100)}%: {value:.2%}')
        ax1.legend()
        ax1.set_title('Value at Risk Analysis', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        
        ax2 = fig.add_subplot(gs[0, 1])
        windows = [30, 60, 120, 252]
        for window in windows:
            if len(returns) > window:
                rolling_ret = returns.rolling(window).mean() * 252
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                ax2.scatter(rolling_vol, rolling_ret, alpha=0.6, s=10, label=f'{window}D')
        
        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Return')
        ax2.set_title('Risk-Return Profile', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        
        ax3 = fig.add_subplot(gs[0, 2])
        downside_metrics = {
            'Max Drawdown': metrics.get('max_drawdown', 0),
            'VaR 95%': metrics.get('var_95', 0),
            'CVaR 95%': metrics.get('cvar_95', 0),
            'Downside Dev': returns[returns < 0].std()
        }
        
        bars = ax3.bar(range(len(downside_metrics)), list(downside_metrics.values()),
                      color=['red', 'orange', 'darkred', 'brown'])
        ax3.set_xticks(range(len(downside_metrics)))
        ax3.set_xticklabels(downside_metrics.keys(), rotation=45)
        ax3.set_title('Downside Risk Metrics', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        
        ax4 = fig.add_subplot(gs[1, :])
        lags = range(1, min(50, len(returns) // 10))
        autocorr = [returns.autocorr(lag) for lag in lags]
        ax4.plot(lags, autocorr, marker='o', markersize=3, linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Return Autocorrelation', fontweight='bold')
        ax4.set_xlabel('Lag (days)')
        ax4.set_ylabel('Autocorrelation')
        ax4.grid(True, alpha=0.3)
        
        
        ax5 = fig.add_subplot(gs[2, 0])
        
        threshold = 2 * returns.std()
        extreme_positive = returns[returns > threshold]
        extreme_negative = returns[returns < -threshold]
        
        ax5.hist([extreme_negative, extreme_positive], bins=20, 
                color=['red', 'green'], alpha=0.7, label=['Extreme Losses', 'Extreme Gains'])
        ax5.set_title('Tail Risk Distribution', fontweight='bold')
        ax5.set_xlabel('Extreme Returns')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        
        ax6 = fig.add_subplot(gs[2, 1:])
        if len(returns) > 30:
            monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            
            years = [idx[0] for idx in monthly_returns.index]
            months = [idx[1] for idx in monthly_returns.index]
            
            if len(set(years)) > 1:  
                pivot_data = pd.DataFrame({
                    'Year': years,
                    'Month': months,
                    'Return': monthly_returns.values
                })
                pivot_table = pivot_data.pivot(index='Year', columns='Month', values='Return')
                
                sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', 
                           center=0, ax=ax6, cbar_kws={'label': 'Monthly Return'})
                ax6.set_title('Monthly Returns Heatmap', fontweight='bold')
        
        plt.suptitle('Comprehensive Risk Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Risk analysis saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_dashboard(
        self,
        equity_curve: pd.Series,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive performance dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        
        #sanitize numeric metric values to avoid inf/NaN propagating to text/patch sizes
        def _m(name, key, fmt):
            return (name, self._safe_numeric(metrics.get(key, 0)), fmt)

        key_metrics = [
            _m('Total Return', 'total_return', '.2%'),
            _m('CAGR', 'cagr', '.2%'),
            _m('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            _m('Max Drawdown', 'max_drawdown', '.2%')
        ]
        
        for i, (name, value, fmt) in enumerate(key_metrics):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, f'{value:{fmt}}', ha='center', va='center', 
                   fontsize=24, fontweight='bold', color='darkblue')
            ax.text(0.5, 0.2, name, ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            
            if 'return' in name.lower() or 'cagr' in name.lower():
                bg_color = 'lightgreen' if value > 0 else 'lightcoral'
            elif 'sharpe' in name.lower():
                bg_color = 'lightgreen' if value > 1 else 'lightyellow' if value > 0 else 'lightcoral'
            else:  
                bg_color = 'lightgreen' if value > -0.1 else 'lightyellow' if value > -0.2 else 'lightcoral'
            
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                     facecolor=bg_color, alpha=0.3))
        
        
        ax_equity = fig.add_subplot(gs[1, :2])
        equity_curve = self._sanitize_series(equity_curve)
        if equity_curve.empty:
            self.logger.warning("No equity curve data for performance dashboard")
        else:
            equity_curve.plot(ax=ax_equity, linewidth=2, color='darkblue')
        ax_equity.set_title('Portfolio Equity Curve', fontweight='bold', fontsize=14)
        ax_equity.grid(True, alpha=0.3)
        ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        
        ax_dd = fig.add_subplot(gs[1, 2:])
        running_max = equity_curve.expanding().max()
        denom = running_max.replace(0, np.nan)
        drawdown = (equity_curve - running_max) / denom
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
        ax_dd.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax_dd.set_title('Drawdown', fontweight='bold', fontsize=14)
        ax_dd.grid(True, alpha=0.3)
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        
        ax_dist = fig.add_subplot(gs[2, :2])
        returns = equity_curve.pct_change(fill_method=None).dropna()
        returns = self._sanitize_series(returns)
        ax_dist.hist(returns, bins=50, alpha=0.7, color='skyblue', density=True)
        ax_dist.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {returns.mean():.3%}')
        ax_dist.set_title('Daily Returns Distribution', fontweight='bold', fontsize=14)
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        
        ax_table = fig.add_subplot(gs[2, 2:])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        
        #sanitize table metric values before formatting
        table_metrics = [
            ['Metric', 'Value'],
            ['Volatility', f"{self._safe_numeric(metrics.get('volatility', 0)):.2%}"],
            ['Sortino Ratio', f"{self._safe_numeric(metrics.get('sortino_ratio', 0)):.3f}"],
            ['Calmar Ratio', f"{self._safe_numeric(metrics.get('calmar_ratio', 0)):.3f}"],
            ['VaR (95%)', f"{self._safe_numeric(metrics.get('var_95', 0)):.2%}"],
            ['Win Rate', f"{self._safe_numeric(metrics.get('win_rate', 0)):.1%}"],
            ['Total Trades', f"{int(self._safe_numeric(metrics.get('total_trades', 0))):,}"],
            ['Profit Factor', f"{self._safe_numeric(metrics.get('profit_factor', 0)):.2f}"],
        ]
        
        table = ax_table.table(cellText=table_metrics, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        
        for i in range(len(table_metrics)):
            if i == 0:  
                table[(i, 0)].set_facecolor('darkblue')
                table[(i, 1)].set_facecolor('darkblue')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('lightgray')
                table[(i, 1)].set_facecolor('white')
        
        ax_table.set_title('Performance Metrics', fontweight='bold', fontsize=14)
        
        plt.suptitle('Performance Dashboard', fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Performance dashboard saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rolling_performance(self, returns: pd.Series, save_path: Optional[str] = None, windows: List[int] = [30, 60, 120, 252]) -> None:
        """Public wrapper for rolling performance plotting to ensure method exists on the instance.
        Delegates to the internal implementation `_plot_rolling_performance`.
        """
        return self._plot_rolling_performance(returns=returns, save_path=save_path, windows=windows)

    def _plot_rolling_performance(
        self,
        returns: pd.Series,
        save_path: Optional[str] = None,
        windows: List[int] = [30, 60, 120, 252]
    ) -> None:
        """Plot rolling performance metrics (implementation).
        This was split to provide a stable public entry point and avoid attribute errors
        when older code or incremental imports are used.
        """
        # sanitize returns
        if returns is None:
            returns = pd.Series(dtype=float)
        # ensure returns are sanitized
        returns = self._sanitize_series(returns)
        if returns.empty:
            self.logger.warning("No returns available for rolling performance plot")
            if save_path:
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.text(0.5, 0.5, 'No return data', horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                plt.close()
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        colors = ['blue', 'red', 'green', 'orange']

        for i, window in enumerate(windows):
            if len(returns) > window:
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
                rolling_sharpe *= np.sqrt(252)
                ax1.plot(rolling_sharpe.index, rolling_sharpe, color=colors[i % len(colors)], label=f'{window}D', linewidth=1.5)
        
        ax1.set_title('Rolling Sharpe Ratio', fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for i, window in enumerate(windows):
            if len(returns) > window:
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                ax2.plot(rolling_vol.index, rolling_vol, color=colors[i % len(colors)], label=f'{window}D', linewidth=1.5)
        
        ax2.set_title('Rolling Volatility', fontweight='bold')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        equity_curve = (1 + returns).cumprod()
        for i, window in enumerate(windows):
            if len(equity_curve) > window:
                rolling_max_dd = []
                for j in range(window, len(equity_curve)):
                    segment = equity_curve.iloc[j-window:j]
                    running_max = segment.expanding().max()
                    drawdown = (segment - running_max) / running_max
                    rolling_max_dd.append(drawdown.min())
                dates = equity_curve.index[window:]
                ax3.plot(dates, rolling_max_dd, color=colors[i % len(colors)], label=f'{window}D', linewidth=1.5)
        
        ax3.set_title('Rolling Maximum Drawdown', fontweight='bold')
        ax3.set_ylabel('Max Drawdown')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        for i, window in enumerate(windows):
            if len(returns) > window:
                rolling_returns = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
                # guard against invalid values
                rolling_returns = rolling_returns.replace([np.inf, -np.inf], np.nan).dropna()
                if not rolling_returns.empty:
                    #computing the  annualized returns robustly using log1p to avoid overflow
                    try:
                        if (rolling_returns > -1).all():
                            annualized_returns = np.exp(np.log1p(rolling_returns) * (252.0 / window)) - 1
                        else:
                            annualized_returns = rolling_returns * (252.0 / window)
                    except Exception:
                        annualized_returns = rolling_returns * (252.0 / window)
                    annualized_returns = pd.Series(annualized_returns, index=rolling_returns.index)
                    annualized_returns = annualized_returns.replace([np.inf, -np.inf], np.nan).dropna()
                    if not annualized_returns.empty:
                        ax4.plot(annualized_returns.index, annualized_returns, color=colors[i % len(colors)], label=f'{window}D', linewidth=1.5)
        
        ax4.set_title('Rolling Annualized Returns', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Annualized Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            self.logger.info(f"Rolling performance saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Extract drawdown periods from drawdown series"""
        periods: List[Dict[str, Any]] = []
        in_drawdown = False
        start_date = None
        min_value = None

        for date, value in drawdown.items():
            if value < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
                min_value = value
            elif value < 0 and in_drawdown:
                # continue drawdown
                min_value = min(min_value, value)
            elif value >= 0 and in_drawdown:
                # drawdown finished
                in_drawdown = False
                try:
                    duration = (date - start_date).days if hasattr(date - start_date, 'days') else 0
                except Exception:
                    duration = 0

                # compute recovery time (days until drawdown reaches non-negative again)
                recovery_date = None
                try:
                    for future_date, future_value in drawdown[date:].items():
                        if future_value >= 0:
                            recovery_date = future_date
                            break
                except Exception:
                    recovery_date = None

                recovery_time = None
                if recovery_date is not None:
                    try:
                        recovery_time = (recovery_date - date).days if hasattr(recovery_date - date, 'days') else None
                    except Exception:
                        recovery_time = None

                periods.append({
                    'start': start_date,
                    'end': date,
                    'duration': duration,
                    'depth': min_value,
                    'recovery_time': recovery_time
                })
                start_date = None
                min_value = None

        # if still in drawdown at the end of series, close it
        if in_drawdown and start_date is not None:
            try:
                duration = (drawdown.index[-1] - start_date).days if hasattr(drawdown.index[-1] - start_date, 'days') else 0
            except Exception:
                duration = 0
            periods.append({
                'start': start_date,
                'end': drawdown.index[-1],
                'duration': duration,
                'depth': min_value if min_value is not None else 0,
                'recovery_time': None
            })

        return periods

