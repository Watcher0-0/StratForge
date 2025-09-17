import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import sys
import os
import glob
import importlib.util


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _load_cpp_core(logger=None):
    try:
        import backtesting_core
        return backtesting_core, True
    except Exception as e:
        expected_tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
        search_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'cpp')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')),
        ]
        so_candidates = []
        for p in search_paths:
            if os.path.isdir(p):
                so_candidates.extend(glob.glob(os.path.join(p, 'backtesting_core*.so')))
        matching = [f for f in so_candidates if expected_tag in os.path.basename(f)]
        mismatched = [f for f in so_candidates if expected_tag not in os.path.basename(f)]

        if matching:
            path = matching[0]
            try:
                spec = importlib.util.spec_from_file_location('backtesting_core', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if logger:
                    logger.info(f"Loaded C++ extension from {path}")
                return module, True
            except Exception as e2:
                if logger:
                    logger.warning(f"Found compiled extension for this Python but failed to load: {e2}")
                return None, False

        if logger:
            if mismatched:
                logger.warning('Found compiled C++ extension(s) but for a different Python ABI:')
                for m in mismatched:
                    logger.warning('  - %s', m)
                logger.warning('Rebuild the C++ extension for the current Python (ABI tag: %s). Example:', expected_tag)
                logger.warning('  %s -m pip install --upgrade setuptools wheel pybind11', sys.executable)
                logger.warning('  %s setup.py build_ext --inplace', sys.executable)
            else:
                logger.warning('C++ extension not found. Build it to enable the optimized core. Example:')
                logger.warning('  %s -m pip install --upgrade setuptools wheel pybind11', sys.executable)
                logger.warning('  %s setup.py build_ext --inplace', sys.executable)
        else:
            print('C++ extension not available and could not be loaded.')
        return None, False


backtesting_core, CPP_CORE_AVAILABLE = _load_cpp_core(logging.getLogger(__name__))

from src.data_loader import DataLoader
from src.performance import PerformanceAnalyzer
from src.visualizer import BacktestVisualizer
from src.risk_manager import RiskManager


class QuantBacktester:
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        slippage: float = 0.001,
        transaction_costs: float = 0.0005,
        mode: str = 'daily',
        parallel: bool = True,
        verbose: bool = False,
        max_pct_per_trade: float = 0.10,
        stop_loss_pct: float = 0.10,
        take_profit_pct: float = 0.30,
        force_cpp: bool = False
    ):
       
        # sanitize parameters and enforce limits
        self.initial_capital = max(1000.0, min(float(initial_capital), 1e9))  # Reasonable capital range
        self.slippage = max(0.0, min(float(slippage), 0.1))  # Cap slippage at 10%
        self.transaction_costs = max(0.0, min(float(transaction_costs), 0.1))  # Cap transaction costs at 10%
        self.mode = mode
        self.parallel = parallel
        self.verbose = verbose
        # allow caller to force using the C++ core (if available)
        self.force_cpp = bool(force_cpp)

        # max_pct_per_trade i.e.- maximum fraction of current cash to deploy per new position
        self.max_pct_per_trade = max(0.01, min(float(max_pct_per_trade), 1.0))  #at least 1%, at most 100%
        self.stop_loss_pct = max(0.0, min(float(stop_loss_pct), 1.0))  #cap stop loss at 100%
        self.take_profit_pct = max(0.0, min(float(take_profit_pct), 10.0))  # Cap take profit at 1000%
        
        #helpers
        self.data_loader = DataLoader()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()
        self.risk_manager = RiskManager()
        
        #Logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # C++ core config(if present)
        self.cpp_available = CPP_CORE_AVAILABLE
        if CPP_CORE_AVAILABLE:
            self.cpp_config = backtesting_core.BacktestConfig()
            self.cpp_config.initial_capital = initial_capital
            self.cpp_config.slippage = slippage
            self.cpp_config.transaction_costs = transaction_costs
            self.cpp_config.num_threads = 4 if parallel else 1
            try:
                self.cpp_config.max_pct_per_trade = self.max_pct_per_trade
                self.cpp_config.stop_loss_pct = self.stop_loss_pct
                self.cpp_config.take_profit_pct = self.take_profit_pct
            except Exception:
                pass
            self.logger.info(" C++ core initialized for maximum performance")
        else:
            self.logger.warning(" Using Python fallback (10-20x slower)")
            if self.force_cpp:
                self.logger.warning("force_cpp=True but C++ core not available; will use Python fallback.")

    def run_backtest(
        self,
        strategy,
        data_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: str = "results",
        force_python: bool = True  # Force Python execution for debugging
    ) -> Dict[str, Any]:
        self.logger.info(f" Starting backtest with {strategy.__class__.__name__}")
        
        
        Path(output_dir).mkdir(exist_ok=True)
        
        
        if data is None:
            if data_path is None:
                raise ValueError("Either data_path or data must be provided")
            self.logger.info(" Loading market data...")
            data = self.data_loader.load_ohlcv_data(
                data_path, start_date, end_date, mode=self.mode
            )
            self.logger.info(f"   → Loaded {len(data):,} data points from {data_path}")
        else:
            self.logger.info(" Using provided data...")
            self.logger.info(f"   → Loaded {len(data):,} data points (provided)")
        
        if data.empty:
            raise ValueError("No data available. Check input parameters.")
        
        
        self.logger.info(" Generating trading signals...")
        signals = strategy.generate_signals(data)
        
        if isinstance(signals, pd.Series):
            signals = signals.values
        
        self.logger.info(f"   → Generated {len(signals):,} signals")
        
        # Debugging: expose signals as Series for inspection when verbose
        try:
            #attempt to create a Series aligned to the data index when lengths match
            if isinstance(signals, np.ndarray) or isinstance(signals, list):
                if len(signals) == len(data):
                    signals_series = pd.Series(signals, index=data.index, name='signals')
                else:
                    signals_series = pd.Series(signals, name='signals')
            elif isinstance(signals, pd.Series):
                signals_series = signals.copy()
            else:
                signals_series = pd.Series(signals, name='signals')
        except Exception:
            signals_series = pd.Series(np.asarray(signals), name='signals')
        if self.verbose:
            try:
                nz = int((signals_series != 0).sum())
            except Exception:
                nz = None
            try:
                total = len(signals_series)
            except Exception:
                total = None
            self.logger.info(f"   → Signals non-zero count: {nz} / {total}")
            try:
                self.logger.debug(f"   → Signals head:\n{signals_series.head().to_string()}")
                self.logger.debug(f"   → Signals tail:\n{signals_series.tail().to_string()}")
            except Exception:
                pass
            #-save signals for offline inspection
            try:
                signals_series.to_csv(Path(output_dir) / 'signals_debug.csv', index=True)
                self.logger.info(f"   → Signals saved to {Path(output_dir) / 'signals_debug.csv'}")
            except Exception:
                pass

        self.logger.info(" Running portfolio simulation...")
        
        start_time = time.time()
        used_cpp = False
        # Use Python backend for debugging unless explicitly forced to C++
        if CPP_CORE_AVAILABLE and self.force_cpp and not force_python:
            try:
                used_cpp = True
                results = self._run_cpp_backtest(data, signals)
            except Exception as e:
                self.logger.warning(f"C++ backtest failed, falling back to Python: {e}")
                used_cpp = False
                # capture C++ exception for diagnostics
                cpp_err = str(e)
                results = self._run_python_backtest(data, signals)
                results['cpp_error'] = cpp_err
        else:
            if self.force_cpp and not CPP_CORE_AVAILABLE:
                self.logger.warning("Force C++ requested but C++ core not available; using Python fallback.")
            results = self._run_python_backtest(data, signals)
            results['cpp_error'] = None

        # record execution time and backend used
        elapsed = time.time() - start_time
        results['backend'] = 'C++' if used_cpp else 'Python'
        results['execution_time'] = elapsed
        # expose cpp availability to caller
        results['cpp_available'] = bool(CPP_CORE_AVAILABLE)
        
        
        self.logger.info("Calculating performance metrics...")
        # Generate performance report with actual portfolio values
        metrics = self.performance_analyzer.calculate_comprehensive_metrics(
            results['equity_curve'], 
            results['trades'],
            initial_capital=self.initial_capital
        )
        
        # Ensure final value matches actual portfolio value
        if 'equity_curve' in results and not results['equity_curve'].empty:
            final_value = float(results['equity_curve'].iloc[-1])
            if np.isfinite(final_value) and final_value > 0:
                metrics['final_value'] = final_value
                
        results['performance_metrics'] = metrics
        
        
        self.logger.info(" Performing risk analysis...")
        risk_analysis = self.risk_manager.analyze_portfolio_risk(
            results['equity_curve'], 
            results['trades']
        )
        results['risk_analysis'] = risk_analysis
        
        
        self.logger.info("Creating visualizations...")
        self.visualizer.create_full_report(
            results['equity_curve'],
            results['trades'], 
            metrics,
            output_dir=output_dir
        )
        
        
        self.logger.info(" Saving results...")
        self._save_results(results, output_dir)
        
        self.logger.info(" Backtest completed successfully!")
        return results
    
    def _run_cpp_backtest(self, data: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        
        
        
        prices = self._prepare_price_matrix(data)
        signal_matrix = self._prepare_signal_matrix(signals, prices.shape)
        symbols = data.columns.tolist() if 'close' not in data.columns else ['ASSET']
        timestamps = data.index.astype(np.int64) // 10**9  
        
        
        prices_np = np.array(prices, dtype=np.float64)
        signals_np = np.array(signal_matrix, dtype=np.float64)
        timestamps_np = np.array(timestamps, dtype=np.float64)
        
        
        start_time = time.time()
        equity_curve_np, trades, states, cpp_metrics = backtesting_core.run_optimized_backtest(
            prices_np, signals_np, symbols, timestamps_np, self.cpp_config
        )
        end_time = time.time()
        
        self.logger.info(f"   → C++ execution time: {end_time - start_time:.3f} seconds")
        
        
        equity_curve = pd.Series(
            equity_curve_np, 
            index=data.index,
            name='portfolio_value'
        )
        
        
        trades_df = self._convert_cpp_trades_to_df(trades)
        
        # convert cpp_metrics (pybind object) to a plain dict for easier consumption in Python
        cpp_metrics_dict = {}
        if cpp_metrics is not None:
            try:
                for key in ['win_rate', 'profit_factor', 'average_win', 'average_loss', 'total_trades', 'var_95', 'cvar_95', 'volatility', 'skewness', 'kurtosis']:
                    try:
                        val = getattr(cpp_metrics, key)
                        # Convert numpy/pybind types to native Python types
                        if hasattr(val, 'item'):
                            val = val.item()
                        cpp_metrics_dict[key] = float(val) if val is not None else None
                    except Exception:
                        cpp_metrics_dict[key] = None
            except Exception:
                cpp_metrics_dict = {}
        
        return {
            'equity_curve': equity_curve,
            'trades': trades_df,
            'portfolio_states': states,
            'cpp_metrics': cpp_metrics,
            'cpp_metrics_dict': cpp_metrics_dict
        }
    
    def _run_python_backtest(self, data: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        
        # Validate and initialize core parameters
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        # Initialize tracking variables with proper types
        portfolio_value = float(self.initial_capital)
        cash = float(self.initial_capital)
        position = 0.0
        avg_entry_price = 0.0
        realized_pnl_cumulative = 0.0
        last_valid_portfolio_value = float(self.initial_capital)
        
        equity_curve = []
        trades = []
        
        # Track metrics for validation
        max_daily_return = 0.05  # 5% max daily return for reality check
        cumulative_cost = 0.0
        trade_count = 0
        
        prices = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
        # sanitize prices: ensure finite positive prices by forward-filling and clipping
        prices = np.asarray(prices, dtype=float)
        if prices.size == 0:
            raise ValueError("No price data available for backtest")
        # replace non-finite with NaN then forward-fill using last valid price, fallback to 1.0
        invalid_mask = ~np.isfinite(prices) | (prices <= 0)
        if invalid_mask.any():
            last_valid = None
            for i in range(len(prices)):
                if invalid_mask[i]:
                    if last_valid is not None:
                        prices[i] = last_valid
                    else:
                        # find first later valid
                        j = i + 1
                        while j < len(prices) and (not np.isfinite(prices[j]) or prices[j] <= 0):
                            j += 1
                        if j < len(prices) and np.isfinite(prices[j]) and prices[j] > 0:
                            prices[i] = prices[j]
                        else:
                            prices[i] = 1.0
                else:
                    last_valid = prices[i]
        
        # sanitize signals array and clamp to [-1,1]
        signals = np.asarray(signals, dtype=float)
        if signals.ndim > 1:
            signals = signals.ravel()
        # replace non-finite signals with 0 and clamp
        signals[~np.isfinite(signals)] = 0.0
        signals = np.clip(signals, -1.0, 1.0)
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = prices[i]
            signal = float(signals[i]) if i < len(signals) else 0.0
            
            # Debugging logs for cash, position, and portfolio value
            self.logger.debug(f"Iteration {i}: Timestamp={timestamp}, Signal={signal}, Current Price={current_price}")
            self.logger.debug(f"Before Update: Cash={cash}, Position={position}, Portfolio Value={portfolio_value}")

            # Ensure the signal value variable used below is always defined
            sig_val = signal

            if not np.isfinite(current_price) or current_price <= 0:
                current_price = prices[i-1] if i > 0 else 1.0
            # Calculate position value and unrealized P&L
            position_value = position * current_price
            unrealized_pnl = 0.0
            if position != 0:
                unrealized_pnl = position * (current_price - avg_entry_price) if position > 0 else position * (avg_entry_price - current_price)
            
            # Calculate total portfolio value including unrealized P&L and track it properly
            market_value = position * current_price
            unrealized_pnl = position * (current_price - avg_entry_price) if position != 0 else 0.0
            portfolio_value = cash + market_value
            
            # Safety checks only for truly invalid states
            if not np.isfinite(portfolio_value) or portfolio_value < 0:
                self.logger.warning('Portfolio value became invalid; using last valid value')
                portfolio_value = last_valid_portfolio_value
                sig_val = 0.0
            
            # Handle bankruptcy case
            if portfolio_value <= 0:
                if position != 0:
                    # Close position at current price
                    closed_qty = abs(position)
                    pnl_close = closed_qty * (current_price - avg_entry_price) if position > 0 else closed_qty * (avg_entry_price - current_price)
                    realized_pnl_cumulative += pnl_close
                    trade_value = abs(position * current_price)
                    
                    # Minimal costs on liquidation
                    slippage_cost = trade_value * self.slippage * 0.5  # Reduced slippage on liquidation
                    commission = trade_value * self.transaction_costs
                    total_costs = slippage_cost + commission
                    
                    # Update cash
                    cash += position * current_price + pnl_close - total_costs
                    trades.append({
                        'timestamp': timestamp,
                        'symbol': 'ASSET',
                        'quantity': -position,
                        'price': current_price,
                        'commission': commission,
                        'slippage_cost': slippage_cost,
                        'side': 'LIQUIDATE',
                        'pnl': float(pnl_close)
                    })
                    position = 0.0
                    
                # Set minimum portfolio value and continue
                portfolio_value = max(100.0, cash)  # Minimum viable portfolio
                remaining = len(data) - i
                for _ in range(remaining):
                    equity_curve.append(portfolio_value)
                break
            sizing_base = portfolio_value  # Use actual portfolio value for position sizing
            
            # Calculate position size based on signal and constraints
            if sig_val != 0:
                # Strictly enforce max position size as percentage of portfolio
                max_position_value = sizing_base * self.max_pct_per_trade
                
                # Calculate available cash considering current position
                available_cash = cash
                if position > 0:
                    available_cash += position_value
                elif position < 0:
                    available_cash -= position_value
                
                # Scale desired position by signal strength
                position_scale = abs(sig_val)  # Signal strength affects position size
                desired_exposure = max_position_value * position_scale
                
                # Calculate target position with strict risk management
                if sig_val > 0:  # Long position
                    # Calculate maximum position size based on available cash
                    max_shares = (available_cash / current_price) * self.max_pct_per_trade if current_price > 0 else 0.0
                    target_position = max_shares * position_scale
                else:  # Short position
                    # Calculate maximum short position
                    max_shares = (available_cash / current_price) * self.max_pct_per_trade * 0.5 if current_price > 0 else 0.0
                    target_position = -max_shares * position_scale

                # Apply portfolio-based position limit as a final check
                max_position_value = portfolio_value * self.max_pct_per_trade
                if abs(target_position * current_price) > max_position_value:
                    target_position = (max_position_value / current_price) * (1 if target_position > 0 else -1)
            else:
                target_position = position
            position_change = target_position - position
            if abs(position_change) > 1e-12:
                # Calculate trade costs with market impact consideration
                trade_value = abs(position_change * current_price)
                
                # Apply progressive slippage based on trade size relative to portfolio
                trade_size_factor = min(1.0, trade_value / portfolio_value)
                effective_slippage = self.slippage * (1 + trade_size_factor)  # Larger trades have more impact
                
                # Calculate costs
                slippage_cost = trade_value * effective_slippage
                commission = trade_value * self.transaction_costs
                
                # Apply reasonable limits to costs
                max_total_cost_pct = 0.05  # Maximum 5% total trading cost
                total_costs = min(
                    slippage_cost + commission,
                    trade_value * max_total_cost_pct
                )
                
                # Adjust position change if costs would exceed available capital
                if total_costs > available_cash * 0.2:  # Don't use more than 20% of available cash for costs
                    scale_factor = (available_cash * 0.2) / total_costs
                    position_change *= scale_factor
                    trade_value = abs(position_change * current_price)
                    total_costs *= scale_factor
                pnl = 0.0

                # Debug the trade that will be executed
                self.logger.debug(f"Executing trade at i={i}: position_change={position_change}, trade_value={trade_value}, cash_before={cash}, avg_entry_price_before={avg_entry_price}")

                if position == 0:
                    avg_entry_price = current_price
                    cash -= position_change * current_price + total_costs
                else:
                    if np.sign(position) == np.sign(target_position) or target_position == 0:
                        if abs(target_position) < abs(position):
                            closed_qty = abs(position) - abs(target_position)
                            if position > 0:
                                pnl = closed_qty * (current_price - avg_entry_price)
                            else:
                                pnl = closed_qty * (avg_entry_price - current_price)
                            realized_pnl_cumulative += pnl
                            cash += closed_qty * current_price + pnl - total_costs
                            if abs(target_position) < 1e-12:
                                avg_entry_price = 0.0
                        else:
                            added_qty = abs(position_change)
                            denom_qty = abs(position) + added_qty
                            if denom_qty > 0.0:
                                avg_entry_price = ((avg_entry_price * abs(position)) + (current_price * added_qty)) / denom_qty
                            else:
                                avg_entry_price = current_price
                            cash -= position_change * current_price + total_costs
                    else:
                        closed_qty = abs(position)
                        if position > 0:
                            pnl_close = closed_qty * (current_price - avg_entry_price)
                        else:
                            pnl_close = closed_qty * (avg_entry_price - current_price)
                        realized_pnl_cumulative += pnl_close
                        cash += closed_qty * current_price + pnl_close - total_costs
                        avg_entry_price = current_price
                
                # Record the trade with calculated costs
                trades.append({
                    'timestamp': timestamp,
                    'symbol': 'ASSET',
                    'quantity': position_change,
                    'price': current_price,
                    'commission': commission,
                    'slippage_cost': slippage_cost,
                    'side': 'BUY' if position_change > 0 else 'SELL',
                    'pnl': float(pnl)
                })
                
                # Update position (already updated above in the trade logic)
                position = target_position

                # Log post-trade state with actual costs
                self.logger.debug(f"Post-trade: cash={cash}, position={position}, avg_entry_price={avg_entry_price}, costs={total_costs}, realized_pnl_cumulative={realized_pnl_cumulative}")
            
            # Calculate final portfolio value with precise P&L tracking
            market_value = position * current_price
            position_cost = position * avg_entry_price if position != 0 else 0
            unrealized_pnl = market_value - position_cost if position != 0 else 0
            
            # Use actual portfolio value without artificial limits
            portfolio_value = cash + market_value
            
            # Only validate truly invalid states
            if not np.isfinite(portfolio_value) or portfolio_value < 0:
                # Try to recover with a sensible value
                if np.isfinite(cash) and cash > 0:
                    portfolio_value = cash
                else:
                    portfolio_value = last_valid_portfolio_value
                if position != 0:
                    sig_val = 0  
            else:
                last_valid_portfolio_value = portfolio_value

            # Update portfolio metrics without artificial limits
            equity_curve.append(portfolio_value)
        equity_curve = pd.Series(equity_curve, index=data.index, name='portfolio_value')
        
        equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan)
        
        equity_curve = equity_curve.fillna(method='ffill')
        
        equity_curve = equity_curve.fillna(self.initial_capital)
        
        if portfolio_value > 0 and np.isfinite(portfolio_value):
            equity_curve.iloc[-1] = portfolio_value
        if np.isfinite(portfolio_value) and portfolio_value > 0:
            equity_curve.iloc[-1] = portfolio_value
        
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty and 'timestamp' in trades_df.columns:
            try:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            except Exception:
                pass
        
        return {
            'equity_curve': equity_curve,
            'trades': trades_df
        }
    
    def _prepare_price_matrix(self, data: pd.DataFrame) -> np.ndarray:
        
        if 'close' in data.columns:
            
            return data[['close']].values
        else:
            
            return data.values
    
    def _prepare_signal_matrix(self, signals: np.ndarray, price_shape: tuple) -> np.ndarray:
        
        if signals.ndim == 1:
            
            return signals.reshape(-1, 1)
        else:
            return signals
    
    def _convert_cpp_trades_to_df(self, cpp_trades) -> pd.DataFrame:
        trades_data = []
        for trade in cpp_trades:
            trades_data.append({
                'timestamp': pd.to_datetime(trade.timestamp, unit='s'),
                'symbol': trade.symbol,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'slippage_cost': trade.slippage_cost,
                'side': trade.side,
                'pnl': trade.pnl
            })
        
        return pd.DataFrame(trades_data)
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        if not results['trades'].empty:
            trades_path = Path(output_dir) / 'trades.csv'
            results['trades'].to_csv(trades_path, index=False)
            self.logger.info(f"   → Trades saved: {trades_path}")
        
        
        metrics = results['performance_metrics']
        
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = value.item()
            else:
                clean_metrics[key] = value
        
        performance_path = Path(output_dir) / 'performance.json'
        with open(performance_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2, default=str)
        self.logger.info(f"   → Performance metrics saved: {performance_path}")
        
        equity_path = Path(output_dir) / 'equity_curve.csv'
        results['equity_curve'].to_csv(equity_path)
        self.logger.info(f"   → Equity curve saved: {equity_path}")
