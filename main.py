import argparse
import sys
import os
from pathlib import Path
import json
import time
from src.backtester import QuantBacktester
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy


def main():
    parser = argparse.ArgumentParser(description='High Performance Quantitative Backtesting Engine')
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        required=True,
        choices=['momentum', 'mean_reversion'],
        help='Strategy to backtest (momentum or mean_reversion)'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_data.csv',
        help='Path to OHLCV CSV data file'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='daily',
        choices=['daily', 'intraday'],
        help='Backtesting frequency mode'
    )
    
    parser.add_argument(
        '--start_date', 
        type=str, 
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date', 
        type=str, 
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial_capital', 
        type=float, 
        default=1000000.0,
        help='Initial portfolio capital'
    )
    
    parser.add_argument(
        '--slippage', 
        type=float, 
        default=0.001,
        help='Slippage factor (default 0.1%)'
    )
    
    parser.add_argument(
        '--transaction_costs', 
        type=float, 
        default=0.0005,
        help='Transaction costs factor (default 0.05%)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("INSTITUTIONAL GRADE BACKTESTING ENGINE")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Data: {args.data}")
    print(f"Mode: {args.mode}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Slippage: {args.slippage:.4f}")
    print(f"Transaction Costs: {args.transaction_costs:.4f}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Parallel Processing: {args.parallel}")
    print("-" * 70)
    
    strategy_map = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy
    }
    
    strategy_class = strategy_map[args.strategy]
    strategy = strategy_class()
    
    backtester = QuantBacktester(
        initial_capital=args.initial_capital,
        slippage=args.slippage,
        transaction_costs=args.transaction_costs,
        mode=args.mode,
        parallel=args.parallel,
        verbose=args.verbose
    )
    
    try:
        start_time = time.time()
        
        
        results = backtester.run_backtest(
            strategy=strategy,
            data_path=args.data,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n Backtest completed in {execution_time:.2f} seconds")
        print("-" * 70)
        print(" PERFORMANCE SUMMARY")
        print("-" * 70)
        
        metrics = results['performance_metrics']
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"CAGR: {metrics['cagr']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"VaR (95%): {metrics['var_95']:.2%}")
        print(f"Total Trades: {metrics['total_trades']:,}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        print("-" * 70)
        print(" OUTPUT FILES:")
        print(f"• Trades: {args.output_dir}/trades.csv")
        print(f"• Performance: {args.output_dir}/performance.json")
        print(f"• Equity Curve: {args.output_dir}/equity_curve.png")
        print(f"• Drawdown Chart: {args.output_dir}/drawdown.png")
        print(f"• Trade Analysis: {args.output_dir}/trade_analysis.png")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n Backtest failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()