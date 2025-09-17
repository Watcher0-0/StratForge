import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
import io


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class DataLoader:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_ohlcv_data(
        self,
        file_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: str = 'daily',
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        self.logger.info(f"Loading data from {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        
        dtype_mapping = self._get_dtype_mapping()
        
        try:
            
            df = pd.read_csv(
                file_path,
                parse_dates=['date'],
                dtype=dtype_mapping,
                low_memory=False
            )
            
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            
            if symbols and 'symbol' in df.columns:
                df = df[df['symbol'].isin(symbols)]
            
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
                
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            
            df.sort_index(inplace=True)
            
            
            self._validate_data_quality(df)
            
            
            if mode == 'intraday':
                df = self._process_intraday_data(df)
            
            self.logger.info(f" Loaded {len(df):,} data points")
            self.logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            if 'symbol' in df.columns:
                symbols_count = df['symbol'].nunique()
                self.logger.info(f"   Assets: {symbols_count}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        
        
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            self.logger.warning("Missing data detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                self.logger.warning(f"   {col}: {count:,} missing values")
        
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_count = (df[col] <= 0).sum()
                if invalid_count > 0:
                    self.logger.warning(f"   {col}: {invalid_count:,} invalid prices (≤ 0)")
        
        
        if all(col in df.columns for col in price_cols):
            inconsistent = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            
            if inconsistent > 0:
                self.logger.warning(f"   OHLC inconsistencies: {inconsistent:,} records")
        
        
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"   Duplicate timestamps: {duplicates:,}")
            
            df.drop_duplicates(keep='last', inplace=True)
    
    def _process_intraday_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        
        df = df.sort_index()
        
        
        time_diffs = df.index.to_series().diff()
        median_interval = time_diffs.median()
        
        
        large_gaps = time_diffs > (median_interval * 5)
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            self.logger.info(f"   Large time gaps detected: {gap_count:,}")
            self.logger.info(f"   Median interval: {median_interval}")
        
        return df
    
    def create_sample_data(
        self,
        output_path: str = "data/sample_data.csv",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        num_assets: int = 1,
        dataset_choice: Optional[str] = None,
        seed: int = 42
    ) -> None:
        np.random.seed(seed)
        
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = pd.to_datetime(end_date)

        if start_date is None:
            start_dt = datetime(2015, 1, 1)
        else:
            start_dt = pd.to_datetime(start_date)

        date_range = pd.bdate_range(start=start_dt, end=end_dt)
        num_days = len(date_range)
        
        all_data = []
        
        for asset_idx in range(num_assets):
            if num_assets == 1:
                if dataset_choice and 'SPY' in dataset_choice:
                    symbol = 'SPY'
                    base_vol = 0.015
                    drift = 0.0005
                elif dataset_choice and 'BTC' in dataset_choice:
                    symbol = 'BTC-USD'
                    base_vol = 0.05
                    drift = 0.001
                elif dataset_choice and 'EURUSD' in dataset_choice:
                    symbol = 'EURUSD'
                    base_vol = 0.005
                    drift = 0.0002
                else:
                    symbol = 'ASSET'
                    base_vol = 0.02
                    drift = 0.0005
            else:
                symbol = f"ASSET_{asset_idx:03d}"
                base_vol = 0.02 + (asset_idx % 3) * 0.005
                drift = 0.0003 + (asset_idx % 5) * 0.0001
            
            initial_price = 100.0 + np.random.uniform(-20, 50)
            returns = np.random.normal(drift, base_vol, num_days)
            trend = np.linspace(0, 0.1 * drift if drift is not None else 0, num_days)
            returns += trend / max(num_days, 1)
            
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices)
            
            
            for i, (date, close) in enumerate(zip(date_range, prices)):
                
                daily_vol = abs(np.random.normal(0, 0.01))
                
                
                open_price = close * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close) * (1 + daily_vol * np.random.uniform(0, 1))
                low_price = min(open_price, close) * (1 - daily_vol * np.random.uniform(0, 1))
                
                
                volume = int(np.random.lognormal(15, 0.5))  
                
                row_data = {
                    'date': date,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close, 2),
                    'volume': volume
                }
                
                all_data.append(row_data)
        
        
        df = pd.DataFrame(all_data)
        df = df.sort_values(['date', 'symbol'])
        
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Sample data created: {output_path}")
        self.logger.info(f"  Records: {len(df):,}")
        self.logger.info(f"  Assets: {num_assets}")
        self.logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    def load_sample_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, num_assets: int = 1, dataset_choice: Optional[str] = None) -> pd.DataFrame:
        """Load sample data, creating it if it doesn't exist."""
        sample_path = "data/sample_data.csv"
        if not Path(sample_path).exists():
            self.create_sample_data(sample_path, start_date=start_date, end_date=end_date, num_assets=num_assets, dataset_choice=dataset_choice)
        else:
            if start_date is not None or end_date is not None:
                
                try:
                    tmp = pd.read_csv(sample_path, parse_dates=['date'])
                    if len(tmp) > 0:
                        file_start = pd.to_datetime(tmp['date'].min())
                        file_end = pd.to_datetime(tmp['date'].max())
                        if (start_date and pd.to_datetime(start_date) < file_start) or (end_date and pd.to_datetime(end_date) > file_end):
                            
                            self.create_sample_data(sample_path, start_date=start_date, end_date=end_date, num_assets=num_assets, dataset_choice=dataset_choice)
                except Exception:
                    self.create_sample_data(sample_path, start_date=start_date, end_date=end_date, num_assets=num_assets, dataset_choice=dataset_choice)

        return self.load_ohlcv_data(sample_path, start_date, end_date)

    def load_csv_data(self, uploaded_file, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load data from uploaded CSV file."""
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  

        df = pd.read_csv(
            io.BytesIO(file_content),
            parse_dates=['date'],
            dtype={
                'open': np.float32,
                'high': np.float32,
                'low': np.float32,
                'close': np.float32,
                'volume': np.float64,
                'symbol': 'category'
            },
            low_memory=False
        )

        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in uploaded file: {missing_cols}")

        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        df.sort_index(inplace=True)
        self._validate_data_quality(df)

        self.logger.info(f" Loaded {len(df):,} data points from uploaded file")
        if len(df) > 0:
            self.logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
        else:
            self.logger.warning(" No data loaded from uploaded file")

        return df

    def _get_dtype_mapping(self) -> Dict[str, type]:
        """Get the dtype mapping for OHLCV data."""
        return {
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.float64,
            'symbol': 'category'
        }


class MarketDataManager:    
    def __init__(self):
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)
    
    def load_multi_asset_data(
        self,
        file_paths: Union[str, List[str]],
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if isinstance(file_paths, str):
            
            return self.data_loader.load_ohlcv_data(
                file_paths, start_date, end_date, symbols=symbols
            )
        
        
        all_data = []
        for file_path in file_paths:
            df = self.data_loader.load_ohlcv_data(
                file_path, start_date, end_date
            )
            all_data.append(df)
        
        
        combined_df = pd.concat(all_data, axis=0, ignore_index=False)
        combined_df.sort_index(inplace=True)
        
        return combined_df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        frequency: str = '1D',
        method: str = 'last'
    ) -> pd.DataFrame:
        if 'symbol' in df.columns:
            
            resampled_data = []
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].drop('symbol', axis=1)
                resampled_symbol = self._resample_single_asset(
                    symbol_data, frequency, method
                )
                resampled_symbol['symbol'] = symbol
                resampled_data.append(resampled_symbol)
            
            return pd.concat(resampled_data, axis=0)
        else:
            
            return self._resample_single_asset(df, frequency, method)
    
    def _resample_single_asset(
        self,
        df: pd.DataFrame,
        frequency: str,
        method: str
    ) -> pd.DataFrame:
        
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        resampled = df.resample(frequency).agg(agg_dict)
        
        
        resampled.dropna(inplace=True)
        
        return resampled
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        method: str = 'simple'
    ) -> pd.Series:
        prices = df[price_col]
        
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns.dropna()
    
    def align_data_timestamps(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = 'outer'
    ) -> Dict[str, pd.DataFrame]:
        
        all_indices = [df.index for df in data_dict.values()]
        
        if method == 'inner':
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
        elif method == 'outer':
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
        else:
            raise ValueError("Method must be 'inner' or 'outer'")
        
        
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_data[symbol] = df.reindex(common_index, method='ffill')
        
        return aligned_data