
from .base_strategy import BaseStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'BaseStrategy',
    'MomentumStrategy', 
    'MeanReversionStrategy'
]


STRATEGY_REGISTRY = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy
}

def get_strategy(strategy_name: str, **kwargs):
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)