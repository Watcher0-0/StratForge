#include "core.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <chrono>

namespace backtesting {

class BacktestingEngineImpl {
public:
    explicit BacktestingEngineImpl(const BacktestConfig& config) : config_(config) {
        if (config_.num_threads > 0) {
            omp_set_num_threads(config_.num_threads);
        }
    }
    BacktestConfig config_;
};

BacktestingEngine::BacktestingEngine(const BacktestConfig& config) 
    : config_(config), impl_(std::make_unique<BacktestingEngineImpl>(config)) {
}

BacktestingEngine::~BacktestingEngine() = default;

std::tuple<Eigen::VectorXd, std::vector<Trade>, std::vector<PortfolioState>>
BacktestingEngine::run_backtest(
    const Eigen::MatrixXd& prices,
    const Eigen::MatrixXd& signals, 
    const std::vector<std::string>& symbols,
    const Eigen::VectorXd& timestamps
) {
    const int num_periods = static_cast<int>(prices.rows());
    const int num_assets = static_cast<int>(prices.cols());

    if (signals.rows() != num_periods || signals.cols() != num_assets) {
        throw std::invalid_argument("Price and signal matrix dimensions must match");
    }
    if (static_cast<int>(symbols.size()) != num_assets) {
        throw std::invalid_argument("Symbol count must match number of assets");
    }

    Eigen::VectorXd equity_curve = Eigen::VectorXd::Zero(num_periods);
    std::vector<Trade> trades;
    std::vector<PortfolioState> portfolio_states;

    trades.reserve(std::max(1, num_periods * num_assets / 10));
    portfolio_states.reserve(num_periods);

    simulate_portfolio_vectorized(
        prices, signals, symbols, timestamps,
        equity_curve, trades, portfolio_states
    );

    return std::make_tuple(equity_curve, trades, portfolio_states);
}

void BacktestingEngine::simulate_portfolio_vectorized(
    const Eigen::MatrixXd& prices,
    const Eigen::MatrixXd& signals,
    const std::vector<std::string>& symbols,
    const Eigen::VectorXd& timestamps,
    Eigen::VectorXd& equity_curve,
    std::vector<Trade>& trades,
    std::vector<PortfolioState>& states
) const {
    const int num_periods = static_cast<int>(prices.rows());
    const int num_assets = static_cast<int>(prices.cols());

    Eigen::VectorXd positions = Eigen::VectorXd::Zero(num_assets);
    Eigen::VectorXd prev_positions = Eigen::VectorXd::Zero(num_assets);
    double cash = config_.initial_capital;
    double total_value = config_.initial_capital;

    double peak_value = config_.initial_capital;
    if (num_periods > 0) equity_curve[0] = config_.initial_capital;

    for (int t = 1; t < num_periods; ++t) {
        const double timestamp = (t < timestamps.size()) ? timestamps[t] : 0.0;
        Eigen::VectorXd current_prices = prices.row(t);
        Eigen::VectorXd current_signals = signals.row(t);
        Eigen::VectorXd prev_prices = prices.row(t-1);

        //calculating target positions (portfolio allocation)
        Eigen::VectorXd target_positions = current_signals.cwiseProduct(
            Eigen::VectorXd::Constant(num_assets, total_value)
        ).cwiseQuotient((current_prices.array() + 1e-12).matrix());
        
        //Position changes(trades to execute)
        Eigen::VectorXd position_changes = target_positions - positions;
        
        //execute trades with transaction costs and slippage
        double trade_costs = 0.0;

        #pragma omp parallel for reduction(+:trade_costs) if(num_assets > 10)
        for (int i = 0; i < num_assets; ++i) {
            if (std::abs(position_changes[i]) > 1e-6) {
                const double quantity = position_changes[i];
                const double price = current_prices[i];
                const double slippage_factor = config_.slippage * std::copysign(1.0, quantity);
                const double execution_price = price * (1.0 + slippage_factor);
                const double trade_value = std::abs(quantity * execution_price);
                const double commission = trade_value * config_.transaction_costs;
                const double slippage_cost = std::abs(quantity) * price * config_.slippage;
                trade_costs += commission + slippage_cost;
                if (omp_get_thread_num() == 0) {
                    Trade trade;
                    trade.timestamp = timestamp;
                    trade.symbol = symbols[i];
                    trade.quantity = quantity;
                    trade.price = execution_price;
                    trade.commission = commission;
                    trade.slippage_cost = slippage_cost;
                    trade.side = (quantity > 0) ? "BUY" : "SELL";
                    trade.pnl = 0.0;
                    #pragma omp critical
                    trades.push_back(trade);
                }
            }
        }

        positions = target_positions;
        cash -= trade_costs;

        Eigen::VectorXd price_changes = current_prices - prev_prices;
        (void)price_changes; 

        double position_value = positions.dot(current_prices);
        total_value = cash + position_value;

        peak_value = std::max(peak_value, total_value);
        double drawdown = (peak_value - total_value) / (peak_value + 1e-12);

        if (t < equity_curve.size()) equity_curve[t] = total_value;

        PortfolioState state;
        state.timestamp = timestamp;
        state.total_value = total_value;
        state.cash = cash;
        state.equity = position_value;
        state.leverage = position_value / (total_value + 1e-12);
        state.drawdown = drawdown;

        for (int i = 0; i < num_assets; ++i) {
            state.positions[symbols[i]] = positions[i];
            state.weights[symbols[i]] = (positions[i] * current_prices[i]) / (total_value + 1e-12);
        }

        states.push_back(state);
        prev_positions = positions;
    }
}

PerformanceMetrics BacktestingEngine::calculate_performance_metrics(
    const Eigen::VectorXd& equity_curve,
    const Eigen::VectorXd& timestamps,
    const std::vector<Trade>& trades
) const {
    PerformanceMetrics metrics{};

    if (equity_curve.size() < 2) return metrics;

    Eigen::VectorXd returns = calculate_returns(equity_curve);
    const double initial_value = equity_curve[0];
    const double final_value = equity_curve[equity_curve.size() - 1];
    metrics.total_return = (final_value - initial_value) / (initial_value + 1e-12);

    const double num_years = (timestamps[timestamps.size()-1] - timestamps[0]) / (365.25 * 24 * 3600);
    if (num_years > 0) {
        metrics.cagr = std::pow(final_value / (initial_value + 1e-12), 1.0 / num_years) - 1.0;
    } else {
        metrics.cagr = 0.0;
    }

    const double mean_return = returns.mean();
    metrics.volatility = std::sqrt(returns.array().square().mean() - mean_return * mean_return) * std::sqrt(252);
    metrics.sharpe_ratio = (metrics.volatility > 0) ? (metrics.cagr) / metrics.volatility : 0.0;

    Eigen::VectorXd negative_returns = returns.array().min(0);
    double downside_variance = negative_returns.array().square().mean();
    double downside_std = std::sqrt(downside_variance) * std::sqrt(252);
    metrics.sortino_ratio = (downside_std > 0) ? (metrics.cagr) / downside_std : 0.0;

    metrics.max_drawdown = calculate_max_drawdown(equity_curve);
    metrics.calmar_ratio = (std::abs(metrics.max_drawdown) > 0) ? metrics.cagr / std::abs(metrics.max_drawdown) : 0.0;

    metrics.var_95 = calculate_var(returns, 0.95);
    metrics.cvar_95 = calculate_cvar(returns, 0.95);

    metrics.total_trades = static_cast<int>(trades.size());

    if (!trades.empty()) {
        std::vector<double> trade_pnls;
        trade_pnls.reserve(trades.size());
        double total_volume = 0.0;
        int winning_trades = 0;
        double total_wins = 0.0;
        double total_losses = 0.0;
        for (const auto& trade : trades) {
            total_volume += std::abs(trade.quantity * trade.price);
            double pnl = trade.quantity * trade.price * 0.001;
            trade_pnls.push_back(pnl);
            if (pnl > 0) { winning_trades++; total_wins += pnl; } else { total_losses += std::abs(pnl); }
        }
        metrics.win_rate = static_cast<double>(winning_trades) / trades.size();
        metrics.profit_factor = (total_losses > 0) ? total_wins / total_losses : 0.0;
        metrics.average_win = (winning_trades > 0) ? total_wins / winning_trades : 0.0;
        metrics.average_loss = (trades.size() - winning_trades > 0) ? total_losses / (trades.size() - winning_trades) : 0.0;
        metrics.turnover = total_volume / ((timestamps.size() > 1 ? (timestamps[timestamps.size()-1] - timestamps[0]) / (365.25 * 24 * 3600) : 1.0) * final_value);
    }

    // Ulcer Index: sqrt(mean(((peak - equity)/peak)^2))
    if (equity_curve.size() > 0) {
        Eigen::VectorXd peaks(equity_curve.size());
        peaks[0] = equity_curve[0];
        for (int i = 1; i < equity_curve.size(); ++i) {
            peaks[i] = std::max(peaks[i-1], equity_curve[i]);
        }
        Eigen::VectorXd drawdowns = (peaks - equity_curve).array() / (peaks.array() + 1e-12);
        metrics.ulcer_index = std::sqrt(drawdowns.array().square().mean());
    } else {
        metrics.ulcer_index = 0.0;
    }

    return metrics;
}

Eigen::VectorXd BacktestingEngine::calculate_returns(const Eigen::VectorXd& prices) const {
    Eigen::VectorXd returns(prices.size() - 1);
    #pragma omp parallel for if(prices.size() > 1000)
    for (int i = 1; i < prices.size(); ++i) {
        returns[i-1] = (prices[i] - prices[i-1]) / (prices[i-1] + 1e-12);
    }
    return returns;
}

double BacktestingEngine::calculate_max_drawdown(const Eigen::VectorXd& equity_curve) const {
    double max_dd = 0.0;
    double peak = equity_curve[0];
    for (int i = 1; i < equity_curve.size(); ++i) {
        peak = std::max(peak, equity_curve[i]);
        double drawdown = (peak - equity_curve[i]) / (peak + 1e-12);
        max_dd = std::max(max_dd, drawdown);
    }
    return max_dd;
}

double BacktestingEngine::calculate_var(const Eigen::VectorXd& returns, double confidence) const {
    std::vector<double> sorted_returns(returns.data(), returns.data() + returns.size());
    std::sort(sorted_returns.begin(), sorted_returns.end());
    int index = static_cast<int>((1.0 - confidence) * sorted_returns.size());
    if (index < 0) index = 0;
    if (index >= static_cast<int>(sorted_returns.size())) index = static_cast<int>(sorted_returns.size()) - 1;
    return sorted_returns[index];
}

double BacktestingEngine::calculate_cvar(const Eigen::VectorXd& returns, double confidence) const {
    double var = calculate_var(returns, confidence);
    double sum = 0.0; int count = 0;
    for (int i = 0; i < returns.size(); ++i) {
        if (returns[i] <= var) { sum += returns[i]; count++; }
    }
    return (count > 0) ? sum / count : 0.0;
}

Eigen::VectorXd BacktestingEngine::calculate_rolling_sharpe(
    const Eigen::VectorXd& returns, 
    int window, 
    double risk_free_rate
) const {
    if (returns.size() < window || window <= 0) {
        return Eigen::VectorXd::Zero(0);
    }
    Eigen::VectorXd out(returns.size() - window + 1);
    for (int i = 0; i <= returns.size() - window; ++i) {
        Eigen::VectorXd seg = returns.segment(i, window);
        double mu = seg.mean() - risk_free_rate / 252.0;
        double sigma = std::sqrt((seg.array() - mu).square().mean()) * std::sqrt(252);
        out[i] = (sigma > 0.0) ? (mu / sigma) : 0.0;
    }
    return out;
}

Eigen::MatrixXd BacktestingEngine::calculate_rolling_metrics(
    const Eigen::VectorXd& equity_curve,
    int window_size
) const {
    if (equity_curve.size() < window_size || window_size <= 0) {
        return Eigen::MatrixXd::Zero(0,0);
    }
    int n = static_cast<int>(equity_curve.size()) - window_size + 1;
    Eigen::MatrixXd metrics(n, 2); // col0 = mean return, col1 = volatility
    Eigen::VectorXd returns = calculate_returns(equity_curve);
    Eigen::VectorXd rolling_sharpes = calculate_rolling_sharpe(returns, window_size);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd seg = returns.segment(i, window_size);
        double mu = seg.mean();
        double sigma = std::sqrt((seg.array() - mu).square().mean()) * std::sqrt(252);
        metrics(i, 0) = mu;
        metrics(i, 1) = sigma;
    }
    return metrics;
}

void BacktestingEngine::process_batch_parallel(
    const Eigen::MatrixXd& data,
    int start_idx,
    int end_idx,
    Eigen::VectorXd& results
) const {
    if (start_idx < 0) start_idx = 0;
    if (end_idx > data.rows()) end_idx = data.rows();
    int len = end_idx - start_idx;
    if (len <= 0) return;
    if (results.size() != len) results = Eigen::VectorXd::Zero(len);
    for (int i = 0; i < len; ++i) {
        // simple placeholder: compute sum of row as a proxy
        results[i] = data.row(start_idx + i).sum();
    }
}

namespace utils {

Eigen::MatrixXd numpy_to_eigen(const double* data, int rows, int cols) {
    return Eigen::Map<const Eigen::MatrixXd>(data, rows, cols);
}

void eigen_to_numpy(const Eigen::MatrixXd& matrix, double* output) {
    std::copy(matrix.data(), matrix.data() + matrix.size(), output);
}

Eigen::VectorXd moving_average(const Eigen::VectorXd& data, int window) {
    Eigen::VectorXd ma = Eigen::VectorXd::Zero(data.size());
    for (int i = window - 1; i < data.size(); ++i) {
        ma[i] = data.segment(i - window + 1, window).mean();
    }
    return ma;
}

Eigen::VectorXd rsi(const Eigen::VectorXd& data, int window) {
    Eigen::VectorXd rsi = Eigen::VectorXd::Zero(data.size());
    Eigen::VectorXd changes = data.tail(data.size() - 1) - data.head(data.size() - 1);
    for (int i = window; i < data.size(); ++i) {
        Eigen::VectorXd segment = changes.segment(i - window, window);
        double avg_gain = segment.cwiseMax(0).mean();
        double avg_loss = -segment.cwiseMin(0).mean();
        if (avg_loss > 0) {
            double rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    return rsi;
}

} // namespace utils
} // namespace backtesting