#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <omp.h>

namespace backtesting {

struct BacktestConfig {
    double initial_capital = 1000000.0;
    double slippage = 0.001;
    double transaction_costs = 0.0005;
    double margin_requirement = 1.0;
    bool enable_shorting = true;
    bool enable_leverage = false;
    double max_leverage = 1.0;
    int num_threads = 1;
};

struct MarketData {
    double timestamp;
    double open;
    double high;
    double low; 
    double close;
    double volume;
    std::string symbol;
};

struct Trade {
    double timestamp;
    std::string symbol;
    double quantity;
    double price;
    double commission;
    double slippage_cost;
    std::string side;  
    double pnl;
};

struct PortfolioState {
    double timestamp;
    double total_value;
    double cash;
    double equity;
    double leverage;
    double drawdown;
    std::unordered_map<std::string, double> positions;
    std::unordered_map<std::string, double> weights;
};

struct PerformanceMetrics {
    double total_return = 0.0;
    double cagr = 0.0;
    double volatility = 0.0;
    double sharpe_ratio = 0.0;
    double sortino_ratio = 0.0;
    double max_drawdown = 0.0;
    double calmar_ratio = 0.0;
    double var_95 = 0.0;
    double cvar_95 = 0.0;
    double beta = 0.0;
    double alpha = 0.0;
    double information_ratio = 0.0;
    double win_rate = 0.0;
    double profit_factor = 0.0;
    double average_win = 0.0;
    double average_loss = 0.0;
    int total_trades = 0;
    double turnover = 0.0;
    double ulcer_index = 0.0;
};

class BacktestingEngine {
public:
    explicit BacktestingEngine(const BacktestConfig& config);
    ~BacktestingEngine();
    
    std::tuple<Eigen::VectorXd, std::vector<Trade>, std::vector<PortfolioState>>
    run_backtest(
        const Eigen::MatrixXd& prices,
        const Eigen::MatrixXd& signals,
        const std::vector<std::string>& symbols,
        const Eigen::VectorXd& timestamps
    );
    
    PerformanceMetrics calculate_performance_metrics(
        const Eigen::VectorXd& equity_curve,
        const Eigen::VectorXd& timestamps,
        const std::vector<Trade>& trades
    ) const;
    
    Eigen::MatrixXd calculate_rolling_metrics(
        const Eigen::VectorXd& equity_curve,
        int window_size
    ) const;

private:
    BacktestConfig config_;
    std::unique_ptr<class BacktestingEngineImpl> impl_;
    
    
    Eigen::VectorXd calculate_returns(const Eigen::VectorXd& prices) const;
    Eigen::VectorXd calculate_rolling_sharpe(
        const Eigen::VectorXd& returns, 
        int window, 
        double risk_free_rate = 0.0
    ) const;
    double calculate_max_drawdown(const Eigen::VectorXd& equity_curve) const;
    double calculate_var(const Eigen::VectorXd& returns, double confidence) const;
    double calculate_cvar(const Eigen::VectorXd& returns, double confidence) const;
    
    
    void simulate_portfolio_vectorized(
        const Eigen::MatrixXd& prices,
        const Eigen::MatrixXd& signals,
        const std::vector<std::string>& symbols,
        const Eigen::VectorXd& timestamps,
        Eigen::VectorXd& equity_curve,
        std::vector<Trade>& trades,
        std::vector<PortfolioState>& states
    ) const;
    
    
    void process_batch_parallel(
        const Eigen::MatrixXd& data,
        int start_idx,
        int end_idx,
        Eigen::VectorXd& results
    ) const;
};

namespace utils {
    Eigen::MatrixXd numpy_to_eigen(const double* data, int rows, int cols);
    
    void eigen_to_numpy(const Eigen::MatrixXd& matrix, double* output);
    
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<std::string>>
    align_market_data(const std::vector<MarketData>& data);
    
    Eigen::VectorXd moving_average(const Eigen::VectorXd& data, int window);
    Eigen::VectorXd bollinger_bands(const Eigen::VectorXd& data, int window, double num_std);
    Eigen::VectorXd rsi(const Eigen::VectorXd& data, int window);
    Eigen::VectorXd macd(const Eigen::VectorXd& data, int fast, int slow, int signal);
}

}