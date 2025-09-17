#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "core.hpp"

namespace py = pybind11;
using namespace backtesting;

PYBIND11_MODULE(backtesting_core, m) {
    m.doc() = "High-performance C++ backtesting engine with Eigen and OpenMP";
    
    py::class_<BacktestConfig>(m, "BacktestConfig")
        .def(py::init<>())
        .def_readwrite("initial_capital", &BacktestConfig::initial_capital)
        .def_readwrite("slippage", &BacktestConfig::slippage)
        .def_readwrite("transaction_costs", &BacktestConfig::transaction_costs)
        .def_readwrite("margin_requirement", &BacktestConfig::margin_requirement)
        .def_readwrite("enable_shorting", &BacktestConfig::enable_shorting)
        .def_readwrite("enable_leverage", &BacktestConfig::enable_leverage)
        .def_readwrite("max_leverage", &BacktestConfig::max_leverage)
        .def_readwrite("num_threads", &BacktestConfig::num_threads);
    

    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("timestamp", &MarketData::timestamp)
        .def_readwrite("open", &MarketData::open)
        .def_readwrite("high", &MarketData::high)
        .def_readwrite("low", &MarketData::low)
        .def_readwrite("close", &MarketData::close)
        .def_readwrite("volume", &MarketData::volume)
        .def_readwrite("symbol", &MarketData::symbol);
    
    py::class_<Trade>(m, "Trade")
        .def(py::init<>())
        .def_readwrite("timestamp", &Trade::timestamp)
        .def_readwrite("symbol", &Trade::symbol)
        .def_readwrite("quantity", &Trade::quantity)
        .def_readwrite("price", &Trade::price)
        .def_readwrite("commission", &Trade::commission)
        .def_readwrite("slippage_cost", &Trade::slippage_cost)
        .def_readwrite("side", &Trade::side)
        .def_readwrite("pnl", &Trade::pnl);
    
    py::class_<PortfolioState>(m, "PortfolioState")
        .def(py::init<>())
        .def_readwrite("timestamp", &PortfolioState::timestamp)
        .def_readwrite("total_value", &PortfolioState::total_value)
        .def_readwrite("cash", &PortfolioState::cash)
        .def_readwrite("equity", &PortfolioState::equity)
        .def_readwrite("leverage", &PortfolioState::leverage)
        .def_readwrite("drawdown", &PortfolioState::drawdown)
        .def_readwrite("positions", &PortfolioState::positions)
        .def_readwrite("weights", &PortfolioState::weights);
    
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("total_return", &PerformanceMetrics::total_return)
        .def_readwrite("cagr", &PerformanceMetrics::cagr)
        .def_readwrite("volatility", &PerformanceMetrics::volatility)
        .def_readwrite("sharpe_ratio", &PerformanceMetrics::sharpe_ratio)
        .def_readwrite("sortino_ratio", &PerformanceMetrics::sortino_ratio)
        .def_readwrite("max_drawdown", &PerformanceMetrics::max_drawdown)
        .def_readwrite("calmar_ratio", &PerformanceMetrics::calmar_ratio)
        .def_readwrite("var_95", &PerformanceMetrics::var_95)
        .def_readwrite("cvar_95", &PerformanceMetrics::cvar_95)
        .def_readwrite("beta", &PerformanceMetrics::beta)
        .def_readwrite("alpha", &PerformanceMetrics::alpha)
        .def_readwrite("information_ratio", &PerformanceMetrics::information_ratio)
        .def_readwrite("win_rate", &PerformanceMetrics::win_rate)
        .def_readwrite("profit_factor", &PerformanceMetrics::profit_factor)
        .def_readwrite("average_win", &PerformanceMetrics::average_win)
        .def_readwrite("average_loss", &PerformanceMetrics::average_loss)
        .def_readwrite("total_trades", &PerformanceMetrics::total_trades)
        .def_readwrite("turnover", &PerformanceMetrics::turnover)
        .def_readwrite("ulcer_index", &PerformanceMetrics::ulcer_index);
    
    
    py::class_<BacktestingEngine>(m, "BacktestingEngine")
        .def(py::init<const BacktestConfig&>())
        .def("run_backtest", &BacktestingEngine::run_backtest,
             "Run vectorized backtest on multi-asset data",
             py::arg("prices"), py::arg("signals"), py::arg("symbols"), py::arg("timestamps"))
        .def("calculate_performance_metrics", &BacktestingEngine::calculate_performance_metrics,
             "Calculate comprehensive performance metrics",
             py::arg("equity_curve"), py::arg("timestamps"), py::arg("trades"))
        .def("calculate_rolling_metrics", &BacktestingEngine::calculate_rolling_metrics,
             "Calculate rolling performance metrics",
             py::arg("equity_curve"), py::arg("window_size"));
    
    m.def("run_optimized_backtest", 
        [](py::array_t<double> prices_np, 
           py::array_t<double> signals_np,
           const std::vector<std::string>& symbols,
           py::array_t<double> timestamps_np,
           const BacktestConfig& config) -> py::tuple {
            
            auto prices_buf = prices_np.request();
            auto signals_buf = signals_np.request();
            auto timestamps_buf = timestamps_np.request();
            
            if (prices_buf.ndim != 2 || signals_buf.ndim != 2 || timestamps_buf.ndim != 1) {
                throw std::runtime_error("Price and signal arrays must be 2D, timestamps must be 1D");
            }
            
            int num_periods = static_cast<int>(prices_buf.shape[0]);
            int num_assets = static_cast<int>(prices_buf.shape[1]);
            
            Eigen::Map<Eigen::MatrixXd> prices_eigen(
                static_cast<double*>(prices_buf.ptr), num_periods, num_assets);
            Eigen::Map<Eigen::MatrixXd> signals_eigen(
                static_cast<double*>(signals_buf.ptr), num_periods, num_assets);
            Eigen::Map<Eigen::VectorXd> timestamps_eigen(
                static_cast<double*>(timestamps_buf.ptr), num_periods);
            
            BacktestingEngine engine(config);
            auto result = engine.run_backtest(
                prices_eigen, signals_eigen, symbols, timestamps_eigen);
            
            Eigen::VectorXd equity_curve = std::get<0>(result);
            std::vector<Trade> trades = std::get<1>(result);
            std::vector<PortfolioState> states = std::get<2>(result);
            
            PerformanceMetrics metrics = engine.calculate_performance_metrics(
                equity_curve, timestamps_eigen, trades);
            
            py::array_t<double> equity_np(equity_curve.size());
            auto buf = equity_np.request();
            double* ptr = static_cast<double*>(buf.ptr);
            std::copy(equity_curve.data(), equity_curve.data() + equity_curve.size(), ptr);
            
            return py::make_tuple(equity_np, trades, states, metrics);
        },
        "High-performance backtesting function with numpy interface",
        py::arg("prices"), py::arg("signals"), py::arg("symbols"), 
        py::arg("timestamps"), py::arg("config"));
    
    auto utils_module = m.def_submodule("utils", "Utility functions");
    
    utils_module.def("moving_average", &utils::moving_average,
                     "Calculate moving average", py::arg("data"), py::arg("window"));
    
    utils_module.def("rsi", &utils::rsi,
                     "Calculate RSI indicator", py::arg("data"), py::arg("window"));
    
    m.attr("__version__") = "1";
    m.attr("__author__") = "Yash Gupta";
    

    #ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    #else
    m.attr("openmp_enabled") = false;
    #endif
}