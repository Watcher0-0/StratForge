Basically this project is a Backtesting Engine for backtesting the quantitative trading strategies with a web and command line interface (using streamlit this time for web).


## Features are-

- Very fast execution due to C++ integration along with python.
- Web demo based on streamlit.
- Supports multiple instruments and indices.
- Includes transaction cost and slippage.
- Visualize strategies in real time.
- You can put your dataset apart from the builtin dataset and backtest on that too.
- You can use your own custom strategies and also download the results.


### Requirements to run this
- Python 3.11 or higher versions..
- C++ compiler(GCC 9+,Clang 10+ or MSVC 2019+)
- CMake 3.12+

now the next steps are 

### To setup and install you need to follow the below given steps

1. Install system dependencies-
```bash
#for Debian based like ubuntu
sudo apt install build-essential cmake libeigen3-dev libgomp1

# for Arch based systems
sudo pacman -S base-devel cmake eigen libgomp

```

2. Install the requirements-
```bash
pip install -r requirements.txt
```

3. Build the project:
```bash
python setup.py build_ext --inplace
```

### To use the web interface

Launch the streamlit web application..or just type this command in the terminal after installing the requirements
```bash
streamlit run streamlit_app.py
```

### If you want to use the command line interface

For automated testing or batch processing use the CLI to run it..
```bash
python main.py --strategy momentum
```

Run with custom parameters like whatever you want
```bash
python main.py --strategy mean_reversion --initial_capital 2000000 --slippage 0.002
```
The outcomes will be stored in the `results` directory which will contain the following things like -
- Performance metrics will be saved in json format
- Trade list in csv file
- Performance charts ofcourse in png

The results will also be available via download from the web interface.

### Settings on the web interface

In the streamlit web interface you can do a number of things including the following: 
- choose your trading strategy (four are builtin) 
- specify your initial capital and position sizing                     
- specify your risk factors for e.g. slippage and transaction costs
- upload and preprocess your data
- you can also switch the dataset and put your data too

### CLI parameters 
for command line usage-
- `--strategy`- use to run the strategy(like momentum,mean_reversion)
- `--data`- input data file path
- `--initial_capital`- starting capital(default is 1,000,000)
- `--slippage`- slippage factor (default is 0.001)
- `--transaction_costs`- trading costs(default is 0.0005)

### Data format
Input data should be a csv file with columns same as -  date,symbol, open,high,low,close,volume. 
You can also drag and drop the csv file.

## Performance metrics

This engine calculates trading metrics including:
- returns i.e. Total Return and CAGR.
- risk measures (like the sharpe ratio,max drawdown and VaR).
- trade statistics i.e win rate and profit factor.

## If you want to do testing

To run the test suite type - 
```bash
python -m pytest tests/
```

## Also there is four builtin strategies if you want to test
- Momentum Strategy.
- Black Scholes Strategy(Mainly this is for future and options but it is modified for the stocks and indices).
- Mean Reversion Strategy and 
- Volatility scaling trend strategy.