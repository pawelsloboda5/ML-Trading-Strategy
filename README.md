# ML Trading Strategy

This project implements a machine learning-based trading strategy using Random Forest classification and technical indicators. It uses historical stock data to train a model and then applies this model in a backtesting environment to evaluate its performance.

## Features

- Data fetching using yfinance
- Feature engineering with various technical indicators
- Machine learning model training with RandomForestClassifier
- Hyperparameter tuning using RandomizedSearchCV
- Backtesting framework using backtrader
- Customizable trading strategy with adjustable parameters

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/ml_trading_strategy.git
cd ml_trading_strategy
pip install -r requirements.txt
```

## Usage

Run the main script to fetch data, train models, and backtest strategies:

```bash
python ml_trading_strategy/main.py
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.