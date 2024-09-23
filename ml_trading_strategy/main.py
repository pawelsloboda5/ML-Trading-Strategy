# main.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import backtrader as bt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler



def get_data(tickers, start_date, end_date):
    """Fetches historical data for the specified tickers and adds technical indicators."""
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    if len(tickers) == 1:
        df = data.copy()
        df.columns = pd.MultiIndex.from_product([tickers, df.columns])
    else:
        df = data.copy()
    
    for ticker in tickers:
        # Simple Moving Averages
        df[(ticker, 'SMA_20')] = df[(ticker, 'Close')].rolling(window=20).mean()
        df[(ticker, 'SMA_50')] = df[(ticker, 'Close')].rolling(window=50).mean()
        
        # Exponential Moving Average
        df[(ticker, 'EMA_20')] = df[(ticker, 'Close')].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index
        delta = df[(ticker, 'Close')].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[(ticker, 'RSI')] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df[(ticker, 'Close')].ewm(span=12, adjust=False).mean()
        exp2 = df[(ticker, 'Close')].ewm(span=26, adjust=False).mean()
        df[(ticker, 'MACD')] = exp1 - exp2
        df[(ticker, 'MACD_Signal')] = df[(ticker, 'MACD')].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df[(ticker, 'BB_middle')] = df[(ticker, 'Close')].rolling(window=20).mean()
        df[(ticker, 'BB_std')] = df[(ticker, 'Close')].rolling(window=20).std()
        df[(ticker, 'BB_upper')] = df[(ticker, 'BB_middle')] + (df[(ticker, 'BB_std')] * 2)
        df[(ticker, 'BB_lower')] = df[(ticker, 'BB_middle')] - (df[(ticker, 'BB_std')] * 2)
        
        # Average True Range (ATR)
        high_low = df[(ticker, 'High')] - df[(ticker, 'Low')]
        high_close = np.abs(df[(ticker, 'High')] - df[(ticker, 'Close')].shift())
        low_close = np.abs(df[(ticker, 'Low')] - df[(ticker, 'Close')].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df[(ticker, 'ATR')] = true_range.rolling(window=14).mean()
        
        # Add momentum
        df[(ticker, 'Momentum')] = df[(ticker, 'Close')] - df[(ticker, 'Close')].shift(5)
        
        # Add rate of change
        df[(ticker, 'ROC')] = df[(ticker, 'Close')].pct_change(5)
        
        # Add Stochastic Oscillator
        low_14 = df[(ticker, 'Low')].rolling(window=14).min()
        high_14 = df[(ticker, 'High')].rolling(window=14).max()
        df[(ticker, 'Stoch_K')] = 100 * (df[(ticker, 'Close')] - low_14) / (high_14 - low_14)
        df[(ticker, 'Stoch_D')] = df[(ticker, 'Stoch_K')].rolling(window=3).mean()
    
    return df

def prepare_features(df, ticker):
    features = pd.DataFrame(index=df.index)
    features['open'] = df[(ticker, 'Open')]
    features['high'] = df[(ticker, 'High')]
    features['low'] = df[(ticker, 'Low')]
    features['close'] = df[(ticker, 'Close')]
    features['volume'] = df[(ticker, 'Volume')]
    features['SMA_20'] = df[(ticker, 'SMA_20')]
    features['SMA_50'] = df[(ticker, 'SMA_50')]
    features['EMA_20'] = df[(ticker, 'EMA_20')]
    features['RSI'] = df[(ticker, 'RSI')]
    features['MACD'] = df[(ticker, 'MACD')]
    features['MACD_Signal'] = df[(ticker, 'MACD_Signal')]
    features['BB_upper'] = df[(ticker, 'BB_upper')]
    features['BB_lower'] = df[(ticker, 'BB_lower')]
    features['ATR'] = df[(ticker, 'ATR')]
    features['Momentum'] = df[(ticker, 'Momentum')]
    features['ROC'] = df[(ticker, 'ROC')]
    features['Stoch_K'] = df[(ticker, 'Stoch_K')]
    features['Stoch_D'] = df[(ticker, 'Stoch_D')]
    
    # Use a 1-day price change for the target
    features['Target'] = np.where(df[(ticker, 'Close')] > df[(ticker, 'Close')].shift(1), 1, 0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.drop(['open', 'high', 'low', 'close', 'volume', 'Target'], axis=1))
    scaled_df = pd.DataFrame(scaled_features, columns=features.drop(['open', 'high', 'low', 'close', 'volume', 'Target'], axis=1).columns, index=features.index)
    scaled_df['Target'] = features['Target']
    
    return features, scaled_df.dropna()

def train_model(features):
    """Train a Random Forest model on the given features with hyperparameter tuning."""
    X = features.drop('Target', axis=1)
    y = features['Target']
    
    print("Training data shape:", X.shape)
    print("Training target distribution:", y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    return best_model, feature_importance

class MLStrategy(bt.Strategy):
    params = (
        ('ticker', None),
        ('threshold', 0.7),
        ('feature_means', None),
        ('feature_stds', None),
        ('stop_loss', 0.04),
        ('take_profit', 0.06),
        ('trailing_stop', 0.05),
        ('min_hold_period', 2),
        ('max_open_positions', 5),
        ('feature_importance', None),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('volume_factor', 1.2),
        ('atr_factor', 1.1),
        ('min_buy_score', 3)  # Minimum score required for a buy signal
    )

    def __init__(self):
        self.model = models[self.params.ticker]
        self.dataclose = self.datas[0].close
        self.datavolume = self.datas[0].volume
        self.order = None
        self.open_positions = []
        self.initial_cash = self.broker.getvalue()
        self.current_value = self.initial_cash
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
            'total_return': 0
        }

        # Add indicators
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.datas[0], period=50)
        self.ema20 = bt.indicators.ExponentialMovingAverage(self.datas[0], period=20)
        self.rsi = bt.indicators.RSI(self.datas[0])
        self.macd = bt.indicators.MACD(self.datas[0])
        self.bbands = bt.indicators.BollingerBands(self.datas[0])
        self.atr = bt.indicators.ATR(self.datas[0])
        self.stoch = bt.indicators.StochasticFull(self.datas[0])
        self.volume_sma = bt.indicators.SimpleMovingAverage(self.datavolume, period=20)

    def next(self):
        self.current_value = self.broker.getvalue()

        features = self.get_features()
        scaled_features = (features - self.p.feature_means) / self.p.feature_stds
        prob = self.model.predict_proba(scaled_features)[0][1]

        # Calculate buy score
        buy_score = 0
        buy_score += 1 if prob > self.p.threshold else 0
        buy_score += 1 if self.sma20[0] > self.sma50[0] else 0
        buy_score += 1 if self.rsi[0] < self.p.rsi_upper else 0
        buy_score += 1 if self.dataclose[0] > self.bbands.lines.mid[0] else 0
        buy_score += 1 if self.macd.lines.macd[0] > self.macd.lines.signal[0] else 0
        buy_score += 1 if self.datavolume[0] > self.volume_sma[0] * self.p.volume_factor else 0
        buy_score += 1 if self.atr[0] > self.atr[-1] * self.p.atr_factor else 0

        buy_signals = buy_score >= self.p.min_buy_score

        sell_signals = (
            prob < 0.5 or
            self.rsi[0] > self.p.rsi_upper or
            self.dataclose[0] < self.bbands.lines.bot[0] or
            self.macd.lines.macd[0] < self.macd.lines.signal[0]
        )

        self.log(f"Close: {self.dataclose[0]:.2f}, Prob: {prob:.2f}, Buy Score: {buy_score}, "
                 f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}, RSI: {self.rsi[0]:.2f}, "
                 f"MACD: {self.macd.macd[0]:.4f}, Signal: {self.macd.signal[0]:.4f}")

        # Check for closing positions
        for position in self.open_positions[:]:
            days_held = (self.data.datetime.date(0) - position['date']).days
            profit_loss = (self.dataclose[0] / position['price'] - 1)

            if (sell_signals or
                (days_held >= self.p.min_hold_period and (
                    profit_loss <= -self.p.stop_loss or 
                    profit_loss >= self.p.take_profit or
                    self.dataclose[0] <= position['price'] * (1 - self.p.trailing_stop)
                ))
            ):
                self.close(self.datas[0])
                self.log(f"SELL EXECUTED, Price: {self.dataclose[0]:.2f}, Profit: {profit_loss:.2%}")
                self.results['total_trades'] += 1
                if profit_loss > 0:
                    self.results['winning_trades'] += 1
                    self.results['total_profit'] += profit_loss
                else:
                    self.results['losing_trades'] += 1
                    self.results['total_loss'] += profit_loss
                
                self.current_value = self.broker.getvalue()
                self.results['total_return'] = (self.current_value / self.initial_cash - 1) * 100
                
                self.open_positions.remove(position)

        # Check for opening new positions
        if buy_signals and len(self.open_positions) < self.p.max_open_positions:
            size = int(self.broker.getcash() * 0.02 / self.dataclose[0])  # Use 2% of available cash
            self.buy(size=size)
            self.open_positions.append({
                'price': self.dataclose[0],
                'date': self.data.datetime.date(0)
            })
            self.log(f"BUY EXECUTED, Price: {self.dataclose[0]:.2f}, Size: {size}")

    def stop(self):
        # Calculate final results
        self.current_value = self.broker.getvalue()
        self.results['final_value'] = self.current_value
        self.results['total_return'] = (self.current_value / self.initial_cash - 1) * 100
        self.results['accuracy'] = self.results['winning_trades'] / self.results['total_trades'] if self.results['total_trades'] > 0 else 0

        print('==== Strategy Finished ====')
        print(f'Final Portfolio Value: {self.results["final_value"]:.2f}')
        print(f'Total Return: {self.results["total_return"]:.2f}%')
        print(f'Total Trades: {self.results["total_trades"]}')
        print(f'Winning Trades: {self.results["winning_trades"]}')
        print(f'Losing Trades: {self.results["losing_trades"]}')
        print(f'Accuracy: {self.results["accuracy"]:.2f}')

    def get_features(self):
        return pd.DataFrame({
            'SMA_20': self.sma20[0],
            'SMA_50': self.sma50[0],
            'EMA_20': self.ema20[0],
            'RSI': self.rsi[0],
            'MACD': self.macd.macd[0],
            'MACD_Signal': self.macd.signal[0],
            'BB_upper': self.bbands.top[0],
            'BB_lower': self.bbands.bot[0],
            'ATR': self.atr[0],
            'Momentum': self.dataclose[0] - self.dataclose[-5],
            'ROC': (self.dataclose[0] - self.dataclose[-5]) / self.dataclose[-5] * 100,
            'Stoch_K': self.stoch.percK[0],
            'Stoch_D': self.stoch.percD[0]
        }, index=[0])

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

def run_backtest(ticker, data, scaled_data, feature_means, feature_stds, feature_importance):
    results = {}
    for threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
        for min_buy_score in [3, 4, 5]:
            cerebro = bt.Cerebro()
            
            data_feed = bt.feeds.PandasData(
                dataname=data,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1
            )
            cerebro.adddata(data_feed)
            cerebro.addstrategy(MLStrategy, 
                                ticker=ticker, 
                                threshold=threshold, 
                                feature_means=feature_means, 
                                feature_stds=feature_stds,
                                feature_importance=feature_importance,
                                min_buy_score=min_buy_score)
            
            initial_cash = 100000.0
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.001)
            
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            
            print(f'\nStarting Portfolio Value (threshold {threshold}, min_buy_score {min_buy_score}): {cerebro.broker.getvalue():.2f}')
            run_results = cerebro.run()
            
            strat = run_results[0]
            final_value = cerebro.broker.getvalue()
            
            print(f'Final Portfolio Value (threshold {threshold}, min_buy_score {min_buy_score}): {final_value:.2f}')
            
            sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
            drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            
            results[(threshold, min_buy_score)] = strat.results.copy()
            results[(threshold, min_buy_score)]['sharpe_ratio'] = sharpe_ratio
            results[(threshold, min_buy_score)]['max_drawdown'] = drawdown
            results[(threshold, min_buy_score)]['final_value'] = final_value
            results[(threshold, min_buy_score)]['total_return'] = ((final_value / initial_cash) - 1) * 100
    
    return results


if __name__ == "__main__":
    tickers_to_compare = ['AMD']
    start_date = '2023-01-01'
    end_date = '2024-09-20'

    print("Fetching data...")
    price_data = get_data(tickers_to_compare, start_date, end_date)
    print(f"Data range: {price_data.index[0]} to {price_data.index[-1]}")
    
    print("Preparing features...")
    all_features = {}
    all_scaled_features = {}
    for ticker in tickers_to_compare:
        all_features[ticker], all_scaled_features[ticker] = prepare_features(price_data, ticker)
        
        # Ensure the index is datetime
        all_features[ticker].index = pd.to_datetime(all_features[ticker].index)
        all_scaled_features[ticker].index = pd.to_datetime(all_scaled_features[ticker].index)
        
        # Sort the data by date
        all_features[ticker] = all_features[ticker].sort_index()
        all_scaled_features[ticker] = all_scaled_features[ticker].sort_index()
    
    print("\nSample of features for", tickers_to_compare[0])
    print(all_scaled_features[tickers_to_compare[0]].head())

    print("\nTraining model...")
    models = {}
    feature_means = {}
    feature_stds = {}
    for ticker in tickers_to_compare:
        print(f"\nTraining model for {ticker}...")
        model, feature_importance = train_model(all_scaled_features[ticker])
        models[ticker] = model
        
        X = all_scaled_features[ticker].drop('Target', axis=1)
        feature_means[ticker] = X.mean()
        feature_stds[ticker] = X.std()
        
    print("\nRunning backtest...")
    for ticker in tickers_to_compare:
        print(f"\nRunning backtest for {ticker}...")
        results = run_backtest(ticker, all_features[ticker], all_scaled_features[ticker], 
                               feature_means[ticker], feature_stds[ticker], feature_importance)
        
        print(f"\nBacktest results for {ticker}:")
        for threshold, result in results.items():
            print(f"\nResults for threshold {threshold}:")
            for key, value in result.items():
                print(f"{key}: {value}")






