import unittest
from ml_trading_strategy import main

class TestMain(unittest.TestCase):
    def test_get_data(self):
        tickers = ['AAPL']
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        data = main.get_data(tickers, start_date, end_date)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()