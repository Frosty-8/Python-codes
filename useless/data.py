import yfinance as yf

ticker = 'MSFT'

data = yf.download(ticker, start='1997-01-01', end='2024-12-31')

data = data.reset_index()

data.to_csv(f'{ticker}.csv',index=False)