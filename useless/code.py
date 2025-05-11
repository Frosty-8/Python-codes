import pandas as pd
import yfinance as yf

# Define date range
start = "2020-01-01"
end = "2024-01-01"

# Fetch stock price data for Apple (AAPL)
df = yf.download("MSFT", start=start, end=end)

# Save to CSV
df.to_csv("time_series_data.csv")

print(df.head())
