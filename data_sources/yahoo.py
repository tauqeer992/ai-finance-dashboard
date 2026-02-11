# data_sources/yahoo.py
import yfinance as yf
import pandas as pd

def get_stock_data(symbol="AAPL", period="6mo", interval="1d"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df
