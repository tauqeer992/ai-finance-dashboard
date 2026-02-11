# data_sources/forex.py
import requests

def get_forex_rate(pair="EURUSD"):
    url = f"https://api.exchangerate.host/latest?base={pair[:3]}&symbols={pair[3:]}"
    response = requests.get(url)
    return response.json()
