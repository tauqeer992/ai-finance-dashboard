from data_sources.binance import get_crypto_data
from analysis.indicators import add_indicators
from analysis.signals import generate_signal
from ai.llm_engine import generate_trade_report
from ui.dashboard import create_dashboard

symbol = "BTC/USDT"

df = get_crypto_data(symbol)
df = add_indicators(df)

signal = generate_signal(df)

indicators = df.tail(1).to_dict()
news = "Latest market conditions"

report = generate_trade_report(symbol, signal, indicators, news)

app = create_dashboard(df, report)

if __name__ == "__main__":
    app.run(debug=True)
