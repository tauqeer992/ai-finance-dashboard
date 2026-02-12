import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import ta
import requests
from openai import OpenAI

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="AI Personal Trading Tool", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =====================================
# SIDEBAR (DEFINE SYMBOL FIRST)
# =====================================
st.sidebar.title("📊 AI Personal Trading Tool")

symbol = st.sidebar.text_input("Asset Symbol (e.g. NVDA, BTC-USD)", "NVDA")

timeframe_option = st.sidebar.selectbox(
    "Timeframe",
    ["1 Day (Intraday)", "1 Month", "3 Months", "1 Year"]
)

if timeframe_option == "1 Day (Intraday)":
    period = "1d"
    interval = "60m"
elif timeframe_option == "1 Month":
    period = "1mo"
    interval = "1h"
elif timeframe_option == "3 Months":
    period = "3mo"
    interval = "1d"
else:
    period = "1y"
    interval = "1d"

# =====================================
# NEWS FUNCTION (DEFINE BEFORE USING)
# =====================================
def fetch_news(query):
    try:
        api_key = st.secrets["NEWS_API_KEY"]
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
        )
        response = requests.get(url)
        data = response.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            if not articles:
                return "No recent news found."
            return "\n".join([f"- {a['title']}" for a in articles])
        else:
            return "News unavailable."
    except Exception:
        return "News fetch error."

# NOW SAFE TO CALL
news_summary = fetch_news(symbol)

# =====================================
# LOAD MARKET DATA
# =====================================
df = yf.download(symbol, period=period, interval=interval, progress=False)

if df.empty:
    st.error("No market data available.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "Date"}, inplace=True)

close_vals = df["Close"]
current_price = float(close_vals.iloc[-1])

# =====================================
# CHART
# =====================================
st.title(f"📈 {symbol} Market Overview")
st.metric("Current Price", f"${current_price:,.2f}")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# =====================================
# RSI
# =====================================
df["RSI"] = ta.momentum.RSIIndicator(close_vals).rsi()
latest_rsi = df["RSI"].dropna().iloc[-1] if not df["RSI"].dropna().empty else None

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines"))
rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)
rsi_fig.update_layout(template="plotly_dark", height=200)
st.plotly_chart(rsi_fig, use_container_width=True)

# =====================================
# MACD
# =====================================
macd_indicator = ta.trend.MACD(close_vals)
df["MACD"] = macd_indicator.macd()
df["MACD_SIGNAL"] = macd_indicator.macd_signal()

latest_macd = df["MACD"].dropna().iloc[-1] if not df["MACD"].dropna().empty else None
latest_signal = df["MACD_SIGNAL"].dropna().iloc[-1] if not df["MACD_SIGNAL"].dropna().empty else None

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines"))
macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines"))
macd_fig.update_layout(template="plotly_dark", height=250)
st.plotly_chart(macd_fig, use_container_width=True)

# =====================================
# SIGNAL
# =====================================
signal = "HOLD"
confidence = 50

if latest_rsi and latest_macd and latest_signal:
    if latest_rsi < 35 and latest_macd > latest_signal:
        signal = "BUY"
        confidence = 75
    elif latest_rsi > 65 and latest_macd < latest_signal:
        signal = "SELL"
        confidence = 75

st.subheader("📊 Rule-Based Signal")

if signal == "BUY":
    st.success(f"🟢 BUY | Confidence: {confidence}%")
elif signal == "SELL":
    st.error(f"🔴 SELL | Confidence: {confidence}%")
else:
    st.warning(f"🟡 HOLD | Confidence: {confidence}%")

# =====================================
# NEWS PANEL
# =====================================
st.subheader("📰 Latest Financial News")
st.text(news_summary)
