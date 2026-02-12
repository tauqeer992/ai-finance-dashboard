import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import ta
import requests
from openai import OpenAI
from datetime import datetime, timedelta

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="AI Personal Trading Tool", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =====================================
# SIDEBAR
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
# NEWS FUNCTION (Finnhub - Cloud Safe)
# =====================================
def fetch_news(symbol):
    try:
        api_key = st.secrets["FINNHUB_API_KEY"]

        today = datetime.today()
        past = today - timedelta(days=7)

        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={symbol}&"
            f"from={past.strftime('%Y-%m-%d')}&"
            f"to={today.strftime('%Y-%m-%d')}&"
            f"token={api_key}"
        )

        response = requests.get(url)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            headlines = [f"- {item['headline']}" for item in data[:5]]
            return "\n".join(headlines)
        else:
            return "No recent news found."

    except Exception:
        return "News unavailable."

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
# FETCH NEWS
# =====================================
news_summary = fetch_news(symbol)

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

# =====================================
# MACD
# =====================================
macd_indicator = ta.trend.MACD(close_vals)
df["MACD"] = macd_indicator.macd()
df["MACD_SIGNAL"] = macd_indicator.macd_signal()

latest_macd = df["MACD"].dropna().iloc[-1] if not df["MACD"].dropna().empty else None
latest_signal = df["MACD_SIGNAL"].dropna().iloc[-1] if not df["MACD_SIGNAL"].dropna().empty else None

# =====================================
# RULE-BASED SIGNAL
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

# =====================================
# AI DECISION ENGINE
# =====================================
def ask_ai_decision():
    prompt = f"""
    You are a professional trading analyst.

    Technical Data:
    Price: {current_price}
    RSI: {latest_rsi}
    MACD: {latest_macd}
    MACD Signal: {latest_signal}

    Recent News:
    {news_summary}

    Provide:
    Decision: BUY or SELL or HOLD
    Confidence: 0-100
    Reason: short explanation
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=300
    )

    return response.output_text

st.subheader("🤖 AI Decision Engine")

if st.button("Generate AI Decision"):
    st.info(ask_ai_decision())

# =====================================
# CUSTOM AI QUERY
# =====================================
st.subheader("💬 Ask the AI")

user_query = st.text_input("Ask about trend, risk, outlook, macro impact...")

if st.button("Ask AI"):
    if user_query:
        prompt = f"""
        Asset: {symbol}
        Price: {current_price}
        RSI: {latest_rsi}
        MACD: {latest_macd}

        News:
        {news_summary}

        User Question:
        {user_query}

        Provide professional trading analysis.
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=400
        )

        st.info(response.output_text)
