import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import ta
from openai import OpenAI

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="AI Personal Trading Tool", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("📊 AI Personal Trading Tool")

symbol = st.sidebar.text_input("Asset Symbol (Use BTC-USD for crypto)", "NVDA")

timeframe_option = st.sidebar.selectbox(
    "Timeframe",
    [
        "Intraday (1 Day)",
        "Short Term (1 Month)",
        "Medium Term (3 Months)",
        "Long Term (1 Year)"
    ]
)

# =====================================
# AUTO PERIOD + INTERVAL
# =====================================
if timeframe_option == "Intraday (1 Day)":
    period = "1d"
    interval = "60m"
elif timeframe_option == "Short Term (1 Month)":
    period = "1mo"
    interval = "1h"
elif timeframe_option == "Medium Term (3 Months)":
    period = "3mo"
    interval = "1d"
else:
    period = "1y"
    interval = "1d"

# =====================================
# LOAD DATA
# =====================================
try:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
except Exception as e:
    st.error(f"Data download error: {e}")
    st.stop()

if df.empty:
    st.error("No market data available.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "Date"}, inplace=True)

df = df.dropna()

# =====================================
# NUMERIC ARRAYS
# =====================================
open_vals = df["Open"].values
high_vals = df["High"].values
low_vals = df["Low"].values
close_vals = df["Close"].values
volume_vals = df["Volume"].values

current_price = float(close_vals[-1])

# =====================================
# HEADER
# =====================================
col1, col2 = st.columns([3, 1])

with col1:
    st.title(f"📈 {symbol} Market Overview")

with col2:
    st.metric("Current Price", f"${current_price:,.2f}")

# =====================================
# CANDLESTICK
# =====================================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=open_vals,
    high=high_vals,
    low=low_vals,
    close=close_vals,
    name="Price"
))

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# =====================================
# RSI
# =====================================
df["RSI"] = ta.momentum.RSIIndicator(pd.Series(close_vals)).rsi()

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["RSI"],
    mode="lines",
    name="RSI"
))

rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)
rsi_fig.update_layout(template="plotly_dark", height=200)
st.plotly_chart(rsi_fig, use_container_width=True)

# =====================================
# MACD
# =====================================
macd_indicator = ta.trend.MACD(pd.Series(close_vals))

df["MACD"] = macd_indicator.macd()
df["MACD_SIGNAL"] = macd_indicator.macd_signal()
df["MACD_HIST"] = macd_indicator.macd_diff()

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MACD"],
    mode="lines",
    name="MACD"
))
macd_fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MACD_SIGNAL"],
    mode="lines",
    name="Signal"
))
macd_fig.add_trace(go.Bar(
    x=df["Date"],
    y=df["MACD_HIST"],
    name="Histogram"
))

macd_fig.update_layout(template="plotly_dark", height=250)
st.plotly_chart(macd_fig, use_container_width=True)

# =====================================
# SAFE INDICATOR EXTRACTION
# =====================================
latest_rsi = None
latest_macd = None
latest_signal = None

if not df["RSI"].dropna().empty:
    latest_rsi = float(df["RSI"].dropna().iloc[-1])

if not df["MACD"].dropna().empty:
    latest_macd = float(df["MACD"].dropna().iloc[-1])

if not df["MACD_SIGNAL"].dropna().empty:
    latest_signal = float(df["MACD_SIGNAL"].dropna().iloc[-1])

# =====================================
# RULE-BASED SIGNAL
# =====================================
signal = "HOLD"
confidence = 50

if latest_rsi is not None and latest_macd is not None and latest_signal is not None:

    if latest_rsi < 35 and latest_macd > latest_signal:
        signal = "BUY"
        confidence = 75

    elif latest_rsi > 65 and latest_macd < latest_signal:
        signal = "SELL"
        confidence = 75

st.markdown("---")
st.subheader("📊 Rule-Based Trading Signal")

if signal == "BUY":
    st.success(f"🟢 BUY | Confidence: {confidence}%")
elif signal == "SELL":
    st.error(f"🔴 SELL | Confidence: {confidence}%")
else:
    st.warning(f"🟡 HOLD | Confidence: {confidence}%")

# =====================================
# AI DECISION ENGINE
# =====================================
def ask_ai_decision():
    prompt = f"""
    You are a professional trading analyst.

    Market Data:
    Price: {current_price}
    RSI: {latest_rsi if latest_rsi is not None else "Not Available"}
    MACD: {latest_macd if latest_macd is not None else "Not Available"}
    MACD Signal: {latest_signal if latest_signal is not None else "Not Available"}

    Respond in this format:

    Decision: BUY or SELL or HOLD
    Confidence: number between 0 and 100
    Reason: short explanation
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=200
    )

    return response.output_text

st.markdown("---")
st.subheader("🤖 AI Decision Engine")

if st.button("Generate AI Decision"):
    try:
        ai_result = ask_ai_decision()
        st.info(ai_result)
    except Exception as e:
        st.error(f"AI Error: {e}")
