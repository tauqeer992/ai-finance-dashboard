import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import ta
#import os
#from dotenv import load_dotenv
from openai import OpenAI

# =====================================
# CONFIG
# =====================================
#load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Pro AI Financial Analyst", layout="wide")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("🤖 Pro AI Financial Analyst")

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
# AUTO PERIOD + INTERVAL LOGIC
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
    st.error("No market data available. Try a different symbol.")
    st.stop()

# Flatten MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# Normalize Date column (handles Date / Datetime / index)
if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "Date"}, inplace=True)

if "index" in df.columns:
    df.rename(columns={"index": "Date"}, inplace=True)

df = df.dropna()

# =====================================
# NUMERIC ARRAYS (Safe Handling)
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
    st.title(f"📊 {symbol} Market Overview")

with col2:
    st.metric("Current Price", f"${current_price:,.2f}")

# =====================================
# CANDLESTICK CHART
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
# KPI METRICS
# =====================================
k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Period High", f"${float(high_vals.max()):,.2f}")

with k2:
    st.metric("Period Low", f"${float(low_vals.min()):,.2f}")

with k3:
    st.metric("Volume (Last)", f"{int(volume_vals[-1]):,}")

# =====================================
# AI ANALYSIS SECTION
# =====================================
st.markdown("---")
st.subheader("🧠 AI Financial Analyst")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask about trend, risk, entry, outlook...")

def ask_ai(question):
    latest_price = float(close_vals[-1])
    latest_rsi = float(df["RSI"].iloc[-1])

    prompt = f"""
    You are a professional financial market analyst.

    Asset: {symbol}
    Current Price: {latest_price}
    RSI: {latest_rsi}

    User Question:
    {question}

    Provide:
    - Technical analysis
    - Risk assessment
    - Short-term outlook
    - Confidence level (%)
    """

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=400
        )
        return response.output_text
    except Exception as e:
        return f"AI Error: {e}"

if user_question:
    reply = ask_ai(user_question)
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("AI", reply))

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 AI:** {message}")
