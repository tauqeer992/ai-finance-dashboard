import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import ta
import requests
import feedparser
from openai import OpenAI
from datetime import datetime, timedelta

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📊 AI Personal Trading Tool")

symbol = st.sidebar.text_input("Asset Symbol (e.g. NVDA, BTC-USD)", "BTC-USD")

timeframe_option = st.sidebar.selectbox(
    "Timeframe",
    [
        "1 Day (Intraday)",
        "1 Week",
        "1 Month",
        "3 Months",
        "1 Year",
    ]
)

# =====================================================
# DATA LOADER (Yahoo Only)
# =====================================================
def load_data(symbol, timeframe):

    if timeframe == "1 Day (Intraday)":
        period = "7d"        # fetch more data for indicators
        interval = "1h"
    elif timeframe == "1 Week":
        period = "7d"
        interval = "4h"
    elif timeframe == "1 Month":
        period = "1mo"
        interval = "4h"
    elif timeframe == "3 Months":
        period = "3mo"
        interval = "1d"
    else:
        period = "1y"
        interval = "1d"

    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    return df


# =====================================================
# NEWS: FINNHUB + RSS FALLBACK
# =====================================================
def fetch_finnhub_news(symbol):
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
            return [item["headline"] for item in data[:5]]

    except Exception:
        pass

    return None


def fetch_rss_news(symbol):
    try:
        rss_url = f"https://news.google.com/rss/search?q={symbol}"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except Exception:
        return []


def fetch_news(symbol):
    headlines = fetch_finnhub_news(symbol)

    if headlines:
        return "\n".join([f"- {h}" for h in headlines])

    rss_headlines = fetch_rss_news(symbol)

    if rss_headlines:
        return "\n".join([f"- {h}" for h in rss_headlines])

    return "No recent news available."


# =====================================================
# LOAD DATA
# =====================================================
df = load_data(symbol, timeframe_option)

if df is None or df.empty:
    st.stop()

close_vals = df["Close"]
current_price = float(close_vals.iloc[-1])

# =====================================================
# FETCH NEWS
# =====================================================
news_summary = fetch_news(symbol)

# =====================================================
# CHART
# =====================================================
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

# =====================================================
# INDICATORS
# =====================================================
df["RSI"] = ta.momentum.RSIIndicator(close_vals).rsi()
macd_indicator = ta.trend.MACD(close_vals)

df["MACD"] = macd_indicator.macd()
df["MACD_SIGNAL"] = macd_indicator.macd_signal()

latest_rsi = df["RSI"].dropna().iloc[-1] if not df["RSI"].dropna().empty else None
latest_macd = df["MACD"].dropna().iloc[-1] if not df["MACD"].dropna().empty else None
latest_signal = df["MACD_SIGNAL"].dropna().iloc[-1] if not df["MACD_SIGNAL"].dropna().empty else None

# =====================================================
# NUMERIC TECHNICAL SCORING
# =====================================================

technical_score = 50  # neutral base

if latest_rsi is not None:
    if latest_rsi < 30:
        technical_score += 20
    elif latest_rsi < 40:
        technical_score += 10
    elif latest_rsi > 70:
        technical_score -= 20
    elif latest_rsi > 60:
        technical_score -= 10

if latest_macd is not None and latest_signal is not None:
    if latest_macd > latest_signal:
        technical_score += 15
    else:
        technical_score -= 15

technical_score = max(0, min(100, technical_score))


# =====================================================
# AI NEWS SENTIMENT → NUMERIC SCORE
# =====================================================

def get_news_sentiment_score(headlines):

    if not headlines:
        return 50, "Neutral"

    prompt = f"""
    Determine overall sentiment from these financial headlines:

    {headlines}

    Respond strictly in this format:
    Sentiment: Bullish
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=50
    )

    result = response.output_text.lower()

    if "bullish" in result:
        return 75, "Bullish"
    elif "bearish" in result:
        return 25, "Bearish"
    else:
        return 50, "Neutral"


news_items = fetch_finnhub_news(symbol)

if news_items:
    headlines = [item["headline"] for item in news_items]
else:
    headlines = []

news_score, news_sentiment = get_news_sentiment_score(headlines)


# =====================================================
# FINAL WEIGHTED CONFIDENCE
# =====================================================

final_score = int((technical_score * 0.7) + (news_score * 0.3))

if final_score > 65:
    final_signal = "BUY"
elif final_score < 35:
    final_signal = "SELL"
else:
    final_signal = "HOLD"


# =====================================================
# DISPLAY RESULTS
# =====================================================

st.subheader("📊 Multi-Factor Signal Engine")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Technical Score", f"{technical_score}/100")

with col2:
    st.metric("News Sentiment", news_sentiment)

with col3:
    st.metric("Final Confidence", f"{final_score}%")

# Progress Bar
st.progress(final_score / 100)

if final_signal == "BUY":
    st.success(f"🟢 BUY | Weighted Confidence: {final_score}%")
elif final_signal == "SELL":
    st.error(f"🔴 SELL | Weighted Confidence: {final_score}%")
else:
    st.warning(f"🟡 HOLD | Weighted Confidence: {final_score}%")
# =====================================================
# NEWS PANEL
# =====================================================
st.subheader("📰 Latest Financial News")
st.text(news_summary)

# =====================================================
# AI DECISION ENGINE
# =====================================================
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

# =====================================================
# CUSTOM AI QUERY
# =====================================================
st.subheader("💬 Ask the AI")

user_query = st.text_input("Ask about trend, macro outlook, ETF impact...")

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
