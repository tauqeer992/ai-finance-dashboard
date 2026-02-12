# =====================================
# PROFESSIONAL NEWSAPI INTEGRATION
# =====================================

news_summary = "News unavailable."

def fetch_news(query):
    try:
        api_key = st.secrets["NEWS_API_KEY"]

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"sortBy=publishedAt&"
            f"language=en&"
            f"pageSize=5&"
            f"apiKey={api_key}"
        )

        response = requests.get(url)
        data = response.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            if not articles:
                return "No recent news found."
            headlines = [f"- {article['title']}" for article in articles]
            return "\n".join(headlines)
        else:
            return f"News API error: {data}"

    except Exception as e:
        return f"News fetch error: {e}"

# Always assign safely
news_summary = fetch_news(symbol)
