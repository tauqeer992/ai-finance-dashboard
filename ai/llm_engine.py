# ai/llm_engine.py

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_trade_report(symbol, signal, indicators, news, user_question=None):

    prompt = f"""
    Asset: {symbol}
    Indicators: {indicators}
    News: {news}

    User Question:
    {user_question}

    Provide professional trading analysis.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=600
    )

    return response.output_text
