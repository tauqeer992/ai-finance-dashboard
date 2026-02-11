# analysis/signals.py

def generate_signal(df):
    latest = df.iloc[-1]
    
    if latest['rsi'] < 30 and latest['macd'] > 0:
        return "BUY"
    elif latest['rsi'] > 70 and latest['macd'] < 0:
        return "SELL"
    else:
        return "HOLD"
