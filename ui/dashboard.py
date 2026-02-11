# ui/dashboard.py

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from ai.llm_engine import generate_trade_report

def create_dashboard(df, initial_report):

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    ))

    app = dash.Dash(__name__)

    app.layout = html.Div([

        html.H1("AI Trading Dashboard"),

        dcc.Graph(figure=fig),

        html.H2("AI Market Analysis"),
        html.Div(initial_report, id="ai-report",
                 style={"whiteSpace": "pre-wrap",
                        "border": "1px solid #ccc",
                        "padding": "15px"}),

        html.Hr(),

        html.H2("Ask AI Agent"),

        dcc.Textarea(
            id="user-input",
            placeholder="Ask about trend, risk, entry point...",
            style={"width": "100%", "height": 100}
        ),

        html.Button("Ask", id="ask-button"),

        html.Div(id="chat-output",
                 style={"whiteSpace": "pre-wrap",
                        "border": "1px solid #aaa",
                        "padding": "15px",
                        "marginTop": "20px"})
    ])

    @app.callback(
        Output("chat-output", "children"),
        Input("ask-button", "n_clicks"),
        State("user-input", "value"),
        prevent_initial_call=True
    )
    def handle_question(n_clicks, user_question):

        if not user_question:
            return "Please enter a question."

        # You can pass df info here for context
        context = df.tail(1).to_dict()

        response = generate_trade_report(
            symbol="BTC/USDT",
            signal="LIVE CONTEXT",
            indicators=context,
            news="Live market context",
        )

        return response

    return app
