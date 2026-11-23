"""SanjoTracker dashboard entry point.

Run from the project root:  streamlit run app/Home.py
Analysis and alerts only: nothing here places orders.
"""

import streamlit as st

st.set_page_config(
    page_title="SanjoTracker", page_icon=":material/candlestick_chart:", layout="wide"
)

page = st.navigation(
    [
        st.Page(
            "app_pages/alerts.py",
            title="Alerts",
            icon=":material/notifications:",
            default=True,
            help="Live alert feed (refreshes every 60 s), market regime and the watchlist.",
        ),
        st.Page(
            "app_pages/charts.py",
            title="Charts",
            icon=":material/candlestick_chart:",
            help="Candlesticks with S/R zones, pivots, breakout markers and indicators.",
        ),
        st.Page(
            "app_pages/scanner.py",
            title="Scanner",
            icon=":material/radar:",
            help="Breakouts, retests and symbols near strong levels right now, with filters.",
        ),
        st.Page(
            "app_pages/backtest.py",
            title="Backtest",
            icon=":material/query_stats:",
            help="Run a strategy with costs: equity, drawdown, trades, regimes, parameter heatmap.",
        ),
        st.Page(
            "app_pages/paper.py",
            title="Paper trading",
            icon=":material/receipt_long:",
            help="Virtual account: open positions, closed trades and results per setup.",
        ),
        st.Page(
            "app_pages/journal.py",
            title="Journal",
            icon=":material/edit_note:",
            help="Notes, tags and ratings for any paper trade, and results by tag.",
        ),
        st.Page(
            "app_pages/settings.py",
            title="Settings",
            icon=":material/settings:",
            help="Edit the watchlist and thresholds (saved to config.yaml).",
        ),
    ],
    position="top",
)
st.title(page.title, icon=page.icon)
page.run()
