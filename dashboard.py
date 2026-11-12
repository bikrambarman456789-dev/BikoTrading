
import streamlit as st  # type: ignore[reportMissingImports]
import pandas as pd
import ccxt
import plotly.graph_objects as go  # type: ignore[reportMissingImports]
import json
import os
from datetime import datetime, timezone

# ==========================================
# BIKOTRADING - PROFESSIONAL PAPER DASHBOARD
# ==========================================

st.set_page_config(
    page_title="BikoTrading Pro",
    page_icon="📈",
    layout="wide"
)

START_BALANCE = 10000.0
RISK_PERCENT = 0.01
STOP_ATR = 1.5
TARGET_ATR = 3.0
FEE_RATE = 0.001

STATE_FILE = "dashboard_state.json"


# ==========================================
# STATE MANAGEMENT
# ==========================================

def load_state():
    default = {
        "balance": START_BALANCE,
        "position": None,
        "trades": []
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                default.update(data)

            if not isinstance(default["trades"], list):
                default["trades"] = []

            return default

        except (json.JSONDecodeError, OSError):
            pass

    return default


def save_state():
    temp_file = STATE_FILE + ".tmp"

    with open(temp_file, "w") as f:
        json.dump(
            st.session_state.bot_state,
            f,
            indent=4
        )

    os.replace(temp_file, STATE_FILE)


if "bot_state" not in st.session_state:
    st.session_state.bot_state = load_state()


# ==========================================
# EXCHANGE CONNECTION
# ==========================================

@st.cache_resource
def get_exchange():
    return ccxt.binance({
        "enableRateLimit": True,
        "timeout": 15000
    })


@st.cache_data(ttl=5)
def get_data(symbol, timeframe):

    candles = get_exchange().fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=200
    )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    # EMA
    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI 14
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[avg_loss == 0, "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss == 0), "RSI"] = 50

    # ATR 14
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    return df


def get_live_price(symbol):

    ticker = get_exchange().fetch_ticker(symbol)

    price = ticker.get("last")

    if price is None or float(price) <= 0:
        raise ValueError("Live price unavailable")

    return float(price)


# ==========================================
# TRADE CLOSING
# ==========================================

def close_trade(exit_price, reason):

    state = st.session_state.bot_state
    position = state["position"]

    if position is None:
        return None

    proceeds = position["qty"] * exit_price

    exit_fee = proceeds * FEE_RATE

    pnl = (
        proceeds
        - exit_fee
        - position["cost"]
        - position["entry_fee"]
    )

    # Return sale proceeds to virtual cash balance
    state["balance"] += proceeds - exit_fee

    trade = {
        "Time": datetime.now(timezone.utc).isoformat(),
        "Market": position["symbol"],
        "Timeframe": position["timeframe"],
        "Side": "LONG",
        "Entry": round(position["entry"], 4),
        "Exit": round(exit_price, 4),
        "Quantity": round(position["qty"], 8),
        "Entry Fee": round(position["entry_fee"], 4),
        "Exit Fee": round(exit_fee, 4),
        "Exit Reason": reason,
        "Net P/L": round(pnl, 4),
        "Balance After": round(state["balance"], 4)
    }

    state["trades"].append(trade)
    state["position"] = None

    save_state()

    return pnl


# ==========================================
# AUTOMATIC SL / TP
# ==========================================

def monitor_position(df, live_price):

    state = st.session_state.bot_state
    position = state["position"]

    if position is None:
        return None

    sl = position["sl"]
    tp = position["tp"]

    entry_candle_ts = position["entry_candle_ts"]

    candle_times = (
        df["timestamp"].astype("int64") // 1_000_000
    )

    later_candles = df[
        candle_times > entry_candle_ts
    ]

    for candle in later_candles.itertuples():

        candle_open = float(candle.open)
        candle_high = float(candle.high)
        candle_low = float(candle.low)

        # Conservative assumption if both levels
        # are touched within the same candle.
        if candle_low <= sl:

            exit_price = min(candle_open, sl)

            return close_trade(
                exit_price,
                "STOP LOSS"
            )

        elif candle_high >= tp:

            exit_price = max(candle_open, tp)

            return close_trade(
                exit_price,
                "TAKE PROFIT"
            )

    if live_price <= sl:
        return close_trade(sl, "STOP LOSS")

    if live_price >= tp:
        return close_trade(tp, "TAKE PROFIT")

    return None


# ==========================================
# P&L CALCULATIONS
# ==========================================

def calculate_pnl(state, current_price=None):

    trades = state["trades"]
    position = state["position"]

    realized = sum(
        float(t.get("Net P/L", 0))
        for t in trades
    )

    unrealized = 0.0

    if position is not None and current_price is not None:

        current_value = (
            position["qty"] * current_price
        )

        exit_fee = current_value * FEE_RATE

        unrealized = (
            current_value
            - position["cost"]
            - position["entry_fee"]
            - exit_fee
        )

    total = realized + unrealized

    wins = sum(
        1 for t in trades
        if float(t.get("Net P/L", 0)) > 0
    )

    losses = sum(
        1 for t in trades
        if float(t.get("Net P/L", 0)) < 0
    )

    win_rate = (
        wins / len(trades) * 100
        if trades else 0
    )

    equity = state["balance"]

    if position is not None and current_price is not None:

        equity += (
            position["qty"] * current_price
            * (1 - FEE_RATE)
        )

    return {
        "realized": realized,
        "unrealized": unrealized,
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "equity": equity
    }


# ==========================================
# HEADER
# ==========================================

st.title("📈 BikoTrading Pro")

st.caption(
    "PROFESSIONAL PAPER TRADING DASHBOARD | "
    "LIVE MARKET DATA | NO REAL ORDERS"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("⚙️ Settings")

    symbol = st.selectbox(
        "Select Market",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m", "1h"]
    )

    st.divider()

    st.subheader("Risk Management")

    st.write("Risk per trade: 1%")
    st.write("Stop Loss: 1.5 ATR")
    st.write("Take Profit: 3 ATR")
    st.write("Fee assumption: 0.1% per side")

    st.divider()

    st.info(
        "This is a virtual paper trading system. "
        "No real exchange orders are placed."
    )

    if st.button(
        "🔄 Reset Paper Account",
        use_container_width=True
    ):

        st.session_state.confirm_reset = True

    if st.session_state.get("confirm_reset", False):

        st.warning(
            "Reset will erase this dashboard's "
            "paper balance, position and trade history."
        )

        if st.button("Confirm Reset"):

            st.session_state.bot_state = {
                "balance": START_BALANCE,
                "position": None,
                "trades": []
            }

            save_state()

            st.session_state.confirm_reset = False

            st.rerun()

        if st.button("Cancel Reset"):
            st.session_state.confirm_reset = False
            st.rerun()


# ==========================================
# MAIN DASHBOARD
# ==========================================

@st.fragment(run_every="5s")
def dashboard():

    state = st.session_state.bot_state

    try:

        # Fetch selected market data
        df = get_data(symbol, timeframe)

        if len(df) < 50:
            st.warning("Waiting for market data.")
            return

        # Completed candle indicators
        candle = df.iloc[-2]

        ema20 = float(candle["EMA20"])
        ema50 = float(candle["EMA50"])
        atr = float(candle["ATR"])
        rsi = float(candle["RSI"])

        live_price = get_live_price(symbol)

        if ema20 > ema50:
            trend = "BULLISH"

        elif ema20 < ema50:
            trend = "BEARISH"

        else:
            trend = "NEUTRAL"

        # ==================================
        # MONITOR OPEN POSITION
        # ==================================

        position = state["position"]
        position_price = None

        if position is not None:

            if (
                position["symbol"] == symbol
                and position["timeframe"] == timeframe
            ):

                monitor_df = df
                monitor_price = live_price

            else:

                monitor_df = get_data(
                    position["symbol"],
                    position["timeframe"]
                )

                monitor_price = get_live_price(
                    position["symbol"]
                )

            old_position = state["position"]

            pnl = monitor_position(
                monitor_df,
                monitor_price
            )

            if old_position is not None and state["position"] is None:

                st.success(
                    f"Automatic trade closed. "
                    f"Net P/L: {pnl:+.2f} USDT"
                )

            if state["position"] is not None:
                position_price = monitor_price

        elif position is not None:
            position_price = live_price

        # Refresh position after monitoring
        position = state["position"]

        # ==================================
        # MARKET OVERVIEW
        # ==================================

        st.subheader("🌐 Market Overview")

        a, b, c, d = st.columns(4)

        a.metric(
            "Live Price",
            f"{live_price:,.2f} USDT"
        )

        b.metric(
            "Market Bias",
            trend
        )

        c.metric(
            "EMA 20",
            f"{ema20:,.2f}"
        )

        d.metric(
            "EMA 50",
            f"{ema50:,.2f}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "RSI (14)",
            f"{rsi:.2f}"
        )

        b.metric(
            "ATR (14)",
            f"{atr:,.2f}"
        )

        c.metric(
            "Virtual Cash",
            f'{state["balance"]:,.2f} USDT'
        )

        if trend == "BULLISH":

            st.success(
                "🟢 BULLISH TREND — EMA20 above EMA50"
            )

        elif trend == "BEARISH":

            st.error(
                "🔴 BEARISH TREND — EMA20 below EMA50"
            )

        else:

            st.warning("NEUTRAL MARKET")

        st.divider()

        # ==================================
        # P&L OVERVIEW
        # ==================================

        st.subheader("💰 P&L Overview")

        pnl_data = calculate_pnl(
            state,
            position_price
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Realized P&L",
            f'{pnl_data["realized"]:+.2f} USDT'
        )

        b.metric(
            "Unrealized P&L",
            f'{pnl_data["unrealized"]:+.2f} USDT'
        )

        c.metric(
            "Total P&L",
            f'{pnl_data["total"]:+.2f} USDT'
        )

        d.metric(
            "Win Rate",
            f'{pnl_data["win_rate"]:.1f}%'
        )

        a, b, c = st.columns(3)

        a.metric(
            "Account Equity",
            f'{pnl_data["equity"]:,.2f} USDT'
        )

        b.metric(
            "Winning Trades",
            pnl_data["wins"]
        )

        c.metric(
            "Losing Trades",
            pnl_data["losses"]
        )

        st.caption(
            f'Closed Trades: {len(state["trades"])} | '
            f'Initial Balance: {START_BALANCE:,.2f} USDT'
        )

        st.divider()

        # ==================================
        # PRICE CHART
        # ==================================

        st.subheader("📊 Market Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlestick"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["EMA20"],
                name="EMA 20",
                line=dict(width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["EMA50"],
                name="EMA 50",
                line=dict(width=2)
            )
        )

        if (
            position is not None
            and position["symbol"] == symbol
        ):

            fig.add_hline(
                y=position["entry"],
                line_dash="dot",
                annotation_text="Entry"
            )

            fig.add_hline(
                y=position["sl"],
                line_dash="dash",
                annotation_text="Stop Loss"
            )

            fig.add_hline(
                y=position["tp"],
                line_dash="dash",
                annotation_text="Take Profit"
            )

        fig.update_layout(
            height=550,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================
        # PAPER TRADING
        # ==================================

        st.divider()

        st.subheader("🎯 Paper Trading")

        position = state["position"]

        if position is None:

            if (
                pd.notna(atr)
                and atr > 0
                and live_price > 0
            ):

                sl = live_price - STOP_ATR * atr

                tp = live_price + TARGET_ATR * atr

                risk_budget = (
                    state["balance"] * RISK_PERCENT
                )

                risk_per_unit = live_price - sl

                qty_by_risk = (
                    risk_budget / risk_per_unit
                )

                available_cash = state["balance"] * 0.99

                qty_by_cash = (
                    available_cash
                    / (live_price * (1 + FEE_RATE))
                )

                qty = min(
                    qty_by_risk,
                    qty_by_cash
                )

                cost = qty * live_price

                entry_fee = cost * FEE_RATE

                x, y, z = st.columns(3)

                x.metric(
                    "Proposed Entry",
                    f"{live_price:,.2f}"
                )

                y.metric(
                    "Proposed Stop Loss",
                    f"{sl:,.2f}"
                )

                z.metric(
                    "Proposed Take Profit",
                    f"{tp:,.2f}"
                )

                st.metric(
                    "Estimated Risk",
                    f"{qty * risk_per_unit:,.2f} USDT"
                )

                st.caption(
                    "Paper BUY opens a virtual LONG position. "
                    "This signal is not a guaranteed trade."
                )

                if st.button(
                    "🟢 PAPER BUY / LONG",
                    use_container_width=True
                ):

                    total_cost = cost + entry_fee

                    if (
                        qty > 0
                        and total_cost <= state["balance"]
                    ):

                        state["balance"] -= total_cost

                        state["position"] = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "entry": live_price,
                            "qty": qty,
                            "cost": cost,
                            "entry_fee": entry_fee,
                            "sl": sl,
                            "tp": tp,
                            "entry_candle_ts": int(
                                df.iloc[-1]["timestamp"].timestamp()
                                * 1000
                            )
                        }

                        save_state()

                        st.success(
                            "Virtual LONG position opened!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Insufficient virtual balance."
                        )

            else:

                st.warning(
                    "ATR data is not ready."
                )

        else:

            st.info(
                f'OPEN POSITION: {position["symbol"]} '
                f'| {position["timeframe"]}'
            )

            x, y, z = st.columns(3)

            x.metric(
                "Entry Price",
                f'{position["entry"]:,.2f}'
            )

            y.metric(
                "Stop Loss",
                f'{position["sl"]:,.2f}'
            )

            z.metric(
                "Take Profit",
                f'{position["tp"]:,.2f}'
            )

            if position_price is not None:

                floating = calculate_pnl(
                    state,
                    position_price
                )["unrealized"]

                st.metric(
                    "Floating P&L",
                    f"{floating:+.2f} USDT"
                )

            if st.button(
                "🔴 CLOSE VIRTUAL TRADE",
                use_container_width=True
            ):

                current_price = get_live_price(
                    position["symbol"]
                )

                result = close_trade(
                    current_price,
                    "MANUAL CLOSE"
                )

                st.success(
                    f"Trade closed. Net P/L: {result:+.2f} USDT"
                )

                st.rerun()

        # ==================================
        # OPEN POSITION DETAILS
        # ==================================

        st.divider()

        st.subheader("📌 Open Position")

        if state["position"] is not None:

            open_position = state["position"]

            open_data = pd.DataFrame([{
                "Market": open_position["symbol"],
                "Timeframe": open_position["timeframe"],
                "Side": "LONG",
                "Entry": open_position["entry"],
                "Quantity": open_position["qty"],
                "Stop Loss": open_position["sl"],
                "Take Profit": open_position["tp"]
            }])

            st.dataframe(
                open_data,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No open positions.")

        # ==================================
        # CLOSED TRADE HISTORY
        # ==================================

        st.divider()

        st.subheader("📚 Closed Trade History")

        trades = state["trades"]

        if trades:

            history = pd.DataFrame(trades)

            st.dataframe(
                history.iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

            csv_data = history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Trade History CSV",
                data=csv_data,
                file_name="bikotrading_trades.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No closed paper trades yet."
            )

        # ==================================
        # FOOTER
        # ==================================

        st.caption(
            "Last Update: "
            + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )

    except Exception as error:

        st.error(
            f"Dashboard Error: {error}"
        )


# ==========================================
# RUN DASHBOARD
# ==========================================

dashboard()

st.divider()

st.caption(
    "BikoTrading Pro | Paper Trading Only | "
    "No real exchange orders | "
    "SL/TP monitoring works only while dashboard is active."
)
