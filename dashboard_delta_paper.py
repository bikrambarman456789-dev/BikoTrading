import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime, timezone

# ==========================================
# BIKOTRADING PRO - PAPER TRADING ONLY
# Delta public market data; NO REAL ORDERS
# ==========================================

st.set_page_config(page_title="BikoTrading Pro", page_icon="📈", layout="wide")

START_BALANCE = 10000.0
RISK_PERCENT = 0.01
STOP_ATR = 1.5
TARGET_ATR = 3.0
FEE_RATE = 0.001
BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"  # Confirmed by the user's candle API test
STATE_FILE = "dashboard_state.json"


# ==========================================
# STATE MANAGEMENT
# ==========================================

def load_state():
    default = {
        "balance": START_BALANCE,
        "position": None,
        "trades": [],
        "last_signal_candle": {}
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                default.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    if not isinstance(default.get("trades"), list):
        default["trades"] = []
    if not isinstance(default.get("last_signal_candle"), dict):
        default["last_signal_candle"] = {}
    if not isinstance(default.get("balance"), (int, float)):
        default["balance"] = START_BALANCE
    return default


def save_state():
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.bot_state, f, indent=4)
    os.replace(temp_file, STATE_FILE)


if "bot_state" not in st.session_state:
    st.session_state.bot_state = load_state()


# ==========================================
# DELTA PUBLIC MARKET DATA (GET ONLY)
# ==========================================

@st.cache_data(ttl=5)
def get_data(symbol, timeframe):
    end_ts = int(time.time())
    # Fetch enough history for EMA50 and ATR14.
    interval_seconds = {
        "5m": 300,
        "15m": 900,
        "1h": 3600
    }[timeframe]
    start_ts = end_ts - (200 * interval_seconds)

    response = requests.get(
        f"{BASE_URL}/v2/history/candles",
        params={
            "resolution": timeframe,
            "symbol": symbol,
            "start": start_ts,
            "end": end_ts
        },
        headers={"Accept": "application/json"},
        timeout=15
    )
    response.raise_for_status()
    payload = response.json()
    candles = payload.get("result", [])

    if not candles:
        raise ValueError(f"No candle data returned for {symbol}.")

    df = pd.DataFrame(candles)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Delta candle response is missing: {', '.join(missing)}")

    df = df[required].copy()
    for column in ["open", "high", "low", "close", "volume", "time"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss == 0), "RSI"] = 50

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


@st.cache_data(ttl=3)
def get_live_price(symbol):
    response = requests.get(
        f"{BASE_URL}/v2/tickers/{symbol}",
        headers={"Accept": "application/json"},
        timeout=15
    )
    response.raise_for_status()
    payload = response.json()
    ticker = payload.get("result", payload)
    if isinstance(ticker, list):
        ticker = ticker[0] if ticker else {}
    if not isinstance(ticker, dict):
        ticker = {}

    # Delta ticker fields can include close or mark_price.
    price = ticker.get("close") or ticker.get("last_price") or ticker.get("mark_price")
    if price is None:
        raise ValueError("Ticker response did not contain a usable price.")
    price = float(price)
    if price <= 0:
        raise ValueError("Delta returned an invalid live price.")
    return price


# ==========================================
# PAPER TRADE ACCOUNTING
# ==========================================

def close_trade(exit_price, reason):
    state = st.session_state.bot_state
    position = state["position"]
    if position is None:
        return None

    proceeds = position["qty"] * float(exit_price)
    exit_fee = proceeds * FEE_RATE
    pnl = proceeds - exit_fee - position["cost"] - position["entry_fee"]
    state["balance"] += proceeds - exit_fee

    state["trades"].append({
        "Time": datetime.now(timezone.utc).isoformat(),
        "Market": position["symbol"],
        "Timeframe": position["timeframe"],
        "Side": "LONG",
        "Entry": round(position["entry"], 4),
        "Exit": round(float(exit_price), 4),
        "Quantity": round(position["qty"], 8),
        "Entry Fee": round(position["entry_fee"], 4),
        "Exit Fee": round(exit_fee, 4),
        "Exit Reason": reason,
        "Net P/L": round(pnl, 4),
        "Balance After": round(state["balance"], 4)
    })
    state["position"] = None
    save_state()
    return pnl


def monitor_position(df, live_price):
    state = st.session_state.bot_state
    position = state["position"]
    if position is None:
        return None

    sl = float(position["sl"])
    tp = float(position["tp"])
    entry_ts = int(position["entry_candle_ts"])

    later_candles = df[df["time"] > entry_ts]
    for candle in later_candles.itertuples():
        candle_open = float(candle.open)
        candle_high = float(candle.high)
        candle_low = float(candle.low)

        # If both levels occur in one candle, assume SL first (conservative).
        if candle_low <= sl:
            return close_trade(min(candle_open, sl), "STOP LOSS")
        if candle_high >= tp:
            return close_trade(max(candle_open, tp), "TAKE PROFIT")

    if live_price <= sl:
        return close_trade(live_price, "STOP LOSS")
    if live_price >= tp:
        return close_trade(live_price, "TAKE PROFIT")
    return None


def calculate_pnl(state, current_price=None):
    trades = state["trades"]
    position = state["position"]
    realized = sum(float(t.get("Net P/L", 0)) for t in trades)
    unrealized = 0.0

    if position is not None and current_price is not None:
        current_value = position["qty"] * current_price
        exit_fee = current_value * FEE_RATE
        unrealized = (
            current_value - position["cost"]
            - position["entry_fee"] - exit_fee
        )

    wins = sum(1 for t in trades if float(t.get("Net P/L", 0)) > 0)
    losses = sum(1 for t in trades if float(t.get("Net P/L", 0)) < 0)
    win_rate = wins / len(trades) * 100 if trades else 0.0
    equity = state["balance"]

    if position is not None and current_price is not None:
        equity += position["qty"] * current_price * (1 - FEE_RATE)

    return {
        "realized": realized,
        "unrealized": unrealized,
        "total": realized + unrealized,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "equity": equity
    }


# ==========================================
# AUTOMATIC EMA CROSSOVER PAPER ENTRY
# ==========================================

def try_automatic_entry(df, live_price, symbol, timeframe):
    state = st.session_state.bot_state
    if state["position"] is not None or len(df) < 52:
        return False

    # Use only fully completed candles. The latest row may still be forming.
    previous = df.iloc[-3]
    signal = df.iloc[-2]
    signal_ts = int(signal["time"])
    signal_key = f"{symbol}_{timeframe}"
    last_processed = state["last_signal_candle"].get(signal_key)

    if last_processed == signal_ts:
        return False

    # Mark this completed candle as checked, even if no crossover occurred.
    state["last_signal_candle"][signal_key] = signal_ts
    save_state()

    crossed_up = (
        float(previous["EMA20"]) <= float(previous["EMA50"])
        and float(signal["EMA20"]) > float(signal["EMA50"])
    )
    atr = float(signal["ATR"])

    if not crossed_up or pd.isna(atr) or atr <= 0 or live_price <= 0:
        return False

    entry = float(live_price)
    sl = entry - STOP_ATR * atr
    tp = entry + TARGET_ATR * atr
    risk_per_unit = entry - sl
    if risk_per_unit <= 0:
        return False

    risk_budget = float(state["balance"]) * RISK_PERCENT
    qty_by_risk = risk_budget / risk_per_unit
    available_cash = float(state["balance"]) * 0.99
    qty_by_cash = available_cash / (entry * (1 + FEE_RATE))
    qty = min(qty_by_risk, qty_by_cash)

    cost = qty * entry
    entry_fee = cost * FEE_RATE
    total_cost = cost + entry_fee

    if qty <= 0 or total_cost > state["balance"]:
        return False

    state["balance"] -= total_cost
    state["position"] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "entry": entry,
        "qty": qty,
        "cost": cost,
        "entry_fee": entry_fee,
        "sl": sl,
        "tp": tp,
        "entry_candle_ts": signal_ts
    }
    save_state()
    return True


# ==========================================
# HEADER AND SIDEBAR
# ==========================================

st.title("📈 BikoTrading Pro")
st.caption("PAPER TRADING DASHBOARD | DELTA PUBLIC MARKET DATA | NO REAL ORDERS")
st.info(
    "Educational simulation only. Uses BTCUSD market data from Delta. "
    "All entries, exits, balance and P&L are virtual."
)
st.divider()

with st.sidebar:
    st.title("⚙️ Settings")
    st.selectbox("Select Market", [SYMBOL], index=0, disabled=True)
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h"], index=0)

    st.divider()
    st.subheader("Risk Management")
    st.write("Virtual risk per trade: 1%")
    st.write("Virtual Stop Loss: 1.5 ATR")
    st.write("Virtual Take Profit: 3 ATR")
    st.write("Simulated fee assumption: 0.1% per side")

    st.divider()
    st.info("Paper-only mode. No API keys, authentication or exchange orders are used.")

    if st.button("🔄 Reset Paper Account", use_container_width=True):
        st.session_state.confirm_reset = True

    if st.session_state.get("confirm_reset", False):
        st.warning("This erases the saved virtual balance, open position and trade history.")
        if st.button("Confirm Reset"):
            st.session_state.bot_state = {
                "balance": START_BALANCE,
                "position": None,
                "trades": [],
                "last_signal_candle": {}
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
        df = get_data(SYMBOL, timeframe)
        if len(df) < 52:
            st.warning("Waiting for enough candle history.")
            return

        # Last completed candle; latest row can be in progress.
        candle = df.iloc[-2]
        ema20 = float(candle["EMA20"])
        ema50 = float(candle["EMA50"])
        atr = float(candle["ATR"])
        rsi = float(candle["RSI"])
        live_price = get_live_price(SYMBOL)

        trend = "BULLISH" if ema20 > ema50 else "BEARISH" if ema20 < ema50 else "NEUTRAL"

        # Monitor an already-open position using its own saved timeframe.
        position_at_start = state["position"]
        position_price = None
        if position_at_start is not None:
            if position_at_start["symbol"] == SYMBOL and position_at_start["timeframe"] == timeframe:
                monitor_df = df
                monitor_price = live_price
            else:
                monitor_df = get_data(position_at_start["symbol"], position_at_start["timeframe"])
                monitor_price = get_live_price(position_at_start["symbol"])

            previous_position = state["position"]
            close_result = monitor_position(monitor_df, monitor_price)
            if previous_position is not None and state["position"] is None:
                st.success(f"Automatic virtual trade closed. Net P/L: {close_result:+.2f} USD")
            if state["position"] is not None:
                position_price = monitor_price

        # Only look for a new entry if there was no position at start.
        # This prevents closing and reopening on the same refresh.
        auto_opened = False
        if position_at_start is None and state["position"] is None:
            auto_opened = try_automatic_entry(df, live_price, SYMBOL, timeframe)
            if auto_opened:
                st.success("EMA20 crossed above EMA50 on a completed candle. Virtual LONG opened automatically.")
                position_price = live_price

        if state["position"] is not None and position_price is None:
            if state["position"]["symbol"] == SYMBOL and state["position"]["timeframe"] == timeframe:
                position_price = live_price
            else:
                position_price = get_live_price(state["position"]["symbol"])

        # Market overview
        st.subheader("🌐 Market Overview")
        a, b, c, d = st.columns(4)
        a.metric("Delta BTCUSD Price", f"{live_price:,.2f} USD")
        b.metric("Market Bias", trend)
        c.metric("EMA 20", f"{ema20:,.2f}")
        d.metric("EMA 50", f"{ema50:,.2f}")

        a, b, c = st.columns(3)
        a.metric("RSI (14)", f"{rsi:.2f}")
        b.metric("ATR (14)", f"{atr:,.2f}")
        c.metric("Virtual Cash", f'{state["balance"]:,.2f} USD')

        if trend == "BULLISH":
            st.success("🟢 BULLISH TREND — EMA20 above EMA50")
        elif trend == "BEARISH":
            st.error("🔴 BEARISH TREND — EMA20 below EMA50")
        else:
            st.warning("NEUTRAL MARKET")

        st.caption("Automatic entry checks only a new, completed candle crossover. It does not enter just because EMA20 remains above EMA50.")
        st.divider()

        # P&L
        st.subheader("💰 P&L Overview")
        pnl_data = calculate_pnl(state, position_price)
        a, b, c, d = st.columns(4)
        a.metric("Realized P&L", f'{pnl_data["realized"]:+.2f} USD')
        b.metric("Unrealized P&L", f'{pnl_data["unrealized"]:+.2f} USD')
        c.metric("Total P&L", f'{pnl_data["total"]:+.2f} USD')
        d.metric("Win Rate", f'{pnl_data["win_rate"]:.1f}%')

        a, b, c = st.columns(3)
        a.metric("Account Equity", f'{pnl_data["equity"]:,.2f} USD')
        b.metric("Winning Trades", pnl_data["wins"])
        c.metric("Losing Trades", pnl_data["losses"])
        st.caption(f'Closed Trades: {len(state["trades"])} | Initial Virtual Balance: {START_BALANCE:,.2f} USD')
        st.divider()

        # Chart
        st.subheader("📊 Market Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Candlestick"
        ))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA20"], name="EMA 20", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA50"], name="EMA 50", line=dict(width=2)))

        position = state["position"]
        if position is not None and position["symbol"] == SYMBOL:
            fig.add_hline(y=position["entry"], line_dash="dot", annotation_text="Entry")
            fig.add_hline(y=position["sl"], line_dash="dash", annotation_text="Stop Loss")
            fig.add_hline(y=position["tp"], line_dash="dash", annotation_text="Take Profit")

        fig.update_layout(
            height=550, template="plotly_dark",
            xaxis_rangeslider_visible=False, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # Paper trading position panel
        st.subheader("🎯 Automatic Paper Trading")
        position = state["position"]
        if position is None:
            st.info("No open virtual position. The dashboard is waiting for a fresh EMA20/EMA50 bullish crossover on a completed candle.")
        else:
            st.info(f'OPEN VIRTUAL POSITION: {position["symbol"]} | {position["timeframe"]} | LONG')
            a, b, c = st.columns(3)
            a.metric("Entry Price", f'{position["entry"]:,.2f} USD')
            b.metric("Stop Loss", f'{position["sl"]:,.2f} USD')
            c.metric("Take Profit", f'{position["tp"]:,.2f} USD')

            if position_price is not None:
                floating = calculate_pnl(state, position_price)["unrealized"]
                st.metric("Floating P&L", f"{floating:+.2f} USD")

            if st.button("🔴 CLOSE VIRTUAL TRADE", use_container_width=True):
                current_price = get_live_price(position["symbol"])
                result = close_trade(current_price, "MANUAL CLOSE")
                st.success(f"Virtual trade closed. Net P/L: {result:+.2f} USD")
                st.rerun()

        st.divider()
        st.subheader("📌 Open Position")
        if state["position"] is not None:
            p = state["position"]
            open_data = pd.DataFrame([{
                "Market": p["symbol"], "Timeframe": p["timeframe"],
                "Side": "LONG", "Entry": p["entry"], "Quantity": p["qty"],
                "Stop Loss": p["sl"], "Take Profit": p["tp"]
            }])
            st.dataframe(open_data, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

        st.divider()
        st.subheader("📚 Closed Trade History")
        if state["trades"]:
            history = pd.DataFrame(state["trades"])
            st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
            csv_data = history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Trade History CSV",
                data=csv_data,
                file_name="bikotrading_trades.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No closed paper trades yet.")

        st.caption("Last Update: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

    except requests.RequestException as error:
        st.error(f"Delta market-data connection error: {error}")
    except Exception as error:
        st.error(f"Dashboard Error: {error}")


dashboard()
st.divider()
st.caption(
    "BikoTrading Pro | Paper Trading Only | No real exchange orders. "
    "Automatic entries and SL/TP monitoring run only while this dashboard session is active."
)
