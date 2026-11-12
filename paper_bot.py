
import ccxt
import pandas as pd
import json
import csv
import os
import time
from datetime import datetime, timezone

# BikoTrading - paper trading only
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
START_BALANCE = 10000.0
RISK_PERCENT = 0.01
FEE_RATE = 0.001  # Assumed 0.1% per side
CHECK_SECONDS = 30

STATE_FILE = "paper_bot_state.json"
TRADES_FILE = "paper_bot_trades.csv"

COLUMNS = [
    "entry_time", "exit_time", "symbol", "side",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "quantity", "gross_pnl", "entry_fee", "exit_fee",
    "net_pnl", "balance_after", "exit_reason"
]

exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 20000
})


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_state(balance, position):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "balance": balance,
            "position": position
        }, f, indent=2)


def load_state():
    if not os.path.exists(STATE_FILE):
        return START_BALANCE, None

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        balance = float(state["balance"])
        position = state.get("position")

        if balance <= 0:
            raise ValueError("Invalid saved balance")

        return balance, position
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        print("State file invalid. Starting with a fresh virtual balance.")
        return START_BALANCE, None


def ensure_csv():
    if not os.path.exists(TRADES_FILE) or os.path.getsize(TRADES_FILE) == 0:
        with open(TRADES_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def get_market_data():
    candles = exchange.fetch_ohlcv(
        SYMBOL, timeframe=TIMEFRAME, limit=100
    )

    df = pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    # Ignore the currently forming candle for EMA crossover signals.
    df = df.iloc[:-1].copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    previous = df.iloc[-2]
    latest = df.iloc[-1]

    # ATR(14)
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = true_range.rolling(14).mean().iloc[-1]

    crossed_up = (
        previous["ema20"] <= previous["ema50"]
        and latest["ema20"] > latest["ema50"]
    )

    return float(latest["close"]), float(latest["ema20"]), \
        float(latest["ema50"]), float(atr), crossed_up, int(latest["time"])


def open_paper_trade(balance, price, atr):
    if pd.isna(atr) or atr <= 0 or price <= 0:
        print("Invalid ATR or price. No trade opened.")
        return balance, None

    stop = price - 1.5 * atr
    target = price + 3.0 * atr

    risk_budget = balance * RISK_PERCENT
    risk_per_unit = price - stop

    if risk_per_unit <= 0:
        return balance, None

    # Limit simulated position value to the virtual balance.
    quantity_by_risk = risk_budget / risk_per_unit
    quantity_by_balance = balance / price
    quantity = min(quantity_by_risk, quantity_by_balance)

    if quantity <= 0:
        return balance, None

    entry_fee = quantity * price * FEE_RATE

    if entry_fee >= balance:
        print("Not enough virtual balance for assumed fee.")
        return balance, None

    balance -= entry_fee

    position = {
        "entry_time": now_text(),
        "entry_price": price,
        "stop_loss": stop,
        "take_profit": target,
        "quantity": quantity,
        "entry_fee": entry_fee
    }

    print("\nPAPER LONG OPENED")
    print(f"Entry price : {price:.2f}")
    print(f"Stop Loss   : {stop:.2f}")
    print(f"Take Profit : {target:.2f}")
    print(f"Quantity    : {quantity:.8f} BTC")
    print(f"Entry fee   : {entry_fee:.4f} USDT")
    print(f"Balance     : {balance:.2f} USDT\n")

    return balance, position


def close_paper_trade(balance, position, exit_price, reason):
    quantity = float(position["quantity"])
    entry = float(position["entry_price"])
    entry_fee = float(position["entry_fee"])

    gross_pnl = (exit_price - entry) * quantity
    exit_fee = exit_price * quantity * FEE_RATE
    net_pnl = gross_pnl - entry_fee - exit_fee

    # Entry fee was deducted when opening; add gross P&L and deduct exit fee.
    balance += gross_pnl - exit_fee

    record = {
        "entry_time": position["entry_time"],
        "exit_time": now_text(),
        "symbol": SYMBOL,
        "side": "LONG",
        "entry_price": round(entry, 8),
        "exit_price": round(exit_price, 8),
        "stop_loss": round(float(position["stop_loss"]), 8),
        "take_profit": round(float(position["take_profit"]), 8),
        "quantity": round(quantity, 8),
        "gross_pnl": round(gross_pnl, 6),
        "entry_fee": round(entry_fee, 6),
        "exit_fee": round(exit_fee, 6),
        "net_pnl": round(net_pnl, 6),
        "balance_after": round(balance, 6),
        "exit_reason": reason
    }

    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(record)

    print("\nPAPER TRADE CLOSED")
    print(f"Reason       : {reason}")
    print(f"Entry        : {entry:.2f}")
    print(f"Exit         : {exit_price:.2f}")
    print(f"Stop Loss    : {position['stop_loss']:.2f}")
    print(f"Take Profit  : {position['take_profit']:.2f}")
    print(f"Gross P/L    : {gross_pnl:+.4f} USDT")
    print(f"Entry fee    : {entry_fee:.4f} USDT")
    print(f"Exit fee     : {exit_fee:.4f} USDT")
    print(f"Net P/L      : {net_pnl:+.4f} USDT")
    print(f"Total balance: {balance:.4f} USDT\n")

    return balance


def main():
    ensure_csv()
    balance, position = load_state()

    print("BikoTrading paper bot started. No real orders.")
    print(f"Symbol={SYMBOL}, timeframe={TIMEFRAME}")
    print(f"Virtual balance={balance:.2f} USDT")
    print("Press Ctrl+C to stop.\n")

    last_signal_candle = None

    while True:
        try:
            price, ema20, ema50, atr, crossed_up, candle_time = \
                get_market_data()

            ticker = exchange.fetch_ticker(SYMBOL)
            live_price = float(ticker["last"] or price)

            print(
                f"{now_text()} | Price={live_price:.2f} "
                f"EMA20={ema20:.2f} EMA50={ema50:.2f} "
                f"Balance={balance:.2f} USDT"
            )

            if position is not None:
                stop = float(position["stop_loss"])
                target = float(position["take_profit"])

                if live_price <= stop:
                    balance = close_paper_trade(
                        balance, position, stop, "STOP_LOSS"
                    )
                    position = None
                elif live_price >= target:
                    balance = close_paper_trade(
                        balance, position, target, "TAKE_PROFIT"
                    )
                    position = None

            elif crossed_up and candle_time != last_signal_candle:
                last_signal_candle = candle_time
                balance, position = open_paper_trade(
                    balance, live_price, atr
                )

            save_state(balance, position)
            time.sleep(CHECK_SECONDS)

        except KeyboardInterrupt:
            save_state(balance, position)
            print("\nBot stopped. State saved.")
            break

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"Market data error: {e}. Retrying in 30 seconds.")
            time.sleep(CHECK_SECONDS)

        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 30 seconds.")
            time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
