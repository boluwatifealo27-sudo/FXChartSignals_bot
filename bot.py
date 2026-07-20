"""
@FXChartSignals_bot — Telegram Forex Signal Bot
-------------------------------------------------
Fetches forex price data (Yahoo Finance), calculates RSI + EMA50,
checks for BUY/SELL signals, draws a candlestick chart with markers,
and sends the chart + signal info to subscribed Telegram users.

Two ways signals reach users:
1. AUTOMATIC — background scan every CHECK_INTERVAL_SECONDS, sends
   only when a fresh EMA crossover happens (sent to all subscribers)
2. ON-DEMAND — any user (subscribed or not) can send /signal, pick
   a pair from buttons, and instantly get a chart with the CURRENT
   market state (BUY / SELL / NEUTRAL), even without a crossover

Indicators used:
- EMA50 (trend direction)
- RSI 14 (momentum / overbought-oversold filter)

DISCLAIMER: This is a technical-indicator based tool for educational/
informational purposes only. It is NOT financial advice, and no
signal system guarantees profit. Always test on a demo account first.
"""

import json
import logging
import os
import time
from datetime import datetime

import mplfinance as mpf
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "XAU/USD": "GC=F",       # Gold futures as a proxy for XAU/USD
}

TIMEFRAME = os.environ.get("TIMEFRAME", "15m")
LOOKBACK_PERIOD = os.environ.get("LOOKBACK_PERIOD", "5d")
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "900"))

DATA_DIR = os.environ.get("DATA_DIR", "/data" if os.path.isdir("/data") else ".")
SUBSCRIBERS_FILE = os.path.join(DATA_DIR, "subscribers.json")
LAST_SIGNAL_FILE = os.path.join(DATA_DIR, "last_signals.json")
CHART_DIR = os.path.join(DATA_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# STORAGE HELPERS
# ----------------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def get_subscribers():
    return load_json(SUBSCRIBERS_FILE, [])


def add_subscriber(chat_id):
    subs = get_subscribers()
    if chat_id not in subs:
        subs.append(chat_id)
        save_json(SUBSCRIBERS_FILE, subs)


def remove_subscriber(chat_id):
    subs = get_subscribers()
    if chat_id in subs:
        subs.remove(chat_id)
        save_json(SUBSCRIBERS_FILE, subs)


def get_last_signals():
    return load_json(LAST_SIGNAL_FILE, {})


def save_last_signal(pair, signal_type, timestamp):
    data = get_last_signals()
    data[pair] = {"type": signal_type, "time": timestamp}
    save_json(LAST_SIGNAL_FILE, data)


# ----------------------------------------------------------------------
# DATA + INDICATORS
# ----------------------------------------------------------------------
def fetch_price_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=LOOKBACK_PERIOD, interval=TIMEFRAME, progress=False
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    return df


def check_crossover_signal(df: pd.DataFrame):
    """Used by the AUTOMATIC scanner — only fires on a fresh crossover."""
    if len(df) < 55:
        return None, None, None, None

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if pd.isna(prev["EMA50"]) or pd.isna(last["EMA50"]):
        return None, None, None, None

    crossed_up = prev["Close"] < prev["EMA50"] and last["Close"] > last["EMA50"]
    crossed_down = prev["Close"] > prev["EMA50"] and last["Close"] < last["EMA50"]

    entry = float(last["Close"])
    atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

    if crossed_up and last["RSI"] < 70:
        return "BUY", entry, entry - 1.5 * atr, entry + 2 * atr

    if crossed_down and last["RSI"] > 30:
        return "SELL", entry, entry + 1.5 * atr, entry - 2 * atr

    return None, None, None, None


def get_current_state(df: pd.DataFrame):
    """
    Used by the ON-DEMAND /signal command — always returns a reading,
    even if there's no fresh crossover. Classifies the CURRENT market
    position relative to EMA50 + RSI so users always get a chart.
    """
    last = df.iloc[-1]
    entry = float(last["Close"])
    rsi_value = float(last["RSI"])
    ema_value = float(last["EMA50"])
    atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

    if entry > ema_value and rsi_value < 70:
        state = "BUY"
        sl = entry - 1.5 * atr
        tp = entry + 2 * atr
    elif entry < ema_value and rsi_value > 30:
        state = "SELL"
        sl = entry + 1.5 * atr
        tp = entry - 2 * atr
    else:
        state = "NEUTRAL"
        sl = entry - 1.5 * atr
        tp = entry + 1.5 * atr

    return state, entry, sl, tp, rsi_value


# ----------------------------------------------------------------------
# CHART GENERATION
# ----------------------------------------------------------------------
def generate_chart(df: pd.DataFrame, pair_name: str, signal_type: str,
                    entry: float, sl: float, tp: float) -> str:
    plot_df = df.tail(100).copy()
    ema_plot = mpf.make_addplot(plot_df["EMA50"], color="orange", width=1.2)

    filename = os.path.join(
        CHART_DIR, f"{pair_name.replace('/', '')}_{int(time.time())}.png"
    )

    hlines = dict(
        hlines=[entry, sl, tp],
        colors=["blue", "red", "green"],
        linestyle="--",
        linewidths=0.8,
    )

    title = f"{pair_name} — {signal_type}"

    mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        addplot=ema_plot,
        hlines=hlines,
        title=title,
        ylabel="Price",
        volume=False,
        savefig=dict(fname=filename, dpi=150, bbox_inches="tight"),
    )
    return filename


# ----------------------------------------------------------------------
# TELEGRAM SENDING (automatic scanner)
# ----------------------------------------------------------------------
async def send_signal_to_subscribers(bot: Bot, pair_name: str, signal_type: str,
                                      entry: float, sl: float, tp: float,
                                      rsi_value: float, chart_path: str):
    caption = (
        f"📊 {pair_name} — {signal_type} Signal\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Entry: {entry:.5f}\n"
        f"Stop Loss: {sl:.5f}\n"
        f"Take Profit: {tp:.5f}\n"
        f"RSI: {rsi_value:.1f}\n\n"
        f"⚠️ Not financial advice. Trade at your own risk."
    )

    for chat_id in get_subscribers():
        try:
            with open(chart_path, "rb") as img:
                await bot.send_photo(chat_id=chat_id, photo=img, caption=caption)
        except Exception as e:
            logger.error(f"Failed to send to {chat_id}: {e}")


# ----------------------------------------------------------------------
# AUTOMATIC SCAN LOOP
# ----------------------------------------------------------------------
async def scan_and_send(bot: Bot):
    last_signals = get_last_signals()

    for pair_name, ticker in PAIRS.items():
        try:
            df = fetch_price_data(ticker)
            df = calculate_indicators(df)
            signal_type, entry, sl, tp = check_crossover_signal(df)

            if signal_type is None:
                continue

            prev = last_signals.get(pair_name)
            now_str = datetime.utcnow().isoformat()
            if prev and prev["type"] == signal_type:
                continue

            rsi_value = float(df.iloc[-1]["RSI"])
            chart_path = generate_chart(df, pair_name, signal_type, entry, sl, tp)

            await send_signal_to_subscribers(
                bot, pair_name, signal_type, entry, sl, tp, rsi_value, chart_path
            )
            save_last_signal(pair_name, signal_type, now_str)
            logger.info(f"Sent {signal_type} signal for {pair_name}")

        except Exception as e:
            logger.error(f"Error processing {pair_name}: {e}")


async def periodic_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_and_send(context.bot)


# ----------------------------------------------------------------------
# TELEGRAM COMMAND HANDLERS
# ----------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_subscriber(update.effective_chat.id)
    pairs_list = ", ".join(PAIRS.keys())
    await update.message.reply_text(
        "✅ You're subscribed to FX Chart Signals!\n\n"
        f"Tracked pairs: {pairs_list}\n"
        f"Timeframe: {TIMEFRAME}\n\n"
        "Commands:\n"
        "/signal — get an instant chart + signal for any pair\n"
        "/stop — unsubscribe from automatic alerts\n"
        "/status — see the last automatic signal per pair\n\n"
        "⚠️ Signals are for informational purposes only, not financial advice."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remove_subscriber(update.effective_chat.id)
    await update.message.reply_text("You've been unsubscribed from automatic signals.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last_signals = get_last_signals()
    if not last_signals:
        await update.message.reply_text("No automatic signals sent yet.")
        return
    lines = [f"{pair}: {info['type']} at {info['time']}" for pair, info in last_signals.items()]
    await update.message.reply_text("Last automatic signals:\n" + "\n".join(lines))


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows a button for each pair. Works for any user, subscribed or not."""
    keyboard = [
        [InlineKeyboardButton(pair, callback_data=f"signal:{pair}")]
        for pair in PAIRS
    ]
    await update.message.reply_text(
        "Pick a pair to get an instant chart + current signal:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def signal_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the button tap: fetches fresh data and sends the chart."""
    query = update.callback_query
    await query.answer()  # acknowledge the tap so the button stops "loading"

    pair_name = query.data.split("signal:", 1)[1]
    ticker = PAIRS.get(pair_name)
    if not ticker:
        await query.message.reply_text("Unknown pair, please try /signal again.")
        return

    await query.message.reply_text(f"⏳ Generating {pair_name} chart...")

    try:
        df = fetch_price_data(ticker)
        df = calculate_indicators(df)
        state, entry, sl, tp, rsi_value = get_current_state(df)
        chart_path = generate_chart(df, pair_name, state, entry, sl, tp)

        emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}[state]
        caption = (
            f"{emoji} {pair_name} — Current: {state}\n"
            f"Timeframe: {TIMEFRAME}\n"
            f"Price: {entry:.5f}\n"
            f"Reference SL: {sl:.5f}\n"
            f"Reference TP: {tp:.5f}\n"
            f"RSI: {rsi_value:.1f}\n\n"
            f"⚠️ Not financial advice. Trade at your own risk."
        )

        with open(chart_path, "rb") as img:
            await query.message.reply_photo(photo=img, caption=caption)

    except Exception as e:
        logger.error(f"Error generating on-demand signal for {pair_name}: {e}")
        await query.message.reply_text(
            f"Sorry, couldn't fetch data for {pair_name} right now. Try again shortly."
        )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not BOT_TOKEN:
        raise SystemExit(
            "Missing TELEGRAM_BOT_TOKEN environment variable.\n"
            "On Railway: Project -> Variables -> add TELEGRAM_BOT_TOKEN.\n"
            "Locally: export TELEGRAM_BOT_TOKEN='123456:ABC-your-token'"
        )

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("signal", signal_command))
    application.add_handler(CallbackQueryHandler(signal_button_callback, pattern=r"^signal:"))

    application.job_queue.run_repeating(
        periodic_scan, interval=CHECK_INTERVAL_SECONDS, first=10
    )

    logger.info("@FXChartSignals_bot started. Polling for commands...")
    application.run_polling()


if __name__ == "__main__":
    main()
