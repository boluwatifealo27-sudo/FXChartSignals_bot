"""
@FXChartSignals_bot — Telegram Forex Signal Bot
-------------------------------------------------
Fetches forex price data (Yahoo Finance), calculates RSI + EMA50,
checks for BUY/SELL signals, draws a candlestick chart with markers,
and sends the chart + signal info to subscribed Telegram users.

Four ways signals reach users:
1. AUTOMATIC — background scan every CHECK_INTERVAL_SECONDS, sends
   only when a fresh EMA crossover happens (sent to all subscribers)
2. ON-DEMAND — any user sends /signal, picks a pair, then a
   timeframe, and instantly gets a chart with the CURRENT market
   state (BUY / SELL / NEUTRAL)
3. REFRESH — a button under every result re-pulls fresh data for
   the same pair/timeframe, or switches timeframe/pair
4. LIVE MODE — a button starts auto-refreshing the same message
   every LIVE_REFRESH_SECONDS with the newest available data, until
   the user taps Stop or MAX_LIVE_UPDATES is reached

NOTE ON "LIVE": Yahoo Finance forex data is delayed by a few minutes,
not tick-level real-time. Live mode re-fetches the newest data Yahoo
has on a fixed interval — it is the freshest free data available,
not a broker-grade real-time feed.

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
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
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

# Timeframes available for on-demand /signal requests.
# label shown to user -> (yfinance interval, matching lookback period)
TIMEFRAMES = {
    "15m": ("15m", "5d"),
    "1h": ("60m", "1mo"),
    "4h": ("1h", "3mo"),   # yfinance has no native 4h; we resample 1h -> 4h below
    "1d": ("1d", "1y"),
}

AUTO_TIMEFRAME = os.environ.get("TIMEFRAME", "15m")
AUTO_LOOKBACK = os.environ.get("LOOKBACK_PERIOD", "5d")
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "900"))

LIVE_REFRESH_SECONDS = int(os.environ.get("LIVE_REFRESH_SECONDS", "60"))
MAX_LIVE_UPDATES = int(os.environ.get("MAX_LIVE_UPDATES", "10"))

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
def fetch_price_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance has no native 4h interval, so build it from 1h candles."""
    resampled = df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    resampled.dropna(inplace=True)
    return resampled


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
    Used by ON-DEMAND requests — always returns a reading, even
    without a fresh crossover. Classifies CURRENT market position
    relative to EMA50 + RSI.
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
def generate_chart(df: pd.DataFrame, pair_name: str, label: str, tf_label: str,
                    entry: float, sl: float, tp: float) -> str:
    plot_df = df.tail(100).copy()
    ema_plot = mpf.make_addplot(plot_df["EMA50"], color="orange", width=1.2)

    filename = os.path.join(
        CHART_DIR, f"{pair_name.replace('/', '')}_{tf_label}_{int(time.time())}.png"
    )

    hlines = dict(
        hlines=[entry, sl, tp],
        colors=["blue", "red", "green"],
        linestyle="--",
        linewidths=0.8,
    )

    title = f"{pair_name} ({tf_label}) — {label}"

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


def get_signal_data(pair_name: str, tf_label: str):
    """Full pipeline: fetch -> indicators -> current state, for a given pair+timeframe."""
    ticker = PAIRS[pair_name]
    interval, period = TIMEFRAMES[tf_label]
    df = fetch_price_data(ticker, interval, period)

    if tf_label == "4h":
        df = resample_to_4h(df)

    df = calculate_indicators(df)
    state, entry, sl, tp, rsi_value = get_current_state(df)
    chart_path = generate_chart(df, pair_name, state, tf_label, entry, sl, tp)
    return state, entry, sl, tp, rsi_value, chart_path


def build_caption(pair_name: str, tf_label: str, state: str, entry: float,
                   sl: float, tp: float, rsi_value: float, live: bool, update_count: int = 0) -> str:
    emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}[state]
    header = f"{emoji} {pair_name} ({tf_label}) — Current: {state}"
    if live:
        header = f"🔴 LIVE  {header}"

    caption = (
        f"{header}\n"
        f"Price: {entry:.5f}\n"
        f"Reference SL: {sl:.5f}\n"
        f"Reference TP: {tp:.5f}\n"
        f"RSI: {rsi_value:.1f}\n"
    )
    if live:
        caption += (
            f"Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC "
            f"(update {update_count}/{MAX_LIVE_UPDATES})\n"
        )
    caption += "\n⚠️ Not financial advice. Trade at your own risk."
    return caption


# ----------------------------------------------------------------------
# TELEGRAM SENDING (automatic scanner)
# ----------------------------------------------------------------------
async def send_signal_to_subscribers(bot: Bot, pair_name: str, signal_type: str,
                                      entry: float, sl: float, tp: float,
                                      rsi_value: float, chart_path: str):
    caption = (
        f"📊 {pair_name} — {signal_type} Signal\n"
        f"Timeframe: {AUTO_TIMEFRAME}\n"
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
            df = fetch_price_data(ticker, AUTO_TIMEFRAME, AUTO_LOOKBACK)
            df = calculate_indicators(df)
            signal_type, entry, sl, tp = check_crossover_signal(df)

            if signal_type is None:
                continue

            prev = last_signals.get(pair_name)
            now_str = datetime.utcnow().isoformat()
            if prev and prev["type"] == signal_type:
                continue

            rsi_value = float(df.iloc[-1]["RSI"])
            chart_path = generate_chart(df, pair_name, signal_type, AUTO_TIMEFRAME, entry, sl, tp)

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
# LIVE MODE JOB
# ----------------------------------------------------------------------
def live_job_name(chat_id: int, message_id: int) -> str:
    return f"live_{chat_id}_{message_id}"


async def live_update_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    data = job.data
    chat_id = data["chat_id"]
    message_id = data["message_id"]
    pair_name = data["pair"]
    tf_label = data["tf"]
    data["count"] += 1

    if data["count"] > MAX_LIVE_UPDATES:
        job.schedule_removal()
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Live mode for {pair_name} ({tf_label}) ended after {MAX_LIVE_UPDATES} updates. "
                     f"Tap 🔴 Go Live again to restart.",
            )
        except Exception:
            pass
        return

    try:
        state, entry, sl, tp, rsi_value, chart_path = get_signal_data(pair_name, tf_label)
        caption = build_caption(pair_name, tf_label, state, entry, sl, tp, rsi_value,
                                 live=True, update_count=data["count"])

        with open(chart_path, "rb") as img:
            await context.bot.edit_message_media(
                chat_id=chat_id,
                message_id=message_id,
                media=InputMediaPhoto(media=img, caption=caption),
                reply_markup=live_result_keyboard(pair_name, tf_label),
            )
    except Exception as e:
        logger.error(f"Live update failed for {pair_name} ({tf_label}): {e}")


# ----------------------------------------------------------------------
# COMMAND HANDLERS
# ----------------------------------------------------------------------
HELP_TEXT = (
    "🤖 *FX Chart Signals — Commands*\n\n"
    "/start — subscribe to automatic signals\n"
    "/signal — get an instant chart for any pair + timeframe\n"
    "/stop — unsubscribe from automatic alerts\n"
    "/status — see the last automatic signal per pair\n"
    "/help — show this menu\n\n"
    "After picking a pair and timeframe:\n"
    "🔄 *Refresh* — re-check the same pair/timeframe once\n"
    "🔴 *Go Live* — auto-updates the same message every "
    f"{LIVE_REFRESH_SECONDS}s for up to {MAX_LIVE_UPDATES} updates\n"
    "⏹ *Stop Live* — ends auto-updating early\n"
    "⏱ *Change Timeframe* / 📊 *Change Pair* — get a different reading\n\n"
    "⚠️ Signals are for informational purposes only, not financial advice. "
    "Live mode uses the freshest data available from the free data source — "
    "it is not a broker-grade real-time tick feed."
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_subscriber(update.effective_chat.id)
    pairs_list = ", ".join(PAIRS.keys())
    await update.message.reply_text(
        "✅ You're subscribed to FX Chart Signals!\n\n"
        f"Tracked pairs: {pairs_list}\n\n"
        "Send /help to see everything I can do.",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


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
    """Step 1: pick a pair."""
    keyboard = [
        [InlineKeyboardButton(pair, callback_data=f"pair:{pair}")]
        for pair in PAIRS
    ]
    await update.message.reply_text(
        "Pick a pair:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def timeframe_keyboard(pair_name: str) -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton(tf, callback_data=f"tf:{pair_name}:{tf}")
        for tf in TIMEFRAMES
    ]
    return InlineKeyboardMarkup([row])


def result_keyboard(pair_name: str, tf_label: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"tf:{pair_name}:{tf_label}")],
            [InlineKeyboardButton("🔴 Go Live", callback_data=f"golive:{pair_name}:{tf_label}")],
            [InlineKeyboardButton("⏱ Change Timeframe", callback_data=f"pair:{pair_name}")],
            [InlineKeyboardButton("📊 Change Pair", callback_data="newsignal")],
        ]
    )


def live_result_keyboard(pair_name: str, tf_label: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⏹ Stop Live", callback_data=f"stoplive:{pair_name}:{tf_label}")]]
    )


async def pair_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 2: user picked a pair, now show timeframe options."""
    query = update.callback_query
    await query.answer()
    pair_name = query.data.split("pair:", 1)[1]
    await query.message.reply_text(
        f"{pair_name} — pick a timeframe:",
        reply_markup=timeframe_keyboard(pair_name),
    )


async def newsignal_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """'Change Pair' button — restart at pair selection."""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton(pair, callback_data=f"pair:{pair}")]
        for pair in PAIRS
    ]
    await query.message.reply_text("Pick a pair:", reply_markup=InlineKeyboardMarkup(keyboard))


async def timeframe_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 3: user picked a timeframe (or hit Refresh) -> generate + send chart."""
    query = update.callback_query
    await query.answer()

    _, pair_name, tf_label = query.data.split(":")
    await query.message.reply_text(f"⏳ Generating {pair_name} ({tf_label}) chart...")

    try:
        state, entry, sl, tp, rsi_value, chart_path = get_signal_data(pair_name, tf_label)
        caption = build_caption(pair_name, tf_label, state, entry, sl, tp, rsi_value, live=False)

        with open(chart_path, "rb") as img:
            await query.message.reply_photo(
                photo=img,
                caption=caption,
                reply_markup=result_keyboard(pair_name, tf_label),
            )

    except Exception as e:
        logger.error(f"Error generating on-demand signal for {pair_name} ({tf_label}): {e}")
        await query.message.reply_text(
            f"Sorry, couldn't fetch data for {pair_name} ({tf_label}) right now. Try again shortly."
        )


async def golive_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts a repeating job that edits this message every LIVE_REFRESH_SECONDS."""
    query = update.callback_query
    await query.answer()

    _, pair_name, tf_label = query.data.split(":")
    chat_id = query.message.chat_id
    message_id = query.message.message_id

    # Turn this specific message into a live-updating one immediately
    try:
        state, entry, sl, tp, rsi_value, chart_path = get_signal_data(pair_name, tf_label)
        caption = build_caption(pair_name, tf_label, state, entry, sl, tp, rsi_value, live=True, update_count=0)

        with open(chart_path, "rb") as img:
            await query.edit_message_media(
                media=InputMediaPhoto(media=img, caption=caption),
                reply_markup=live_result_keyboard(pair_name, tf_label),
            )
    except Exception as e:
        logger.error(f"Failed to start live mode for {pair_name} ({tf_label}): {e}")
        await query.message.reply_text("Couldn't start live mode, please try again.")
        return

    # Cancel any existing live job for this exact message, then schedule a fresh one
    job_name = live_job_name(chat_id, message_id)
    for existing in context.job_queue.get_jobs_by_name(job_name):
        existing.schedule_removal()

    context.job_queue.run_repeating(
        live_update_job,
        interval=LIVE_REFRESH_SECONDS,
        first=LIVE_REFRESH_SECONDS,
        data={"chat_id": chat_id, "message_id": message_id, "pair": pair_name, "tf": tf_label, "count": 0},
        name=job_name,
    )


async def stoplive_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, pair_name, tf_label = query.data.split(":")
    chat_id = query.message.chat_id
    message_id = query.message.message_id

    job_name = live_job_name(chat_id, message_id)
    jobs = context.job_queue.get_jobs_by_name(job_name)
    for job in jobs:
        job.schedule_removal()

    await query.message.reply_text(f"⏹ Live mode stopped for {pair_name} ({tf_label}).")
    try:
        await query.edit_message_reply_markup(reply_markup=result_keyboard(pair_name, tf_label))
    except Exception:
        pass


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
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("signal", signal_command))

    application.add_handler(CallbackQueryHandler(pair_button_callback, pattern=r"^pair:"))
    application.add_handler(CallbackQueryHandler(timeframe_button_callback, pattern=r"^tf:"))
    application.add_handler(CallbackQueryHandler(newsignal_button_callback, pattern=r"^newsignal$"))
    application.add_handler(CallbackQueryHandler(golive_button_callback, pattern=r"^golive:"))
    application.add_handler(CallbackQueryHandler(stoplive_button_callback, pattern=r"^stoplive:"))

    application.job_queue.run_repeating(
        periodic_scan, interval=CHECK_INTERVAL_SECONDS, first=10
    )

    logger.info("@FXChartSignals_bot started. Polling for commands...")
    application.run_polling()


if __name__ == "__main__":
    main()
