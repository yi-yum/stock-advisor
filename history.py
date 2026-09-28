import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

# SQLite database file
DB_FILE = "history.db"
# Legacy JSON file (for migration)
LEGACY_JSON_FILE = "history.json"


def _get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    # Create history table with all necessary columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            asset_type TEXT,
            currency TEXT,
            current_price REAL,
            change_pct REAL,
            rsi REAL,
            volume_ratio REAL,
            return_5d REAL,
            pct_from_52w_high REAL,
            analysis TEXT,
            claude_prompt TEXT,
            coin TEXT,
            direction TEXT,
            score REAL,
            funding_rate REAL,
            oi_change_6h REAL,
            ls_ratio REAL,
            rr_long REAL,
            rr_short REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def _migrate_from_json():
    """Migrate data from legacy history.json to SQLite if it exists."""
    if not os.path.exists(LEGACY_JSON_FILE):
        return

    try:
        with open(LEGACY_JSON_FILE, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)

        if not legacy_data:
            os.remove(LEGACY_JSON_FILE)
            return

        conn = _get_connection()
        cursor = conn.cursor()

        for record in legacy_data:
            cursor.execute("""
                INSERT INTO history (
                    timestamp, symbol, asset_type, currency, current_price,
                    change_pct, rsi, volume_ratio, return_5d, pct_from_52w_high,
                    analysis, claude_prompt, coin, direction, score,
                    funding_rate, oi_change_6h, ls_ratio, rr_long, rr_short
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp"),
                record.get("symbol"),
                record.get("asset_type"),
                record.get("currency"),
                record.get("current_price"),
                record.get("change_pct"),
                record.get("rsi"),
                record.get("volume_ratio"),
                record.get("return_5d"),
                record.get("pct_from_52w_high"),
                record.get("analysis"),
                record.get("claude_prompt"),
                record.get("coin"),
                record.get("direction"),
                record.get("score"),
                record.get("funding_rate"),
                record.get("oi_change_6h"),
                record.get("ls_ratio"),
                record.get("rr_long"),
                record.get("rr_short"),
            ))

        conn.commit()
        conn.close()

        # Remove the legacy JSON file after successful migration
        os.remove(LEGACY_JSON_FILE)

    except Exception as e:
        print(f"Migration error: {e}")


def load_history() -> list:
    """
    Load all history records from SQLite.
    Returns a list of dictionaries in the same format as before.
    """
    _init_db()
    _migrate_from_json()

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM history
        ORDER BY timestamp DESC
        LIMIT 100
    """)

    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to regular dictionaries
    history = []
    for row in rows:
        record = dict(row)
        # Remove internal database columns
        record.pop("id", None)
        record.pop("created_at", None)
        # Remove None values to keep the format clean (matching original JSON behavior)
        record = {k: v for k, v in record.items() if v is not None}
        history.append(record)

    return history


def save_result(stock_data: dict, analysis: str, claude_prompt: str):
    """
    Save a stock analysis result to the database.
    Deletes any existing record for the same symbol before inserting.
    """
    _init_db()

    symbol = stock_data["symbol"]

    conn = _get_connection()
    cursor = conn.cursor()

    # Delete existing record for this symbol (keep only the latest)
    cursor.execute("DELETE FROM history WHERE symbol = ?", (symbol,))

    # Insert new record
    cursor.execute("""
        INSERT INTO history (
            timestamp, symbol, asset_type, currency, current_price,
            change_pct, rsi, volume_ratio, return_5d, pct_from_52w_high,
            analysis, claude_prompt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        stock_data["symbol"],
        stock_data["asset_type"],
        stock_data["currency"],
        stock_data["current_price"],
        stock_data["change_pct"],
        stock_data["rsi"],
        stock_data["volume_ratio"],
        stock_data["return_5d"],
        stock_data["pct_from_52w_high"],
        analysis,
        claude_prompt,
    ))

    # Enforce max 100 records
    cursor.execute("""
        DELETE FROM history WHERE id IN (
            SELECT id FROM history ORDER BY timestamp DESC LIMIT -1 OFFSET 100
        )
    """)

    conn.commit()
    conn.close()


def save_short_result(btc_data: dict, analysis: str, claude_prompt: str):
    """
    Save a short-term (crypto) analysis result to the database.
    Deletes any existing record for the same symbol before inserting.
    """
    _init_db()

    coin = btc_data.get("coin", "BTC")
    symbol = f"{coin}-SHORT"

    funding = btc_data.get("funding") or {}
    oi = btc_data.get("oi") or {}
    ls = btc_data.get("ls_ratio") or {}
    h1 = (btc_data.get("tf_data") or {}).get("1h") or {}

    conn = _get_connection()
    cursor = conn.cursor()

    # Delete existing record for this symbol (keep only the latest)
    cursor.execute("DELETE FROM history WHERE symbol = ?", (symbol,))

    # Insert new record
    cursor.execute("""
        INSERT INTO history (
            timestamp, symbol, asset_type, currency, current_price,
            change_pct, rsi, volume_ratio, return_5d, pct_from_52w_high,
            coin, direction, score, funding_rate, oi_change_6h, ls_ratio,
            rr_long, rr_short, analysis, claude_prompt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        symbol,
        "short",
        "$",
        btc_data.get("entry_ref"),
        None,
        h1.get("rsi"),
        None,
        None,
        None,
        coin,
        btc_data.get("direction"),
        btc_data.get("score"),
        funding.get("funding_rate"),
        oi.get("oi_change_6h"),
        ls.get("ratio"),
        btc_data.get("rr_long"),
        btc_data.get("rr_short"),
        analysis,
        claude_prompt,
    ))

    # Enforce max 100 records
    cursor.execute("""
        DELETE FROM history WHERE id IN (
            SELECT id FROM history ORDER BY timestamp DESC LIMIT -1 OFFSET 100
        )
    """)

    conn.commit()
    conn.close()


def delete_record(symbol: str, timestamp: Optional[str] = None):
    """
    Delete record(s) from the database.
    If timestamp is provided, delete only that specific record.
    Otherwise, delete all records for the symbol.
    """
    _init_db()

    conn = _get_connection()
    cursor = conn.cursor()

    if timestamp:
        # Delete specific record
        cursor.execute(
            "DELETE FROM history WHERE symbol = ? AND timestamp = ?",
            (symbol, timestamp)
        )
    else:
        # Delete all records for this symbol
        cursor.execute("DELETE FROM history WHERE symbol = ?", (symbol,))

    conn.commit()
    conn.close()
