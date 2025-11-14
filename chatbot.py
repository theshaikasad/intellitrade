import pandas as pd
import yfinance as yf
import streamlit as st
import logging
import os
import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
import html
from PIL import Image
import io
import base64
import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from pathlib import Path
import altair as alt
import textwrap
from urllib.parse import quote

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

st.set_page_config(page_title="AI Financial Advisor", page_icon="📈", layout="wide")

# Try to import NewsAPI
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

# Try to import cv2 (opencv)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import FAISS with proper error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
except Exception as e:
    FAISS_AVAILABLE = False

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Patch for huggingface_hub compatibility
try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = hf_hub_download
except ImportError:
    pass

# Import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import Tesseract OCR with pytesseract
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False

# -------------------------
# CONFIG & LOGGING
# -------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def require_env(name: str, mask: bool = False) -> str:
    """Fetch a required environment variable or raise a helpful error."""
    value = os.getenv(name)
    if value:
        if mask:
            logger.debug("Loaded secret env var %s", name)
        return value
    raise RuntimeError(f"Environment variable '{name}' must be set before launching the app.")

HF_API_KEY = require_env("HF_API_KEY", mask=True)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    logger.warning("NEWSAPI_KEY not set; market news features will be limited.")
MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DURATION_MINUTES = 30
DB_FILE = "portfolio_cache.db"
FMP_API_KEY = os.getenv("FMP_API_KEY")
VALIDATION_CACHE_DAYS = 7
NEWS_CACHE_DAYS = 30
SLOW_QUERY_THRESHOLD_MS = 100

SAMPLE_PORTFOLIO_ROWS = [
    {"Ticker": "TCS.NS", "Quantity": 10, "Average Price": 3250.0},
    {"Ticker": "INFY.NS", "Quantity": 15, "Average Price": 1475.0},
    {"Ticker": "RELIANCE.NS", "Quantity": 8, "Average Price": 2380.0},
    {"Ticker": "HDFCBANK.NS", "Quantity": 12, "Average Price": 1525.0},
    {"Ticker": "ICICIBANK.NS", "Quantity": 20, "Average Price": 980.0},
]
SAMPLE_PORTFOLIO_DF = pd.DataFrame(SAMPLE_PORTFOLIO_ROWS)
SAMPLE_PORTFOLIO_CSV = SAMPLE_PORTFOLIO_DF.to_csv(index=False)
SAMPLE_PORTFOLIO_DATA_URI = f"data:text/csv;charset=utf-8,{quote(SAMPLE_PORTFOLIO_CSV)}"
SAMPLE_PORTFOLIO_FILENAME = "sample_portfolio.csv"



class TimedCursor(sqlite3.Cursor):
    def execute(self, sql, parameters=()):
        start = time.perf_counter()
        result = super().execute(sql, parameters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > SLOW_QUERY_THRESHOLD_MS:
            logger.warning("Slow query (%.2f ms): %s", elapsed_ms, sql)
        return result

    def executemany(self, sql, seq_of_parameters):
        start = time.perf_counter()
        result = super().executemany(sql, seq_of_parameters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > SLOW_QUERY_THRESHOLD_MS:
            logger.warning("Slow batch query (%.2f ms): %s", elapsed_ms, sql)
        return result


class TimedConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        if factory is None:
            factory = TimedCursor
        return super().cursor(factory)


class SQLiteConnectionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

    def get_connection(self, timeout: int = 30) -> sqlite3.Connection:
        with self._lock:
            conn = sqlite3.connect(
                self.db_path,
                timeout=timeout,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
                factory=TimedConnection,
            )
        conn.row_factory = sqlite3.Row
        return conn


@st.cache_resource
def get_connection_manager() -> SQLiteConnectionManager:
    return SQLiteConnectionManager(DB_FILE)


# -------------------------
# DATABASE SETUP
# -------------------------

def init_database() -> None:
    """Initialise SQLite database for caching and session data."""
    manager = get_connection_manager()
    with manager.get_connection(timeout=60) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_cache (
                ticker TEXT PRIMARY KEY,
                price_inr REAL,
                price_usd REAL,
                company_name TEXT,
                currency TEXT,
                change_percent REAL,
                volume INTEGER,
                last_updated TIMESTAMP,
                market_cap REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS market_indices (
                index_name TEXT PRIMARY KEY,
                current_value REAL,
                change_percent REAL,
                last_updated TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                session_id INTEGER PRIMARY KEY,
                portfolio_json TEXT,
                updated_portfolio_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_validation (
                ticker TEXT PRIMARY KEY,
                is_valid INTEGER NOT NULL,
                exchange TEXT,
                last_checked TIMESTAMP NOT NULL,
                delisting_date TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS news_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                news_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT NOT NULL UNIQUE,
                source TEXT,
                published_date TIMESTAMP,
                sentiment_score REAL,
                cached_at TIMESTAMP NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_news_cache_ticker
            ON news_cache(ticker)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_news_cache_published
            ON news_cache(published_date)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_ticker
            ON stock_cache(ticker)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_updated
            ON stock_cache(last_updated)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_messages(session_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_portfolio_session
            ON portfolio_snapshots(session_id)
            """
        )

        conn.commit()
        try:
            cursor.execute("VACUUM")
        except sqlite3.DatabaseError as exc:
            logger.warning("VACUUM skipped: %s", exc)
    logger.info("Database initialisation completed")

def get_cached_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached stock data if fresh (within CACHE_DURATION_MINUTES)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ticker, price_inr, price_usd, company_name, currency,
                   change_percent, volume, last_updated, market_cap
            FROM stock_cache
            WHERE ticker = ?
              AND datetime(last_updated) > datetime('now', '-' || ? || ' minutes')
            """,
            (ticker, CACHE_DURATION_MINUTES),
        )
        result = cursor.fetchone()

    if result:
        return {
            "ticker": result[0],
            "current_price_inr": result[1],
            "current_price_usd": result[2],
            "company_name": result[3],
            "currency": result[4],
            "change_percent": result[5],
            "volume": result[6],
            "cached": True,
            "cache_time": result[7],
            "market_cap": result[8],
        }
    return None

def save_stock_to_cache(records: Iterable[Dict[str, Any]]) -> None:
    """Persist stock data in the cache table using batch inserts."""
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, Iterable):
        return

    prepared_rows: List[Tuple[Any, ...]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        ticker = record.get("ticker")
        if not ticker:
            continue
        prepared_rows.append(
            (
                ticker,
                record.get("current_price_inr"),
                record.get("current_price_usd"),
                record.get("company_name"),
                record.get("currency"),
                record.get("change_percent"),
                record.get("volume"),
                record.get("market_cap"),
            )
        )

    if not prepared_rows:
        return

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO stock_cache (
                ticker,
                price_inr,
                price_usd,
                company_name,
                currency,
                change_percent,
                volume,
                last_updated,
                market_cap
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            prepared_rows,
        )
        conn.commit()

def get_db_connection(timeout: int = 30) -> sqlite3.Connection:
    return get_connection_manager().get_connection(timeout)

def generate_salt():
    return secrets.token_hex(16)

def hash_password(password: str, salt: str) -> str:
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        150_000
    )
    return base64.b64encode(hashed).decode("utf-8")

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    computed_hash = hash_password(password, salt)
    return hmac.compare_digest(stored_hash, computed_hash)

def create_user(username: str, password: str):
    salt = generate_salt()
    password_hash = hash_password(password, salt)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, salt)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, salt)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return False, "Failed to create account. Please try again."
    finally:
        conn.close()

def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, username, password_hash, salt
            FROM users
            WHERE username = ?
            """,
            (username,)
        )
        row = cursor.fetchone()
        if row and verify_password(password, row["salt"], row["password_hash"]):
            return {"id": row["id"], "username": row["username"]}
        return None
    finally:
        conn.close()

def create_chat_session(user_id: int, title: str | None = None) -> int:
    if not title:
        title = datetime.now().strftime("Session %d %b %Y %H:%M")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO chat_sessions (user_id, title)
            VALUES (?, ?)
            """,
            (user_id, title)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_user_sessions(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, title, created_at, last_active
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY COALESCE(last_active, created_at) DESC, id DESC
            """,
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_chat_messages(session_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (session_id,)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
    finally:
        conn.close()

def save_chat_message(session_id: int, role: str, content: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO chat_messages (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content)
        )
        cursor.execute(
            """
            UPDATE chat_sessions
            SET last_active = datetime('now')
            WHERE id = ?
            """,
            (session_id,)
        )
        conn.commit()
    finally:
        conn.close()

def save_portfolio_snapshot(session_id: int, portfolio_df: pd.DataFrame | None, updated_df: pd.DataFrame | None):
    portfolio_json = portfolio_df.to_json(orient="records") if isinstance(portfolio_df, pd.DataFrame) else None
    updated_json = updated_df.to_json(orient="records") if isinstance(updated_df, pd.DataFrame) else None
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO portfolio_snapshots (session_id, portfolio_json, updated_portfolio_json, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(session_id) DO UPDATE SET
                portfolio_json = excluded.portfolio_json,
                updated_portfolio_json = excluded.updated_portfolio_json,
                updated_at = datetime('now')
            """,
            (session_id, portfolio_json, updated_json)
        )
        cursor.execute(
            """
            UPDATE chat_sessions
            SET last_active = datetime('now')
            WHERE id = ?
            """,
            (session_id,)
        )
        conn.commit()
    finally:
        conn.close()

def get_portfolio_snapshot(session_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT portfolio_json, updated_portfolio_json
            FROM portfolio_snapshots
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        if not row:
            return pd.DataFrame(), pd.DataFrame()

        def json_to_df(payload):
            if not payload:
                return pd.DataFrame()
            try:
                data = json.loads(payload)
                if isinstance(data, list) and data:
                    return pd.DataFrame(data)
                if isinstance(data, list):
                    return pd.DataFrame()
                return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Failed to parse portfolio snapshot: {e}")
                return pd.DataFrame()

        return json_to_df(row["portfolio_json"]), json_to_df(row["updated_portfolio_json"])
    finally:
        conn.close()

# --- Hard-coded Knowledge Bases ---
FINANCIAL_KNOWLEDGE = {
    "portfolio_management": {
        "Diversification Rules": "Allocate across large, mid, and small-cap equities plus debt and gold so no single asset exceeds about 20-25% of portfolio value. Review sector overlap to avoid concentration risk, especially in correlated industries.",
        "Rebalancing Triggers": "Schedule rebalancing every 6-12 months or when an asset class drifts more than 5% from its target weight. Use systematic rebalancing orders to shift gains from overheated segments into underweighted allocations.",
        "Position Sizing": "Cap any single equity position at 5-7% of total portfolio unless a deliberate high-conviction call is documented. For trading-oriented positions, risk no more than 1-2% of capital per trade using stop-loss levels.",
    },
    "indian_market_specifics": {
        "Taxation (LTCG/STCG Rates)": "Equity gains held under 12 months are taxed at 15% plus surcharge and cess as Short-Term Capital Gains. Long-Term Capital Gains above ₹1 lakh annually attract 10% tax without indexation, so harvest gains gradually to stay within thresholds.",
        "Market Hours": "NSE and BSE cash sessions run 9:15 AM to 3:30 PM IST with a 9:00-9:08 AM pre-open auction for price discovery. Plan large orders near mid-session when liquidity peaks and spreads are tighter.",
        "Securities Transaction Tax (STT)": "STT on equity delivery is 0.1% each side, intraday is 0.025% on sell trades, and equity options charge ₹1 per lakh on sell premium. Incorporate these costs into break-even calculations before placing trades.",
    },
    "risk_metrics": {
        "Beta": "Beta measures a stock's volatility relative to the market; values above 1 imply amplified swings. Use beta to scale position size—trim exposures when portfolio beta exceeds your risk tolerance.",
        "Sharpe Ratio": "Sharpe ratio evaluates excess return per unit of volatility; a reading above 1 is generally acceptable, while 2+ signals strong risk-adjusted performance. Compare Sharpe across strategies to prioritize capital deployment.",
        "Max Drawdown": "Max drawdown captures the largest peak-to-trough loss and highlights worst-case capital erosion. Set maximum acceptable drawdown levels (e.g., 15%) to trigger defensive actions like hedging or cutting exposure.",
    },
    "sector_analysis": {
        "Financial Services": "Banks and NBFCs drive earnings from credit growth and net interest margins; monitor RBI policy, credit costs, and CASA ratios. Favor diversified lenders with strong capital adequacy during rate tightening cycles.",
        "Information Technology": "Indian IT services benefit from USD-denominated contracts; revenue visibility hinges on US and EU enterprise tech budgets. Track deal wins, attrition, and margin guidance to gauge sector momentum.",
        "Consumer Staples": "FMCG players rely on distribution reach and rural demand; raw material inflation directly impacts margins. Accumulate quality names on input cost corrections and watch volume growth for early demand signals.",
    },
    "market_indicators": {
        "Nifty 50": "The Nifty 50 captures large-cap Indian performance; use it as the benchmark for equity allocation and hedging decisions. Compare your portfolio returns and beta to Nifty weights to maintain alignment with the core market.",
        "Bank Nifty": "Bank Nifty reflects major banking stocks and is sensitive to credit growth and interest-rate expectations. Deploy Bank Nifty futures or options to hedge concentrated financial exposure or to trade RBI policy outcomes.",
        "India VIX": "India VIX measures implied volatility; readings above 20 imply heightened market stress while sub-12 indicates complacency. Adjust option strategies—sell premium in high VIX regimes with defined risk, buy protection when VIX is abnormally low.",
    },
}

# -------------------------
# OCR PREPROCESSING
# -------------------------
def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    if CV2_AVAILABLE:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return processed
    else:
        from PIL import ImageEnhance, ImageFilter
        gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        return np.array(sharpened)

# -------------------------
# TESSERACT OCR EXTRACTION
# -------------------------
def extract_text_from_image_tesseract(image_file):
    """Extract text from uploaded image using Tesseract OCR"""
    if not TESSERACT_AVAILABLE:
        return []
    try:
        image_file.seek(0)
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_img = preprocess_image_for_ocr(image)
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(processed_img, output_type=Output.DICT, config=custom_config)
        
        extracted_data = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                confidence = int(ocr_data['conf'][i])
                if confidence > 0:
                    extracted_data.append({
                        'text': text,
                        'confidence': confidence / 100.0,
                        'bbox': {
                            'left': ocr_data['left'][i],
                            'top': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        },
                        'line_num': ocr_data['line_num'][i],
                        'word_num': ocr_data['word_num'][i]
                    })
        
        return extracted_data
    except Exception as e:
        logger.error(f"Tesseract OCR extraction error: {e}")
        return []

def parse_portfolio_from_ocr(ocr_results):
    """Parse portfolio data from OCR results - optimized for broker app screenshots"""
    portfolio_entries = []
    seen_tickers = {}
    
    sorted_results = sorted(ocr_results, key=lambda x: (x.get('line_num', 0), x.get('word_num', 0)))
    
    # Group text by line number
    lines_dict = {}
    for item in sorted_results:
        line_num = item.get('line_num', 0)
        if line_num not in lines_dict:
            lines_dict[line_num] = []
        lines_dict[line_num].append(item['text'])
    
    # Skip words that aren't stock tickers
    skip_words = {
        'PORTFOLIO', 'POSITION', 'OPEN', 'QTY', 'AVG', 'VAL', 'CURR', 'MKT', 
        'REALIZED', 'UNREALIZED', 'HOME', 'MENU', 'WATCHLIST', 'ORDERS', 'TOOLS',
        'STOCKS', 'DEMAT', 'HOLDINGS', 'NIFTY', 'BANKNIFTY', 'BETA', 'TRY', 'NOW',
        'CLICK', 'HERE', 'P&L', 'U.', 'ENABLED', 'ACCOUNT', 'FOR', 'YOUR', 'HAS', 'BEEN'
    }
    
    line_numbers = sorted(lines_dict.keys())
    i = 0
    
    while i < len(line_numbers):
        current_line_num = line_numbers[i]
        current_line = ' '.join(lines_dict[current_line_num])
        
        # Enhanced ticker detection - prioritize uppercase sequences
        ticker_matches = re.finditer(r'\b([A-Z][A-Z0-9\-&]{1,14})\b', current_line)
        
        for ticker_match in ticker_matches:
            ticker = ticker_match.group(1)
            
            # Skip if already processed or in skip list
            if ticker in skip_words or ticker in seen_tickers:
                continue
            
            # Look for Qty and Avg in nearby lines
            quantity = None
            avg_price = None
            
            # Search in current line and next 5 lines
            for offset in range(6):
                if i + offset >= len(line_numbers):
                    break
                    
                check_line_num = line_numbers[i + offset]
                check_line = ' '.join(lines_dict[check_line_num])
                
                # Pattern 1: "Qty X Avg Y.YY"
                qty_avg_match = re.search(r'Qty\s+(\d+)\s+Avg\s+([\d,.]+)', check_line, re.IGNORECASE)
                if qty_avg_match:
                    quantity = int(qty_avg_match.group(1))
                    avg_price = float(qty_avg_match.group(2).replace(',', ''))
                    break
                
                # Pattern 2: Separate Qty and Avg
                if quantity is None:
                    qty_match = re.search(r'Qty\s+(\d+)', check_line, re.IGNORECASE)
                    if qty_match:
                        quantity = int(qty_match.group(1))
                
                if avg_price is None:
                    avg_match = re.search(r'Avg\s+([\d,.]+)', check_line, re.IGNORECASE)
                    if avg_match:
                        avg_price = float(avg_match.group(1).replace(',', ''))
            
            # Valid entry found
            if quantity and avg_price and quantity > 0 and avg_price > 0:
                seen_tickers[ticker] = True
                portfolio_entries.append({
                    'ticker': ticker,
                    'quantity': quantity,
                    'avg_price': avg_price
                })
        
        i += 1
    
    # Fallback: consolidated text parsing
    if len(portfolio_entries) < 5:
        fallback_entries = parse_portfolio_from_consolidated_text(sorted_results)
        for entry in fallback_entries:
            if entry['ticker'] not in seen_tickers:
                portfolio_entries.append(entry)
                seen_tickers[entry['ticker']] = True
    
    return portfolio_entries

def parse_portfolio_from_consolidated_text(ocr_results):
    """Fallback: parse from consolidated text"""
    portfolio_entries = []
    all_text = ' '.join([item['text'] for item in ocr_results])
    
    # More flexible pattern
    pattern = r'([A-Z][A-Z0-9\-&]{1,14})\b[^A-Z]{0,150}?Qty\s+(\d+)\s+Avg\s+([\d,.]+)'
    
    skip_words = {
        'PORTFOLIO', 'POSITION', 'OPEN', 'QTY', 'AVG', 'VAL', 'CURR', 
        'MKT', 'REALIZED', 'NIFTY', 'BANKNIFTY', 'P&L', 'U.'
    }
    
    matches = re.finditer(pattern, all_text, re.IGNORECASE)
    seen = set()
    
    for match in matches:
        ticker = match.group(1)
        quantity = int(match.group(2))
        avg_price = float(match.group(3).replace(',', ''))
        
        if ticker not in skip_words and ticker not in seen and quantity > 0 and avg_price > 0:
            seen.add(ticker)
            portfolio_entries.append({
                'ticker': ticker,
                'quantity': quantity,
                'avg_price': avg_price
            })
    
    return portfolio_entries

def create_csv_from_ocr(portfolio_entries):
    """Create DataFrame from parsed portfolio entries"""
    if not portfolio_entries:
        return pd.DataFrame()
    
    df = pd.DataFrame(portfolio_entries)
    
    if 'avg_price' not in df.columns:
        df['avg_price'] = 0.0
    
    return df

# -------------------------
# CURRENCY CONVERSION
# -------------------------
@st.cache_data(ttl=3600)
def get_usd_to_inr_rate():
    """Get current USD to INR exchange rate"""
    session = get_http_session()
    try:
        response = session.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['rates']['INR']
        else:
            return 83.0
    except Exception as e:
        logger.error(f"Currency API error: {e}")
        return 83.0

def usd_to_inr(usd_amount):
    """Convert USD to INR"""
    if usd_amount is None or usd_amount == 0:
        return 0
    exchange_rate = get_usd_to_inr_rate()
    return usd_amount * exchange_rate

def format_inr(amount):
    """Format amount in Indian Rupee style"""
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.2f}"

# -------------------------
# INITIALIZE CLIENTS
# -------------------------
@st.cache_resource
def initialize_clients():
    try:
        hf_client = InferenceClient(token=HF_API_KEY)
        if NEWSAPI_AVAILABLE and NEWSAPI_KEY:
            news_client = NewsApiClient(api_key=NEWSAPI_KEY)
        else:
            news_client = None
        return hf_client, news_client
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        return None, None

hf_client, news_client = initialize_clients()

@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "intelliTrade/1.0"})
    return session

# -------------------------
# STOCK DATA
# -------------------------
def _lookup_company_name(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT company_name
        FROM stock_cache
        WHERE ticker = ?
        """,
        (ticker,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def validate_and_normalize_ticker(ticker: str) -> Dict[str, Any]:
    """Validate a ticker symbol against yfinance and cache the result."""
    original = ticker.strip().upper()
    if not original:
        return {
            "original": ticker,
            "normalized": None,
            "is_valid": False,
            "exchange": None,
            "name": None,
            "error": "Ticker symbol is empty.",
        }

    base_symbol = original.split(".")[0]
    variant_candidates = []
    seen: Set[str] = set()
    for variant in [base_symbol, f"{base_symbol}.NS", f"{base_symbol}.BO"]:
        variant = variant.strip().upper()
        if variant and variant not in seen:
            variant_candidates.append(variant)
            seen.add(variant)

    now = datetime.utcnow()
    last_error: Optional[str] = None
    delisted_on: Optional[str] = None

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for variant in variant_candidates:
            cursor.execute(
                """
                SELECT ticker, is_valid, exchange, last_checked, delisting_date
                FROM stock_validation
                WHERE ticker = ?
                """,
                (variant,),
            )
            cached = cursor.fetchone()
            if cached:
                last_checked_raw = cached["last_checked"]
                try:
                    last_checked_dt = datetime.fromisoformat(last_checked_raw)
                except (TypeError, ValueError):
                    try:
                        last_checked_dt = datetime.strptime(last_checked_raw, "%Y-%m-%d %H:%M:%S")
                    except (TypeError, ValueError):
                        last_checked_dt = None
                if last_checked_dt and now - last_checked_dt <= timedelta(days=VALIDATION_CACHE_DAYS):
                    if cached["is_valid"]:
                        company_name = _lookup_company_name(conn, variant)
                        return {
                            "original": original,
                            "normalized": variant,
                            "is_valid": True,
                            "exchange": cached["exchange"],
                            "name": company_name,
                            "error": None,
                            "delisting_date": cached["delisting_date"],
                            "source": "cache",
                        }
                    else:
                        delisted_on = cached["delisting_date"]
                        last_error = f"No active market data for {variant}"
                        continue

            try:
                yf_ticker = yf.Ticker(variant)
                info = yf_ticker.info or {}
            except Exception as exc:
                last_error = f"{variant}: {exc}"
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO stock_validation (ticker, is_valid, exchange, last_checked, delisting_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (variant, 0, None, now.isoformat(), None),
                )
                conn.commit()
                continue

            regular_price = info.get("regularMarketPrice")
            exchange = info.get("exchange") or info.get("fullExchangeName") or info.get("market")
            company_name = info.get("longName") or info.get("shortName") or info.get("symbol")

            delisting_raw = info.get("delistingDate") or info.get("delisting_date")
            delisting_str: Optional[str] = None
            if isinstance(delisting_raw, (int, float)) and delisting_raw > 0:
                try:
                    delisting_str = datetime.utcfromtimestamp(delisting_raw).date().isoformat()
                except (OverflowError, ValueError):
                    delisting_str = None
            elif isinstance(delisting_raw, str):
                delisting_str = delisting_raw

            if regular_price is not None:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO stock_validation (ticker, is_valid, exchange, last_checked, delisting_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (variant, 1, exchange, now.isoformat(), delisting_str),
                )
                conn.commit()
                return {
                    "original": original,
                    "normalized": variant,
                    "is_valid": True,
                    "exchange": exchange,
                    "name": company_name,
                    "error": None,
                    "delisting_date": delisting_str,
                    "source": "fresh",
                }

            last_error = f"{variant}: regularMarketPrice unavailable"
            if delisting_str:
                delisted_on = delisting_str

            cursor.execute(
                """
                INSERT OR REPLACE INTO stock_validation (ticker, is_valid, exchange, last_checked, delisting_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (variant, 0, exchange, now.isoformat(), delisting_str),
            )
            conn.commit()

    friendly_error = last_error or f"Unable to validate ticker '{original}'."
    if delisted_on:
        friendly_error = f"Ticker '{original}' appears to be delisted. Last known trading date: {delisted_on}."

    return {
        "original": original,
        "normalized": None,
        "is_valid": False,
        "exchange": None,
        "name": None,
        "error": friendly_error,
        "delisting_date": delisted_on,
        "source": "fresh",
    }

def fetch_current_stock_data(ticker: str):
    """Fetches current stock data with intelligent caching"""
    validation = validate_and_normalize_ticker(ticker)
    if not validation.get("is_valid"):
        error_message = validation.get("error") or "Ticker is invalid or delisted."
        logger.warning(f"Skipping data fetch for invalid ticker '{ticker}': {error_message}")
        return {
            "ticker": validation.get("original", ticker),
            "error": error_message,
            "status": "invalid_ticker",
            "validation": validation,
        }

    ticker_norm = validation["normalized"]
    delisting_date = validation.get("delisting_date")
    friendly_error = validation.get("error") or f"Unable to fetch live data for {ticker_norm}."
    
    # Check cache first
    cached_data = get_cached_stock_data(ticker_norm)
    if cached_data:
        logger.info(f"Using cached data for {ticker_norm}")
        if validation.get("exchange"):
            cached_data.setdefault("exchange", validation["exchange"])
        if validation.get("name") and not cached_data.get("company_name"):
            cached_data["company_name"] = validation["name"]
        cached_data.setdefault("validation", validation)
        return cached_data

    # Fetch fresh data
    try:
        stock = yf.Ticker(ticker_norm)

        # --- Primary price/volume via history with graceful fallbacks ---
        history_error: Optional[Exception] = None
        hist = pd.DataFrame()
        try:
            hist = stock.history(period="5d")
        except Exception as hist_exc:
            history_error = hist_exc

        close_series = None
        volume_series = None
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            candidate = hist
            if isinstance(hist.columns, pd.MultiIndex):
                try:
                    candidate = hist.xs(ticker_norm, level=1, axis=1)
                except (KeyError, IndexError, ValueError):
                    candidate = hist
            if "Close" in candidate:
                close_series = candidate["Close"]
            if "Volume" in candidate:
                volume_series = candidate["Volume"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            if isinstance(volume_series, pd.DataFrame):
                volume_series = volume_series.iloc[:, 0]

        current_price = None
        prev_price = None
        if close_series is not None and not close_series.empty:
            current_price = _safe_float(close_series.iloc[-1])
            if len(close_series) > 1:
                prev_price = _safe_float(close_series.iloc[-2])

        volume_value: Optional[int] = None
        if volume_series is not None and not volume_series.empty:
            try:
                volume_value = int(float(volume_series.iloc[-1]))
            except (TypeError, ValueError):
                volume_value = None

        # --- Fallback to fast_info when history is unavailable ---
        fast_info: Dict[str, Any] = {}
        try:
            fast_info = getattr(stock, "fast_info", {}) or {}
        except Exception:
            fast_info = {}

        if current_price is None and fast_info:
            current_price = _safe_float(
                fast_info.get("lastPrice")
                or fast_info.get("regularMarketPrice")
                or fast_info.get("previousClose")
            )
        if prev_price is None:
            prev_price = _safe_float(fast_info.get("previousClose")) if fast_info else None
        if volume_value is None and fast_info:
            candidate_volume = (
                fast_info.get("lastVolume")
                or fast_info.get("volume")
                or fast_info.get("tenDayAverageVolume")
            )
            try:
                if candidate_volume is not None:
                    volume_value = int(float(candidate_volume))
            except (TypeError, ValueError):
                volume_value = None

        if current_price is None:
            msg = f"Unable to retrieve price data for {ticker_norm}"
            if history_error:
                msg += f" (history error: {history_error})"
            raise ValueError(msg)

        if prev_price is None or prev_price == 0:
            prev_price = current_price

        change = (current_price - prev_price) if (current_price is not None and prev_price is not None) else 0.0
        change_percent = (change / prev_price * 100.0) if prev_price else 0.0

        # --- Metadata lookups ---
        info: Dict[str, Any] = {}
        try:
            info = stock.info or {}
        except Exception as info_exc:
            logger.debug("stock.info lookup failed for %s: %s", ticker_norm, info_exc)
            info = {}

        stock_currency = info.get("currency") or fast_info.get("currency") or "INR"

        if stock_currency == "USD":
            price_inr = usd_to_inr(current_price)
            price_usd = current_price
        else:
            price_inr = current_price
            price_usd = None

        company_name = (
            validation.get("name")
            or info.get("longName")
            or info.get("shortName")
            or fast_info.get("shortName")
            or ticker_norm
        )

        market_cap = info.get("marketCap") or fast_info.get("marketCap") or 0

        result = {
            "ticker": ticker_norm,
            "company_name": company_name,
            "current_price_usd": float(price_usd) if price_usd is not None else None,
            "current_price_inr": round(float(price_inr), 2),
            "change_percent": round(float(change_percent), 2),
            "volume": volume_value if volume_value is not None else 0,
            "currency": stock_currency,
            "market_cap": market_cap,
            "cached": False,
            "exchange": validation.get("exchange"),
            "validation": validation,
        }
        
        # Save to cache
        save_stock_to_cache([result])
        
        return result
        
    except Exception as e:
        error_message = str(e)
        if any(keyword in error_message.lower() for keyword in ["no data", "not found", "possibly delisted"]):
            logger.warning(f"No live data for {ticker_norm}: {error_message}")
        else:
            logger.error(f"Error fetching {ticker_norm}: {error_message}")
        
        # Try cache as fallback
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM stock_cache WHERE ticker = ?",
                (ticker_norm,),
            )
            result = cursor.fetchone()
        
        if result:
            cache_time_raw = result[7]
            cache_dt = _parse_db_timestamp(cache_time_raw)
            if cache_dt:
                age_text = _format_time_ago(cache_dt)
                cache_message = (
                    f"⚠️ Couldn't fetch {ticker_norm} data. Using cached price from {age_text}."
                )
            else:
                cache_message = f"⚠️ Couldn't fetch {ticker_norm} data. Using cached price."
            return {
                "ticker": result[0],
                "current_price_inr": result[1],
                "current_price_usd": result[2],
                "company_name": result[3],
                "currency": result[4],
                "change_percent": result[5],
                "volume": result[6],
                "cached": True,
                "cache_time": cache_time_raw,
                "exchange": validation.get("exchange"),
                "validation": validation,
                "warning_message": cache_message,
            }
        if delisting_date:
            friendly_error = (
                f"Ticker '{validation.get('original', ticker)}' appears to be delisted. "
                f"Last known trading date: {delisting_date}."
            )
        
        return {
            "error": friendly_error,
            "ticker": ticker_norm,
            "status": "not_found",
            "validation": validation,
        }



# -------------------------
# NEWS FETCH
# -------------------------

POSITIVE_SENTIMENT_KEYWORDS: Set[str] = {
    "profit",
    "gain",
    "gains",
    "growth",
    "rally",
    "surge",
    "bullish",
    "upgrade",
    "upgrades",
    "beat",
    "beats",
}

NEGATIVE_SENTIMENT_KEYWORDS: Set[str] = {
    "loss",
    "losses",
    "fall",
    "falls",
    "drop",
    "drops",
    "crash",
    "crashes",
    "bearish",
    "downgrade",
    "downgrades",
    "miss",
    "misses",
    "decline",
    "declines",
}


def analyze_sentiment(text: str | None) -> float:
    """Return a sentiment score between -1.0 and 1.0 using simple keyword matching."""
    if not text:
        return 0.0

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in POSITIVE_SENTIMENT_KEYWORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_SENTIMENT_KEYWORDS)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0

    score = (positive_hits - negative_hits) / total_hits
    return max(-1.0, min(1.0, score))


def _compute_article_sentiment(title: str | None, description: str | None) -> float:
    title_score = analyze_sentiment(title)
    description_score = analyze_sentiment(description)

    weights = 0.0
    weighted_total = 0.0

    if title:
        weights += 2.0
        weighted_total += 2.0 * title_score
    if description:
        weights += 1.0
        weighted_total += 1.0 * description_score

    if weights == 0:
        return 0.0

    combined = weighted_total / weights
    return max(-1.0, min(1.0, combined))


def _sentiment_indicator(score: Optional[float]) -> Tuple[str, str]:
    if score is None:
        return "➖ Neutral", "neutral"
    if score >= 0.3:
        return "📈 Positive", "positive"
    if score <= -0.3:
        return "📉 Negative", "negative"
    return "➖ Neutral", "neutral"


def get_average_sentiment_for_tickers(
    tickers: Iterable[str],
    days: int = NEWS_CACHE_DAYS,
) -> Dict[str, Tuple[float, int]]:
    unique_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker})
    if not unique_tickers:
        return {}

    placeholders = ",".join("?" for _ in unique_tickers)
    if not placeholders:
        return {}

    params: List[Any] = list(unique_tickers)
    params.append(f"-{days} days")

    query = f"""
        SELECT
            ticker,
            AVG(sentiment_score) AS avg_score,
            COUNT(sentiment_score) AS news_count
        FROM news_cache
        WHERE ticker IN ({placeholders})
          AND sentiment_score IS NOT NULL
          AND datetime(published_date) >= datetime('now', ?)
        GROUP BY ticker
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    sentiment_map: Dict[str, Tuple[float, int]] = {}
    for row in rows:
        ticker = row["ticker"]
        avg_score = row["avg_score"]
        count = row["news_count"]
        if ticker and avg_score is not None and count:
            sentiment_map[ticker] = (float(avg_score), int(count))

    return sentiment_map


def _format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _parse_db_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    return None


def _format_time_ago(timestamp: datetime) -> str:
    delta = datetime.utcnow() - timestamp
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _prepare_market_result(
    value: Any,
    change: Any,
    change_pct: Any,
    last_updated: Optional[datetime],
    is_cached: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    value_f = _safe_float(value)
    change_pct_f = _safe_float(change_pct)
    change_f = _safe_float(change)

    entry: Dict[str, Any] = {
        "value": round(value_f, 2) if value_f is not None else None,
        "change": round(change_f, 2) if change_f is not None else None,
        "change_pct": round(change_pct_f, 2) if change_pct_f is not None else None,
        "last_updated": last_updated.isoformat() if last_updated else None,
        "is_cached": is_cached,
    }
    if error:
        entry["error"] = error
    return entry


def _normalize_published_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, (int, float)):
        try:
            return _format_datetime(datetime.utcfromtimestamp(float(value)))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        working = text.replace("T", " ")
        if working.endswith("Z"):
            working = working[:-1]
        for sep in ("+", "-"):
            if sep in working[10:]:
                working = working.split(sep, 1)[0]
                break
        candidates = (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        )
        for fmt in candidates:
            try:
                dt = datetime.strptime(working, fmt)
                return _format_datetime(dt)
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            return _format_datetime(dt)
        except (ValueError, AttributeError):
            return None
    return None


def clean_news_cache(conn: sqlite3.Connection, days: int = NEWS_CACHE_DAYS) -> None:
    cutoff = _format_datetime(datetime.utcnow() - timedelta(days=days))
    conn.execute(
        """
        DELETE FROM news_cache
        WHERE datetime(cached_at) < datetime(?)
        """,
        (cutoff,),
    )
    conn.commit()


def save_news_to_cache(
    articles: Iterable[Dict[str, Any]],
    ticker: Optional[str],
    news_type: str,
    cache_days: int = NEWS_CACHE_DAYS,
) -> None:
    rows: List[Tuple[Any, ...]] = []
    normalized_ticker = ticker.upper() if ticker else None
    now_str = _format_datetime(datetime.utcnow())

    for article in articles:
        title = (article.get("title") or "").strip()
        url = (article.get("url") or "").strip()
        if not title or not url:
            continue
        description = article.get("description") or article.get("summary")
        published = (
            _normalize_published_date(
                article.get("published")
                or article.get("published_date")
                or article.get("publishedAt")
            )
            or now_str
        )
        sentiment = article.get("sentiment_score")
        if sentiment is None:
            sentiment = _compute_article_sentiment(title, description)
        else:
            try:
                sentiment = float(sentiment)
            except (TypeError, ValueError):
                sentiment = _compute_article_sentiment(title, description)
        source = article.get("source")
        rows.append(
            (
                normalized_ticker,
                news_type,
                title,
                description,
                url,
                source,
                published,
                sentiment,
                now_str,
            )
        )

    if not rows:
        return

    with get_db_connection() as conn:
        cursor = conn.cursor()
        clean_news_cache(conn, cache_days)
        cursor.executemany(
            """
            INSERT OR IGNORE INTO news_cache (
                ticker,
                news_type,
                title,
                description,
                url,
                source,
                published_date,
                sentiment_score,
                cached_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def get_cached_news(
    ticker: Optional[str],
    days: int = NEWS_CACHE_DAYS,
    limit: int = 10,
    news_type: str = "stock",
) -> List[Dict[str, Any]]:
    normalized_ticker = ticker.upper() if ticker else None
    cutoff = _format_datetime(datetime.utcnow() - timedelta(days=days))

    with get_db_connection() as conn:
        cursor = conn.cursor()
        clean_news_cache(conn, days)

        params: List[Any] = [news_type, cutoff]
        query = """
            SELECT title, description, url, source, published_date, sentiment_score
            FROM news_cache
            WHERE news_type = ?
              AND datetime(published_date) >= datetime(?)
        """
        if normalized_ticker:
            query += " AND ticker = ?"
            params.append(normalized_ticker)
        else:
            query += " AND ticker IS NULL"

        query += " ORDER BY datetime(published_date) DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

    articles: List[Dict[str, Any]] = []
    for row in rows:
        sentiment = row["sentiment_score"]
        if sentiment is not None:
            try:
                sentiment = float(sentiment)
            except (TypeError, ValueError):
                sentiment = None
        articles.append(
            {
                "title": row["title"],
                "description": row["description"],
                "url": row["url"],
                "published": row["published_date"],
                "source": row["source"],
                "sentiment_score": sentiment,
            }
        )

    return articles


def _shorten_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _fetch_news_via_newsapi(query: str, max_articles: int, lookback_days: int, sort_by: str = "relevancy") -> List[Dict[str, str]]:
    if not news_client:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        response = news_client.get_everything(
            q=query,
            language="en",
            sort_by=sort_by,
            page_size=max_articles,
            from_param=from_date,
        )
        articles: List[Dict[str, str]] = []
        for article in response.get("articles", []):
            title = article.get("title")
            description = article.get("description")
            if not title or not description:
                continue
            articles.append(
                {
                    "title": title,
                    "description": _shorten_text(description),
                    "url": article.get("url", ""),
                    "published": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", ""),
                }
            )
        return articles
    except Exception as exc:
        logger.warning(f"NewsAPI fetch failed for query '{query}': {exc}")
        return []


def _fetch_news_via_fmp(tickers: List[str] | None, max_articles: int) -> List[Dict[str, str]]:
    if not FMP_API_KEY:
        return []
    session = get_http_session()
    try:
        params = {
            "apikey": FMP_API_KEY,
            "limit": max_articles,
        }
        if tickers:
            params["tickers"] = ",".join({t for t in tickers if t})
        response = session.get(
            "https://financialmodelingprep.com/api/v3/stock_news",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        articles: List[Dict[str, str]] = []
        for item in payload or []:
            title = item.get("title")
            summary = item.get("text")
            if not title or not summary:
                continue
            articles.append(
                {
                    "title": title,
                    "description": _shorten_text(summary),
                    "url": item.get("url", ""),
                    "published": item.get("publishedDate", ""),
                    "source": item.get("site", ""),
                }
            )
            if len(articles) >= max_articles:
                break
        return articles
    except Exception as exc:
        logger.warning(f"FMP news fetch failed: {exc}")
        return []


def _fetch_news_via_yfinance(ticker: str, max_articles: int) -> List[Dict[str, str]]:
    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news or []
        articles: List[Dict[str, str]] = []
        for item in raw_news:
            title = item.get("title")
            if not title:
                continue
            summary = item.get("summary") or item.get("content") or ""
            published = item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published_dt = datetime.fromtimestamp(published)
                published_str = published_dt.isoformat()
            else:
                published_str = ""
            articles.append(
                {
                    "title": title,
                    "description": _shorten_text(summary),
                    "url": item.get("link", ""),
                    "published": published_str,
                    "source": item.get("publisher", ""),
                }
            )
            if len(articles) >= max_articles:
                break
        return articles
    except Exception as exc:
        logger.warning(f"Yahoo Finance news fallback failed for {ticker}: {exc}")
        return []


def _dedupe_articles(articles: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for article in articles:
        title = (article.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
        if len(unique) >= limit:
            break
    return unique


@st.cache_data(ttl=600)
def fetch_stock_news(ticker: str, max_articles: int = 3):
    """Fetch news for a specific stock with multi-provider fallbacks."""
    desired_count = max(max_articles, 15)
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()

    cached_articles = get_cached_news(clean_ticker, days=NEWS_CACHE_DAYS, limit=desired_count)
    collected: List[Dict[str, Any]] = list(_dedupe_articles(cached_articles, desired_count))

    if len(collected) >= desired_count:
        return collected[:desired_count]

    existing_urls = {article.get("url") for article in collected if article.get("url")}

    provider_fetchers = (
        lambda: _fetch_news_via_newsapi(f"{clean_ticker} stock", max_articles=desired_count * 2, lookback_days=7),
        lambda: _fetch_news_via_fmp([clean_ticker], max_articles=desired_count * 2),
        lambda: _fetch_news_via_yfinance(clean_ticker, max_articles=desired_count * 2),
    )

    for fetcher in provider_fetchers:
        articles = fetcher()
        if not articles:
            continue

        fresh_articles = []
        for article in articles:
            url = article.get("url")
            if url and url not in existing_urls:
                existing_urls.add(url)
                sentiment = article.get("sentiment_score")
                if sentiment is None:
                    sentiment = _compute_article_sentiment(
                        article.get("title"),
                        article.get("description") or article.get("summary"),
                    )
                enriched_article = {**article, "sentiment_score": sentiment}
                fresh_articles.append(enriched_article)

        if not fresh_articles:
            continue

        collected.extend(fresh_articles)
        collected = _dedupe_articles(collected, desired_count)

        save_news_to_cache(fresh_articles, clean_ticker, "stock")

        if len(collected) >= desired_count:
            break

    if collected:
        return collected[:desired_count]

    logger.info(f"No news found for {clean_ticker} across providers or cache")
    return []


@st.cache_data(ttl=1800)
def fetch_market_news(max_articles: int = 5):
    """Fetch broader market news with fallbacks for resiliency."""
    desired_count = max(max_articles, 15)
    cached_articles = get_cached_news(None, days=NEWS_CACHE_DAYS, limit=desired_count, news_type="market")
    collected: List[Dict[str, Any]] = list(_dedupe_articles(cached_articles, desired_count))
    existing_urls = {article.get("url") for article in collected if article.get("url")}

    if len(collected) >= desired_count:
        return collected[:desired_count]

    query = "India stock market OR NSE OR BSE OR Nifty OR Sensex"
    provider_fetchers = [
        lambda: _fetch_news_via_newsapi(query, max_articles=desired_count * 2, lookback_days=3, sort_by="publishedAt"),
        lambda: _fetch_news_via_fmp(None, max_articles=desired_count * 2),
    ]

    for fetcher in provider_fetchers:
        articles = fetcher()
        if not articles:
            continue

        fresh_articles = []
        for article in articles:
            url = article.get("url")
            if url and url not in existing_urls:
                existing_urls.add(url)
                sentiment = article.get("sentiment_score")
                if sentiment is None:
                    sentiment = _compute_article_sentiment(
                        article.get("title"),
                        article.get("description") or article.get("summary"),
                    )
                enriched_article = {**article, "sentiment_score": sentiment}
                fresh_articles.append(enriched_article)

        if not fresh_articles:
            continue

        collected.extend(fresh_articles)
        collected = _dedupe_articles(collected, desired_count)

        save_news_to_cache(fresh_articles, None, "market")

        if len(collected) >= desired_count:
            return collected[:desired_count]

    for index_symbol in ("^NSEI", "^BSESN", "^GSPC", "^IXIC", "^FTSE"):
        articles = _fetch_news_via_yfinance(index_symbol, max_articles=desired_count * 2)
        if not articles:
            continue

        fresh_articles = []
        for article in articles:
            url = article.get("url")
            if url and url not in existing_urls:
                existing_urls.add(url)
                sentiment = article.get("sentiment_score")
                if sentiment is None:
                    sentiment = _compute_article_sentiment(
                        article.get("title"),
                        article.get("description") or article.get("summary"),
                    )
                enriched_article = {**article, "sentiment_score": sentiment}
                fresh_articles.append(enriched_article)

        if not fresh_articles:
            continue

        collected.extend(fresh_articles)
        collected = _dedupe_articles(collected, desired_count)
        save_news_to_cache(fresh_articles, None, "market")

        if len(collected) >= desired_count:
            return collected[:desired_count]

    if collected:
        return collected[:desired_count]

    logger.info("No market news returned from providers, cache empty")
    return []


def get_nifty50_stocks() -> List[str]:
    """Get Nifty 50 stock list with basic metrics"""
    return [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "BAJFINANCE.NS",
        "ASIANPAINT.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "AXISBANK.NS",
    ]


@st.cache_data(ttl=3600)
def get_stock_recommendations(max_stocks: int = 5) -> List[Dict[str, Any]]:
    """Get top performing stocks from Nifty 50"""
    tickers = get_nifty50_stocks()
    recommendations: List[Dict[str, Any]] = []

    for ticker in tickers[:20]:  # Check first 20 to save time
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            hist = stock.history(period="1mo")

            if hist.empty:
                continue

            current_price = float(hist["Close"].iloc[-1])
            month_ago_price = float(hist["Close"].iloc[0])
            if month_ago_price == 0:
                continue
            momentum = ((current_price - month_ago_price) / month_ago_price) * 100

            recommendations.append(
                {
                    "ticker": ticker,
                    "name": info.get("longName", ticker),
                    "price": current_price,
                    "momentum": momentum,
                    "pe_ratio": info.get("trailingPE", 0) or 0,
                    "market_cap": info.get("marketCap", 0) or 0,
                }
            )
        except Exception:
            continue

    recommendations.sort(key=lambda x: x.get("momentum", 0), reverse=True)
    return recommendations[:max_stocks]

# -------------------------
# PORTFOLIO PROCESSING
# -------------------------
# PORTFOLIO PROCESSING
# -------------------------
def parse_portfolio_csv(df):
    """Parse the portfolio CSV and standardize column names"""
    column_mapping = {
        'Stock': 'ticker', 'stock': 'ticker', 'Ticker': 'ticker', 'ticker': 'ticker',
        'Qty': 'quantity', 'qty': 'quantity', 'Quantity': 'quantity', 'quantity': 'quantity',
        'Avg': 'avg_price', 'avg': 'avg_price', 'Average': 'avg_price', 'avg_price': 'avg_price',
    }

    df_renamed = df.rename(columns=column_mapping)
    required_cols = ['ticker', 'quantity']
    for col in required_cols:
        if col not in df_renamed.columns:
            st.error(f"Missing required column: {col}")
            return None

    df_clean = df_renamed.copy()
    df_clean['quantity'] = pd.to_numeric(df_clean['quantity'], errors='coerce')
    if 'avg_price' in df_clean.columns:
        df_clean['avg_price'] = pd.to_numeric(df_clean['avg_price'], errors='coerce')
    else:
        df_clean['avg_price'] = 0.0

    df_clean = df_clean[df_clean['quantity'] > 0]
    df_clean = df_clean.dropna(subset=['ticker'])

    return df_clean

def update_portfolio_with_current_data(portfolio_df):
    """Update portfolio with current market data"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    updated_rows = []
    tickers_to_fetch = portfolio_df['ticker'].unique()
    total_tickers = len(tickers_to_fetch)
    if total_tickers == 0:
        return pd.DataFrame()

    progress_text = st.empty()
    progress_bar = st.progress(0)

    live_data: dict[str, dict[str, Any]] = {}
    cache_warnings: Set[str] = set()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_current_stock_data, ticker): ticker for ticker in tickers_to_fetch}
        for idx, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                live_data[ticker] = future.result()
            except Exception as exc:
                logger.error(f"Failed to fetch {ticker}: {exc}")
                live_data[ticker] = {"error": str(exc)}
            progress = (idx + 1) / total_tickers
            progress_bar.progress(progress)
            progress_text.text(f"Fetching data: {idx + 1}/{total_tickers} stocks")

    progress_text.empty()
    progress_bar.empty()
    
    for _, row in portfolio_df.iterrows():
        ticker = row.get('ticker', '').strip()
        if not ticker:
            continue
        
        quantity = float(row.get('quantity', 0))
        if quantity <= 0:
            continue
        
        avg_price = float(row.get('avg_price', 0)) if pd.notna(row.get('avg_price')) else 0
        
        current_data = live_data.get(ticker, {})

        if 'error' not in current_data:
            current_price_inr = current_data['current_price_inr']
            market_value = current_price_inr * quantity
            
            if avg_price and avg_price > 0:
                invested_value = avg_price * quantity
                unrealized_pl = market_value - invested_value
                unrealized_pl_percent = (unrealized_pl / invested_value * 100) if invested_value > 0 else 0
            else:
                unrealized_pl = 0
                unrealized_pl_percent = 0
            
            updated_row = {
                'ticker': current_data['ticker'],
                'company_name': current_data['company_name'],
                'quantity': int(quantity),
                'avg_price': round(float(avg_price), 2),
                'current_price_inr': round(float(current_price_inr), 2),
                'current_price_usd': current_data.get('current_price_usd'),
                'change_percent': round(float(current_data['change_percent']), 2),
                'market_value_inr': round(float(market_value), 2),
                'unrealized_pl_inr': round(float(unrealized_pl), 2),
                'unrealized_pl_percent': round(float(unrealized_pl_percent), 2),
                'currency': current_data['currency'],
                'is_cached': current_data.get('cached', False)
            }
            if current_data.get('warning_message'):
                cache_warnings.add(current_data['warning_message'])
        else:
            fallback_price = float(avg_price) if avg_price else 0
            market_value = fallback_price * quantity
            updated_row = {
                'ticker': ticker,
                'company_name': ticker.replace('.NS', ''),
                'quantity': int(quantity),
                'avg_price': float(avg_price) if avg_price else 0,
                'current_price_inr': fallback_price,
                'current_price_usd': None,
                'change_percent': 0.0,
                'market_value_inr': market_value,
                'unrealized_pl_inr': 0.0,
                'unrealized_pl_percent': 0.0,
                'currency': 'INR',
                'error': current_data['error'],
                'is_cached': False
            }
            if current_data.get('error'):
                cache_warnings.add(current_data['error'])
        
        updated_rows.append(updated_row)
        
    for warning in sorted(cache_warnings):
        st.warning(warning)
        
    return pd.DataFrame(updated_rows)

def build_allocation_chart(portfolio_df: pd.DataFrame):
    if portfolio_df.empty or 'market_value_inr' not in portfolio_df.columns:
        return None
    chart_df = (
        portfolio_df
        .sort_values('market_value_inr', ascending=False)
        .head(8)
        .copy()
    )
    chart_df['Allocation'] = chart_df['market_value_inr']
    chart_df['Ticker'] = chart_df['ticker']
    if chart_df['Allocation'].sum() == 0:
        return None
    chart = (
        alt.Chart(chart_df)
        .mark_arc(innerRadius=60, outerRadius=140)
        .encode(
            theta=alt.Theta(field='Allocation', type='quantitative', stack=True),
            color=alt.Color(
                field='Ticker',
                type='nominal',
                legend=alt.Legend(labelColor='#e2e8f0', title=None)
            ),
            tooltip=[
                alt.Tooltip('Ticker:N', title='Ticker'),
                alt.Tooltip('company_name:N', title='Company'),
                alt.Tooltip('Allocation:Q', title='Market Value (₹)', format=',.0f')
            ]
        )
        .properties(height=320, width=360)
        .configure_legend(
            symbolType='circle',
            labelFontSize=12
        )
        .configure_view(strokeOpacity=0)
    )
    return chart

def build_pl_chart(portfolio_df: pd.DataFrame):
    if portfolio_df.empty or 'unrealized_pl_inr' not in portfolio_df.columns:
        return None
    winners = portfolio_df.nlargest(5, 'unrealized_pl_inr')
    losers = portfolio_df.nsmallest(5, 'unrealized_pl_inr')
    chart_df = pd.concat([winners, losers]).drop_duplicates(subset=['ticker']).copy()
    chart_df['Ticker'] = chart_df['ticker']
    chart_df['Unrealized P&L (₹)'] = chart_df['unrealized_pl_inr']
    chart_df['Company'] = chart_df['company_name']
    if chart_df.empty:
        return None
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=6)
        .encode(
            y=alt.Y('Ticker:N', sort='-x', title=None),
            x=alt.X('Unrealized P&L (₹):Q', title='Unrealized P&L (₹)', axis=alt.Axis(format=',.0f')),
            color=alt.condition(
                alt.datum["Unrealized P&L (₹)"] >= 0,
                alt.value('#34d399'),
                alt.value('#f87171')
            ),
            tooltip=[
                alt.Tooltip('Ticker:N', title='Ticker'),
                alt.Tooltip('Company:N', title='Company'),
                alt.Tooltip('Unrealized P&L (₹):Q', title='Unrealized P&L (₹)', format=',.0f')
            ]
        )
        .properties(height=320, width=360)
        .configure_view(strokeOpacity=0)
    )
    return chart


def calculate_portfolio_health_score(portfolio_df: pd.DataFrame) -> Tuple[int, Dict[str, int]]:
    """Compute a 0-100 portfolio health score based on diversification, performance, volatility, sizing, and sentiment."""
    if portfolio_df.empty:
        breakdown = {
            "diversification": 0,
            "performance": 0,
            "volatility": 0,
            "position_sizing": 0,
            "sentiment": 0,
        }
        return 0, breakdown

    df = portfolio_df.copy()

    weights = {
        "diversification": 20,
        "performance": 20,
        "volatility": 20,
        "position_sizing": 20,
        "sentiment": 20,
    }

    breakdown: Dict[str, int] = {}

    # Diversification: penalize concentration >15% per stock or >30% per sector
    diversification_score = weights["diversification"]
    if "market_value_inr" in df.columns and df["market_value_inr"].sum() > 0:
        total_value = df["market_value_inr"].sum()
        df["allocation_pct"] = df["market_value_inr"] / total_value * 100
        worst_stock = df["allocation_pct"].max()
        if worst_stock > 30:
            diversification_score -= 15
        elif worst_stock > 20:
            diversification_score -= 10
        elif worst_stock > 15:
            diversification_score -= 5

        if "sector" in df.columns:
            sector_allocations = df.groupby("sector")["market_value_inr"].sum() / total_value * 100
            worst_sector = sector_allocations.max()
            if worst_sector > 40:
                diversification_score -= 15
            elif worst_sector > 35:
                diversification_score -= 10
            elif worst_sector > 30:
                diversification_score -= 5
    else:
        diversification_score = int(weights["diversification"] * 0.5)
    diversification_score = max(0, diversification_score)
    breakdown["diversification"] = diversification_score

    # Performance: average unrealized P&L percent
    performance_score = weights["performance"]
    if "unrealized_pl_percent" in df.columns and df["unrealized_pl_percent"].notna().any():
        avg_return = df["unrealized_pl_percent"].dropna().mean()
        if avg_return >= 20:
            performance_score = weights["performance"]
        elif avg_return >= 10:
            performance_score = int(weights["performance"] * 0.85)
        elif avg_return >= 0:
            performance_score = int(weights["performance"] * 0.7)
        elif avg_return >= -10:
            performance_score = int(weights["performance"] * 0.45)
        else:
            performance_score = int(weights["performance"] * 0.25)
    else:
        performance_score = int(weights["performance"] * 0.6)
    breakdown["performance"] = max(0, performance_score)

    # Volatility: count high beta holdings
    volatility_score = weights["volatility"]
    if "beta" in df.columns and df["beta"].notna().any():
        high_beta_count = (df["beta"] > 1.5).sum()
        total_holdings = len(df)
        high_beta_ratio = high_beta_count / total_holdings if total_holdings else 0
        if high_beta_ratio >= 0.6:
            volatility_score = int(weights["volatility"] * 0.3)
        elif high_beta_ratio >= 0.4:
            volatility_score = int(weights["volatility"] * 0.5)
        elif high_beta_ratio >= 0.25:
            volatility_score = int(weights["volatility"] * 0.7)
    else:
        volatility_score = int(weights["volatility"] * 0.7)
    breakdown["volatility"] = max(0, volatility_score)

    # Position sizing: reward 8-15 stocks, penalize <5 or >25
    position_score = weights["position_sizing"]
    holdings_count = len(df)
    if holdings_count < 5:
        position_score = int(weights["position_sizing"] * 0.3)
    elif holdings_count < 8:
        position_score = int(weights["position_sizing"] * 0.6)
    elif holdings_count <= 15:
        position_score = weights["position_sizing"]
    elif holdings_count <= 20:
        position_score = int(weights["position_sizing"] * 0.85)
    elif holdings_count <= 25:
        position_score = int(weights["position_sizing"] * 0.65)
    else:
        position_score = int(weights["position_sizing"] * 0.45)
    breakdown["position_sizing"] = max(0, position_score)

    # Sentiment: use cached news sentiment
    sentiment_score = weights["sentiment"]
    sentiment_map = get_average_sentiment_for_tickers(df["ticker"].tolist())
    if sentiment_map:
        total_articles = sum(count for _, count in sentiment_map.values())
        weighted_sentiment = sum(score * count for score, count in sentiment_map.values())
        avg_sentiment = weighted_sentiment / total_articles if total_articles else 0.0
        if avg_sentiment >= 0.4:
            sentiment_score = weights["sentiment"]
        elif avg_sentiment >= 0.2:
            sentiment_score = int(weights["sentiment"] * 0.85)
        elif avg_sentiment >= -0.1:
            sentiment_score = int(weights["sentiment"] * 0.65)
        elif avg_sentiment >= -0.3:
            sentiment_score = int(weights["sentiment"] * 0.45)
        else:
            sentiment_score = int(weights["sentiment"] * 0.25)
    else:
        sentiment_score = int(weights["sentiment"] * 0.6)
    breakdown["sentiment"] = max(0, sentiment_score)

    total_score = sum(breakdown.values())
    total_score = max(0, min(100, total_score))
    return total_score, breakdown


def _portfolio_health_status(score: int) -> Tuple[str, str]:
    if score >= 80:
        return "🟢 Excellent", "excellent"
    if score >= 60:
        return "🟡 Good", "good"
    if score >= 40:
        return "🟠 Fair", "fair"
    return "🔴 Needs Attention", "poor"

# -------------------------
# RAG SYSTEM
# -------------------------
@st.cache_resource
def load_embedder():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        device = None
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
            except Exception as torch_error:
                logger.warning("Unable to determine optimal torch device: %s", torch_error)
        kwargs = {}
        if device:
            logger.info("Loading sentence transformer on %s", device)
            kwargs["device"] = device
        else:
            logger.info("Loading sentence transformer on cpu")
        model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
        return model
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None

class EnhancedRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.documents = []
        self.embeddings = None

    def build_knowledge_base(self, portfolio_df, include_news=True):
        documents = []
        for category, topics in FINANCIAL_KNOWLEDGE.items():
            category_label = category.replace("_", " ").title()
            for topic, content in topics.items():
                documents.append(f"{category_label} - {topic}: {content}")

        if not portfolio_df.empty:
            total_stocks = len(portfolio_df)
            total_value = portfolio_df['market_value_inr'].sum()
            total_pl = portfolio_df['unrealized_pl_inr'].sum()
            documents.append(f"Portfolio Summary: {total_stocks} stocks, total value {format_inr(total_value)}, total P&L {format_inr(total_pl)}")

            try:
                top_stocks = get_stock_recommendations(max_stocks=3)
                for stock in top_stocks:
                    price = float(stock.get('price') or 0.0)
                    momentum = float(stock.get('momentum') or 0.0)
                    pe_ratio = float(stock.get('pe_ratio') or 0.0)
                    rec_text = (
                        f"Recommended Stock: {stock.get('name', stock.get('ticker', 'Unknown'))} "
                        f"({stock.get('ticker', '')}) - Current price ₹{price:.2f}, "
                        f"1-month momentum {momentum:+.2f}%, P/E ratio {pe_ratio:.1f}"
                    )
                    documents.append(rec_text)
            except Exception:
                pass

            for _, row in portfolio_df.iterrows():
                if 'error' not in row:
                    ticker = row.get('ticker', '')
                    company = row.get('company_name', ticker)
                    quantity = row.get('quantity', 0)
                    avg_price = row.get('avg_price', 0)
                    
                    entry = f"You hold {int(quantity)} shares of {company} ({ticker}) with an average purchase price of ₹{avg_price:.2f}. The current price is ₹{row['current_price_inr']:.2f}, with a market value of {format_inr(row['market_value_inr'])}. This holding has an unrealized profit/loss of {row['unrealized_pl_percent']:.2f}%."
                    
                    documents.append(entry)

        self.documents = documents
        
        if self.embedder and documents:
            try:
                embeddings = self.embedder.encode(documents, convert_to_numpy=True, show_progress_bar=False)
                self.embeddings = np.array(embeddings, dtype=np.float32)
                
                if FAISS_AVAILABLE:
                    try:
                        self.index = faiss.IndexFlatL2(embeddings.shape[1])
                        self.index.add(self.embeddings)
                    except:
                        self.index = None
                
                return True
            except Exception as e:
                logger.error(f"Error building knowledge base: {e}")
                return True
        elif documents:
            return True
        return False

    def get_relevant_context(self, query: str, k: int = 7):
        if not self.documents:
            return []
        
        if not self.embedder:
            query_lower = query.lower()
            relevant_docs = []
            for doc in self.documents:
                if any(word in doc.lower() for word in query_lower.split()):
                    relevant_docs.append(doc)
            return relevant_docs[:k]
        
        try:
            query_emb = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
            query_emb = np.array(query_emb, dtype=np.float32)
            
            if self.index is not None and FAISS_AVAILABLE:
                try:
                    distances, indices = self.index.search(query_emb, min(k, len(self.documents)))
                    return [self.documents[i] for i in indices[0] if i < len(self.documents)]
                except:
                    pass
            
            if self.embeddings is not None:
                query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
                doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
                similarities = np.dot(query_norm, doc_norms.T)[0]
                top_k_indices = np.argsort(similarities)[::-1]
                return [self.documents[i] for i in top_k_indices[:k] if i < len(self.documents)]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
        
        query_lower = query.lower()
        relevant_docs = []
        for doc in self.documents:
            if any(word in doc.lower() for word in query_lower.split()):
                relevant_docs.append(doc)
        return relevant_docs[:k]

def reset_session_state_for_new_session():
    st.session_state.messages = []
    st.session_state.portfolio_df = pd.DataFrame()
    st.session_state.updated_portfolio_df = pd.DataFrame()
    st.session_state.extracted_portfolio = None
    st.session_state.rag_system = None
    st.session_state.ocr_results = []

def load_session_into_state(session_id: int):
    reset_session_state_for_new_session()
    st.session_state.active_session_id = session_id
    st.session_state.messages = get_chat_messages(session_id) or []
    
    base_df, updated_df = get_portfolio_snapshot(session_id)
    st.session_state.portfolio_df = base_df if not base_df.empty else pd.DataFrame()
    st.session_state.updated_portfolio_df = updated_df if not updated_df.empty else pd.DataFrame()
    st.session_state.extracted_portfolio = base_df.copy() if not base_df.empty else None
    
    if not st.session_state.updated_portfolio_df.empty:
        embedder = load_embedder()
        rag = EnhancedRAG(embedder)
        rag.build_knowledge_base(st.session_state.updated_portfolio_df, include_news=True)
        st.session_state.rag_system = rag

def logout_user():
    reset_session_state_for_new_session()
    st.session_state.pop("auth_user", None)
    st.session_state.pop("active_session_id", None)
    st.session_state.pop("active_session_select", None)

def format_session_label(session: dict) -> str:
    title = session.get("title") or f"Session {session.get('id')}"
    last_active = session.get("last_active") or session.get("created_at")
    try:
        if last_active:
            label_time = datetime.fromisoformat(last_active).strftime("%d %b %Y %H:%M")
        else:
            label_time = "Unknown"
    except Exception:
        label_time = last_active or "Unknown"
    return f"{title} • {label_time}"

# -------------------------
# LLM CALL
# -------------------------
def get_financial_advice(hf_client, user_query: str, context_docs: list):
    if not hf_client:
        return "Please upload portfolio & build knowledge base first."

    portfolio_context: List[str] = []
    news_context: List[str] = []
    knowledge_context: List[str] = []

    portfolio_keywords = ("hold", "shares", "quantity", "price")
    news_keywords = ("news", "article", "published")

    for doc in context_docs or []:
        doc_lower = doc.lower()
        if any(keyword in doc_lower for keyword in portfolio_keywords):
            portfolio_context.append(doc)
        elif any(keyword in doc_lower for keyword in news_keywords):
            news_context.append(doc)
        else:
            knowledge_context.append(doc)

    def _format_section(label: str, docs: List[str]) -> str:
        if not docs:
            return f"{label} (0 entries)\n- None"
        total = len(docs)
        top_docs = docs[: min(5, total)]
        items = "\n".join(f"- {entry}" for entry in top_docs)
        return f"{label} ({min(5, total)} of {total} entries)\n{items}"

    structured_context = "\n\n".join(
        [
            _format_section("PORTFOLIO DATA", portfolio_context),
            _format_section("NEWS DATA", news_context),
            _format_section("KNOWLEDGE DATA", knowledge_context),
        ]
    )

    portfolio_count = len(portfolio_context)
    news_count = len(news_context)
    knowledge_count = len(knowledge_context)
    total_context_docs = len(context_docs or [])

    def _confidence_badge(doc_count: int) -> str:
        if doc_count > 5:
            return "🟢 **High confidence**"
        if 2 <= doc_count <= 5:
            return "🟡 **Medium confidence**"
        return "🔴 **Limited data - upload portfolio for better advice**"

    def _portfolio_freshness_minutes() -> Optional[int]:
        df = st.session_state.get("updated_portfolio_df")
        if df is None or df.empty:
            return None
        if "cache_time" not in df.columns:
            return 0
        timestamps: List[datetime] = []
        for value in df["cache_time"].dropna():
            if not value:
                continue
            parsed = _parse_db_timestamp(value)
            if parsed:
                timestamps.append(parsed)
        if not timestamps:
            return 0
        oldest = min(timestamps)
        delta = datetime.utcnow() - oldest
        return max(int(delta.total_seconds() // 60), 0)

    freshness_minutes = _portfolio_freshness_minutes()
    if freshness_minutes is None:
        freshness_text = "⏰ Portfolio data unavailable"
    elif freshness_minutes == 0:
        freshness_text = "⏰ Using live data (updated just now)"
    else:
        freshness_text = f"⏰ Using data from {freshness_minutes} minutes ago"

    confidence_line = _confidence_badge(total_context_docs)

    system_prompt = """You are a friendly, expert Indian financial advisor specializing in portfolio analysis and stock market-driven investment advice. Your job is to help users understand and optimize their portfolios with clear, actionable, and well-explained guidance grounded in current market conditions.

COMMUNICATION STYLE:
- Warm, conversational tone — sound like a trusted friend who knows finance
- Explain jargon in simple language (e.g., "Rebalancing just means shifting money to keep your allocation on track")
- Use relatable analogies: "Diversification is like not putting all your eggs in one basket"
- Be encouraging and empathetic: celebrate wins, acknowledge worries
- Keep sentences concise; avoid overly formal or technical phrasing

1. PORTFOLIO ANALYSIS (when holdings are available):
   - Begin with an upbeat summary: value, overall P&L, top 3 positions, concentration flags
   - Cite concrete numbers: "Your TCS stake is ₹45,230 (15% of the portfolio) and is up 12% — nice work!"
   - Compare holdings to benchmarks: "TCS beat the Nifty IT index by ~3% this month"
   - Spotlight diversification gaps and explain why they matter
   - Tie every recommendation to current market behaviour (earnings, sector moves, macro news)

2. USER-FRIENDLY ADVICE STRUCTURE:
   - 🎯 Quick Action: [One simple step + why it helps]
   - 📊 Your Portfolio: [What’s healthy, what needs attention]
   - 📈 Market Update: [How today’s market climate affects these holdings]
   - ⚖️ What This Means: [Plain-language risk vs reward summary]
   - 🔄 Smart Moves: [Rebalancing / allocation tweaks, explained gently]
   - ⚠️ Things to Watch: [Risks stated calmly — no fear-mongering]
   - 💰 Tax Tips: [Explain LTCG/STCG impact in everyday terms]
   - 📅 When to Check Back: [Friendly reminder about next review]
   - 🧠 How I Analyzed This: [1-3 bullets outlining the evidence, metrics, and news used]

3. WHEN NO PORTFOLIO IS LOADED:
   - Welcome warmly and explain how to get personal advice
   - Offer high-level market insight (Nifty/Sensex, sector trends)
   - Suggest starter ideas with simple reasoning
   - Encourage uploading holdings for deeper analysis

4. MARKET DATA MADE SIMPLE:
   - Use everyday comparisons: "Your portfolio gained 10% vs Nifty’s 8%"
   - Explain moves briefly: "Market fell 2% today on inflation concerns"
   - Contextualize news and macro events plainly
   - Mention volatility gently: "VIX is elevated, so swings may continue"
   - Flag upcoming events (budget, RBI meet, earnings) that could impact holdings

5. ALWAYS INCLUDE (IN PLAIN LANGUAGE):
   - Tax impact of suggested trades
   - Risk level (Low/Medium/High) with a brief explanation
   - Time horizon (Short/Medium/Long) suited for the idea
   - Market rationale anchoring the recommendation
   - Confidence level tied to data quality and market clarity

6. EXPLAINABILITY REQUIREMENTS:
   - Add a dedicated section (🧠 How I Analyzed This) that lists the key portfolio metrics, market indices, news signals, and knowledge-base insights you used
   - Spell out any calculations or comparisons performed (e.g., weight %, gain %, index vs holding)
   - Call out cached data with wording like "Based on prices from {time} ago"
   - When data is missing or uncertain, state it plainly ("I don’t have recent prices for…")

7. FORMAT FOR CLARITY:
   - Use friendly headers with emojis
   - Keep paragraphs short; prefer bullet lists
   - Drop in simple tables for comparisons when useful
   - Use bold text to highlight critical numbers or takeaways

8. TONE & SAFETY:
   - Stay positive, supportive, and honest
   - Recognize emotions: "I know volatility can be stressful..."
   - Avoid guarantees; highlight uncertainties respectfully
   - Always mention data freshness when using cached numbers

9. ENDING THE CONVERSATION:
   - Close with "💡 What I'd do: [One clear, kind suggestion]"
   - Make the next step feel approachable ("Take 10 minutes to…")

10. LENGTH GUIDELINES:
   - Default to ~300 words; expand to 400-600 for full portfolio reviews when asked
   - Keep the voice natural — never robotic or overly formal

Remember: You’re guiding real people. Be friendly, transparent about your reasoning, and show the thinking behind every recommendation."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{structured_context}\n\nUser Question: {user_query}"}
    ]

    try:
        response = hf_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=3500,
            temperature=0.1,
            top_p=0.9
        )
        if response and response.choices:
            content = response.choices[0].message.content or "No response generated."
            content = re.sub(r"<br\s*/?>", "\n", content)
        else:
            content = "No response from AI."
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Error generating AI advice. Please try again."

    sources_footer = (
        "\n\n"
        f"📚 Sources: {portfolio_count} portfolio holdings, {news_count} news articles, {knowledge_count} knowledge base entries"
    )
    final_response = (
        f"{confidence_line}\n{freshness_text}\n\n{content}{sources_footer}\n\n"
        "⚠️ **Disclaimer:** For educational purposes only. All amounts in INR."
    )
    return final_response

# -------------------------
# FETCH LIVE MARKET INDICES
# -------------------------
@st.cache_data(ttl=300)
def fetch_live_market_indices(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch live Indian market indices with SQLite caching and graceful fallbacks."""
    # Indian index symbols with fallback options
    indices = {
        "NIFTY 50": ["^NSEI", "NSEI.NS", "NIFTYBEES.NS"],
        "SENSEX": ["^BSESN", "BSESN.NS"],
        "NIFTY Bank": ["^NSEBANK", "NSEBANK.NS", "BANKBEES.NS"],
        "NIFTY IT": ["^CNXIT", "CNXIT.NS", "ITBEES.NS"],
    }

    now = datetime.utcnow()
    freshness_cutoff = now - timedelta(minutes=5)

    results: Dict[str, Dict[str, Any]] = {}
    stale_indices: Dict[str, Dict[str, Any]] = {}

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT index_name, current_value, change_percent, last_updated
            FROM market_indices
            """
        )
        db_rows = {row["index_name"]: row for row in cursor.fetchall()}

    for name, symbol in indices.items():
        row = db_rows.get(name)
        last_updated_dt: Optional[datetime] = None
        row_data: Optional[Dict[str, Any]] = None

        if row:
            last_updated_dt = _parse_db_timestamp(row["last_updated"])
            value_f = _safe_float(row["current_value"])
            change_pct_f = _safe_float(row["change_percent"])
            change_estimate = (
                value_f * (change_pct_f / 100.0)
                if value_f is not None and change_pct_f is not None
                else None
            )
            row_data = {
                "value": value_f,
                "change_pct": change_pct_f,
                "change": change_estimate,
                "last_updated": last_updated_dt,
            }

            if (
                last_updated_dt
                and not force_refresh
                and last_updated_dt >= freshness_cutoff
            ):
                results[name] = _prepare_market_result(
                    value_f,
                    change_estimate,
                    change_pct_f,
                    last_updated_dt,
                    True,
                )
                continue

        stale_indices[name] = {"symbol": symbol, "previous": row_data}

    fresh_rows: List[Tuple[Any, ...]] = []

    for name, info in stale_indices.items():
        symbol_list = info["symbol"] if isinstance(info["symbol"], list) else [info["symbol"]]
        previous = info.get("previous")
        fetched = False
        
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if hist.empty:
                    continue

                current = _safe_float(hist["Close"].iloc[-1])
                prev = (
                    _safe_float(hist["Close"].iloc[-2])
                    if len(hist) > 1
                    else current
                )
                if current is None or prev is None:
                    continue

                change = current - prev
                change_pct = (change / prev * 100.0) if prev else 0.0
                entry_time = now

                results[name] = _prepare_market_result(
                    current, change, change_pct, entry_time, False
                )
                fresh_rows.append(
                    (
                        name,
                        round(current, 6),
                        round(change_pct, 6),
                        entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )
                fetched = True
                break
            except Exception as exc:
                logger.debug(f"Failed to fetch {name} with symbol {symbol}: {exc}")
                continue
        
        if not fetched:
            error_msg = f"Could not fetch {name} with any available symbol"
            logger.error(f"Error fetching {name}: {error_msg}")
            if previous:
                results[name] = _prepare_market_result(
                    previous.get("value"),
                    previous.get("change"),
                    previous.get("change_pct"),
                    previous.get("last_updated"),
                    True,
                    error=error_msg,
                )
            else:
                results[name] = _prepare_market_result(
                    None, None, None, None, True, error=error_msg
                )

    if fresh_rows:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO market_indices (index_name, current_value, change_percent, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(index_name) DO UPDATE SET
                    current_value = excluded.current_value,
                    change_percent = excluded.change_percent,
                    last_updated = excluded.last_updated
                """,
                fresh_rows,
            )
            conn.commit()
    
    return results

# -------------------------
# STREAMLIT UI
# -------------------------

# Custom CSS for better UI
st.markdown("""
<style>
:root {
    --page-bg: #f5f9ff;
    --panel-bg: #ffffff;
    --panel-border: rgba(37, 99, 235, 0.12);
    --surface-soft: #eef4ff;
    --surface-muted: #dbe8ff;
    --accent: #2563eb;
    --accent-2: #1d4ed8;
    --text: #0f172a;
    --text-muted: #4b5563;
}
[data-testid="stAppViewContainer"] > .main {
    background: var(--page-bg);
    color: var(--text);
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}
.hero {
    background: var(--panel-bg);
    padding: 2.25rem;
    border-radius: 24px;
    border: 1px solid var(--panel-border);
    box-shadow: 0 20px 40px rgba(37, 99, 235, 0.08);
    color: var(--text);
}
.hero h1 {
        font-size: 2.5rem;
    margin-bottom: 0.75rem;
    color: var(--text);
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 90%;
    margin-bottom: 1.1rem;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.hero-badge {
    background: rgba(37, 99, 235, 0.12);
    color: var(--accent);
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    background: rgba(37, 99, 235, 0.16);
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.section-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 1.6rem;
    margin-top: 1.5rem;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
}
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title span {
    font-size: 1.4rem;
    color: var(--accent);
}
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.9rem;
}
.status-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    color: var(--text);
}
.status-card:hover {
    transform: translateY(-4px);
    border-color: rgba(37, 99, 235, 0.25);
    box-shadow: 0 16px 32px rgba(37, 99, 235, 0.12);
}
.status-card .status-label {
        font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.status-card .status-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
}
.status-card.success {
    border-color: rgba(34, 197, 94, 0.35);
}
.status-card.warning {
    border-color: rgba(250, 204, 21, 0.35);
}
.status-card.danger {
    border-color: rgba(248, 113, 113, 0.35);
}
.status-card.info {
    border-color: rgba(37, 99, 235, 0.35);
}
.market-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.1rem;
    margin-top: 0.5rem;
}
.market-card {
    background: var(--panel-bg);
    border-radius: 18px;
    border: 1px solid var(--panel-border);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    color: var(--text);
}
.market-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
}
.market-card .label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.market-card .value {
    font-size: 1.6rem;
    font-weight: 600;
    margin-top: 0.25rem;
    color: var(--text);
}
.market-card .delta {
    margin-top: 0.7rem;
    font-size: 0.95rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.market-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.65rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.8rem;
    background: var(--surface-soft);
    border: 1px solid var(--panel-border);
}
.market-status.market-status-live {
    background: rgba(59, 130, 246, 0.18);
    border-color: rgba(59, 130, 246, 0.35);
    color: #1d4ed8;
}
.market-status.market-status-cached {
    background: rgba(148, 163, 184, 0.18);
    border-color: rgba(148, 163, 184, 0.35);
    color: #334155;
}
.health-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.65rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    background: var(--surface-soft);
    border: 1px solid var(--panel-border);
}
.health-badge.health-excellent {
    background: rgba(34, 197, 94, 0.18);
    border-color: rgba(34, 197, 94, 0.35);
    color: #15803d;
}
.health-badge.health-good {
    background: rgba(250, 204, 21, 0.18);
    border-color: rgba(250, 204, 21, 0.35);
    color: #b45309;
}
.health-badge.health-fair {
    background: rgba(249, 115, 22, 0.18);
    border-color: rgba(249, 115, 22, 0.35);
    color: #c2410c;
}
.health-badge.health-poor {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.4);
    color: #991b1b;
}
.health-score-value {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.market-error {
    margin-top: 0.6rem;
    font-size: 0.78rem;
    color: #b91c1c;
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.25);
    border-radius: 12px;
    padding: 0.4rem 0.6rem;
    display: inline-block;
}
.delta-up {
    color: #15803d;
}
.delta-down {
    color: #b91c1c;
}
.kpi-wrap {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.kpi-pill {
    background: var(--panel-bg);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    border: 1px solid var(--panel-border);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
}
.kpi-pill span {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.kpi-pill strong {
    font-size: 1.3rem;
    color: var(--text);
}
.list-stack {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    margin-top: 0.6rem;
}
.list-item {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 0.9rem 1.05rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    color: var(--text);
}
.list-item .label span {
    font-size: 0.85rem;
    color: var(--text-muted);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    color: var(--text);
    border-right: 1px solid var(--panel-border);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: var(--text) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    color: var(--text) !important;
    background: #ffffff !important;
    border: 1px solid var(--panel-border) !important;
}
.sidebar-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1.1rem;
    color: var(--text);
}
.chat-hint {
    margin-top: 1rem;
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(37, 99, 235, 0.2);
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    color: var(--text);
    font-size: 0.9rem;
}
.stButton > button,
.stDownloadButton > button,
.stFileUploader label {
    color: #0f172a !important;
    background: #e2edff !important;
    border: 1px solid rgba(37, 99, 235, 0.25) !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.12) !important;
    font-weight: 600 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFileUploader label:hover {
    background: #d7e6ff !important;
    border-color: rgba(37, 99, 235, 0.35) !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.16) !important;
}
[data-testid="stChatInput"] {
    background: var(--panel-bg);
    border-top: 1px solid var(--panel-border);
}
[data-testid="stChatMessage"] {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}
[data-testid="stChatMessage"] > div:nth-child(1) {
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.news-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin-top: 0.5rem;
}
.news-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    color: var(--text);
}
.sentiment-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.65rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    background: var(--surface-soft);
    border: 1px solid var(--panel-border);
}
.sentiment-tag.sentiment-positive {
    background: rgba(34, 197, 94, 0.16);
    border-color: rgba(34, 197, 94, 0.35);
    color: #15803d;
}
.sentiment-tag.sentiment-neutral {
    background: rgba(148, 163, 184, 0.18);
    border-color: rgba(148, 163, 184, 0.35);
    color: #334155;
}
.sentiment-tag.sentiment-negative {
    background: rgba(248, 113, 113, 0.16);
    border-color: rgba(248, 113, 113, 0.35);
    color: #b91c1c;
}
.sentiment-overall {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.7rem;
}
.sentiment-count {
    font-size: 0.78rem;
    color: var(--text-muted);
}
.sentiment-value {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.25rem;
}
.news-card p {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.4;
}
.news-card a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
    font-size: 0.86rem;
}
.instructions-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 1.4rem;
    margin-top: 1.2rem;
    color: var(--text);
}
.instructions-card h3 {
    color: var(--text);
    margin-bottom: 0.8rem;
}
[data-testid="stMetric"] {
    background: var(--panel-bg);
    padding: 1.2rem;
    border-radius: 16px;
    border: 1px solid var(--panel-border);
}
[data-testid="stVegaLiteChart"] {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 1rem;
}
[data-testid="stVegaLiteChart"] text,
[data-testid="stVegaLiteChart"] .mark-text {
    fill: #0f172a !important;
}
.ocr-line {
    display: block;
    margin-bottom: 0.25rem;
    padding: 0.35rem 0.6rem;
    border-radius: 8px;
    background: var(--surface-soft);
    border: 1px solid var(--panel-border);
    color: var(--text);
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.82rem;
}
.stTabs button,
.stTabs button span {
    color: var(--text) !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] select {
    color: var(--text) !important;
    background: #ffffff !important;
    border: 1px solid var(--panel-border) !important;
}

form#form-login_form button,
form#form-signup_form button {
    color: #0f172a !important;
    background: #e2edff !important;
    border: 1px solid rgba(37, 99, 235, 0.25) !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.12) !important;
    font-weight: 600 !important;
}

form#form-login_form button:hover,
form#form-signup_form button:hover {
    background: #d7e6ff !important;
    border-color: rgba(37, 99, 235, 0.35) !important;
}

form#form-login_form button:disabled,
form#form-signup_form button:disabled {
    color: rgba(15, 23, 42, 0.45) !important;
    background: rgba(226, 232, 240, 0.8) !important;
    border-color: rgba(148, 163, 184, 0.4) !important;
    box-shadow: none !important;
}

.stTextInput input,
.stPassword input,
.stNumberInput input,
[data-baseweb="textarea"] textarea {
    background: #ffffff !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px !important;
    padding: 0.75rem 0.85rem !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextInput input:focus,
.stPassword input:focus,
.stNumberInput input:focus,
[data-baseweb="textarea"] textarea:focus {
    border-color: rgba(37, 99, 235, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

.stTextInput input::placeholder,
.stPassword input::placeholder,
[data-baseweb="textarea"] textarea::placeholder {
    color: rgba(71, 85, 105, 0.6) !important;
}

.stTextInput > label,
.stPassword > label,
.stSelectbox > label,
.stRadio > label,
.stFileUploader > label {
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: var(--text) !important;
}

.info-tip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(37, 99, 235, 0.15);
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 6px;
    cursor: help;
}

.suggestion-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.85rem;
    margin: 1rem 0 1.5rem 0;
}

.suggestion-buttons .stButton button {
    border-radius: 16px;
    border: 1px solid rgba(37, 99, 235, 0.18);
    background: var(--panel-bg);
    color: var(--accent);
    font-weight: 600;
    padding: 1rem;
    text-align: left;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
}

.suggestion-buttons .stButton button:hover {
    border-color: var(--accent);
    background: rgba(37, 99, 235, 0.08);
}

</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()

# Session defaults
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame()
if "updated_portfolio_df" not in st.session_state:
    st.session_state.updated_portfolio_df = pd.DataFrame()
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []
if "extracted_portfolio" not in st.session_state:
    st.session_state.extracted_portfolio = None
if "show_help" not in st.session_state:
    st.session_state.show_help = False

exchange_rate = get_usd_to_inr_rate()
greeting_name = st.session_state.auth_user["username"] if st.session_state.auth_user else "Investor"
tracked_holdings = len(st.session_state.updated_portfolio_df) if not st.session_state.updated_portfolio_df.empty else 0
message_count = len(st.session_state.messages)
session_label = st.session_state.auth_user["username"] if st.session_state.auth_user else "Guest mode"

hero_cols = st.columns([3, 2], gap="large")
with hero_cols[0]:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-pill">AI Wealth Workspace</div>
            <h1>Welcome back, {greeting_name}. Elevate your portfolio decisions with confident insights.</h1>
            <p class="hero-subtitle">
                Upload holdings, capture screenshots with OCR, and receive research-grade guidance that blends live market data, cached intelligence, and contextual AI.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Live INR pricing & caching</span>
                <span class="hero-badge">Personalized news briefs</span>
                <span class="hero-badge">Chat-based portfolio intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_cols[1]:
    st.markdown(
        f"""
        <div class="section-card" style="margin-top:0;">
            <div class="section-title"><span>🧭</span>Session Snapshot</div>
            <div class="kpi-wrap">
                <div class="kpi-pill">
                    <span>Signed in as</span>
                    <strong>{session_label}</strong>
                </div>
                <div class="kpi-pill">
                    <span>Cache duration</span>
                    <strong>{CACHE_DURATION_MINUTES} min</strong>
                </div>
                <div class="kpi-pill">
                    <span>USD → INR</span>
                    <strong>₹{exchange_rate:.2f}</strong>
                </div>
                <div class="kpi-pill">
                    <span>Tracked holdings</span>
                    <strong>{tracked_holdings}</strong>
                </div>
                <div class="kpi-pill">
                    <span>Chat messages</span>
                    <strong>{message_count}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

status_items = [
    ("Tesseract OCR", "Ready", "success", "📸") if TESSERACT_AVAILABLE else ("Tesseract OCR", "Install locally", "danger", "📸"),
    ("Embeddings", "Sentence Transformers", "success", "🧠") if SENTENCE_TRANSFORMERS_AVAILABLE else ("Embeddings", "Fallback mode", "warning", "🧠"),
    ("Vector Search", "FAISS Acceleration", "success", "🗂️") if FAISS_AVAILABLE else ("Vector Search", "Cosine similarity", "info", "🗂️"),
    ("News API", "Connected", "success", "🗞️") if news_client else ("News API", "Unavailable", "warning", "🗞️"),
    ("Cache Window", f"{CACHE_DURATION_MINUTES} min", "info", "⏱️"),
]

status_cards_html = "".join(
    f"""
    <div class="status-card {css_class}">
        <div class="status-icon">{icon}</div>
        <div class="status-label">{label}</div>
        <div class="status-value">{value}</div>
    </div>
    """
    for label, value, css_class, icon in status_items
).strip()

st.markdown(
    f"""
    <div class="section-card">
        <div class="section-title"><span>🛠️</span>Runtime Checks</div>
        <div class="status-grid">
            {status_cards_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    header_cols = st.columns([5, 1])
    with header_cols[0]:
        st.markdown("<div class='section-title'><span>🌐</span>Live Market Pulse</div>", unsafe_allow_html=True)
    with header_cols[1]:
        refresh_market = st.button("🔄 Refresh", key="refresh_market")

    if refresh_market:
        fetch_live_market_indices.clear()

    market_data = fetch_live_market_indices(force_refresh=refresh_market)

    if market_data:
        market_cards: List[str] = []
        for name, data in market_data.items():
            value_f = _safe_float(data.get("value"))
            value_display = f"{value_f:,.2f}" if value_f is not None else "--"

            change_pct_value = _safe_float(data.get("change_pct"))
            if change_pct_value is None:
                arrow = "➖"
                delta_class = "delta-up"
                change_pct_display = "--"
            else:
                arrow = "▲" if change_pct_value >= 0 else "▼"
                delta_class = "delta-up" if change_pct_value >= 0 else "delta-down"
                change_pct_display = f"{abs(change_pct_value):.2f}%"

            last_updated_str = data.get("last_updated")
            last_updated_dt = _parse_db_timestamp(last_updated_str)
            if data.get("is_cached"):
                if last_updated_dt:
                    age_text = _format_time_ago(last_updated_dt)
                    status_badge = f"<span class='market-status market-status-cached'>💾 Cached ({age_text})</span>"
                else:
                    status_badge = "<span class='market-status market-status-cached'>💾 Cached</span>"
            else:
                status_badge = "<span class='market-status market-status-live'>🆕 Live</span>"

            error_html = ""
            if data.get("error"):
                error_html = f"<div class='market-error'>⚠️ {html.escape(str(data['error']))}</div>"

            card_html = (
                "<div class=\"market-card\">"
                "<div class=\"market-card-header\">"
                f"<div class=\"label\">{html.escape(name)}</div>"
                f"{status_badge}"
                "</div>"
                f"<div class=\"value\">{value_display}</div>"
                f"<div class=\"delta {delta_class}\">{arrow} {change_pct_display}</div>"
                f"{error_html}"
                "</div>"
            )
            market_cards.append(card_html)

        cards_html = "\n".join(market_cards)
        st.markdown(
            f"<div class=\"market-grid\">{cards_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p>Market data is currently unavailable.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# Stock Recommendations Section
    st.markdown(
        """
<div class=\"section-card\">
    <div class=\"section-title\"><span>🎯</span>Top Stock Picks</div>
        </div>
        """,
unsafe_allow_html=True,
)

if st.button("🔍 Get Recommendations", key="get_recommendations"):
    with st.spinner("Analyzing Nifty 50 stocks..."):
        recommendations = get_stock_recommendations(max_stocks=5)

        if recommendations:
            for i, stock in enumerate(recommendations, 1):
                momentum_value = stock.get("momentum", 0)
                momentum_color = "positive" if momentum_value > 0 else "negative"
                momentum_symbol = "📈" if momentum_value > 0 else "📉"
                name = stock.get("name") or stock.get("ticker", "")
                price = stock.get("price") or 0.0

                st.markdown(
                    f"""
                    <div class="list-item">
                        <div class="label">
                            <strong>#{i} {stock.get('ticker', '').replace('.NS', '')}</strong>
                            <span>{name}</span>
                        </div>
                        <div class="value {momentum_color}">
                            {momentum_symbol} {momentum_value:+.2f}% | ₹{price:.2f}
                        </div>
                    </div>
                    """
                    ,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No recommendations available right now.")
# Sidebar
with st.sidebar:
    st.header("🔐 Account")
    user = st.session_state.auth_user
    
    if user:
        st.success(f"Signed in as {user['username']}")
        sessions = get_user_sessions(user["id"])
        session_map = {session["id"]: session for session in sessions}
        
        if not sessions:
            st.info("No chat sessions yet.")
            if st.button("Start First Session", width="stretch"):
                new_session_id = create_chat_session(user["id"])
                load_session_into_state(new_session_id)
                st.rerun()
        else:
            session_ids = [session["id"] for session in sessions]
            if st.session_state.active_session_id not in session_ids:
                load_session_into_state(session_ids[0])
            current_session_id = st.session_state.active_session_id
            selected_session_id = st.selectbox(
                "Chat Sessions",
                options=session_ids,
                index=session_ids.index(current_session_id) if current_session_id in session_ids else 0,
                format_func=lambda sid: format_session_label(session_map[sid]),
                key="active_session_select"
            )
            if selected_session_id != current_session_id:
                load_session_into_state(selected_session_id)
                st.rerun()
            
            if st.button("Start New Session", use_container_width=True):
                new_session_id = create_chat_session(user["id"])
                load_session_into_state(new_session_id)
                st.rerun()
        
        if st.button("Log out", use_container_width=True):
            logout_user()
            st.success("Logged out successfully.")
            st.rerun()
        
        st.markdown("---")
    else:
        st.markdown("---")
        auth_mode = st.radio("Account Action", ["Login", "Sign Up"])
        
        if auth_mode == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Log in")
            if submit_login:
                if not username or not password:
                    st.error("Please enter username and password.")
                else:
                    authenticated_user = authenticate_user(username.strip(), password)
                    if authenticated_user:
                        st.session_state.auth_user = authenticated_user
                        sessions = get_user_sessions(authenticated_user["id"])
                        if sessions:
                            load_session_into_state(sessions[0]["id"])
                        else:
                            new_session_id = create_chat_session(authenticated_user["id"])
                            load_session_into_state(new_session_id)
                        st.success("Logged in successfully.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        else:
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit_signup = st.form_submit_button("Create account")
            if submit_signup:
                username_clean = new_username.strip()
                if not username_clean or not password or not confirm_password:
                    st.error("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(username_clean) < 3:
                    st.error("Username must be at least 3 characters long.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long.")
                else:
                    success, message = create_user(username_clean, password)
                    if success:
                        authenticated_user = authenticate_user(username_clean, password)
                        if authenticated_user:
                            st.session_state.auth_user = authenticated_user
                            new_session_id = create_chat_session(authenticated_user["id"])
                            load_session_into_state(new_session_id)
                            st.success("Account created and logged in.")
                            st.rerun()
                        else:
                            st.warning("Account created, but automatic login failed. Please log in manually.")
                    else:
                        st.error(message)
    
if st.session_state.auth_user:
    st.header("💼 Portfolio Management")
    st.info(
        f"🔄 Cache Duration: {CACHE_DURATION_MINUTES} minutes\n\n⏰ Prices refresh automatically every {CACHE_DURATION_MINUTES} minutes"
    )
    
    upload_options = ["Upload CSV File"]
    if TESSERACT_AVAILABLE:
        upload_options.append("Upload Screenshot Images (OCR)")
    
    upload_method = st.radio(
        "Upload Method:",
        upload_options,
        help="Choose to upload a CSV file or extract data from portfolio screenshots"
    )
    
    df_to_process = pd.DataFrame()
    
    if upload_method == "Upload CSV File":
        st.download_button(
            "📥 Download Sample CSV",
            SAMPLE_PORTFOLIO_CSV,
            file_name=SAMPLE_PORTFOLIO_FILENAME,
            mime="text/csv",
            key="sample_csv_download_sidebar",
        )
        uploaded_file = st.file_uploader(
            "Upload Portfolio CSV",
            type="csv",
            help="CSV should contain columns: Stock/Ticker, Qty/Quantity, and optionally Avg/Average price",
        )
        
        if uploaded_file:
            try:
                portfolio_df_from_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV loaded with {len(portfolio_df_from_upload)} rows")
                st.session_state.extracted_portfolio = portfolio_df_from_upload
            except Exception as e:
                logger.error("Failed to read uploaded CSV: %s", e)
                sample_link = f"[link]({SAMPLE_PORTFOLIO_DATA_URI})"
                st.error(
                    f"❌ Invalid CSV format. Required columns: ticker, quantity. Download sample: {sample_link}"
                )
    
    elif upload_method == "Upload Screenshot Images (OCR)" and TESSERACT_AVAILABLE:
        st.info("📸 Upload screenshots of your portfolio. Tesseract OCR will extract stock data.")
        
        uploaded_images = st.file_uploader(
            "Upload Portfolio Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload one or more screenshots of your portfolio"
        )
        
        if uploaded_images:
            st.write(f"📁 {len(uploaded_images)} image(s) uploaded")
            
            if st.button("🔍 Extract Data from Images", type="primary"):
                all_portfolio_entries = []
                st.session_state.ocr_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, image_file in enumerate(uploaded_images):
                    status_text.text(f"Processing image {idx+1}/{len(uploaded_images)}...")
                    ocr_data = extract_text_from_image_tesseract(image_file)
                    
                    if ocr_data:
                        st.session_state.ocr_results.extend(ocr_data)
                        entries = parse_portfolio_from_ocr(ocr_data)
                        all_portfolio_entries.extend(entries)
                        
                        with st.expander(f"📄 Preview: {image_file.name}"):
                            st.write(f"Extracted {len(ocr_data)} text elements")
                            st.write(f"Found {len(entries)} stock entries")
                            if entries:
                                preview_df = pd.DataFrame(entries)
                                st.dataframe(preview_df)
                
                    progress_bar.progress((idx + 1) / len(uploaded_images))
                
                status_text.empty()
                
                if all_portfolio_entries:
                    portfolio_df_from_upload = create_csv_from_ocr(all_portfolio_entries)
                    
                    if not portfolio_df_from_upload.empty:
                        st.success(f"✅ Extracted {len(portfolio_df_from_upload)} stock entries from images!")
                        st.session_state.extracted_portfolio = portfolio_df_from_upload
                        
                        st.subheader("📊 Extracted Portfolio Data")
                        st.dataframe(portfolio_df_from_upload)
                        
                        csv_data = portfolio_df_from_upload.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Extracted CSV",
                            data=csv_data,
                            file_name=f"extracted_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No valid portfolio data could be extracted")
                else:
                    st.warning("⚠️ No portfolio entries found in the images. Please ensure images are clear and contain portfolio data.")

    if st.button("🚀 Load and Analyze Portfolio", type="primary", use_container_width=True):
        extracted = st.session_state.get("extracted_portfolio")
        if extracted is not None and not extracted.empty:
            df_to_process = extracted
            st.info(f"Loading {len(df_to_process)} stocks from extracted data...")
        else:
            st.warning("⚠️ No portfolio data available. Please upload a CSV or extract from images first.")

        if not df_to_process.empty:
            parsed_df = parse_portfolio_csv(df_to_process)
            if parsed_df is not None and not parsed_df.empty:
                st.session_state.portfolio_df = parsed_df
                
                st.info("📊 Fetching live prices (using smart cache)...")
                updated_df = update_portfolio_with_current_data(parsed_df)
                st.session_state.updated_portfolio_df = updated_df
                
                cached_count = sum(1 for _, row in updated_df.iterrows() if row.get('is_cached', False))
                fresh_count = len(updated_df) - cached_count
                
                st.success(
                    f"✅ Updated {len(updated_df)} stocks | 🆕 Fresh: {fresh_count} | 💾 Cached: {cached_count}"
                )

                if st.session_state.active_session_id:
                    save_portfolio_snapshot(st.session_state.active_session_id, parsed_df, updated_df)

                with st.spinner("🧠 Building knowledge base..."):
                    embedder = load_embedder()
                    rag = EnhancedRAG(embedder)
                    if rag.build_knowledge_base(updated_df, include_news=False):
                        st.session_state.rag_system = rag
                        st.success("🧠 Knowledge base built successfully")
                    else:
                        st.warning("⚠️ Failed to build knowledge base")
            else:
                st.error("❌ Failed to parse portfolio data")

if st.session_state.auth_user:
    if not st.session_state.updated_portfolio_df.empty:
        df = st.session_state.updated_portfolio_df.copy()
        total_value = df["market_value_inr"].sum()
        total_pl = df["unrealized_pl_inr"].sum()
        denominator = total_value - total_pl
        total_pl_percent = (total_pl / denominator * 100) if denominator > 0 else 0
        avg_return = df["unrealized_pl_percent"].mean()
        avg_return_display = f"{avg_return:.2f}%" if pd.notna(avg_return) else "N/A"
        pl_class = "positive" if total_pl >= 0 else "negative"
        pl_symbol = "+" if total_pl >= 0 else ""

        health_score, health_breakdown = calculate_portfolio_health_score(df)
        health_label, health_class = _portfolio_health_status(health_score)
        health_badge = f"<span class='health-badge health-{health_class}'>{health_label}</span>"

        breakdown_order = [
            ("diversification", "Diversification"),
            ("performance", "Performance"),
            ("volatility", "Volatility"),
            ("position_sizing", "Position sizing"),
            ("sentiment", "News sentiment"),
        ]

        metrics_html = textwrap.dedent(
            f"""
            <div class="section-card">
                <div class="section-title"><span>📊</span>Performance Snapshot</div>
                <div class="kpi-wrap">
                    <div class="kpi-pill">
                        <span>Portfolio Value</span>
                        <strong>{format_inr(total_value)}</strong>
                    </div>
                    <div class="kpi-pill">
                        <span>Total P&L</span>
                        <strong class="{pl_class}">{pl_symbol}{format_inr(total_pl)}</strong>
                        <div class="kpi-footnote">{pl_symbol}{total_pl_percent:.2f}% total</div>
                    </div>
                    <div class="kpi-pill">
                        <span>Holdings</span>
                        <strong>{len(df)}</strong>
                    </div>
                    <div class="kpi-pill">
                        <span>Avg Return</span>
                        <strong>{avg_return_display}</strong>
                    </div>
                    <div class="kpi-pill">
                        <span>Portfolio Health <span class="info-tip" title="Score blends diversification, performance, volatility, position sizing, and news sentiment (0-100).">?</span></span>
                        <strong class="health-score-value">{health_score}/100 {health_badge}</strong>
                    </div>
                </div>
            </div>
            """
        )
        st.markdown(metrics_html, unsafe_allow_html=True)

        breakdown_lines = "\n".join(
            f"- **{label}**: {health_breakdown.get(key, 0)}/20"
            for key, label in breakdown_order
        )
        with st.expander("🔍 Portfolio Health Breakdown"):
            st.markdown(
                f"{breakdown_lines}\n\n_Total score: **{health_score}/100**_",
                unsafe_allow_html=False,
        )

        overview_tab, holdings_tab, news_tab = st.tabs(["Overview", "Holdings", "News & Insights"])

        with overview_tab:
            chart_cols = st.columns(2, gap="large")
            with chart_cols[0]:
                st.markdown("<div class='section-title'><span>🧾</span>Allocation Mix</div>", unsafe_allow_html=True)
                allocation_chart = build_allocation_chart(df)
                if allocation_chart:
                    st.altair_chart(allocation_chart, theme="streamlit")
                else:
                    st.info("Add more holdings to visualize allocation.")

            with chart_cols[1]:
                st.markdown("<div class='section-title'><span>📈</span>Gain / Loss Ladder</div>", unsafe_allow_html=True)
                pl_chart = build_pl_chart(df)
                if pl_chart:
                    st.altair_chart(pl_chart, theme="streamlit")
                else:
                    st.info("Unrealized P&L will appear once holdings have pricing data.")

            sentiment_summary = get_average_sentiment_for_tickers(df["ticker"].tolist())
            if sentiment_summary:
                name_map = df.groupby("ticker")["company_name"].first().to_dict()
                total_articles = sum(count for _, count in sentiment_summary.values())
                weighted_sum = sum(score * count for score, count in sentiment_summary.values())
                overall_avg = weighted_sum / total_articles if total_articles else 0.0
                overall_label, overall_class = _sentiment_indicator(overall_avg)
                overall_badge = f"<span class='sentiment-tag sentiment-{overall_class}'>{overall_label} ({overall_avg:+.2f})</span>"
                overall_footer = (
                    f"<span class='sentiment-count'>{total_articles} articles analysed</span>"
                    if total_articles
                    else ""
        )

                ordered_sentiments = sorted(
                    sentiment_summary.items(),
                    key=lambda item: item[1][0],
                    reverse=True,
        )
                sentiment_items = "".join(
            f"""
                    <div class="list-item">
                        <div class="label">
                            <strong>{ticker}</strong>
                            <span>{name_map.get(ticker, "N/A")}</span>
            </div>
                        <div class="value sentiment-value">
                            <span class="sentiment-tag sentiment-{_sentiment_indicator(score)[1]}">{_sentiment_indicator(score)[0]} ({score:+.2f})</span>
                            <span class="sentiment-count">{count} articles</span>
            </div>
            </div>
                    """
                    for ticker, (score, count) in ordered_sentiments
                ).strip()

                st.markdown(
            f"""
                    <div class="section-card">
                        <div class="section-title"><span>📰</span>News Sentiment Pulse</div>
                        <div class="sentiment-overall">
                            {overall_badge}
                            {overall_footer}
            </div>
                        <div class="list-stack">
                            {sentiment_items}
            </div>
            </div>
            """,
                    unsafe_allow_html=True,
        )

            movers_cols = st.columns(2, gap="large")
            top_performers = df.nlargest(5, "unrealized_pl_percent")[
                ["ticker", "company_name", "unrealized_pl_percent", "market_value_inr"]
            ]
            underperformers = df.nsmallest(5, "unrealized_pl_percent")[
                ["ticker", "company_name", "unrealized_pl_percent", "market_value_inr"]
            ]

            top_html = (
                "".join(
                    f"""
                    <div class="list-item">
                        <div class="label">
                            <strong>{row['ticker']}</strong>
                            <span>{row.get('company_name', 'N/A')}</span>
                        </div>
                        <div class="value {'positive' if row['unrealized_pl_percent'] >= 0 else 'negative'}">
                            {row['unrealized_pl_percent']:+.2f}% · {format_inr(row['market_value_inr'])}
                        </div>
                    </div>
                    """
                    for _, row in top_performers.iterrows()
                )
                if not top_performers.empty
                else "<p>No gainers detected yet.</p>"
            )

            under_html = (
                "".join(
                    f"""
                    <div class="list-item">
                        <div class="label">
                            <strong>{row['ticker']}</strong>
                            <span>{row.get('company_name', 'N/A')}</span>
                        </div>
                        <div class="value {'positive' if row['unrealized_pl_percent'] >= 0 else 'negative'}">
                            {row['unrealized_pl_percent']:+.2f}% · {format_inr(row['market_value_inr'])}
                        </div>
                    </div>
                    """
                    for _, row in underperformers.iterrows()
                )
                if not underperformers.empty
                else "<p>Great job—no laggards spotted.</p>"
            )

            with movers_cols[0]:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <div class="section-title"><span>🏆</span>Top Movers</div>
                        <div class="list-stack">{top_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with movers_cols[1]:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <div class="section-title"><span>📉</span>Watch List</div>
                        <div class="list-stack">{under_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with holdings_tab:
            st.markdown("<div class='section-title'><span>📋</span>Detailed Holdings</div>", unsafe_allow_html=True)

            display_df = df.copy()
            if "quantity" in display_df.columns:
                display_df["quantity"] = display_df["quantity"].astype(int)

            display_df["Current Price"] = display_df.apply(
                lambda row: f"₹{row['current_price_inr']:.2f} {'💾' if row.get('is_cached', False) else '🆕'}",
                axis=1,
            )
            display_df["P&L"] = display_df.apply(
                lambda row: f"{format_inr(row['unrealized_pl_inr'])} ({row['unrealized_pl_percent']:+.2f}%)",
                axis=1,
            )

            st.dataframe(
                display_df[
                    [
                        "ticker",
                        "company_name",
                        "quantity",
                        "avg_price",
                        "Current Price",
                        "market_value_inr",
                        "P&L",
                    ]
                ],
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "company_name": st.column_config.TextColumn("Company"),
                    "quantity": st.column_config.NumberColumn("Qty", format="%.0f"),
                    "avg_price": st.column_config.NumberColumn("Avg Price (₹)", format="₹%.2f"),
                    "Current Price": st.column_config.TextColumn("Current Price"),
                    "market_value_inr": st.column_config.NumberColumn("Market Value (₹)", format="₹%,.0f"),
                    "P&L": st.column_config.TextColumn("P&L"),
                },
                hide_index=True,
                use_container_width=True,
            )

        with news_tab:
            st.markdown("<div class='section-title'><span>📰</span>News & Sentiment</div>", unsafe_allow_html=True)
            market_news = fetch_market_news(max_articles=4)

            if market_news:
                cards_html = ""
                for article in market_news:
                    title = article.get("title", "Untitled")
                    description = article.get("description", "No summary available.")
                    url = article.get("url")
                    sentiment = article.get("sentiment_score")
                    sentiment_label, sentiment_class = _sentiment_indicator(sentiment)
                    sentiment_text = sentiment_label if sentiment is None else f"{sentiment_label} ({sentiment:+.2f})"
                    link_html = f"<a href='{url}' target='_blank'>Read more →</a>" if url else ""
                    cards_html += textwrap.dedent(
                        f"""
                        <div class="news-card">
                            <div class="sentiment-tag sentiment-{sentiment_class}">{sentiment_text}</div>
                            <h4>{title}</h4>
                            <p>{description}</p>
                            {link_html}
                        </div>
                        """
                    )

                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class='news-grid'>
                        {cards_html}
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )
            else:
                st.info("No fresh news available right now. Try again in a few minutes.")

    else:
        st.info("Upload or analyze a portfolio to see insights, tables, and news.")
        st.markdown(
            """
            **You can still ask about:**
            - How the Indian market is performing today
            - Differences between LTCG and STCG tax rules
            - Diversification or SIP strategies without a portfolio
            """
        )
else:
    st.info("Log in to upload portfolio data and access management tools.")
    st.markdown(
        textwrap.dedent(
            """
            <div class="instructions-card">
                <h3>🚀 Getting Started</h3>
                <p>
                    Bring your holdings into the workspace with a CSV export or clean screenshots.
                    Once uploaded, we will enrich your data with live prices, momentum insights, and AI-powered guidance.
                </p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True
    )
    st.markdown(
        """
        **Try asking even before you upload:**
        - What are the best stocks to buy right now?
        - How is the Indian market performing today?
        - Explain LTCG and STCG tax rules for equity investors
        """
    )
    
    if TESSERACT_AVAILABLE:
        tab1, tab2, tab3 = st.tabs(["📄 CSV Upload", "📸 OCR Upload", "⚙️ Features"])
    else:
        tab1, tab2 = st.tabs(["📄 CSV Upload", "⚙️ Features"])
    
    with tab1:
        st.markdown(
            """
            **Upload a portfolio CSV with the following columns:**
            - `Ticker` or `Stock`
            - `Quantity` or `Qty`
            - Optional: `Avg Price`, `Average`, `Average Price` <span class="info-tip" title="Include your average acquisition price per share. Leave blank if you're unsure.">?</span>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "📥 Download Sample CSV",
            SAMPLE_PORTFOLIO_CSV,
            file_name=SAMPLE_PORTFOLIO_FILENAME,
            mime="text/csv",
            key="sample_csv_download_onboarding",
        )
    
    if TESSERACT_AVAILABLE:
        with tab2:
            st.markdown(
                f"""
                **Upload portfolio screenshots for automatic OCR extraction.**

                ✅ Supported formats: **PNG, JPG, JPEG**  
                ✅ Tips: use crisp images, strong contrast, and include complete rows.

                OCR will extract:
                - Stock tickers or symbols
            - Quantity values
            - Average purchase prices
            
                *Smart caching keeps prices fresh for {CACHE_DURATION_MINUTES} minutes to speed up repeated analyses.*
                """
            )
        
        with tab3:
            st.markdown(
                f"""
                ### ✨ What you get
                - **Smart caching** of prices for {CACHE_DURATION_MINUTES} minutes
                - **Live Yahoo Finance pricing** with INR conversion
                - **P&L analysis**, momentum leaders, and laggards
                - **Personalized AI advisor** augmented with your holdings
                - **News briefs** tailored to your portfolio
                """
            )
    else:
        with tab2:
            st.markdown(
                """
                ### 📸 Enable OCR
                Install Tesseract OCR to unlock screenshot parsing.  
                Until then, upload a CSV to get started.
                """
            )

    if st.button("🆘 Help", key="sidebar_help"):
        st.session_state.show_help = not st.session_state.show_help

    with st.expander("🆘 Help & Onboarding", expanded=st.session_state.show_help):
        st.markdown(
            """
            **How to upload CSV**
            - Use the 📥 sample CSV as a template
            - Keep columns: `Ticker`, `Quantity`, and optional `Average Price`

            **How to use OCR**
            - Upload clear screenshots under the OCR option
            - Click **🔍 Extract Data from Images** to convert tables into a CSV

            **What to ask**
            - "What are the best stocks to buy right now?"
            - "How is the Indian market performing today?"
            - "Explain LTCG and STCG tax rules"

            **Feature overview**
            - Portfolio health scoring with diversification checks
            - Cached + live INR pricing, with refresh controls
            - AI advisor citing portfolio, news, and knowledge sources
                """
            )

# Chat Interface
st.header("💬 AI Financial Advisor")
st.markdown(
    "<div class='chat-hint'>💡 Tip: Ask for rebalancing suggestions, scenario analysis, or risk checks on specific holdings.</div>",
    unsafe_allow_html=True
)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def handle_chat_query(user_query: str) -> None:
    """Render chat interaction for a given user query."""
    st.session_state.messages.append({"role": "user", "content": user_query})
    if st.session_state.active_session_id:
        save_chat_message(st.session_state.active_session_id, "user", user_query)

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.rag_system and not st.session_state.updated_portfolio_df.empty:
            with st.spinner("🧠 Building knowledge base..."):
                relevant_docs = st.session_state.rag_system.get_relevant_context(user_query, k=7)
                response_text = get_financial_advice(hf_client, user_query=user_query, context_docs=relevant_docs)
                st.markdown(response_text)

                preview_docs = relevant_docs[:3]
                if preview_docs:
                    with st.expander("View Context Used"):
                        for doc in preview_docs:
                            st.write(doc)
                        remaining = len(relevant_docs) - len(preview_docs)
                        if remaining > 0:
                            st.caption(f"... and {remaining} more sources")
                else:
                    with st.expander("View Context Used"):
                        st.caption("No additional context was retrieved for this answer.")

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            if st.session_state.active_session_id:
                save_chat_message(st.session_state.active_session_id, "assistant", response_text)
        else:
            fallback_message = (
                "Please upload your portfolio CSV or images first, then click 'Load and Analyze Portfolio' to get personalized advice."
                "\n\nYou can still ask about:\n"
                "- How is the Indian market performing today?\n"
                "- Explain LTCG and STCG tax rules\n"
                "- Diversification or SIP strategies without holdings"
            )
            st.info(fallback_message)
            st.session_state.messages.append({"role": "assistant", "content": fallback_message})
            if st.session_state.active_session_id:
                save_chat_message(st.session_state.active_session_id, "assistant", fallback_message)


# OCR debug view
if st.session_state.ocr_results:
    with st.expander("🔍 View Raw OCR Results"):
        st.write(f"Total text segments extracted: {len(st.session_state.ocr_results)}")

        high_conf = [r for r in st.session_state.ocr_results if r["confidence"] >= 0.8]
        med_conf = [r for r in st.session_state.ocr_results if 0.5 <= r["confidence"] < 0.8]
        low_conf = [r for r in st.session_state.ocr_results if r["confidence"] < 0.5]

        st.write(f"✅ High confidence (≥80%): {len(high_conf)}")
        st.write(f"⚠️ Medium confidence (50-80%): {len(med_conf)}")
        st.write(f"❌ Low confidence (<50%): {len(low_conf)}")

        st.subheader("Extracted Text (First 50 items)")
        for idx, result in enumerate(st.session_state.ocr_results[:50]):
            confidence_icon = "✅" if result["confidence"] >= 0.8 else "⚠️" if result["confidence"] >= 0.5 else "❌"
            st.text(f"{idx+1}. {confidence_icon} {result['text']} (conf: {result['confidence']:.2%})")

if not st.session_state.messages:
    st.markdown("**Suggested questions to get started:**")
    suggestion_container = st.container()
    with suggestion_container:
        st.markdown("<div class='suggestion-buttons'>", unsafe_allow_html=True)
        suggestion_cols = st.columns(3)
        suggested_questions = [
            "What are the best stocks to buy right now?",
            "How is the Indian market performing today?",
            "Explain LTCG and STCG tax rules",
        ]
        suggestion_clicked: Optional[str] = None
        for idx, (col, question) in enumerate(zip(suggestion_cols, suggested_questions)):
            if col.button(question, key=f"suggestion_{idx}", use_container_width=True):
                suggestion_clicked = question
        st.markdown("</div>", unsafe_allow_html=True)
    if suggestion_clicked:
        handle_chat_query(suggestion_clicked)

if user_input := st.chat_input("Ask about your portfolio, specific stocks, or market trends..."):
    handle_chat_query(user_input)

















