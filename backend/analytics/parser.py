import io
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "side",
    "asset",
    "quantity",
    "entry_price",
    "exit_price",
    "pnl",
    "balance",
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column name variants to our required schema."""
    colmap = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=colmap)

    # common synonyms
    synonyms = {
        "buy/sell": "side",
        "buysell": "side",
        "instrument": "asset",
        "symbol": "asset",
        "qty": "quantity",
        "size": "quantity",
        "entry": "entry_price",
        "entryprice": "entry_price",
        "exit": "exit_price",
        "exitprice": "exit_price",
        "p/l": "pnl",
        "pl": "pnl",
        "profit": "pnl",
        "profit_loss": "pnl",
        "account_balance": "balance",
    }
    df = df.rename(columns={k: v for k, v in synonyms.items() if k in df.columns})

    return df

def parse_trade_file(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded CSV/Excel file into a normalized DataFrame.

    Supported:
    - .csv
    - .xlsx, .xls
    """
    name = getattr(uploaded_file, "name", "").lower()
    content = uploaded_file.read()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    df = normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {REQUIRED_COLUMNS}")

    # Type conversions
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["asset"] = df["asset"].astype(str).str.upper().str.strip()

    numeric_cols = ["quantity", "entry_price", "exit_price", "pnl", "balance"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp"] + numeric_cols)

    # Sort for time-series analysis
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
