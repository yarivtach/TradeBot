import yfinance as yf
import pandas as pd
import pandas_ta as ta
from typing import Optional, List, Dict

class TechnicalAnalysis:
    """
    Comprehensive technical-analysis helper for OHLCV data and indicators.
    """
    def __init__(self, symbol: str,
                 start: str = "2015-01-01",
                 interval: str = "1d",
                 auto_adjust: bool = True):             # FIX: explicit kwarg
        self.symbol = symbol.upper()
        self.start = start
        self.interval = interval
        self.auto_adjust = auto_adjust                # FIX: store flag
        self.data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """
        Download and clean historical OHLCV quotes using yfinance.
        Always returns a *single-level* column index:  open, high, low, close, volume
        """
        raw = yf.download(self.symbol,
                          start=self.start,
                          interval=self.interval,
                          auto_adjust=self.auto_adjust)

        # ── FLATTEN ──────────────────────────────────────────────────────────
        if isinstance(raw.columns, pd.MultiIndex):
            # Decide which level is the price field
            level0 = raw.columns.get_level_values(0).str.lower()
            price_names = {"open", "high", "low", "close", "adj close", "volume"}
            if set(level0[:6]).issubset(price_names):      # ('open', 'tqqq')
                raw.columns = raw.columns.get_level_values(0)
            else:                                         # ('tqqq', 'open')
                raw.columns = raw.columns.get_level_values(1)

        # keep only standard OHLCV columns, drop dividends / splits
        raw = raw[[c for c in raw.columns
                   if c.lower() in {"open", "high", "low", "close", "volume"}]]

        raw.columns = raw.columns.str.lower()
        raw = raw.dropna(how="all")
        self.data = raw
        return self.data

    # ------------------------------------------------------------------
    # INDICATORS
    # ------------------------------------------------------------------
    def compute_indicators(self) -> pd.DataFrame:
        """Append a broad set of technical indicators to self.data."""
        if self.data is None or self.data.empty:
            self.load_data()
        df = self.data.copy()

        # Reference Series explicitly
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

        # --- Moving Averages ------------------------------------------------
        for length in (20, 50, 100, 200):
            df[f"sma_{length}"] = ta.sma(close, length=length)

        for length in (12, 26):
            df[f"ema_{length}"] = ta.ema(close, length=length)

        df["vwma_20"] = ta.vwma(close, vol, length=20)

        # --- MACD -----------------------------------------------------------
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        else:
            df[["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]] = pd.NA

        # --- Oscillators ----------------------------------------------------
        df["rsi_14"] = ta.rsi(close, length=14)
        stoch = ta.stoch(high, low, close, k=14, d=3)
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
        df["mom_10"] = ta.mom(close, length=10)
        df["roc_10"] = ta.roc(close, length=10)

        # --- Volatility -----------------------------------------------------
        bb = ta.bbands(close, length=20, std=2)
        kc = ta.kc(high, low, close, length=20, scalar=2.0)
        for tbl in (bb, kc):
            if tbl is not None:
                df = pd.concat([df, tbl], axis=1)

        df["atr_14"] = ta.atr(high, low, close, length=14)

        # --- Trend strength -------------------------------------------------
        adx = ta.adx(high, low, close, length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)
        df["cci_20"] = ta.cci(high, low, close, length=20)

        # --- Ichimoku Cloud -------------------------------------------------
        ichi = ta.ichimoku(high, low, close)
        if isinstance(ichi, tuple):
            ichi = ichi[0]
        if ichi is not None:
            df = pd.concat([df, ichi], axis=1)

        # --- Parabolic SAR, Donchian, SuperTrend ---------------------------
        has_cols = all([col in df.columns for col in ["high", "low", "close"]])
        def enough_valid(col):
            if col in df.columns and isinstance(df[col], pd.Series):
                return df[col].notna().sum() >= 2
            return False
        enough_high = enough_valid("high")
        enough_low = enough_valid("low")
        enough_close = enough_valid("close")
        if has_cols and enough_high and enough_low and enough_close:
            psar = ta.psar(high, low, close, step=0.02, max_step=0.2)
            if psar is not None and "PSARl_0.02_0.2" in psar.columns:
                df["psar"] = psar["PSARl_0.02_0.2"]
            else:
                df["psar"] = pd.NA
        else:
            df["psar"] = pd.NA
        donch = ta.donchian(high, low, lower_length=20, upper_length=20)
        if donch is not None:
            df = pd.concat([df, donch], axis=1)
        st = ta.supertrend(high, low, close, length=10, multiplier=3)
        if st is not None:
            df = pd.concat([df, st], axis=1)

        # --- Volume ---------------------------------------------------------
        if (
            "close" in df.columns and "volume" in df.columns and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["obv"] = ta.obv(close, vol)
        else:
            df["obv"] = pd.NA

        if (
            all([col in df.columns for col in ["high", "low", "close", "volume"]]) and
            isinstance(df["high"], pd.Series) and isinstance(df["low"], pd.Series) and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["high"].notna().sum() > 0 and df["low"].notna().sum() > 0 and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["cmf_20"] = ta.cmf(high, low, close, vol, length=20)
        else:
            df["cmf_20"] = pd.NA

        if (
            all([col in df.columns for col in ["high", "low", "close", "volume"]]) and
            isinstance(df["high"], pd.Series) and isinstance(df["low"], pd.Series) and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["high"].notna().sum() > 0 and df["low"].notna().sum() > 0 and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["mfi_14"] = ta.mfi(high, low, close, vol, length=14)
        else:
            df["mfi_14"] = pd.NA

        # ---------------------------------------------------------------
        #  AFTER *all* indicators have been concatenated
        # ---------------------------------------------------------------
        # FIX: give the most-used signals short, predictable aliases
        if {"MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"}.issubset(df.columns):
            df["macd"]        = df["MACD_12_26_9"]
            df["macd_signal"] = df["MACDs_12_26_9"]
            df["macd_hist"]   = df["MACDh_12_26_9"]

        if {"STOCHk_14_3_3", "STOCHd_14_3_3"}.issubset(df.columns):
            df["stoch_k"] = df["STOCHk_14_3_3"]
            df["stoch_d"] = df["STOCHd_14_3_3"]

        # FIX: make every column name lower-case for easy look-ups
        df.columns = [str(c).lower() for c in df.columns]

        # keep early rows but be sure 'close' itself is present
        close_col = next((col for col in df.columns if col.lower() == "close"), None)
        if close_col:
            self.data = df.dropna(how="all", subset=[close_col])
        else:
            self.data = df.dropna(how="all")
        return self.data

    # ------------------------------------------------------------------
    # QUICK LOOKS
    # ------------------------------------------------------------------
    def latest_signals(self) -> Dict:
        if self.data is None or self.data.empty:
            self.compute_indicators()
        return self.data.iloc[-1].to_dict()

    def plot(self, columns: Optional[List[str]] = None, rows: int = 2):
        """
        Plot price plus chosen indicators using mplfinance.
        Args:
            columns: Which indicator columns to include beneath the price pane.
            rows:    Number of additional subplots (max equal to len(columns)).
        """
        import mplfinance as mpf
        if self.data is None or self.data.empty:
            self.compute_indicators()

        plot_cols = columns or ["sma_50", "sma_200", "rsi_14"]
        addplots = []
        for i, col in enumerate(plot_cols):
            if col not in self.data.columns:
                continue
            addplots.append(
                mpf.make_addplot(self.data[col],
                                 panel=(i // rows) + 1)        # FIX: simpler panel calc
            )
        mpf.plot(self.data, type="candle", style="yahoo",
                 addplot=addplots, volume=True,
                 title=self.symbol) 