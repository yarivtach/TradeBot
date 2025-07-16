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
        """Download and clean historical OHLCV quotes using yfinance."""
        try:
            raw = yf.download(self.symbol,
                               start=self.start,
                               interval=self.interval,
                               auto_adjust=self.auto_adjust)
            if raw.empty:
                raise ValueError(f"No data for {self.symbol}")
            raw = raw.dropna().rename(columns=str.lower)
            self.data = raw
            return self.data
        except Exception as exc:
            print(f"[TechnicalAnalysis] load_data() → {exc}")
            self.data = pd.DataFrame()
            return self.data

    # ------------------------------------------------------------------
    # INDICATORS
    # ------------------------------------------------------------------
    def compute_indicators(self) -> pd.DataFrame:
        """Append a broad set of technical indicators to self.data."""
        if self.data is None or self.data.empty:
            self.load_data()
        df = self.data.copy()

        # --- Moving Averages ------------------------------------------------
        for length in (20, 50, 100, 200):
            df[f"sma_{length}"] = ta.sma(df.close, length=length)

        for length in (12, 26):
            df[f"ema_{length}"] = ta.ema(df.close, length=length)

        df["vwma_20"] = ta.vwma(df.close, df.volume, length=20)

        # --- MACD -----------------------------------------------------------
        macd = ta.macd(df.close, fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)        # FIX: simpler – keep original names
        else:
            df[["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]] = pd.NA

        # --- Oscillators ----------------------------------------------------
        df["rsi_14"] = ta.rsi(df.close, length=14)
        stoch = ta.stoch(df.high, df.low, df.close, k=14, d=3)
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
        df["mom_10"] = ta.mom(df.close, length=10)
        df["roc_10"] = ta.roc(df.close, length=10)

        # --- Volatility -----------------------------------------------------
        bb = ta.bbands(df.close, length=20, std=2)
        kc = ta.kc(df.high, df.low, df.close, length=20, scalar=2.0)
        for tbl in (bb, kc):
            if tbl is not None:
                df = pd.concat([df, tbl], axis=1)

        df["atr_14"] = ta.atr(df.high, df.low, df.close, length=14)

        # --- Trend strength -------------------------------------------------
        adx = ta.adx(df.high, df.low, df.close, length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)
        df["cci_20"] = ta.cci(df.high, df.low, df.close, length=20)

        # --- Ichimoku Cloud -------------------------------------------------
        ichi = ta.ichimoku(df.high, df.low, df.close)
        if isinstance(ichi, tuple):
            ichi = ichi[0]                            # pandas-ta >=0.3.14 returns tuple
        if ichi is not None:
            df = pd.concat([df, ichi], axis=1)

        # --- Parabolic SAR, Donchian, SuperTrend ---------------------------
        has_cols = all([col in df.columns for col in ["high", "low", "close"]])

        def enough_valid(col):
            # Ensure col is a Series, not a DataFrame
            if col in df.columns and isinstance(df[col], pd.Series):
                return df[col].notna().sum() >= 2
            return False

        enough_high = enough_valid("high")
        enough_low = enough_valid("low")
        enough_close = enough_valid("close")

        if has_cols and enough_high and enough_low and enough_close:
            psar = ta.psar(df.high, df.low, df.close, step=0.02, max_step=0.2)
            if psar is not None and "PSARl_0.02_0.2" in psar.columns:
                df["psar"] = psar["PSARl_0.02_0.2"]
            else:
                df["psar"] = pd.NA
        else:
            df["psar"] = pd.NA
        donch = ta.donchian(df.high, df.low, lower_length=20, upper_length=20)
        if donch is not None:
            df = pd.concat([df, donch], axis=1)
        st = ta.supertrend(df.high, df.low, df.close, length=10, multiplier=3)
        if st is not None:
            df = pd.concat([df, st], axis=1)

        # --- Volume ---------------------------------------------------------
        if (
            "close" in df.columns and "volume" in df.columns and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["obv"] = ta.obv(df.close, df.volume)
        else:
            df["obv"] = pd.NA

        if (
            all([col in df.columns for col in ["high", "low", "close", "volume"]]) and
            isinstance(df["high"], pd.Series) and isinstance(df["low"], pd.Series) and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["high"].notna().sum() > 0 and df["low"].notna().sum() > 0 and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["cmf_20"] = ta.cmf(df.high, df.low, df.close, df.volume, length=20)
        else:
            df["cmf_20"] = pd.NA

        if (
            all([col in df.columns for col in ["high", "low", "close", "volume"]]) and
            isinstance(df["high"], pd.Series) and isinstance(df["low"], pd.Series) and
            isinstance(df["close"], pd.Series) and isinstance(df["volume"], pd.Series) and
            df["high"].notna().sum() > 0 and df["low"].notna().sum() > 0 and
            df["close"].notna().sum() > 0 and df["volume"].notna().sum() > 0
        ):
            df["mfi_14"] = ta.mfi(df.high, df.low, df.close, df.volume, length=14)
        else:
            df["mfi_14"] = pd.NA

        # -------------------------------------------------------------------
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns]
        # Do **not** drop all rows: keep early bars for completeness.
        df.columns = [str(col) for col in df.columns]
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