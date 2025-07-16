import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Sequence, Optional

pd.set_option('display.float_format', '{:.6f}'.format)

try:
    from .technical import TechnicalAnalysis
except ImportError:
    from technical import TechnicalAnalysis


class TechnicalAnalysisTester:
    """
    Integration-test & visual sanity-check for a TechnicalAnalysis instance.
    Saves summary and plots to a specified output directory.
    """
    _price_cols = {"open", "high", "low", "close",
                   "adj close", "adj_close", "volume"}   # FIX: added 'adj_close'

    def __init__(self, ta_obj: "TechnicalAnalysis", out_dir: str):
        if getattr(ta_obj, "data", None) is None:
            raise AttributeError("ta_obj must expose a .data attribute")
        self.ta = ta_obj
        self.df: pd.DataFrame = ta_obj.data.copy(deep=False)
        if self.df.empty:
            raise ValueError("TechnicalAnalysis.data is empty – run load_data() first")

        self.indicators: List[str] = [
            c for c in self.df.columns if c.lower() not in self._price_cols
        ]
        if not self.indicators:
            raise ValueError("No indicator columns found – did you forget compute_indicators()?")

        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # BASIC TESTS
    # ------------------------------------------------------------------
    def _check_exists(self, cols: Sequence[str]):
        missing = [c for c in cols if c not in self.df.columns]
        assert not missing, f"Missing expected columns: {missing}"

    def _check_finite(self, cols: Sequence[str], window: int = 252):
        recent = self.df[cols].tail(window)
        if recent.isna().all().all():
            raise AssertionError(f"All values NaN for columns {cols} in last {window} rows")
        assert np.isfinite(recent.values).any(), (
            f"No finite values detected for {cols} in last {window} rows"
        )

    def run_checks(self, verbose: bool = True) -> str:
        self._check_exists(self.indicators)
        for col in self.indicators:
            self._check_finite([col])

        summary = self.latest_summary()
        summary_path = os.path.join(self.out_dir,
                                    f"{self.ta.symbol}_latest_summary.csv")
        summary.to_csv(summary_path)

        if verbose:
            print(f"✓ {len(self.indicators)} indicators validated.")
            print(summary.T)
        return summary.to_string()

    def _resolve_column(self, requested: str) -> str:             # FIX: helper
        """Return the actual column name that matches `requested` (case-insensitive)."""
        for c in self.df.columns:
            if c.lower() == requested.lower():
                return c
        # fallback: partial match (e.g. "macd" → "macd_12_26_9")
        matches = [c for c in self.df.columns if requested.lower() in c.lower()]
        if matches:
            return matches[0]
        raise KeyError(f"{requested} not found in DataFrame")

    def plot(self, indicator: str, *, sub_plot=False, price_col="close", figsize: tuple = (12, 6), save: bool = True):
        ind_col  = self._resolve_column(indicator)                # FIX
        price_col = self._resolve_column(price_col)               # FIX

        fig, ax = plt.subplots(figsize=figsize)
        self.df[price_col].plot(ax=ax, lw=1.2, label=price_col.title())

        target_ax = ax if not sub_plot else fig.add_subplot(212, sharex=ax)
        self.df[ind_col].plot(ax=target_ax, lw=1,
                              color="tab:orange" if sub_plot else "tab:red",
                              label=ind_col.upper())
        target_ax.set_ylabel(ind_col.upper())
        target_ax.legend(loc="upper right" if sub_plot else "upper left")
        ax.set_title(f"{self.ta.symbol} – {ind_col.upper()} vs. {price_col.title()}")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.out_dir,
                                     f"{self.ta.symbol}_{ind_col}.png"))
        plt.close(fig)

    def latest_summary(self) -> pd.DataFrame:
        # FIX: pick the last row that has *any* finite value
        latest_row = self.df[self.indicators].replace([np.inf, -np.inf], np.nan)
        latest_row = latest_row.dropna(how="all").iloc[-1:]
        return latest_row.T.rename(columns={latest_row.index[-1]: "latest_value"})


if __name__ == "__main__":
    OUT_DIR = r"d:\_WORK\Yariv\Projects\TradeBot\tradebot\analysis\Tests\Result"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Example usage: test TQQQ
    symbol = "TQQQ"
    ta = TechnicalAnalysis(symbol, start="2019-01-01")
    ta.load_data()
    ta.compute_indicators()

    tester = TechnicalAnalysisTester(ta, OUT_DIR)

    # Quick numeric summaries ------------------------------------------------
    ranges = {"day": 1, "week": 5, "month": 21, "50d": 50, "200d": 200}
    today = datetime.date.today().strftime("%Y%m%d")

    all_summaries = []
    for label, days in ranges.items():
        block = tester.df.tail(days)
        if block.empty:
            continue
        last  = block.iloc[-1][tester.indicators]
        mean  = block.mean(numeric_only=True)[tester.indicators]
        txt = pd.DataFrame({"last": last, "mean": mean}).to_string()
        all_summaries.append(f"--- {label.upper()} ---\n{txt}\n")
    # Write all summaries to one file
    summary_path = os.path.join(OUT_DIR, f"{ta.symbol}_all_summaries_{today}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_summaries))
    print(f"Saved all summaries to: {summary_path}")

    # Plots ------------------------------------------------------------------
    for ind in ["rsi_14", "macd", "sma_50", "sma_200"]:
        try:
            tester.plot(ind, sub_plot=(ind == "macd"))
        except Exception as e:
            print(f"Plotting {ind} failed – {e}") 