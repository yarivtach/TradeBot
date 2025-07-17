import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


try:
    from .fundamental import FundamentalAnalysis
except ImportError:
    from fundamental import FundamentalAnalysis

class FundamentalAnalysisTester:
    def __init__(self, fa_obj: FundamentalAnalysis):
        assert fa_obj.ratios is not None, "compute_ratios() first"
        self.fa = fa_obj
        self.df = fa_obj.ratios

    def basic_checks(self):
        # Example: last year’s ROE shouldn't be negative for > 3 consecutive yrs
        bad = (self.df["roe"].tail(3) < 0).sum()
        assert bad <= 1, "ROE negative too often"

    def plot_ratio(self, ratio: str):

        if self.df.empty or ratio not in self.df.columns or self.df[ratio].dropna().empty:
            print(f"No data to plot for {self.fa.symbol} – {ratio}")
            return
        self.df[ratio].plot(kind="bar", title=f"{self.fa.symbol} – {ratio}")
        plt.tight_layout(); plt.show()


def get_big7_weighted_fundamentals_as_tqqq_proxy(out_dir):
    """
    Analyze the Big 7 tech stocks and return their QQQ-weighted mean fundamental ratios as a proxy for TQQQ.
    """
    BIG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    BIG7_WEIGHTS = {
        "AAPL": 0.12,
        "MSFT": 0.10,
        "GOOGL": 0.07,
        "AMZN": 0.06,
        "META": 0.04,
        "NVDA": 0.08,
        "TSLA": 0.04
    }
    all_ratios = []
    for symbol in BIG7:
        print(f"Analyzing {symbol}...")
        fa7 = FundamentalAnalysis(symbol)
        fa7.load_statements()
        fa7.compute_ratios()
        if fa7.ratios is not None and not fa7.ratios.empty:
            all_ratios.append(fa7.ratios.iloc[-1])  # Use latest year
        else:
            print(f"No data for {symbol}.")

    if all_ratios:
        big7_df = pd.DataFrame(all_ratios, index=BIG7)
        # Weighted average
        weights = pd.Series(BIG7_WEIGHTS)
        weighted = (big7_df.mul(weights, axis=0)).sum(skipna=True)
        print("--- BIG 7 WEIGHTED FUNDAMENTALS (TQQQ Proxy) ---")
        print(weighted)
        # Save as TXT in 'parameter: value' format (one per line)
        with open(os.path.join(out_dir, "tqqq_weighted_fundamentals.txt"), "w", encoding="utf-8") as f:
            for param, value in weighted.items():
                f.write(f"{param}: {value}\n")
        # Append actual TQQQ price from yfinance

        try:
            tqqq_price = yf.Ticker("TQQQ").history(period="1d")["Close"].iloc[-1]
        except Exception:
            tqqq_price = None
        with open(os.path.join(out_dir, "tqqq_weighted_fundamentals.txt"), "a", encoding="utf-8") as f:
            f.write(f"actual_price: {tqqq_price}\n")
        # Plot only if data exists
        import matplotlib.pyplot as plt
        for ratio in ["roe", "gross_margin", "pe_ttm", "rev_yoy"]:
            if ratio in big7_df.columns and not big7_df[ratio].dropna().empty:
                big7_df[ratio].plot(kind="bar", title=f"Big 7 – {ratio}")
                plt.tight_layout(); plt.show()
            else:
                print(f"No data to plot for Big 7 – {ratio}")
        # Save the full table too
        big7_df.to_csv(os.path.join(out_dir, "big7_fundamentals_table.csv"))
        return weighted
    else:
        print("No data for Big 7.")
        return None


if __name__ == "__main__":
    fa = FundamentalAnalysis("TQQQ")
    fa.load_statements()
    fa.compute_ratios()
    print(fa.latest_summary().head(20))

    # Persist for the tester / dashboard
    out = r"d:\_WORK\Yariv\Projects\TradeBot\tradebot\analysis\Tests\Result"
    os.makedirs(out, exist_ok=True)
    # Example tester usage
    tester = FundamentalAnalysisTester(fa)
    tester.basic_checks()
    tester.plot_ratio("roe")

    # --- Big 7 Weighted Analysis as TQQQ Proxy ---
    tqqq_fundamentals = get_big7_weighted_fundamentals_as_tqqq_proxy(out)
    if tqqq_fundamentals is not None:
        print("\nTQQQ Fundamental Proxy (Big 7 Weighted):")
        print(tqqq_fundamentals) 