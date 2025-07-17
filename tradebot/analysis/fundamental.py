import pandas as pd
import yfinance as yf
import numpy as np
import os
from typing import Optional, Dict

class FundamentalAnalysis:
    """
    Pulls financial statements with yfinance and derives common ratios.
    Supports annual or quarterly periods.
    """
    FIELD_MAP = {
        "revenue": "Total Revenue",
        "gross_profit": "Gross Profit",
        "oper_income": "Operating Income",
        "net_income": "Net Income",
        "equity": "Total Stockholder Equity",
        "assets": "Total Assets",
        "debt": "Total Debt",
        "eps": "Basic EPS",
        "fcf": "Free Cash Flow",
        "inventory": "Inventory",
        "current_assets": "Total Current Assets",
        "current_liabilities": "Total Current Liabilities",
        "ebit": "EBIT",
        "ebitda": "EBITDA",
        "cash": "Cash And Cash Equivalents",
    }

    def __init__(self, symbol: str, period: str = "annual"):
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self.period = period
        self.is_: Optional[pd.DataFrame] = None
        self.bs_: Optional[pd.DataFrame] = None
        self.cf_: Optional[pd.DataFrame] = None
        self.ratios: Optional[pd.DataFrame] = None

    def load_statements(self) -> None:
        """Download financial statements (annual or quarterly)."""
        if self.period == "annual":
            self.is_ = self.ticker.income_stmt
            self.bs_ = self.ticker.balance_sheet
            self.cf_ = self.ticker.cash_flow
        else:
            self.is_ = self.ticker.income_stmt(q=True)
            self.bs_ = self.ticker.balance_sheet(q=True)
            self.cf_ = self.ticker.cash_flow(q=True)
        self.is_, self.bs_, self.cf_ = [
            df.T if isinstance(df, pd.DataFrame) else pd.DataFrame()
            for df in (self.is_, self.bs_, self.cf_)
        ]

    def safe_replace(self, series_or_scalar, to_replace, value):
        if isinstance(series_or_scalar, pd.Series):
            return series_or_scalar.replace(to_replace, value)
        elif isinstance(series_or_scalar, (float, int)):
            return value if series_or_scalar == to_replace else series_or_scalar
        else:
            return series_or_scalar

    def compute_ratios(self) -> pd.DataFrame:
        # Special case: TQQQ or other ETF
        if self.symbol == "TQQQ":

            # Weighted Big 7 logic (copied from tests_fundamental.py)
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
                fa7 = FundamentalAnalysis(symbol)
                fa7.load_statements()
                fa7.compute_ratios()
                if fa7.ratios is not None and not fa7.ratios.empty:
                    all_ratios.append(fa7.ratios.iloc[-1])
            if all_ratios:
                big7_df = pd.DataFrame(all_ratios, index=BIG7)
                weights = pd.Series(BIG7_WEIGHTS)
                weighted = (big7_df.mul(weights, axis=0)).sum(skipna=True)
                # Save as TXT in the same style as technical
                out_dir = r"D:\_WORK\Yariv\Projects\TradeBot\tradebot\analysis\Result"
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "tqqq_weighted_fundamentals.txt"), "w", encoding="utf-8") as f:
                    for param, value in weighted.items():
                        f.write(f"{param}: {value}\n")
                # (No CSV export)
                self.ratios = pd.DataFrame(weighted).T
                self.ratios.index = [self.symbol]
                return self.ratios
            else:
                self.ratios = pd.DataFrame()
                return self.ratios
        if any(df is None or df.empty for df in (self.is_, self.bs_, self.cf_)):
            self.load_statements()
        is_, bs_, cf_ = self.is_, self.bs_, self.cf_
        ratios = pd.DataFrame(index=is_.index)
        # Profitability
        equity = self.safe_replace(bs_.get(self.FIELD_MAP["equity"], np.nan), 0, np.nan)
        assets = self.safe_replace(bs_.get(self.FIELD_MAP["assets"], np.nan), 0, np.nan)
        debt = self.safe_replace(bs_.get(self.FIELD_MAP["debt"], 0), 0, np.nan)
        revenue = is_.get(self.FIELD_MAP["revenue"], np.nan)
        net_income = is_.get(self.FIELD_MAP["net_income"], np.nan)
        ratios["gross_margin"] = is_.get(self.FIELD_MAP["gross_profit"], np.nan) / revenue
        ratios["oper_margin"] = is_.get(self.FIELD_MAP["oper_income"], np.nan) / revenue
        ratios["net_margin"] = net_income / revenue
        ratios["roe"] = net_income / equity
        ratios["roa"] = net_income / assets
        ratios["roic"] = net_income / (debt + equity)
        # Growth
        ratios["rev_yoy"] = revenue.pct_change() if isinstance(revenue, pd.Series) else np.nan
        eps = is_.get(self.FIELD_MAP["eps"], np.nan)
        ratios["eps_yoy"] = eps.pct_change() if isinstance(eps, pd.Series) else np.nan
        fcf = cf_.get(self.FIELD_MAP["fcf"], np.nan)
        ratios["fcf_yoy"] = fcf.pct_change() if isinstance(fcf, pd.Series) else np.nan
        # Liquidity & Solvency
        current_assets = self.safe_replace(bs_.get(self.FIELD_MAP["current_assets"], np.nan), 0, np.nan)
        current_liabilities = self.safe_replace(bs_.get(self.FIELD_MAP["current_liabilities"], np.nan), 0, np.nan)
        inventory = bs_.get(self.FIELD_MAP["inventory"], 0)
        ratios["current_ratio"] = current_assets / current_liabilities
        ratios["quick_ratio"] = (current_assets - inventory) / current_liabilities
        ebit = is_.get(self.FIELD_MAP["ebit"], np.nan)
        interest_exp = is_.get("Interest Expense", np.nan)
        interest_exp = self.safe_replace(interest_exp, 0, np.nan)
        ratios["interest_cov"] = ebit / interest_exp
        ratios["debt_equity"] = debt / equity
        ebitda = is_.get(self.FIELD_MAP["ebitda"], np.nan)
        cash = bs_.get(self.FIELD_MAP["cash"], 0)
        net_debt = debt - cash
        ebitda = self.safe_replace(ebitda, 0, np.nan)
        ratios["net_debt_ebitda"] = net_debt / ebitda
        # Cash-flow quality
        ratios["fcf_margin"] = fcf / revenue
        ratios["fcf_conv"] = fcf / net_income
        # Dividend / buy-backs
        try:
            shares = self.ticker.get_shares_full(start="2000-01-01")
            shares.index = shares.index.year
            ratios["shares_out"] = shares
            ratios["buyback_pct"] = -shares.pct_change()
        except Exception:
            ratios["shares_out"] = np.nan
            ratios["buyback_pct"] = np.nan
        # Valuation (uses latest market price)
        try:
            price = self.ticker.history(period="1d")["Close"].iloc[-1]
        except Exception:
            price = np.nan
        ratios["price"] = price
        ratios["pe_ttm"] = price / self.ticker.info.get("trailingEps", np.nan)
        try:
            ratios["ps_ttm"] = price / (revenue.iloc[-1] / shares.iloc[-1])
            ratios["pb"] = price / (equity.iloc[-1] / shares.iloc[-1])
            ev = (price * shares.iloc[-1] +
                  debt.iloc[-1] -
                  cash.iloc[-1])
            ratios["ev_ebitda"] = ev / ebitda.iloc[-1]
            ratios["ev_fcf"] = ev / fcf.iloc[-1]
        except Exception:
            ratios["ps_ttm"] = ratios["pb"] = ratios["ev_ebitda"] = ratios["ev_fcf"] = np.nan
        self.ratios = ratios.round(4)
        return self.ratios

    def latest_summary(self) -> pd.Series:
        if self.ratios is None:
            self.compute_ratios()
        if self.ratios.empty:
            print(f"No fundamental data available for {self.symbol}.")
            return pd.Series(dtype=float)
        return self.ratios.iloc[-1].dropna()

    def to_csv(self, path: str) -> None:
        if self.ratios is None:
            self.compute_ratios()
        self.ratios.to_csv(path)

