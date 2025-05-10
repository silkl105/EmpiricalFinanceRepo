from __future__ import annotations

import itertools
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import diagnostic as smd
from statsmodels.stats import sandwich_covariance as sw
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score
from scipy.stats import spearmanr, kendalltau


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def annualised_sharpe_ratio(rets: pd.Series, riskfree: float = 0.0) -> float:
    """
    Return the annualised Sharpe ratio of *rets* (which are already excess
    returns, i.e. net of the risk-free rate).

    Parameters
    ----------
    rets :
        Monthly excess returns.
    riskfree :
        Ignored (kept for backwards compatibility).

    Notes
    -----
    Uses 12√ to translate from monthly to yearly statistics.
    """
    mu = rets.mean()
    sigma = rets.std()
    return np.sqrt(12) * mu / sigma if sigma != 0 else np.nan


def newey_west_tvals(model: sm.RegressionResults, max_lag: int | None = None) -> pd.Series:
    """
    Compute Newey-West HAC t statistics for a fitted statsmodels OLS result.

    Parameters
    ----------
    model :
        Fitted statsmodels regression result.
    max_lag :
        Newey-West truncation lag; if None use floor(4·T^1/4).
    """
    if max_lag is None:
        max_lag = int(np.floor(4 * model.nobs ** 0.25))
    hac_cov = sw.cov_hac(model, nlags=max_lag)
    return model.params / np.sqrt(np.diag(hac_cov))


# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TechnicalTradingCase:
    _tree_param_dist = {
        "max_depth": [None] + list(range(3, 14)),
        "min_samples_split": [2, 5, 10, 20, 50, 100],
        "min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "max_features": [None, "sqrt", "log2", 0.3, 0.5, 0.7],
        "max_leaf_nodes": [None, 5, 10, 20, 50, 100],
    }
    """
    Implements the complete momentum-strategy analysis up to (and including)
    task #7 of the case study.

    Parameters
    ----------
    root_dir :
        Path where FF5.csv, FF_MOM.csv, and 48_Industry_Portfolios.CSV
        are stored.
    """
    root_dir: pathlib.Path | str = "."
    use_excess: bool = True

    # --------------------------------------------------------------------- #
    def numeric_rank_strategy(self, q: float = 0.10) -> tuple[pd.Series, sm.RegressionResults]:
        """
        In-sample long-short using numeric decile ranks instead of 30 dummies.
        Collapses each signal's decile into a numeric 1–10 variable,
        fits cross-sectional OLS on these three ranks (no intercept),
        and constructs a q/1-q long-short portfolio.
        """
        df_next = self.industries.shift(-1)
        y = df_next.stack(future_stack=True)

        z12 = self.deciles["Z12"].stack(future_stack=True)
        z6 = self.deciles["Z6"].stack(future_stack=True)
        ret1 = self.deciles["Ret1"].stack(future_stack=True)

        panel = pd.concat([y, z12, z6, ret1], axis=1, keys=["ret","Z12","Z6","Ret1"]).dropna()
        y_aligned = panel["ret"]
        X = panel[["Z12","Z6","Ret1"]]

        result = sm.OLS(y_aligned, X).fit()
        preds = X.dot(result.params).unstack()

        ls = []
        for t, row in preds.iterrows():
            rnk = row.rank(method="first")
            n = rnk.count()
            if n == 0:
                ls.append(np.nan)
                continue
            long = row[rnk > n * (1 - q)]
            short = row[rnk <= n * q]
            ret_row = df_next.loc[t]
            if long.empty or short.empty:
                ls.append(np.nan)
            else:
                ls.append(ret_row[long.index].mean() - ret_row[short.index].mean())

        returns = pd.Series(ls, index=preds.index, name="RankedStrategy")
        return returns, result

    def regression_tree_strategy(
        self, q: float = 0.10, n_iter: int = 50
    ) -> tuple[pd.Series, RandomizedSearchCV]:
        """
        In-sample long-short returns using a regression tree on numeric rank features.
        Hyperparameter tuning via RandomizedSearchCV with n_iter iterations.

        Returns
        -------
        returns :
            Series of long-short excess returns per month.
        search :
            Fitted RandomizedSearchCV object (best_estimator_, best_score_).
        """
        # Next-month returns
        df_next = self.industries.shift(-1)
        y = df_next.stack(future_stack=True)

        # Numeric rank signals
        z12 = self.deciles["Z12"].stack(future_stack=True)
        z6 = self.deciles["Z6"].stack(future_stack=True)
        ret1 = self.deciles["Ret1"].stack(future_stack=True)

        # Align and drop missing
        panel = pd.concat([y, z12, z6, ret1], axis=1, keys=["ret","Z12","Z6","Ret1"]).dropna()
        y_aligned = panel["ret"]
        X = panel[["Z12","Z6","Ret1"]].astype(float)

        # Hyperparameter search space
        param_dist = {
            "max_depth": [None] + list(range(3, 14)),
            "min_samples_split": [2, 5, 10, 20, 50, 100],
            "min_samples_leaf": [1, 2, 5, 10, 20, 50],
            "max_features": [None, "sqrt", "log2", 0.3, 0.5, 0.7],
            "max_leaf_nodes": [None, 5, 10, 20, 50, 100],
        }
        tree = DecisionTreeRegressor(random_state=0)
        scorer = make_scorer(r2_score)
        search = RandomizedSearchCV(
            tree, param_dist, n_iter=n_iter, cv=5,
            scoring=scorer, random_state=10, n_jobs=-1,
        )
        search.fit(X, y_aligned)
        model = search.best_estimator_

        # Forecast expected returns per month × industry
        preds = []
        for t in df_next.index:
            df_feat = pd.DataFrame({
                "Z12": self.deciles["Z12"].loc[t].astype(float),
                "Z6": self.deciles["Z6"].loc[t].astype(float),
                "Ret1": self.deciles["Ret1"].loc[t].astype(float)
            })
            preds.append(model.predict(df_feat))
        exp_ret = pd.DataFrame(preds, index=df_next.index, columns=df_next.columns)

        # Build long-short portfolio returns
        ls = []
        for t, row in exp_ret.iterrows():
            rnk = row.rank(method="first")
            n = rnk.count()
            if n == 0:
                ls.append(np.nan)
                continue
            long_idx = rnk > n * (1 - q)
            short_idx = rnk <= n * q
            ret_row = df_next.loc[t]
            if not long_idx.any() or not short_idx.any():
                ls.append(np.nan)
            else:
                ls.append(ret_row[long_idx].mean() - ret_row[short_idx].mean())

        returns = pd.Series(ls, index=exp_ret.index, name="TreeStrategy")
        return returns, search

    def oos_dummy_strategy(
        self, train_window: int = 36, q: float = 0.10, start_year: int = 2001
    ) -> tuple[pd.Series, float]:
        """
        Out-of-sample long-short using 30 dummy regression with a rolling window.

        Parameters
        ----------
        train_window :
            Number of months to train on (e.g. 36).
        q :
            Fraction to long/short.
        start_year :
            First year to start OOS evaluation (e.g. 2001 for post-2000).

        Returns
        -------
        ls_returns :
            Series of OOS long-short excess returns.
        oos_r2 :
            R² of stacked OOS predictions vs. actual returns.
        """
        dates = self.industries.index
        df_next = self.industries.shift(-1)

        oos_positions = [
            i for i, d in enumerate(dates)
            if d.year >= start_year and i >= train_window
        ]
        ls_list = []
        y_true, y_pred = [], []

        for i in oos_positions:
            t = dates[i]
            train_idx = dates[i - train_window : i]
            # Training data
            y_train = df_next.loc[train_idx].stack(future_stack=True)
            # Build separate dummy matrices for each signal at train dates
            dmatrices = []
            for sig in ("Z12", "Z6", "Ret1"):
                dec_ser = self.deciles[sig].loc[train_idx].stack(future_stack=True)
                dmatrices.append(pd.get_dummies(dec_ser, prefix=f"{sig}D").astype(float))
            X_train = pd.concat(dmatrices, axis=1)
            # Align y_train and X_train
            full_train = pd.concat([y_train, X_train], axis=1).dropna()
            y_train_aligned = full_train.iloc[:, 0]
            X_train_aligned = full_train.iloc[:, 1:]
            model = sm.OLS(y_train_aligned, X_train_aligned).fit()

            # Forecast for date t
            # Build OOS design matrix for each industry
            assets = df_next.columns
            X_t = pd.DataFrame(0.0, index=assets, columns=model.params.index)
            for sig in ("Z12", "Z6", "Ret1"):
                dec_vals = self.deciles[sig].loc[t]
                for asset, dec in dec_vals.items():
                    col = f"{sig}D_{int(dec)}"
                    if col in X_t.columns:
                        X_t.at[asset, col] = 1.0
            # Predict expected returns
            pred = X_t.dot(model.params)
            # Record actual and predicted for R²
            y_true.extend(df_next.loc[t].values.tolist())
            y_pred.extend(pred.values.tolist())

            exp_ret = pred
            rnk = exp_ret.rank(method="first")
            n = rnk.count()
            longs = exp_ret[rnk > n * (1 - q)].index
            shorts = exp_ret[rnk <= n * q].index
            ret_row = df_next.loc[t]
            ls = np.nan
            if longs.size and shorts.size:
                ls = ret_row[longs].mean() - ret_row[shorts].mean()
            ls_list.append(ls)

        ls_series = pd.Series(ls_list, index=dates[oos_positions], name="DummyOOS")
        # Compute OOS R², dropping NaNs
        y_arr = np.array(y_true, dtype=float)
        pred_arr = np.array(y_pred, dtype=float)
        valid = ~np.isnan(y_arr) & ~np.isnan(pred_arr)
        if valid.any():
            oos_r2 = r2_score(y_arr[valid], pred_arr[valid])
        else:
            oos_r2 = np.nan
        return ls_series, oos_r2

    def oos_numeric_rank_strategy(
        self, train_window: int = 36, q: float = 0.10, start_year: int = 2001
    ) -> tuple[pd.Series, float]:
        """
        Out-of-sample long-short using numeric rank regression with a rolling window.
        """
        dates = self.industries.index
        df_next = self.industries.shift(-1)

        oos_positions = [
            i for i, d in enumerate(dates)
            if d.year >= start_year and i >= train_window
        ]
        ls_list = []
        y_true, y_pred = [], []

        for i in oos_positions:
            t = dates[i]
            train_idx = dates[i - train_window : i]
            # Training data
            y_train = df_next.loc[train_idx].stack(future_stack=True)
            X_train = pd.concat([
                self.deciles[sig].loc[train_idx].stack(future_stack=True).rename(sig)
                for sig in ("Z12", "Z6", "Ret1")
            ], axis=1).dropna()
            y_train = y_train.loc[X_train.index]
            model = sm.OLS(y_train, X_train).fit()

            # Forecast for date t
            X_t = pd.DataFrame({
                sig: self.deciles[sig].loc[t].astype(float)
                for sig in ("Z12", "Z6", "Ret1")
            })
            pred = X_t.dot(model.params).values
            y_true.extend(df_next.loc[t].values.tolist())
            y_pred.extend(pred.tolist())

            exp_ret = pd.Series(pred, index=df_next.columns)
            rnk = exp_ret.rank(method="first")
            n = rnk.count()
            longs = exp_ret[rnk > n * (1 - q)].index
            shorts = exp_ret[rnk <= n * q].index
            ret_row = df_next.loc[t]
            ls = np.nan
            if longs.size and shorts.size:
                ls = ret_row[longs].mean() - ret_row[shorts].mean()
            ls_list.append(ls)

        ls_series = pd.Series(ls_list, index=dates[oos_positions], name="RankedOOS")
        # Compute OOS R², dropping NaNs
        y_arr = np.array(y_true, dtype=float)
        pred_arr = np.array(y_pred, dtype=float)
        valid = ~np.isnan(y_arr) & ~np.isnan(pred_arr)
        if valid.any():
            oos_r2 = r2_score(y_arr[valid], pred_arr[valid])
        else:
            oos_r2 = np.nan
        return ls_series, oos_r2

    def oos_regression_tree_strategy(
        self, train_window: int = 36, q: float = 0.10, start_year: int = 2001
    ) -> tuple[pd.Series, float]:
        """
        Out-of-sample long-short using regression-tree with a rolling window.
        """
        dates = self.industries.index
        df_next = self.industries.shift(-1)

        oos_positions = [
            i for i, d in enumerate(dates)
            if d.year >= start_year and i >= train_window
        ]
        ls_list = []
        y_true, y_pred = [], []

        for i in oos_positions:
            t = dates[i]
            train_idx = dates[i - train_window : i]
            # Training data
            y_train = df_next.loc[train_idx].stack(future_stack=True)
            X_train = pd.concat([
                self.deciles[sig].loc[train_idx].stack(future_stack=True).rename(sig)
                for sig in ("Z12", "Z6", "Ret1")
            ], axis=1).astype(float).dropna()
            y_train = y_train.loc[X_train.index]

            # CV on training
            param_dist = self._tree_param_dist  # assume defined above or replicate
            tree = DecisionTreeRegressor(random_state=0)
            search = RandomizedSearchCV(
                tree, param_dist, n_iter=50, cv=5,
                scoring=make_scorer(r2_score), random_state=0, n_jobs=-1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_

            # Forecast for date t
            X_t = pd.DataFrame({
                sig: self.deciles[sig].loc[t].astype(float)
                for sig in ("Z12", "Z6", "Ret1")
            })
            pred = model.predict(X_t)
            y_true.extend(df_next.loc[t].values.tolist())
            y_pred.extend(pred.tolist())

            exp_ret = pd.Series(pred, index=df_next.columns)
            rnk = exp_ret.rank(method="first")
            n = rnk.count()
            longs = exp_ret[rnk > n * (1 - q)].index
            shorts = exp_ret[rnk <= n * q].index
            ret_row = df_next.loc[t]
            ls = np.nan
            if longs.size and shorts.size:
                ls = ret_row[longs].mean() - ret_row[shorts].mean()
            ls_list.append(ls)

        ls_series = pd.Series(ls_list, index=dates[oos_positions], name="TreeOOS")
        # Compute OOS R², dropping NaNs
        y_arr = np.array(y_true, dtype=float)
        pred_arr = np.array(y_pred, dtype=float)
        valid = ~np.isnan(y_arr) & ~np.isnan(pred_arr)
        if valid.any():
            oos_r2 = r2_score(y_arr[valid], pred_arr[valid])
        else:
            oos_r2 = np.nan
        return ls_series, oos_r2
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self.root_dir = pathlib.Path(self.root_dir)
        self._load_and_prepare_data()
        self._build_all_signals()

    # ------------------------- #
    # Task 1 — performance      #
    # ------------------------- #
    def performance_table(self, formation: str = "Z12") -> pd.DataFrame:
        """
        Compute mean, std, and Sharpe ratio of the long-short strategy for the
        full sample and consecutive non-overlapping 10-year windows.

        Parameters
        ----------
        formation :
            Which signal to base the decile strategy on: "Z12", "Z6",
            or "Ret1".

        Returns
        -------
        table :
            Multi-index (metric x subperiod) DataFrame.
        """
        ls = self.long_short_returns[formation]
        groups = {"Full sample": slice(None)}
        # build 10‑year non‑overlapping windows
        years = sorted({d.year for d in ls.index})
        windows = list(zip(years[::10], years[9::10]))
        for lo, hi in windows:
            groups[f"{lo}-{hi}"] = (ls.index.year >= lo) & (ls.index.year <= hi)

        rows = {}
        for label, mask in groups.items():
            sample = ls.loc[mask]
            rows[("Mean Return (%)", label)] = sample.mean() * 100
            rows[("Std. Deviation of Returns (%)", label)] = sample.std() * 100
            rows[("Sharpe Ratio", label)] = annualised_sharpe_ratio(sample)

        table = pd.Series(rows).unstack()
        return table

    # ------------------------- #
    # Task 2 — constant α test  #
    # ------------------------- #
    def constant_alpha_test(self, formation: str = "Z12") -> sm.RegressionResults:
        """
        Regress long-short returns on a constant to test profitability.

        Returns the fitted OLS RegressionResults instance (with HAC t-stats
        added as .t_hac attribute).
        """
        y = self.long_short_returns_factor[formation].dropna()
        if y.empty:
            raise ValueError("No observations available for constant‑alpha test.")
        X = np.ones((len(y), 1))  # explicit 2‑D constant
        model = sm.OLS(y, X, hasconst=True).fit()
        model.t_hac = newey_west_tvals(model)
        return model

    def decile_monotonicity_plot(self, formation: str = "Z12") -> None:
        """
        Plot mean next-month returns (in percent) for each formation decile.
        """
        means = self.decile_mean_returns[formation] * 100  # percent
        ax = means.plot(kind="bar")
        plt.title(f"Next-month return vs. {formation} decile")
        plt.ylabel("Mean return [%]")
        plt.xlabel("Decile (1 = worst, 10 = best)")
        # annotate each bar with its value
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.2f}%", 
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center", va="bottom")
        plt.tight_layout()
        # Monotonicity tests
        ranks = means.index.to_numpy()
        vals = means.to_numpy()
        rho, p_s = spearmanr(ranks, vals)
        tau, p_k = kendalltau(ranks, vals)
        print(f"Spearman's ρ = {rho:.3f}, p-value = {p_s:.3f}")
        print(f"Kendall's τ = {tau:.3f}, p-value = {p_k:.3f}")

    # ------------------------- #
    # Task 3 — FF(5)+MOM α test #
    # ------------------------- #
    def fama_french_regression(
        self, formation: str = "Z12", include_mom: bool = True
    ) -> sm.RegressionResults:
        """
        Regress excess long-short returns on Fama-French 5 factors (optionally
        plus the momentum factor).

        Parameters
        ----------
        formation :
            Signal underlying the long-short strategy.
        include_mom :
            If True include UMD (momentum) in the RHS.

        Returns
        -------
        Fitted regression result.
        """
        y = self.long_short_returns_factor[formation]
        X = self.factors.copy()
        if not include_mom:
            X = X.drop(columns="MOM")
        mask = y.notna()
        y = y[mask]
        X = X.loc[mask]
        X = sm.add_constant(X)
        res = sm.OLS(y, X).fit()
        res.t_hac = newey_west_tvals(res)
        return res

    # ------------------------- #
    # Task 4 — dummy equivalence#
    # ------------------------- #
    def dummy_regression(self, signal: str = "Z12") -> sm.RegressionResults:
        """
        Equivalent dummy regression of individual industry returns on decile
        dummies derived from signal (without a constant).
        """
        # Next-month returns, stacked
        df_next = self.industries.shift(-1)
        y = df_next.stack(dropna=False)

        # Corresponding decile assignment
        dec = self.deciles[signal]
        dec_ser = dec.stack(dropna=False)

        # Align and drop missing
        full = pd.concat([y, dec_ser], axis=1, keys=["ret", "dec"]).dropna()
        y_aligned = full["ret"]
        dec_aligned = full["dec"].astype("Int64")

        # Build dummy matrix without intercept
        dummies = pd.get_dummies(dec_aligned, prefix=f"{signal}D").astype(float)

        # Fit OLS, preserving names
        res = sm.OLS(y_aligned, dummies).fit()
        return res

    def check_univariate_dummy(self, formation: str = "Z12") -> pd.DataFrame:
        """
        Verify that univariate dummy regression coefficients equal the decile-mean returns.

        Returns a DataFrame with columns:
        - coeff: γ̂_j from the dummy regression (no intercept)
        - mean_return: average next-month return in decile j
        """
        # 1) Run the univariate dummy regression
        res = self.dummy_regression(formation)
        coeffs = res.params  # Series indexed by dummy names like "Z12D_1",…

        # 2) Grab the decile means (from the bar-chart data)
        means = self.decile_mean_returns[formation]

        # 3) Build a comparison table
        data = {
            'coeff': [coeffs.get(f"{formation}D_{j}", np.nan) for j in range(1, 11)],
            'mean_return': [means.get(j, np.nan) for j in range(1, 11)],
        }
        df = pd.DataFrame(data, index=range(1, 11))
        df.index.name = 'decile'
        return df

    # ------------------------- #
    # Task 5/6 — other signals  #
    # ------------------------- #
    def multivariate_dummy_regression(self) -> sm.RegressionResults:
        """
        Regress next-month individual returns on the full 30 dummy variables
        (Z12, Z6, Ret1 x 10 deciles each).
        """
        # Stack only non-missing returns
        y = self.industries.shift(-1).stack(future_stack=True)

        # Build dummy matrix
        dmatrices = [
            pd.get_dummies(
                self.deciles[sig].stack(future_stack=True), prefix=f"{sig}D"
            ).astype(float)
            for sig in ("Z12", "Z6", "Ret1")
        ]
        X = pd.concat(dmatrices, axis=1)

        # Align Y and X, dropping any rows with missing data
        full = pd.concat([y, X], axis=1).dropna()
        y = full.iloc[:, 0]
        X = full.iloc[:, 1:]
        # Fit with pandas objects to preserve column names for exog_names
        res = sm.OLS(y, X).fit()
        return res

    # ------------------------- #
    # Task 7 — refined strategy #
    # ------------------------- #
    def refined_strategy_returns(self, q: float = 0.10) -> pd.Series:
        """
        Build long-short returns using the expected-return forecasts from the
        full 30-dummy regression.

        Parameters
        ----------
        q :
            Fraction of assets to allocate to each side (e.g. 0.10 = decile).

        Returns
        -------
        Series of long-short excess returns per month.
        """
        res = self.multivariate_dummy_regression()
        # actual next-month returns for alignment
        df_next = self.industries.shift(-1)
        betas = pd.Series(res.params, index=res.model.exog_names)

        # Expected return per asset & month
        preds = []
        for sig, prefix in zip(("Z12", "Z6", "Ret1"), ("Z12D", "Z6D", "Ret1D")):
            # Build dummies only on factor-aligned sample
            dec = self.deciles[sig].loc[df_next.index]
            dser = dec.stack().astype("Int64")
            d = pd.get_dummies(dser, prefix=prefix).astype(float)
            # Align to estimated betas, fill columns that didn’t appear with zero
            d = d.reindex(columns=betas.index, fill_value=0)
            preds.append(d.multiply(betas, axis=1).sum(axis=1))

        exp_ret = sum(preds).unstack()  # T × N

        # Each month: rank expected return, go long top q%, short bottom q%
        ls = []
        for t, row in exp_ret.iterrows():
            rnk = row.rank(method="first")
            n = rnk.count()
            if n == 0:
                ls.append(np.nan)
                continue
            long = row[rnk > n * (1 - q)]
            short = row[rnk <= n * q]
            if long.empty or short.empty:
                ls.append(np.nan)
                continue
            # retrieve next-month actual returns
            ret_row = df_next.loc[t]
            ls_ret = ret_row[long.index].mean() - ret_row[short.index].mean()
            ls.append(ls_ret)

        return pd.Series(ls, index=exp_ret.index, name="RefinedLS")

    # --------------------------------------------------------------------- #
    # Internal helper methods                                               #
    # --------------------------------------------------------------------- #
    def _load_and_prepare_data(self) -> None:
        """Load all CSV files and perform cleaning / alignment."""
        ff5 = (
            pd.read_csv(self.root_dir / "FF5.csv", index_col=0)
            .rename_axis("DATE")
            .rename(columns=lambda c: c.strip())
        )
        mom = (
            pd.read_csv(self.root_dir / "FF_MOM.csv", index_col=0)
            .rename_axis("DATE")
            .rename(columns=lambda c: c.strip())
        )
        ind = (
            pd.read_csv(self.root_dir / "48_Industry_Portfolios.CSV", index_col=0)
            .rename_axis("DATE")
            .rename(columns=lambda c: c.strip())
        )

        # Replace -99.99 missing codes with NaN and scale %
        ind.replace(-99.99, np.nan, inplace=True)
        ind /= 100.0
        ff5.iloc[:, :] /= 100.0
        mom /= 100.0


        # Build datetime index (YYYYMM → Timestamp at month end)
        to_dt = lambda x: pd.to_datetime(str(int(x)) + "01") + pd.offsets.MonthEnd(0)
        ff5.index = ff5.index.map(to_dt)
        mom.index = mom.index.map(to_dt)
        ind.index = ind.index.map(to_dt)
        # Preserve full industry history (raw returns) before aligning to factors
        self.industries_full = ind.copy()

        # Merge factors (add MOM, compute excess market if needed)
        ff5["MOM"] = mom["Mom"]
        self.factors = ff5.drop(columns="RF").dropna()

        # Align on common dates
        ind = ind.loc[self.factors.index]
        # Extract risk-free rates
        rf = ff5.loc[ind.index, "RF"].values.reshape(-1, 1)
        # Choose excess or absolute returns per user setting
        if self.use_excess:
            self.industries = ind.sub(rf, axis=0)
        else:
            self.industries = ind.copy()
        # Store risk-free series
        self.rf = ff5.loc[ind.index, "RF"]
        # (industries_full already set above before aligning to factors)

    def _build_all_signals(self) -> None:
        """Pre-compute Z12, Z6, and Ret1 signals, their deciles, and LS returns."""
        r = self.industries_full  # shorthand

        # cumulative log returns skipping month t‑1
        def cum_log_ret(df: pd.DataFrame, M: int) -> pd.DataFrame:
            log1p = np.log1p(df)
            return log1p.rolling(M).sum().shift(1) - log1p.shift(1)

        z12 = cum_log_ret(r, 12).rename(columns=lambda c: f"{c}")
        z6 = cum_log_ret(r, 6)
        ret1 = r.shift(1)

        self.signals = {"Z12": z12, "Z6": z6, "Ret1": ret1}

        # decile ranks (1 = worst, 10 = best)
        self.deciles: dict[str, pd.DataFrame] = {
            k: v.rank(axis=1, pct=True, method="first").apply(
                lambda s: np.ceil(s * 10), axis=1
            )
            for k, v in self.signals.items()
        }

        # decile mean next-month returns (avg cross-sectional per decile)
        self.decile_mean_returns = {}
        df_next = self.industries.shift(-1)
        for k, dec in self.deciles.items():
            # compute monthly mean returns for each decile group
            decile_means = pd.DataFrame(
                {d: df_next.where(dec == d).mean(axis=1) for d in range(1, 11)},
                index=df_next.index,
            )
            # average over time for each decile
            self.decile_mean_returns[k] = decile_means.mean()

        # long–short excess returns (full-history, long D10, short D1)
        self.long_short_returns = {}
        nxt_full = self.industries_full.shift(-1)
        for k, dec in self.deciles.items():
            long = nxt_full.where(dec == 10).mean(axis=1)
            short = nxt_full.where(dec == 1).mean(axis=1)
            self.long_short_returns[k] = long - short

        # long–short returns restricted to factor sample (for regressions)
        self.long_short_returns_factor = {}
        for k, dec in self.deciles.items():
            nxt = self.industries.shift(-1)          # factor‑aligned panel
            ls = (
                nxt.where(dec == 10).mean(axis=1)
                - nxt.where(dec == 1).mean(axis=1)
            )
            self.long_short_returns_factor[k] = ls

    def export_data(self) -> dict[str, pd.DataFrame]:
        """
        Export all intermediate DataFrames for inspection.

        Returns
        -------
        A dictionary containing:
        - 'factors': Fama-French factor DataFrame
        - 'industries_full': full-industry returns panel (DataFrame)
        - 'signals': dict of signal DataFrames ('Z12', 'Z6', 'Ret1')
        - 'deciles': dict of decile-rank DataFrames ('Z12', 'Z6', 'Ret1')
        - 'dummies': dict of dummy matrices for each signal ('Z12', 'Z6', 'Ret1')
        """
        data: dict[str, pd.DataFrame] = {}
        # Core data
        data['factors'] = self.factors.copy()
        data['industries_full'] = self.industries_full.copy()
        # Signals and deciles
        data['signals'] = {k: df.copy() for k, df in self.signals.items()}
        data['deciles'] = {k: df.copy() for k, df in self.deciles.items()}
        # Dummy matrices
        dummies: dict[str, pd.DataFrame] = {}
        for k in ('Z12', 'Z6', 'Ret1'):
            ser = self.deciles[k].stack().astype("Int64")
            dummies[k] = pd.get_dummies(ser, prefix=f"{k}D").astype(float)
        data['dummies'] = dummies
        return data

    # --------------------------------------------------------------------- #
    # Script entry point                                                    #
    # --------------------------------------------------------------------- #
    def run_all(self) -> None:  # pragma: no cover
        """
        Convenience driver that executes every step and prints / plots key
        outputs. Intended for ad‑hoc sanity checks, **not** for production.
        """
        print("=== Performance table (12‑month momentum) ===")
        print(self.performance_table("Z12").round(2))

        print("\n=== Constant‑α test (HAC t‑stat) ===")
        res = self.constant_alpha_test("Z12")
        print(
            f"α̂ = {res.params.iloc[0]:.4%}, t‑HAC = {res.t_hac.iloc[0]:.2f}, "
            f"p‑value = {res.pvalues.iloc[0]:.4f}"
        )

        print("\n=== Univariate decile dummy check ===")
        print(self.check_univariate_dummy("Z12").round(4).to_string())

        print("\n=== FF5 regression ===")
        res = self.fama_french_regression("Z12", include_mom=False)
        print(res.summary())

        print("\n=== FF5 + MOM regression ===")
        res_mom = self.fama_french_regression("Z12", include_mom=True)
        print(res_mom.summary())

        print("\n=== 30-dummy regression strategy (in-sample) ===")
        res30 = self.multivariate_dummy_regression()
        ls_ref = self.refined_strategy_returns()
        print(f"Adj-R² = {res30.rsquared_adj:.4f}")
        print(f"Sharpe = {annualised_sharpe_ratio(ls_ref):.2f}")

        # ---------- numeric-rank strategy ----------
        print("\n=== Numeric-rank strategy (in-sample) ===")
        ls_rank, res_rank = self.numeric_rank_strategy()
        print(f"Adj-R² = {res_rank.rsquared_adj:.4f}")
        print(f"Sharpe = {annualised_sharpe_ratio(ls_rank):.2f}")

        # ---------- regression-tree strategy ----------
        print("\n=== Regression-tree strategy (in-sample) ===")
        ls_tree, search_tree = self.regression_tree_strategy()
        print(f"CV R² = {search_tree.best_score_:.4f}")
        print(f"Sharpe = {annualised_sharpe_ratio(ls_tree):.2f}")

        # ---------- FF regressions ----------
        # Optional decile monotonicity plots for all signals
        for formation in ("Z12", "Z6", "Ret1"):
            self.decile_monotonicity_plot(formation)
            plt.show()
        # ---------- OOS strategies (post-2000) ----------
        print("\n=== OOS Dummy strategy ===")
        ls_do, r2_do = self.oos_dummy_strategy()
        print(f"OOS R² = {r2_do:.4f}, Sharpe = {annualised_sharpe_ratio(ls_do):.2f}")

        print("\n=== OOS Numeric-rank strategy ===")
        ls_ro, r2_ro = self.oos_numeric_rank_strategy()
        print(f"OOS R² = {r2_ro:.4f}, Sharpe = {annualised_sharpe_ratio(ls_ro):.2f}")

        print("\n=== OOS Regression-tree strategy ===")
        ls_to, r2_to = self.oos_regression_tree_strategy()
        print(f"OOS R² = {r2_to:.4f}, Sharpe = {annualised_sharpe_ratio(ls_to):.2f}")