"""
AI Finance Guard — pipeline zgodny z notebookiem Fabric (pandas zamiast Spark).
Wyniki zapisywane do katalogu out/ (odpowiednik Files/afg/out).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def last_day_ts(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series)
    return t.dt.to_period("M").dt.to_timestamp("M")


def load_feature_train(
    feature_path: str, train_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(feature_path, low_memory=False)
    train = pd.read_csv(train_path, low_memory=False)
    feat["ProductID"] = feat["ProductID"].astype(str)
    train["ProductID"] = train["ProductID"].astype(str)
    feat["EOM"] = last_day_ts(feat["EOM"])
    train["EOM"] = last_day_ts(train["EOM"])
    if feat["KPI_OrdersIn_Qty"].dtype == object:
        feat["KPI_OrdersIn_Qty"] = to_num(feat["KPI_OrdersIn_Qty"])
    feat["KPI_OrdersIn_Qty"] = pd.to_numeric(feat["KPI_OrdersIn_Qty"], errors="coerce")
    return feat, train


def validate_contract(feat2: pd.DataFrame, train2: pd.DataFrame) -> dict[str, Any]:
    dup_feat = int((feat2.groupby(["ProductID", "EOM"]).size() > 1).sum())
    dup_train = int((train2.groupby(["ProductID", "EOM"]).size() > 1).sum())
    k_feat = feat2[["ProductID", "EOM"]].drop_duplicates()
    k_train = train2[["ProductID", "EOM"]].drop_duplicates()
    only_in_feat = len(
        k_feat.merge(
            k_train, on=["ProductID", "EOM"], how="left", indicator=True
        ).query('_merge == "left_only"')
    )
    only_in_train = len(
        k_train.merge(
            k_feat, on=["ProductID", "EOM"], how="left", indicator=True
        ).query('_merge == "left_only"')
    )
    eom_min = feat2["EOM"].min()
    eom_max = feat2["EOM"].max()
    n_sku = feat2["ProductID"].nunique()
    return {
        "dup_feat": dup_feat,
        "dup_train": dup_train,
        "only_in_feat": only_in_feat,
        "only_in_train": only_in_train,
        "eom_min": eom_min,
        "eom_max": eom_max,
        "n_sku": n_sku,
    }


def build_horizons(feat2: pd.DataFrame, forecast_h: int = 6) -> pd.DataFrame:
    feat2 = feat2.sort_values(["ProductID", "EOM"]).copy()
    parts = []
    for h in range(1, forecast_h + 1):
        d = feat2.copy()
        d["y"] = d.groupby("ProductID")["KPI_OrdersIn_Qty"].shift(-h)
        d["Forecast_EOM"] = d["EOM"].apply(
            lambda e: (pd.Timestamp(e) + pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0)
        )
        d["FC_Year"] = d["Forecast_EOM"].dt.year
        d["FC_Month"] = d["Forecast_EOM"].dt.month
        d["FC_Quarter"] = d["Forecast_EOM"].dt.quarter
        d["Horizon_M"] = h
        d["Target"] = "OrdersIn"
        d = d[d["y"].notna()]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def select_cutoffs(feat2: pd.DataFrame, forecast_h: int, n_cutoffs: int) -> list:
    max_eom = feat2["EOM"].max()
    last_cutoff = (
        pd.Timestamp(max_eom) - pd.DateOffset(months=forecast_h)
    ) + pd.offsets.MonthEnd(0)
    last_cutoff = last_cutoff.date()
    eligible = (
        feat2["EOM"]
        .drop_duplicates()
        .sort_values()
    )
    eligible = eligible[eligible <= pd.Timestamp(last_cutoff)]
    eligible_list = [pd.Timestamp(x).date() for x in eligible.tolist()]
    return eligible_list[-n_cutoffs:] if eligible_list else []


def prepare_h_pd(h_all: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    h_pd = h_all.copy()
    h_pd["EOM"] = pd.to_datetime(h_pd["EOM"])
    h_pd["Forecast_EOM"] = pd.to_datetime(h_pd["Forecast_EOM"])
    h_pd["ProductID"] = h_pd["ProductID"].astype(str)
    EXCLUDE = {"Target", "Horizon_M", "y", "EOM", "Forecast_EOM", "RokMiesiac"}
    feature_cols = [c for c in h_pd.columns if c not in EXCLUDE]
    base_cat = ["ProductID", "HierarchyLevel", "ProductLine", "Family"]
    for c in feature_cols:
        if c in base_cat:
            h_pd[c] = (
                h_pd[c].astype(str).replace({"nan": "UNKNOWN"}).fillna("UNKNOWN")
            )
        else:
            if h_pd[c].dtype == "object" or str(h_pd[c].dtype) == "string":
                h_pd[c] = to_num(h_pd[c])
            h_pd[c] = pd.to_numeric(h_pd[c], errors="coerce")
    cat_cols = [c for c in base_cat if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    return h_pd, feature_cols, cat_idx


def fit_quantile_with_es(
    train_df: pd.DataFrame,
    alpha: float,
    feature_cols: list[str],
    cat_idx: list[int],
    *,
    cb_iterations: int,
    cb_verbose: int,
    cb_depth: int,
    cb_lr: float,
    seed: int,
    val_months: int,
    early_stopping_rounds: int,
    min_val_rows: int,
) -> CatBoostRegressor:
    max_eom = train_df["EOM"].max()
    val_cut = (max_eom - pd.DateOffset(months=val_months)) + pd.offsets.MonthEnd(0)
    tr = train_df[train_df["EOM"] <= val_cut]
    va = train_df[train_df["EOM"] > val_cut]
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=cb_iterations,
        learning_rate=cb_lr,
        depth=cb_depth,
        random_seed=seed,
        verbose=cb_verbose,
        allow_writing_files=False,
    )
    tr_pool = Pool(tr[feature_cols], tr["y"].values, cat_features=cat_idx)
    if len(va) >= min_val_rows and len(tr) > 0:
        va_pool = Pool(va[feature_cols], va["y"].values, cat_features=cat_idx)
        model.fit(
            tr_pool,
            eval_set=va_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model.fit(tr_pool)
    return model


def pinball(y: np.ndarray, yhat: np.ndarray, tau: float) -> np.ndarray:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.where(y >= yhat, tau * (y - yhat), (1 - tau) * (yhat - y))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    denom = np.sum(y)
    return float("nan") if denom <= 0 else float(np.sum(np.abs(y - yhat)) / denom)


def rolling_backtest(
    h_pd: pd.DataFrame,
    feature_cols: list[str],
    cat_idx: list[int],
    cutoffs: list,
    forecast_h: int,
    *,
    cb_iterations: int,
    cb_verbose: int,
    cb_depth: int,
    cb_lr: float,
    seed: int,
    val_months: int,
    early_stopping_rounds: int,
    min_val_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        for h in range(1, forecast_h + 1):
            df_h = h_pd[h_pd["Horizon_M"] == h].copy()
            train_limit = (cutoff - pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0)
            train_df = df_h[df_h["EOM"] <= train_limit]
            test_df = df_h[df_h["EOM"] == cutoff]
            if len(train_df) == 0 or len(test_df) == 0:
                continue
            m50 = fit_quantile_with_es(
                train_df,
                0.5,
                feature_cols,
                cat_idx,
                cb_iterations=cb_iterations,
                cb_verbose=cb_verbose,
                cb_depth=cb_depth,
                cb_lr=cb_lr,
                seed=seed,
                val_months=val_months,
                early_stopping_rounds=early_stopping_rounds,
                min_val_rows=min_val_rows,
            )
            m90 = fit_quantile_with_es(
                train_df,
                0.9,
                feature_cols,
                cat_idx,
                cb_iterations=cb_iterations,
                cb_verbose=cb_verbose,
                cb_depth=cb_depth,
                cb_lr=cb_lr,
                seed=seed,
                val_months=val_months,
                early_stopping_rounds=early_stopping_rounds,
                min_val_rows=min_val_rows,
            )
            p50 = np.clip(m50.predict(test_df[feature_cols]), 0, None)
            p90 = np.clip(m90.predict(test_df[feature_cols]), 0, None)
            for i in range(len(test_df)):
                rows.append(
                    {
                        "Run_Cutoff_EOM": cutoff.date(),
                        "Forecast_EOM": test_df["Forecast_EOM"].iloc[i].date(),
                        "Horizon_M": int(h),
                        "ProductID": test_df["ProductID"].iloc[i],
                        "Actual_OrdersIn": float(test_df["y"].iloc[i]),
                        "Pred_OrdersIn_P50": float(p50[i]),
                        "Pred_OrdersIn_P90": float(p90[i]),
                    }
                )
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, pd.DataFrame()
    bt["AE_P50"] = np.abs(bt["Actual_OrdersIn"] - bt["Pred_OrdersIn_P50"])
    bt["Bias_P50_row"] = bt["Pred_OrdersIn_P50"] - bt["Actual_OrdersIn"]
    bt["Pinball_P90_row"] = pinball(
        bt["Actual_OrdersIn"].values, bt["Pred_OrdersIn_P90"].values, 0.9
    )
    bt["Coverage_P90_row"] = (
        bt["Actual_OrdersIn"] <= bt["Pred_OrdersIn_P90"]
    ).astype(int)
    metrics = []
    for h in sorted(bt["Horizon_M"].unique()):
        g = bt[bt["Horizon_M"] == h]
        metrics.append(
            {
                "Target": "OrdersIn",
                "Horizon_M": int(h),
                "MAE_P50": float(g["AE_P50"].mean()),
                "WAPE_P50": wape(g["Actual_OrdersIn"].values, g["Pred_OrdersIn_P50"].values),
                "Bias_P50": float(g["Bias_P50_row"].mean()),
                "Pinball_P90": float(g["Pinball_P90_row"].mean()),
                "Coverage_P90": float(g["Coverage_P90_row"].mean()),
                "N": int(len(g)),
            }
        )
    metrics_summary = pd.DataFrame(metrics).sort_values("Horizon_M")
    return bt, metrics_summary


def calibrate_p90(bt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = bt.copy()
    bt["Horizon_M"] = pd.to_numeric(bt["Horizon_M"], errors="coerce").astype(int)

    def calibrate_scale(g: pd.DataFrame, target_cov: float = 0.90) -> float:
        a = g["Actual_OrdersIn"].values
        p50 = g["Pred_OrdersIn_P50"].values
        p90 = g["Pred_OrdersIn_P90"].values
        denom = p90 - p50
        s_req = []
        for i in range(len(a)):
            if np.isnan(a[i]) or np.isnan(p50[i]) or np.isnan(p90[i]):
                continue
            if a[i] <= p50[i]:
                s_req.append(0.0)
            elif denom[i] > 1e-9:
                s_req.append((a[i] - p50[i]) / denom[i])
        if len(s_req) == 0:
            return float("nan")
        return float(np.quantile(np.array(s_req), target_cov))

    cal = pd.DataFrame(
        [
            {
                "Horizon_M": int(h),
                "Scale_s": calibrate_scale(bt[bt["Horizon_M"] == h].dropna()),
            }
            for h in sorted(bt["Horizon_M"].unique())
        ]
    ).sort_values("Horizon_M")
    bt = bt.merge(cal, on="Horizon_M", how="left")
    bt["Pred_OrdersIn_P90_cal"] = bt["Pred_OrdersIn_P50"] + bt["Scale_s"] * (
        bt["Pred_OrdersIn_P90"] - bt["Pred_OrdersIn_P50"]
    )
    return bt, cal


FORECAST_H_DEFAULT = 6


def simulate_plan_service(
    demand6: list, fg0: Any, lt_months: Any, forecast_h: int = FORECAST_H_DEFAULT
):
    demand = np.array(demand6, dtype=float)
    demand = np.nan_to_num(demand, nan=0.0)
    demand = np.clip(demand, 0, None)
    fg = max(float(fg0) if pd.notna(fg0) else 0.0, 0.0)
    lt = max(int(np.ceil(lt_months)) if pd.notna(lt_months) else 0, 0)
    releases = np.zeros(forecast_h, dtype=float)
    lost = np.zeros(forecast_h, dtype=float)
    inv = fg
    for t in range(1, forecast_h + 1):
        if t <= lt:
            served = min(demand[t - 1], inv)
            lost[t - 1] = demand[t - 1] - served
            inv = inv - served
        else:
            arrival_req = max(0.0, demand[t - 1] - inv)
            releases[t - lt - 1] = arrival_req
            lost[t - 1] = 0.0
            inv = (inv + arrival_req) - demand[t - 1]
    lost6 = float(lost.sum())
    fill6 = 1.0 - (lost6 / max(float(demand.sum()), 1.0))
    L = min(max(lt, 1), forecast_h)
    lost_lt = float(lost[:L].sum())
    fill_lt = 1.0 - (lost_lt / max(float(demand[:L].sum()), 1.0))
    return releases, lost6, fill6, lost_lt, fill_lt


def service_and_plan_v2(
    feat: pd.DataFrame,
    train: pd.DataFrame,
    bt: pd.DataFrame,
    forecast_h: int = FORECAST_H_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = bt.copy()
    bt["Run_Cutoff_EOM"] = pd.to_datetime(bt["Run_Cutoff_EOM"]).dt.date
    bt["ProductID"] = bt["ProductID"].astype(str)
    bt["Horizon_M"] = pd.to_numeric(bt["Horizon_M"], errors="coerce").astype(int)
    bt["Pred_OrdersIn_P90"] = to_num(bt["Pred_OrdersIn_P90"].astype(str))

    state = feat[
        ["ProductID", "EOM", "OPS_FG_Qty_EOM", "LT_LeadTimeMonths_Used", "LT_PW_Qty"]
    ].copy()
    state["Run_Cutoff_EOM"] = pd.to_datetime(state["EOM"]).dt.date
    state["ProductID"] = state["ProductID"].astype(str)
    state["OPS_FG_Qty_EOM"] = to_num(state["OPS_FG_Qty_EOM"].astype(str))
    state["LT_LeadTimeMonths_Used"] = to_num(
        state["LT_LeadTimeMonths_Used"].astype(str)
    )
    state["LT_PW_Qty"] = to_num(state["LT_PW_Qty"].astype(str))
    state = state.drop(columns=["EOM"])

    train_kpi = train[
        ["ProductID", "EOM", "KPI_LostSales_Qty_LT", "KPI_FillRate_Qty_LT"]
    ].copy()
    train_kpi["Run_Cutoff_EOM"] = pd.to_datetime(train_kpi["EOM"]).dt.date
    train_kpi["ProductID"] = train_kpi["ProductID"].astype(str)
    train_kpi["KPI_LostSales_Qty_LT"] = to_num(
        train_kpi["KPI_LostSales_Qty_LT"].astype(str)
    )
    train_kpi["KPI_FillRate_Qty_LT"] = to_num(
        train_kpi["KPI_FillRate_Qty_LT"].astype(str)
    )
    train_kpi = train_kpi.drop(columns=["EOM"])

    demand = (
        bt.pivot_table(
            index=["Run_Cutoff_EOM", "ProductID"],
            columns="Horizon_M",
            values="Pred_OrdersIn_P90",
            aggfunc="mean",
        )
        .reindex(columns=[1, 2, 3, 4, 5, 6])
        .reset_index()
    )

    df = demand.merge(state, on=["Run_Cutoff_EOM", "ProductID"], how="left").merge(
        train_kpi, on=["Run_Cutoff_EOM", "ProductID"], how="left"
    )

    service_rows, plan_rows = [], []
    for _, r in df.iterrows():
        demand6 = [float(r.get(i, 0.0) or 0) for i in [1, 2, 3, 4, 5, 6]]
        releases, lost6, fill6, lost_lt, fill_lt = simulate_plan_service(
            demand6, r["OPS_FG_Qty_EOM"], r["LT_LeadTimeMonths_Used"], forecast_h
        )
        service_rows.append(
            {
                "Run_Cutoff_EOM": r["Run_Cutoff_EOM"],
                "ProductID": r["ProductID"],
                "LostSales_LT_Plan": lost_lt,
                "FillRate_LT_Plan": fill_lt,
                "KPI_LostSales_Qty_LT": r["KPI_LostSales_Qty_LT"],
                "KPI_FillRate_Qty_LT": r["KPI_FillRate_Qty_LT"],
                "Diff_LostSales_LT": (lost_lt - r["KPI_LostSales_Qty_LT"]),
                "Diff_FillRate_LT": (fill_lt - r["KPI_FillRate_Qty_LT"]),
                "LostSales_6M_Plan": lost6,
                "FillRate_6M_Plan": fill6,
            }
        )
        base = r["LT_PW_Qty"]
        if pd.isna(base) or base <= 0:
            base = float(np.mean(demand6)) if np.mean(demand6) > 0 else 1.0
        cutoff = pd.Timestamp(r["Run_Cutoff_EOM"])
        for m in range(1, 7):
            fc_eom = (cutoff + pd.DateOffset(months=m)) + pd.offsets.MonthEnd(0)
            rec = float(releases[m - 1])
            label = (
                "STOP" if rec <= 0.10 * base else ("SLOW" if rec <= 0.90 * base else "SPEED")
            )
            plan_rows.append(
                {
                    "Run_Cutoff_EOM": r["Run_Cutoff_EOM"],
                    "ProductID": r["ProductID"],
                    "Forecast_EOM": fc_eom.date(),
                    "RecommendedProductionQty": rec,
                    "ActionLabel": label,
                    "LostSales_6M": lost6,
                    "FillRate_6M": fill6,
                }
            )
    return pd.DataFrame(service_rows), pd.DataFrame(plan_rows)


def service_metrics_cal(
    demand6: list, fg0: Any, lt_months: Any, forecast_h: int = FORECAST_H_DEFAULT
):
    demand = np.array(demand6, dtype=float)
    demand = np.nan_to_num(demand, nan=0.0)
    demand = np.clip(demand, 0, None)
    fg = max(float(fg0) if pd.notna(fg0) else 0.0, 0.0)
    lt = max(int(np.ceil(lt_months)) if pd.notna(lt_months) else 0, 0)
    inv = fg
    lost = np.zeros(forecast_h, dtype=float)
    for t in range(1, forecast_h + 1):
        if t <= lt:
            served = min(demand[t - 1], inv)
            lost[t - 1] = demand[t - 1] - served
            inv = inv - served
        else:
            arrival_req = max(0.0, demand[t - 1] - inv)
            inv = inv + arrival_req - demand[t - 1]
            lost[t - 1] = 0.0
    lost6 = float(lost.sum())
    fill6 = 1.0 - (lost6 / max(float(demand.sum()), 1.0))
    L = min(max(lt, 1), forecast_h)
    lost_lt = float(lost[:L].sum())
    fill_lt = 1.0 - (lost_lt / max(float(demand[:L].sum()), 1.0))
    return lost6, fill6, lost_lt, fill_lt


def service_eval_calibrated(
    feat: pd.DataFrame,
    train: pd.DataFrame,
    bt_cal: pd.DataFrame,
    forecast_h: int = FORECAST_H_DEFAULT,
) -> pd.DataFrame:
    bt = bt_cal.copy()
    bt["Run_Cutoff_EOM"] = pd.to_datetime(bt["Run_Cutoff_EOM"]).dt.date
    bt["ProductID"] = bt["ProductID"].astype(str)
    bt["Horizon_M"] = pd.to_numeric(bt["Horizon_M"], errors="coerce").astype(int)
    bt["Pred_OrdersIn_P90_cal"] = to_num(bt["Pred_OrdersIn_P90_cal"].astype(str))

    state = feat[
        ["ProductID", "EOM", "OPS_FG_Qty_EOM", "LT_LeadTimeMonths_Used"]
    ].copy()
    state["Run_Cutoff_EOM"] = pd.to_datetime(state["EOM"]).dt.date
    state["ProductID"] = state["ProductID"].astype(str)
    state["OPS_FG_Qty_EOM"] = to_num(state["OPS_FG_Qty_EOM"].astype(str))
    state["LT_LeadTimeMonths_Used"] = to_num(
        state["LT_LeadTimeMonths_Used"].astype(str)
    )
    state = state.drop(columns=["EOM"])

    train_kpi = train[
        ["ProductID", "EOM", "KPI_LostSales_Qty_LT", "KPI_FillRate_Qty_LT"]
    ].copy()
    train_kpi["Run_Cutoff_EOM"] = pd.to_datetime(train_kpi["EOM"]).dt.date
    train_kpi["ProductID"] = train_kpi["ProductID"].astype(str)
    train_kpi["KPI_LostSales_Qty_LT"] = to_num(
        train_kpi["KPI_LostSales_Qty_LT"].astype(str)
    )
    train_kpi["KPI_FillRate_Qty_LT"] = to_num(
        train_kpi["KPI_FillRate_Qty_LT"].astype(str)
    )
    train_kpi = train_kpi.drop(columns=["EOM"])

    demand = (
        bt.pivot_table(
            index=["Run_Cutoff_EOM", "ProductID"],
            columns="Horizon_M",
            values="Pred_OrdersIn_P90_cal",
            aggfunc="mean",
        )
        .reindex(columns=[1, 2, 3, 4, 5, 6])
        .reset_index()
    )

    df = demand.merge(state, on=["Run_Cutoff_EOM", "ProductID"], how="left").merge(
        train_kpi, on=["Run_Cutoff_EOM", "ProductID"], how="left"
    )

    rows = []
    for _, r in df.iterrows():
        d6 = [float(r.get(i, 0.0) or 0) for i in [1, 2, 3, 4, 5, 6]]
        lost6, fill6, lost_lt, fill_lt = service_metrics_cal(
            d6, r["OPS_FG_Qty_EOM"], r["LT_LeadTimeMonths_Used"], forecast_h
        )
        rows.append(
            {
                "Run_Cutoff_EOM": r["Run_Cutoff_EOM"],
                "ProductID": r["ProductID"],
                "LostSales_6M_Plan_cal": lost6,
                "FillRate_6M_Plan_cal": fill6,
                "LostSales_LT_Plan_cal": lost_lt,
                "FillRate_LT_Plan_cal": fill_lt,
                "KPI_LostSales_Qty_LT": r["KPI_LostSales_Qty_LT"],
                "KPI_FillRate_Qty_LT": r["KPI_FillRate_Qty_LT"],
                "Diff_LostSales_LT_cal": (lost_lt - r["KPI_LostSales_Qty_LT"]),
                "Diff_FillRate_LT_cal": (fill_lt - r["KPI_FillRate_Qty_LT"]),
            }
        )
    return pd.DataFrame(rows)


def fit_quantile_final(
    train_df: pd.DataFrame,
    alpha: float,
    feat_cols: list[str],
    cat_cols: list[str],
    *,
    cb_iterations: int,
    cb_verbose: int,
    cb_depth: int,
    cb_lr: float,
    seed: int,
    val_months: int,
    early_stopping_rounds: int,
    min_val_rows: int,
) -> CatBoostRegressor:
    cat_idx = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
    return fit_quantile_with_es(
        train_df,
        alpha,
        feat_cols,
        cat_idx,
        cb_iterations=cb_iterations,
        cb_verbose=cb_verbose,
        cb_depth=cb_depth,
        cb_lr=cb_lr,
        seed=seed,
        val_months=val_months,
        early_stopping_rounds=early_stopping_rounds,
        min_val_rows=min_val_rows,
    )


def prepare_feat_for_final(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    feat["EOM"] = pd.to_datetime(feat["EOM"])
    feat["ProductID"] = feat["ProductID"].astype(str)
    base_cat = ["ProductID", "HierarchyLevel", "ProductLine", "Family"]
    for c in base_cat:
        if c in feat.columns:
            feat[c] = feat[c].astype(str).replace({"nan": "UNKNOWN"}).fillna("UNKNOWN")
    for c in feat.columns:
        if c in base_cat or c in ["EOM", "RokMiesiac"]:
            continue
        if feat[c].dtype == "object" or str(feat[c].dtype) == "string":
            feat[c] = to_num(feat[c])
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    return feat


def final_forecast_and_plan(
    feat: pd.DataFrame,
    cal: pd.DataFrame,
    *,
    forecast_h: int,
    cb_iterations: int,
    cb_verbose: int,
    cb_depth: int,
    cb_lr: float,
    seed: int,
    val_months: int,
    early_stopping_rounds: int,
    min_val_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = prepare_feat_for_final(feat)
    ASOF = feat["EOM"].max()
    origin = feat[feat["EOM"] == ASOF].copy()
    cal = cal.copy()
    cal["Horizon_M"] = pd.to_numeric(cal["Horizon_M"], errors="coerce").astype(int)
    cal["Scale_s"] = to_num(cal["Scale_s"].astype(str))

    base_cat = ["ProductID", "HierarchyLevel", "ProductLine", "Family"]
    all_fc = []
    for h in range(1, forecast_h + 1):
        d = feat.sort_values(["ProductID", "EOM"]).copy()
        d["y"] = d.groupby("ProductID")["KPI_OrdersIn_Qty"].shift(-h)
        d = d.dropna(subset=["y"]).copy()
        d["Forecast_EOM"] = (
            d["EOM"] + pd.DateOffset(months=h)
        ) + pd.offsets.MonthEnd(0)
        d["FC_Year"] = d["Forecast_EOM"].dt.year
        d["FC_Month"] = d["Forecast_EOM"].dt.month
        d["FC_Quarter"] = d["Forecast_EOM"].dt.quarter

        o = origin.copy()
        o["Forecast_EOM"] = (o["EOM"] + pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0)
        o["FC_Year"] = o["Forecast_EOM"].dt.year
        o["FC_Month"] = o["Forecast_EOM"].dt.month
        o["FC_Quarter"] = o["Forecast_EOM"].dt.quarter

        drop_cols = {"EOM", "RokMiesiac", "y", "Forecast_EOM"}
        feat_cols = [c for c in d.columns if c not in drop_cols]
        cat_cols = [c for c in base_cat if c in feat_cols]

        m50 = fit_quantile_final(
            d,
            0.5,
            feat_cols,
            cat_cols,
            cb_iterations=cb_iterations,
            cb_verbose=cb_verbose,
            cb_depth=cb_depth,
            cb_lr=cb_lr,
            seed=seed,
            val_months=val_months,
            early_stopping_rounds=early_stopping_rounds,
            min_val_rows=min_val_rows,
        )
        m90 = fit_quantile_final(
            d,
            0.9,
            feat_cols,
            cat_cols,
            cb_iterations=cb_iterations,
            cb_verbose=cb_verbose,
            cb_depth=cb_depth,
            cb_lr=cb_lr,
            seed=seed,
            val_months=val_months,
            early_stopping_rounds=early_stopping_rounds,
            min_val_rows=min_val_rows,
        )

        p50 = np.clip(m50.predict(o[feat_cols]), 0, None)
        p90 = np.clip(m90.predict(o[feat_cols]), 0, None)
        row_cal = cal[cal["Horizon_M"] == h]
        s = (
            float(row_cal["Scale_s"].values[0])
            if len(row_cal)
            else 1.0
        )
        if np.isnan(s):
            s = 1.0
        p90_cal = p50 + s * (p90 - p50)

        all_fc.append(
            pd.DataFrame(
                {
                    "Run_Cutoff_EOM": ASOF.date(),
                    "ProductID": o["ProductID"].values,
                    "Forecast_EOM": o["Forecast_EOM"].dt.date.values,
                    "Horizon_M": h,
                    "Pred_OrdersIn_P50": p50,
                    "Pred_OrdersIn_P90": p90,
                    "Scale_s": s,
                    "Pred_OrdersIn_P90_cal": p90_cal,
                }
            )
        )

    forecast_final = pd.concat(all_fc, ignore_index=True)

    st = origin.set_index("ProductID")[
        ["OPS_FG_Qty_EOM", "LT_LeadTimeMonths_Used", "LT_PW_Qty"]
    ].copy()
    st["OPS_FG_Qty_EOM"] = to_num(st["OPS_FG_Qty_EOM"].astype(str))
    st["LT_LeadTimeMonths_Used"] = to_num(st["LT_LeadTimeMonths_Used"].astype(str))
    st["LT_PW_Qty"] = to_num(st["LT_PW_Qty"].astype(str))

    demand = (
        forecast_final.pivot_table(
            index="ProductID",
            columns="Horizon_M",
            values="Pred_OrdersIn_P90_cal",
            aggfunc="mean",
        )
        .reindex(columns=[1, 2, 3, 4, 5, 6])
    )

    def simulate_releases(d6, fg0, lt_months):
        d6 = np.array(d6, dtype=float)
        d6 = np.nan_to_num(d6, nan=0.0)
        d6 = np.clip(d6, 0, None)
        fg = max(float(fg0) if pd.notna(fg0) else 0.0, 0.0)
        lt = max(int(np.ceil(lt_months)) if pd.notna(lt_months) else 0, 0)
        inv = fg
        releases = np.zeros(forecast_h, dtype=float)
        lost = np.zeros(forecast_h, dtype=float)
        for t in range(1, forecast_h + 1):
            if t <= lt:
                served = min(d6[t - 1], inv)
                lost[t - 1] = d6[t - 1] - served
                inv = inv - served
            else:
                arrival_req = max(0.0, d6[t - 1] - inv)
                releases[t - lt - 1] = arrival_req
                inv = inv + arrival_req - d6[t - 1]
                lost[t - 1] = 0.0
        lost6 = float(lost.sum())
        fill6 = 1.0 - (lost6 / max(float(d6.sum()), 1.0))
        return releases, lost6, fill6

    plan_rows = []
    for sku in demand.index:
        if sku not in st.index:
            continue
        d6 = demand.loc[sku].values.tolist()
        fg0 = st.loc[sku, "OPS_FG_Qty_EOM"]
        lt = st.loc[sku, "LT_LeadTimeMonths_Used"]
        base = st.loc[sku, "LT_PW_Qty"]
        if pd.isna(base) or base <= 0:
            base = float(np.mean(d6)) if np.mean(d6) > 0 else 1.0
        rel, lost6, fill6 = simulate_releases(d6, fg0, lt)
        for m in range(1, 7):
            fc_eom = (pd.Timestamp(ASOF) + pd.DateOffset(months=m)) + pd.offsets.MonthEnd(
                0
            )
            rec = float(rel[m - 1])
            label = (
                "STOP" if rec <= 0.10 * base else ("SLOW" if rec <= 0.90 * base else "SPEED")
            )
            plan_rows.append(
                {
                    "Run_Cutoff_EOM": ASOF.date(),
                    "ProductID": sku,
                    "Forecast_EOM": fc_eom.date(),
                    "RecommendedProductionQty": rec,
                    "ActionLabel": label,
                    "LT_LeadTimeMonths_Used": float(lt) if pd.notna(lt) else np.nan,
                    "LostSales_6M": lost6,
                    "FillRate_6M": fill6,
                }
            )

    plan_final = pd.DataFrame(plan_rows)
    return forecast_final, plan_final


@dataclass
class PipelineConfig:
    feature_path: str
    train_path: str
    out_dir: str = "out"
    forecast_h: int = 6
    n_cutoffs: int = 3
    include_final_forecast: bool = True
    cb_iterations: int = 800
    cb_verbose: int = 0
    cb_depth: int = 8
    cb_lr: float = 0.05
    seed: int = 42
    val_months: int = 6
    early_stopping_rounds: int = 80
    min_val_rows: int = 30


@dataclass
class PipelineResult:
    validation: dict
    h_sample: pd.DataFrame
    horizon_counts: pd.DataFrame
    cutoffs: list
    bt: pd.DataFrame
    metrics_summary: pd.DataFrame
    bt_cal: pd.DataFrame
    cal: pd.DataFrame
    service_eval_v2: pd.DataFrame
    plan_bt_v2: pd.DataFrame
    svc_cal: pd.DataFrame
    forecast_final: pd.DataFrame
    plan_final: pd.DataFrame
    asof: Any = None
    paths_written: dict[str, str] = field(default_factory=dict)


def ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    ensure_out(cfg.out_dir)
    feat, train = load_feature_train(cfg.feature_path, cfg.train_path)
    feat2 = feat.copy()
    train2 = train.copy()

    validation = validate_contract(feat2, train2)
    h_all = build_horizons(feat2, cfg.forecast_h)
    horizon_counts = (
        h_all.groupby("Horizon_M").size().reset_index(name="n_rows")
    )
    h_sample = h_all[h_all["Horizon_M"] == 1][["EOM", "Forecast_EOM"]].head(5)

    cutoffs = select_cutoffs(feat2, cfg.forecast_h, cfg.n_cutoffs)
    h_pd, feature_cols, cat_idx = prepare_h_pd(h_all)

    bt, metrics_summary = rolling_backtest(
        h_pd,
        feature_cols,
        cat_idx,
        cutoffs,
        cfg.forecast_h,
        cb_iterations=cfg.cb_iterations,
        cb_verbose=cfg.cb_verbose,
        cb_depth=cfg.cb_depth,
        cb_lr=cfg.cb_lr,
        seed=cfg.seed,
        val_months=cfg.val_months,
        early_stopping_rounds=cfg.early_stopping_rounds,
        min_val_rows=cfg.min_val_rows,
    )

    paths = {}
    if not bt.empty:
        p = os.path.join(cfg.out_dir, "AFG_ML_EVAL_BACKTEST.csv")
        save_csv(bt, p)
        paths["backtest"] = p
    if not metrics_summary.empty:
        p = os.path.join(cfg.out_dir, "AFG_ML_METRICS_SUMMARY.csv")
        save_csv(metrics_summary, p)
        paths["metrics"] = p

    bt_cal = pd.DataFrame()
    cal = pd.DataFrame()
    service_eval_v2 = pd.DataFrame()
    plan_bt_v2 = pd.DataFrame()
    svc_cal = pd.DataFrame()
    forecast_final = pd.DataFrame()
    plan_final = pd.DataFrame()
    asof = feat2["EOM"].max()

    if not bt.empty:
        bt_cal, cal = calibrate_p90(bt)
        p = os.path.join(cfg.out_dir, "P90_Calibration_Scales.csv")
        save_csv(cal, p)
        paths["calibration"] = p
        p = os.path.join(cfg.out_dir, "AFG_ML_EVAL_BACKTEST_CAL.csv")
        save_csv(bt_cal, p)
        paths["backtest_cal"] = p

        service_eval_v2, plan_bt_v2 = service_and_plan_v2(
            feat, train, bt, cfg.forecast_h
        )
        save_csv(
            service_eval_v2,
            os.path.join(cfg.out_dir, "AFG_ML_SERVICE_EVAL_v2.csv"),
        )
        save_csv(
            plan_bt_v2,
            os.path.join(cfg.out_dir, "Plan_Production_Backtest_v2.csv"),
        )
        paths["service_v2"] = os.path.join(cfg.out_dir, "AFG_ML_SERVICE_EVAL_v2.csv")
        paths["plan_v2"] = os.path.join(cfg.out_dir, "Plan_Production_Backtest_v2.csv")

        svc_cal = service_eval_calibrated(feat, train, bt_cal, cfg.forecast_h)
        save_csv(svc_cal, os.path.join(cfg.out_dir, "AFG_ML_SERVICE_EVAL_cal.csv"))
        paths["service_cal"] = os.path.join(cfg.out_dir, "AFG_ML_SERVICE_EVAL_cal.csv")

        if cfg.include_final_forecast:
            forecast_final, plan_final = final_forecast_and_plan(
                feat,
                cal,
                forecast_h=cfg.forecast_h,
                cb_iterations=cfg.cb_iterations,
                cb_verbose=cfg.cb_verbose,
                cb_depth=cfg.cb_depth,
                cb_lr=cfg.cb_lr,
                seed=cfg.seed,
                val_months=cfg.val_months,
                early_stopping_rounds=cfg.early_stopping_rounds,
                min_val_rows=cfg.min_val_rows,
            )
            p_fc = os.path.join(cfg.out_dir, "Forecast_Demand_FINAL.csv")
            p_pl = os.path.join(cfg.out_dir, "Plan_Production_FINAL.csv")
            save_csv(forecast_final, p_fc)
            save_csv(plan_final, p_pl)
            paths["forecast_final"] = p_fc
            paths["plan_final"] = p_pl
            asof = (
                forecast_final["Run_Cutoff_EOM"].iloc[0]
                if len(forecast_final)
                else asof
            )

    return PipelineResult(
        validation=validation,
        h_sample=h_sample,
        horizon_counts=horizon_counts,
        cutoffs=cutoffs,
        bt=bt,
        metrics_summary=metrics_summary,
        bt_cal=bt_cal,
        cal=cal,
        service_eval_v2=service_eval_v2,
        plan_bt_v2=plan_bt_v2,
        svc_cal=svc_cal,
        forecast_final=forecast_final,
        plan_final=plan_final,
        asof=asof,
        paths_written=paths,
    )
