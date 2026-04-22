"""
AI Finance Guard — dashboard Streamlit (prognozy CatBoost, serwis, plan produkcji).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from afg.pipeline import PipelineConfig, run_pipeline

BASE = Path(__file__).resolve().parent


def css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .block-container { padding-top: 1.2rem; max-width: 1200px; }
        h1 { letter-spacing: -0.02em; font-weight: 700 !important; }
        .hero {
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.12) 0%, rgba(15, 23, 42, 0.95) 45%, rgba(245, 158, 11, 0.08) 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0 0 0.35rem 0; font-size: 1.75rem; color: #f8fafc; }
        .hero p { margin: 0; color: #94a3b8; font-size: 0.95rem; }
        .badge {
            display: inline-block;
            background: rgba(20, 184, 166, 0.2);
            color: #5eead4;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(71, 85, 105, 0.5);
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


TAB_INTRO = {
    "a": """
**Walidacja i horyzonty** — tu sprawdzasz, czy dane wejściowe są spójne, oraz jak zbudowane są cele prognozy na kolejne miesiące.

- Metryki u góry strony: brak duplikatów klucza (ProductID + EOM), zgodność zbiorów feature vs train oraz ostatni dostępny miesiąc (ASOF).
- Dla każdego horyzontu *h* = 1…6 model przewiduje **KPI_OrdersIn_Qty** za *h* miesięcy względem danego EOM; data docelowa to zawsze **koniec miesiąca** (*Forecast_EOM*).
- Wykres słupkowy pokazuje, ile wierszy treningowych zostaje po usunięciu obserwacji bez przyszłej wartości (brak *y* na końcu historii SKU).
""",
    "b": """
**Backtest i metryki** — ocena jakości modelu CatBoost w ustawieniu *rolling*: dla wybranych dat odcięcia trenujesz na przeszłości i porównujesz prognozy z faktycznymi zamówieniami w punkcie testowym.

- **P50 (mediana)** — typowa prognoza; **MAE** i **WAPE** mówią o błędzie absolutnym i względnym, **Bias** czy model nie systematycznie przewyższa lub zaniża popyt.
- **P90** — górny kwantyl; **Coverage P90** to udział przypadków, gdzie rzeczywistość ≤ prognoza P90 (dla dobrze skalibrowanego modelu blisko **0,9**).
- Wykres *Actual vs Pred* pokazuje rozrzut punktów wokół linii idealnej (fragment próbki).
""",
    "c": """
**Kalibracja P90** — dopasowanie górnego kwantyla tak, aby w próbie backtestu **pokrycie** było zbliżone do 90% na każdym horyzoncie.

- Wzór: **P90_cal = P50 + s_h · (P90 − P50)**; współczynnik *s_h* jest szacowany z danych backtestu osobno dla każdego *h*.
- Tabela ze skalami *Scale_s* oraz wykres *Coverage po kalibracji* — im bliżej linii 0,9, tym stabilniejsze „górne” estymacje popytu pod planowanie zabezpieczeniowe.
""",
    "d": """
**Serwis i plan** — symulacja uproszczonego łańcucha: popyt z prognozy (najpierw **P90**, potem wersja **P90_cal**), stany **FG**, **lead time** oraz porównanie do KPI z danych źródłowych.

- Sekcja *v2*: popyt z backtestu **Pred_OrdersIn_P90**; histogram różnic *LostSales* i ranking SKU pomagają zobaczyć, gdzie plan odbiega od KPI.
- *Skalibrowany*: ten sam mechanizm przy popycie **P90_cal** — scatter KPI vs plan dla **Fill rate LT** pokazuje zgodność po kalibracji kwantyla.
- Etykiety **STOP / SLOW / SPEED** w planie produkcji wynikają z porównania zalecanego wolumenu do bazy (*LT_PW_Qty* lub średniego popytu).
""",
    "e": """
**Prognoza FINAL** — modele wytrenowane na **pełnej historii** do ostatniego EOM; jedna wspólna data odcięcia (*Run_Cutoff_EOM* = ASOF) dla wszystkich SKU.

- Wykres sumuje prognozy po wszystkich produktach dla kolejnych miesięcy (**Σ P50** vs **Σ P90 cal**) — obraz agregatu portfela.
- **Plan produkcji FINAL** używa popytu **P90_cal** i reguł jak w backteście; wykres kołowy to rozkład rekomendacji akcji, tabela — szczegóły per SKU i miesiąc.
""",
}


def coverage_by_horizon(bt_cal: pd.DataFrame) -> pd.DataFrame | None:
    if bt_cal is None or bt_cal.empty or "Pred_OrdersIn_P90_cal" not in bt_cal.columns:
        return None
    rows = []
    for h in sorted(bt_cal["Horizon_M"].unique()):
        g = bt_cal[bt_cal["Horizon_M"] == h]
        cov = (g["Actual_OrdersIn"] <= g["Pred_OrdersIn_P90_cal"]).mean()
        rows.append({"Horizon_M": int(h), "Coverage_P90_cal": float(cov)})
    return pd.DataFrame(rows)


def main():
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "error" not in st.session_state:
        st.session_state["error"] = None

    st.set_page_config(
        page_title="AI Finance Guard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css()

    st.markdown(
        """
        <div class="hero">
            <div class="badge">CatBoost · P50 / P90 · Lakehouse-ready</div>
            <h1>AI Finance Guard</h1>
            <p>Dashboard prognoz zamówień (OrdersIn), backtestu rolling, kalibracji kwantyli oraz planu produkcji i metryk serwisu.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Dane i parametry ML")
        default_feat = BASE / "AFG_ML_FEATUREONLY_SKU_EOM_STRICT.csv"
        default_train = BASE / "AFG_ML_TRAIN_SKU_EOM_STRICT.csv"
        feat_path = st.text_input("Plik cech (CSV)", str(default_feat))
        train_path = st.text_input("Plik treningowy (CSV)", str(default_train))
        out_dir = st.text_input("Katalog wyjściowy (out)", str(BASE / "out"))

        st.subheader("CatBoost")
        n_cutoffs = st.slider("Liczba cutoffów (backtest)", 1, 12, 3)
        cb_iterations = st.slider("Iteracje (CB_ITERATIONS)", 200, 3000, 800, 100)
        cb_depth = st.slider("Głębokość drzewa", 4, 12, 8)
        cb_lr = st.number_input("Learning rate", 0.01, 0.3, 0.05, 0.01)
        include_final = st.checkbox(
            "Prognoza FINAL + Plan_Production_FINAL (najdłuższy etap)",
            value=True,
            help="Wyłącz, aby szybciej zobaczyć backtest i kalibrację bez ponownego treningu 6×2 modeli na pełnej historii.",
        )

        run_btn = st.button("Uruchom pipeline", type="primary", use_container_width=True)

    if run_btn:
        cfg = PipelineConfig(
            feature_path=feat_path,
            train_path=train_path,
            out_dir=out_dir,
            n_cutoffs=n_cutoffs,
            cb_iterations=cb_iterations,
            cb_depth=cb_depth,
            cb_lr=cb_lr,
            include_final_forecast=include_final,
        )
        with st.spinner("Trening modeli i generowanie planów — może potrwać kilka minut…"):
            try:
                result = run_pipeline(cfg)
                st.session_state["result"] = result
                st.session_state["error"] = None
            except Exception as e:
                st.session_state["error"] = str(e)
                st.session_state["result"] = None

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    res = st.session_state.get("result")
    if res is None:
        st.info(
            "Wgraj ścieżki do plików CSV (domyślnie pliki z folderu projektu) i kliknij **Uruchom pipeline**."
        )
        return

    v = res.validation
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKU (unikalne)", f"{v.get('n_sku', '—')}")
    c2.metric("dup_feat / dup_train", f"{v.get('dup_feat', '—')} / {v.get('dup_train', '—')}")
    c3.metric("only_in_feat / only_in_train", f"{v.get('only_in_feat', '—')} / {v.get('only_in_train', '—')}")
    c4.metric("ASOF (EOM max)", str(v.get("eom_max", "—"))[:10])

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
        [
            "Walidacja i horyzonty",
            "Backtest i metryki",
            "Kalibracja P90",
            "Serwis i plan",
            "Prognoza FINAL",
        ]
    )

    with tab_a:
        st.markdown(TAB_INTRO["a"].strip())
        st.divider()
        st.subheader("Zakres dat (feature)")
        st.write(
            {
                "EOM min": str(v.get("eom_min"))[:10],
                "EOM max": str(v.get("eom_max"))[:10],
            }
        )
        st.subheader("Przykład Forecast_EOM (h=1)")
        st.dataframe(res.h_sample, use_container_width=True)
        st.subheader("Wolumen wierszy per horyzont")
        fig_h = px.bar(
            res.horizon_counts,
            x="Horizon_M",
            y="n_rows",
            color="n_rows",
            color_continuous_scale="Teal",
        )
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            font_color="#e2e8f0",
            xaxis_title="Horizon_M",
            yaxis_title="Liczba wierszy",
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption(f"Wybrane cutoffy: {res.cutoffs}")

    with tab_b:
        st.markdown(TAB_INTRO["b"].strip())
        st.divider()
        if res.metrics_summary is None or res.metrics_summary.empty:
            st.warning("Brak metryk — backtest nie wygenerował wierszy (sprawdź cutoffy i dane).")
        else:
            m = res.metrics_summary.copy()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=m["Horizon_M"], y=m["MAE_P50"], name="MAE P50", marker_color="#14b8a6"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=m["Horizon_M"],
                    y=m["Coverage_P90"],
                    name="Coverage P90",
                    mode="lines+markers",
                    line=dict(color="#f59e0b"),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title="MAE (P50) i pokrycie P90 wg horyzontu",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_yaxes(title_text="MAE", secondary_y=False)
            fig.update_yaxes(title_text="Coverage", secondary_y=True, range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(
                m,
                x="Horizon_M",
                y=["WAPE_P50", "Bias_P50"],
                markers=True,
                color_discrete_sequence=["#38bdf8", "#a78bfa"],
            )
            fig2.update_layout(
                title="WAPE i bias (P50)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(m, use_container_width=True)

            if not res.bt.empty:
                sample = res.bt.head(200)
                fig_sc = px.scatter(
                    sample,
                    x="Actual_OrdersIn",
                    y="Pred_OrdersIn_P50",
                    color="Horizon_M",
                    opacity=0.65,
                    color_continuous_scale="Teal",
                )
                fig_sc.add_trace(
                    go.Scatter(
                        x=[0, sample["Actual_OrdersIn"].max()],
                        y=[0, sample["Actual_OrdersIn"].max()],
                        mode="lines",
                        name="Ideal",
                        line=dict(dash="dash", color="#64748b"),
                    )
                )
                fig_sc.update_layout(
                    title="Actual vs Pred P50 (fragment)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.8)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab_c:
        st.markdown(TAB_INTRO["c"].strip())
        st.divider()
        if res.cal is None or res.cal.empty:
            st.warning("Brak kalibracji — uruchom pełny backtest.")
        else:
            st.dataframe(res.cal, use_container_width=True)
            cov_df = coverage_by_horizon(res.bt_cal)
            if cov_df is not None:
                fig_c = px.line(
                    cov_df,
                    x="Horizon_M",
                    y="Coverage_P90_cal",
                    markers=True,
                    title="Coverage po kalibracji P90_cal (cel ~0.90)",
                )
                fig_c.add_hline(y=0.9, line_dash="dash", line_color="#f59e0b")
                fig_c.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.8)",
                    font_color="#e2e8f0",
                    yaxis_range=[0, 1.05],
                )
                st.plotly_chart(fig_c, use_container_width=True)

    with tab_d:
        st.markdown(TAB_INTRO["d"].strip())
        st.divider()
        cleft, cright = st.columns(2)
        if res.service_eval_v2 is not None and not res.service_eval_v2.empty:
            se = res.service_eval_v2.copy()
            fig_ls = px.histogram(
                se,
                x="Diff_LostSales_LT",
                nbins=40,
                color_discrete_sequence=["#14b8a6"],
                title="Rozkład różnicy LostSales LT (plan − KPI)",
            )
            fig_ls.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
            )
            cleft.plotly_chart(fig_ls, use_container_width=True)

            agg = (
                se.groupby("ProductID", as_index=False)
                .agg(
                    mean_fill=("FillRate_6M_Plan", "mean"),
                    mean_lost=("LostSales_6M_Plan", "mean"),
                )
                .sort_values("mean_lost", ascending=False)
                .head(25)
            )
            fig_b = px.bar(
                agg,
                x="ProductID",
                y="mean_lost",
                color="mean_fill",
                color_continuous_scale="RdYlGn",
                title="Top SKU wg średniego LostSales 6M (backtest)",
            )
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
                xaxis_tickangle=-45,
            )
            cright.plotly_chart(fig_b, use_container_width=True)

            with st.expander("Tabela serwisu v2 (pierwsze 500 wierszy)"):
                st.dataframe(se.head(500), use_container_width=True)
        else:
            st.info("Brak danych serwisu v2.")

        st.subheader("Serwis skalibrowany (P90_cal)")
        if res.svc_cal is not None and not res.svc_cal.empty:
            sc = res.svc_cal.copy()
            fig_sc2 = px.scatter(
                sc,
                x="KPI_FillRate_Qty_LT",
                y="FillRate_LT_Plan_cal",
                color="Diff_FillRate_LT_cal",
                color_continuous_scale="RdYlGn",
                opacity=0.75,
                title="Fill rate LT: KPI vs plan (kalibracja)",
            )
            fig_sc2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_sc2, use_container_width=True)
            st.dataframe(sc.head(300), use_container_width=True)
        else:
            st.caption("Brak danych.")

    with tab_e:
        st.markdown(TAB_INTRO["e"].strip())
        st.divider()
        if res.forecast_final is None or res.forecast_final.empty:
            st.warning("Brak prognozy końcowej.")
        else:
            fc = res.forecast_final.copy()
            st.metric("Run cutoff (FINAL)", str(res.asof))
            agg_fc = (
                fc.groupby("Forecast_EOM", as_index=False)
                .agg(
                    p50=("Pred_OrdersIn_P50", "sum"),
                    p90cal=("Pred_OrdersIn_P90_cal", "sum"),
                )
                .sort_values("Forecast_EOM")
            )
            fig_fc = go.Figure()
            fig_fc.add_trace(
                go.Scatter(
                    x=agg_fc["Forecast_EOM"].astype(str),
                    y=agg_fc["p50"],
                    name="Σ P50",
                    line=dict(color="#38bdf8"),
                )
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=agg_fc["Forecast_EOM"].astype(str),
                    y=agg_fc["p90cal"],
                    name="Σ P90 cal",
                    line=dict(color="#f59e0b"),
                )
            )
            fig_fc.update_layout(
                title="Agregacja prognozy (suma po SKU) — kolejne miesiące",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font_color="#e2e8f0",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            if res.plan_final is not None and not res.plan_final.empty:
                pl = res.plan_final.copy()
                dist = (
                    pl.groupby("ActionLabel").size().reset_index(name="n")
                )
                fig_pie = px.pie(
                    dist,
                    values="n",
                    names="ActionLabel",
                    hole=0.45,
                    color_discrete_map={
                        "STOP": "#ef4444",
                        "SLOW": "#f59e0b",
                        "SPEED": "#22c55e",
                    },
                )
                fig_pie.update_layout(
                    title="Rekomendacje akcji (plan FINAL)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.dataframe(pl.head(400), use_container_width=True)

    with st.expander("Zapisane pliki CSV"):
        st.json(res.paths_written if res.paths_written else {})

    st.caption(
        "AI Finance Guard — lokalny odpowiednik notebooka Fabric; wyniki w katalogu `out/` gotowe do importu Delta (dbo.*)."
    )


if __name__ == "__main__":
    main()
