from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import BVG_TARGET_ISSUERS, BVG_URL, DATA_PATH, LOG_PATH
from app.data import (
    aggregate_daily,
    download_and_clean_bvg,
    get_company_history,
    load_master_dataset,
)
from app.logs import read_log, run_global_catchup, should_run_catchup
from app.ui import (
    render_top_bar,
    setup_page,
    show_price_chart,
    show_quick_history,
    show_traceability_table,
    build_cards,
)
from app.viz import build_plotly


def main() -> None:
    setup_page()
    base_df = load_master_dataset(Path(DATA_PATH))
    tab_companies = list(BVG_TARGET_ISSUERS)

    try:
        log_df = read_log(Path(LOG_PATH))
        last_run = None
        if not log_df.empty:
            last_run = pd.to_datetime(log_df["timestamp_utc"], errors="coerce").max()
        last_run_str = (
            last_run.isoformat() if isinstance(last_run, pd.Timestamp) and pd.notna(last_run) else None
        )
    except Exception:  # noqa: BLE001
        last_run_str = None

    if render_top_bar(last_run_str):
        try:
            with st.spinner("Descargando y limpiando datos BVG..."):
                bvg_df = download_and_clean_bvg(BVG_URL, BVG_TARGET_ISSUERS)
                daily_df = aggregate_daily(bvg_df)

            log_df = read_log(Path(LOG_PATH))
            if should_run_catchup(log_df, daily_df):
                with st.spinner("Ejecutando catch-up y calculando inferencias..."):
                    run_global_catchup(Path(LOG_PATH), daily_df, model_name="h5")
            else:
                st.info("Catch-up ya está actualizado con el último viernes disponible.")

            st.success("Proceso completado: bitácora actualizada.")
        except ConnectionError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

    if tab_companies:
        tabs = st.tabs(tab_companies)
        for tab, company in zip(tabs, tab_companies, strict=False):
            with tab:
                try:
                    history = get_company_history(base_df, company)
                    build_cards(log_df, company)
                    show_quick_history(history, company)
                    fig = build_plotly(history, log_df, company)
                    show_price_chart(fig)
                    company_log = log_df.loc[
                        (log_df["company"] == company)
                        & (log_df["model_family"].isin(["classical", "quantum"]))
                    ].copy()
                    show_traceability_table(company_log)
                except Exception as exc:  # noqa: BLE001
                    st.warning(str(exc))
    else:
        st.warning("No hay empresas disponibles para visualizar.")



if __name__ == "__main__":
    main()
