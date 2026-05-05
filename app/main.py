from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import BVG_TARGET_ISSUERS, BVG_URL, DATA_PATH, LOG_PATH
from app.data import (
    aggregate_daily,
    download_and_clean_bvg,
    get_company_history,
    load_master_dataset,
)
from app.logs import catchup_loop, read_log
from app.ui import (
    action_button,
    setup_page,
    show_live_accuracy,
    show_metrics_header,
    show_price_chart,
    show_quick_history,
    show_traceability_table,
    show_warning_if_insufficient_data,
    sidebar_config,
)
from app.viz import build_price_figure, compute_accuracy


def main() -> None:
    setup_page()
    base_df = load_master_dataset(Path(DATA_PATH))
    companies = sorted(base_df["empresa"].dropna().unique().tolist())
    sidebar_state = sidebar_config(
        companies,
        dataset_path=str(DATA_PATH),
    )

    if action_button():
        try:
            with st.spinner("Descargando y limpiando datos BVG..."):
                bvg_df = download_and_clean_bvg(BVG_URL, BVG_TARGET_ISSUERS)
                daily_df = aggregate_daily(bvg_df)

            if show_warning_if_insufficient_data(daily_df):
                st.stop()

            with st.spinner("Ejecutando catch-up y calculando inferencias..."):
                catchup_loop(
                    Path(LOG_PATH),
                    daily_df,
                    sidebar_state.model_family,
                    "h5",
                )

            st.success("Proceso completado: bitácora actualizada.")
        except ConnectionError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

    if sidebar_state.company:
        try:
            history = get_company_history(base_df, sidebar_state.company)
            show_quick_history(history, sidebar_state.company)
            log_df = read_log(Path(LOG_PATH))
            fig, _history_plot = build_price_figure(
                history,
                log_df,
                sidebar_state.company,
            )
            show_price_chart(fig)
        except Exception as exc:  # noqa: BLE001
            st.warning(str(exc))

    show_metrics_header()
    try:
        log_df = read_log(Path(LOG_PATH))
        accuracy_live = compute_accuracy(log_df)
        show_live_accuracy(accuracy_live)
        show_traceability_table(log_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))


if __name__ == "__main__":
    main()
