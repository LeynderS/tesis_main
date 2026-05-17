from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import BVG_TARGET_ISSUERS, BVG_URL, DATA_PATH, LOG_PATH, LAST_RUN_PATH, ROOT
from app.data import (
    aggregate_daily,
    download_and_clean_bvg,
    get_company_history,
    load_master_dataset,
)
from app.logs import read_log, run_global_catchup, should_run_catchup
from app.state import read_last_run, write_last_run, format_last_run
from app.ui import (
    render_top_bar,
    setup_page,
    show_price_chart,
    show_quick_history,
    show_traceability_table,
    build_cards,
    show_freeze_metrics,
)
from app.retrain import (
    retrain_classical_model,
    should_retrain,
    should_retrain_quantum,
)
from app.viz import build_plotly


def main() -> None:
    setup_page()
    base_df = load_master_dataset(Path(DATA_PATH))
    tab_companies = list(BVG_TARGET_ISSUERS)

    try:
        log_df = read_log(Path(LOG_PATH))
        last_run = read_last_run(Path(LAST_RUN_PATH))
        last_run_str = format_last_run(last_run)
    except Exception:  # noqa: BLE001
        last_run_str = "—"

    if render_top_bar(last_run_str):
        try:
            write_last_run(Path(LAST_RUN_PATH))
            with st.spinner("Descargando y limpiando datos BVG..."):
                bvg_df = download_and_clean_bvg(BVG_URL, BVG_TARGET_ISSUERS)
                daily_df = aggregate_daily(bvg_df)

            log_df = read_log(Path(LOG_PATH))
            if should_run_catchup(log_df, daily_df):
                with st.spinner("Ejecutando catch-up y calculando inferencias..."):
                    run_global_catchup(Path(LOG_PATH), daily_df, base_df, model_name="h5")
                st.success("Proceso completado: bitácora actualizada.")
                st.rerun()
            else:
                st.info("Catch-up ya está actualizado con el último viernes disponible.")
        except ConnectionError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

    freeze_csv = ROOT / "results" / "fase4_comparativa" / "BVG_subfase8_comparativa_h5.csv"
    show_freeze_metrics(freeze_csv)

    # Retrain gate after catchup
    for company in tab_companies:
        classical_drift = should_retrain(log_df, company, "classical")
        quantum_msg = should_retrain_quantum(log_df, company)
        if classical_drift:
            resolved = log_df.loc[
                (log_df["company"] == company)
                & (log_df["model_family"] == "classical")
                & (log_df["status"].isin(["ACIERTO", "FALLO"]))
            ]
            trigger_acc = 0.0
            if len(resolved) >= 4:
                trigger_acc = float((resolved.tail(4)["status"] == "ACIERTO").mean())
            st.warning(
                f"**{company}** — El modelo clásico muestra deriva "
                f"(rolling accuracy = {trigger_acc:.2f})."
            )
            if st.button(
                "Reentrenar modelo clásico",
                key=f"retrain_classical_{company}",
            ):
                try:
                    with st.spinner(
                        f"Reentrenando modelo clásico para {company}..."
                    ):
                        result = retrain_classical_model(
                            company, base_df, trigger_accuracy=trigger_acc
                        )
                    st.success(
                        f"Reentrenamiento completado. "
                        f"Nuevo test accuracy: {result['test_accuracy']:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error en reentrenamiento: {exc}")
        if quantum_msg:
            st.warning(quantum_msg)

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
