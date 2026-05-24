from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import COMPANIES, BVG_URL, DATA_PATH, LOG_PATH, LAST_RUN_PATH, ROOT
from app.data import (
    aggregate_daily,
    download_and_clean_bvg,
    get_company_history,
    load_master_dataset,
    merge_and_build_features,
)
from app.logs import read_log, run_global_catchup, should_run_catchup, BundleCache, CatchupResult, detect_model_version, is_gate_active, derive_training_cutoff
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
from app.retrain import retrain_classical_model
from app.viz import build_plotly
from bvg_core.config import PROCESSED_DIR
from bvg_core.dataset import get_latest_dataset_version


def main() -> None:
    setup_page()

    if "bundle_cache" not in st.session_state:
        st.session_state.bundle_cache = BundleCache()

    # Load latest versioned dataset or fallback to legacy master
    versioned_path = get_latest_dataset_version()
    dataset_version = None
    if versioned_path:
        base_df = load_master_dataset(versioned_path)
        dataset_version = versioned_path.name
    else:
        base_df = load_master_dataset(Path(DATA_PATH))

    tab_companies = list(COMPANIES)

    log_df = pd.DataFrame()
    try:
        log_df = read_log(Path(LOG_PATH))
        last_run = read_last_run(Path(LAST_RUN_PATH))
        last_run_str = format_last_run(last_run)
    except Exception:  # noqa: BLE001
        last_run_str = "—"

    gate_active = is_gate_active(log_df, tab_companies, ("classical", "quantum"))

    merged_df = None
    catchup_result: CatchupResult | None = None

    if render_top_bar(last_run_str, gate_active=gate_active):
        try:
            write_last_run(Path(LAST_RUN_PATH))
            with st.spinner("Descargando y limpiando datos BVG..."):
                bvg_df = download_and_clean_bvg(BVG_URL, COMPANIES)
                daily_df = aggregate_daily(bvg_df)

            with st.spinner("Fusionando datos y reconstruyendo features..."):
                merged_df = merge_and_build_features(base_df, daily_df, horizons=[5])

            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            versioned_csv = PROCESSED_DIR / f"BVG_features_svc_master_v{date_str}.csv"
            merged_df.to_csv(versioned_csv, index=False)

            dataset_version = versioned_csv.name
            st.info(f"Dataset versionado: {dataset_version}")

            log_df = read_log(Path(LOG_PATH))
            if should_run_catchup(log_df, daily_df):
                with st.spinner("Ejecutando catch-up y calculando inferencias..."):
                    catchup_result = run_global_catchup(
                        Path(LOG_PATH), daily_df, merged_df, model_name="h5",
                        cache=st.session_state.bundle_cache,
                    )
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

    # Use merged data when available, otherwise fall back to base
    work_df = merged_df if merged_df is not None else base_df

    # Row-count retrain gate — shown whenever the persisted log indicates a boundary
    if gate_active:
        st.warning(
            "Lote de 4 semanas completado. Es obligatorio reentrenar TODOS los modelos "
            "(Clásico y Cuántico) para continuar."
        )
        if st.button("Chequear reentrenados"):
            all_retrained = True
            missing_items = []
            for company in COMPANIES:
                classical_retrained = detect_model_version(company, "classical", "h5") != "original"
                quantum_retrained = detect_model_version(company, "quantum", "h5") != "original"
                if not classical_retrained:
                    all_retrained = False
                    missing_items.append(f"{company} (clásico)")
                if not quantum_retrained:
                    all_retrained = False
                    missing_items.append(f"{company} (cuántico)")
            if all_retrained:
                st.success("Ambos modelos reentrenados detectados para todas las empresas. Catch-up puede continuar.")
            else:
                st.error(f"Falta reentrenar: {', '.join(missing_items)}")

        if st.button("Reentrenar modelos clásicos"):
            training_cutoff = derive_training_cutoff(log_df)
            for company in tab_companies:
                try:
                    with st.spinner(f"Reentrenando modelo clásico para {company}..."):
                        result = retrain_classical_model(
                            company,
                            work_df,
                            trigger_accuracy=0.0,
                            dataset_version=dataset_version,
                            training_cutoff=training_cutoff,
                        )
                    st.success(
                        f"{company}: reentrenamiento completado. "
                        f"Nuevo test accuracy: {result['test_accuracy']:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{company}: error en reentrenamiento: {exc}")
            st.session_state.bundle_cache.clear()

        st.caption(
            "El modelo cuántico debe reentrenarse manualmente desde el notebook "
            "correspondiente (Fase4_ModeladoCuantico)."
        )

    if tab_companies:
        tabs = st.tabs(tab_companies)
        for tab, company in zip(tabs, tab_companies, strict=False):
            with tab:
                try:
                    # Full history for the chart (not limited to 30 rows)
                    full_history = work_df.loc[
                        work_df["empresa"] == company
                    ].sort_values("fecha").copy()
                    # Quick history table still uses 30-row window
                    quick_history = get_company_history(work_df, company)
                    build_cards(log_df, company)
                    show_quick_history(quick_history, company)
                    fig = build_plotly(full_history, log_df, company, gate_active=gate_active)
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
