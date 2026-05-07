from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import (
    THEME_ACCENT_CLASSICAL,
    THEME_ACCENT_QUANTUM
)


# ── Theme-aware CSS ──────────────────────────────────────────────────────────
_CSS = """
<style>
/* Cards: border-only so they work in both dark and light */
.app-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
    transition: box-shadow 0.2s ease;
}
.app-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Badges */
.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

/* Section headers with colored left border */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 12px 0;
    padding-left: 12px;
    border-left: 4px solid;
}
.section-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
}

/* Metric grid inside cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 12px;
}
.metric-item {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    background: rgba(128,128,128,0.06);
}
.metric-label {
    font-size: 11px;
    opacity: 0.65;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
}

/* Compact table */
.compact-table {
    font-family: "JetBrains Mono", "Fira Mono", "Consolas", monospace;
    font-size: 12px;
}

/* Status chips */
.chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}
</style>
"""


def setup_page() -> None:
    st.set_page_config(page_title="Validación Experimental", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    
    st.title("Validación Experimental")
    st.caption("Comparativo Clásico vs Cuántico — Horizonte t+5")
    
    


def render_top_bar(last_run: str | None) -> bool:
    with st.container():
        c1, c2 = st.columns([1,6])
        with c1:
            clicked = st.button(
                "Actualizar BVG + Catch-up",
                type="secondary",
                width='content',
            )
        with c2:
            label = "Sin ejecuciones previas" if not last_run else f"🕐 {last_run}"
            st.write(f"Última ejecución: {label}")
        return clicked
    return False


# ── Sections ─────────────────────────────────────────────────────────────────

def section_header(emoji: str, title: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="section-header" style="border-color:{color};">
            <span style="font-size:22px;">{emoji}</span>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_freeze_metrics(csv_path: Path) -> None:
    section_header("🧊", "Métricas de Validación (Modelos Congelados)", "#10b981")
    if not csv_path.exists():
        st.info("No se encontró el archivo de métricas de validación.")
        return
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.info("El archivo de métricas está vacío.")
            return
        display = df.copy()
        if "accuracy" in display.columns:
            display["accuracy"] = display["accuracy"].apply(
                lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "—"
            )
        if "f1" in display.columns:
            display["f1"] = display["f1"].apply(
                lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "—"
            )
        if "created_at_utc" in display.columns:
            display["created_at_utc"] = pd.to_datetime(
                display["created_at_utc"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Error al cargar métricas de validación: {exc}")


# ── Model cards ──────────────────────────────────────────────────────────────

def _compute_accuracy_f1(resolved_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if resolved_df.empty:
        return None, None
    y_true = (resolved_df["ret_fwd_h5"] > 0).astype(int)
    y_pred = resolved_df["y_pred"].astype(int)
    accuracy = float((y_true == y_pred).mean())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return accuracy, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, float(f1)


def build_cards(log_df: pd.DataFrame, company: str) -> None:
    section_header("🎯", "Comparativo de Modelos", "#3b82f6")
    if log_df.empty:
        st.info("No hay registros para la empresa seleccionada.")
        return

    left, right = st.columns(2)

    for col, family, label, accent in (
        (left, "classical", "Clásico", THEME_ACCENT_CLASSICAL),
        (right, "quantum", "Cuántico", THEME_ACCENT_QUANTUM),
    ):
        with col:
            family_df = log_df.loc[
                (log_df["company"] == company) & (log_df["model_family"] == family)
            ].copy()

            if family_df.empty:
                st.markdown(
                    f"""
                    <div class="app-card" style="border-color:{accent}40;">
                        <span class="badge-pill" style="background:{accent};color:#fff;">{label}</span>
                        <p style="margin-top:12px; opacity:0.6;">Sin datos disponibles.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue

            family_df["fecha_t5"] = pd.to_datetime(family_df["fecha_t5"], errors="coerce")
            latest = family_df.loc[family_df["fecha_t5"].idxmax()].copy()
            direction = "▲ UP" if int(latest["y_pred"]) == 1 else "▼ DOWN"
            proba_up = latest.get("proba_up")
            proba_label = "—" if pd.isna(proba_up) else f"{float(proba_up):.1%}"

            resolved = family_df.loc[
                family_df["ret_fwd_h5"].notna() & family_df["y_pred"].notna()
            ].copy()
            accuracy, f1 = _compute_accuracy_f1(resolved)
            acc_label = "—" if accuracy is None else f"{accuracy:.1%}"
            f1_label = "—" if f1 is None else f"{f1:.1%}"

            st.markdown(
                f"""
                <div class="app-card" style="border-color:{accent}40;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-pill" style="background:{accent};color:#fff;">{label}</span>
                        <span style="font-size:12px; opacity:0.6;">
                            {pd.to_datetime(latest["fecha_t5"]).date().isoformat()}
                        </span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-label">Dirección</div>
                            <div class="metric-value" style="color:{accent};">{direction}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Proba Up</div>
                            <div class="metric-value">{proba_label}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Accuracy</div>
                            <div class="metric-value">{acc_label}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">F1 Score</div>
                            <div class="metric-value">{f1_label}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Price chart ────────────────────────────────────────────────────────────────

def show_price_chart(fig: object) -> None:
    section_header("📈", "Gráfico de Precios + Predicciones", "#f59e0b")
    st.plotly_chart(fig, use_container_width=True)


# ── Quick history ──────────────────────────────────────────────────────────────

def show_quick_history(history_df: pd.DataFrame, company: str) -> None:
    section_header("📋", "Historial Reciente", "#6366f1")
    if history_df.empty:
        st.info("No hay historial disponible.")
        return
    drop_cols = [c for c in history_df.columns if c in ("target_up_h1", "target_up_h20")]
    display_df = history_df.drop(columns=drop_cols, errors="ignore").tail(10).copy()
    st.caption(f"Últimas {len(display_df)} filas para {company}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Traceability ───────────────────────────────────────────────────────────────

def _pivot_traceability(log_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot from one-row-per-model to one-row-per-event."""
    if log_df.empty:
        return log_df

    df = log_df.copy()
    df = df.dropna(subset=["company", "fecha_t", "fecha_t5", "model_family"]).copy()

    df["fecha_t"] = pd.to_datetime(df["fecha_t"], errors="coerce").dt.date.astype(str)
    df["fecha_t5"] = pd.to_datetime(df["fecha_t5"], errors="coerce").dt.date.astype(str)
    df["y_pred_label"] = df["y_pred"].map({1: "SUBE", 0: "BAJA"})

    grouped = df.groupby(["company", "fecha_t", "horizonte"])
    rows = []
    for (company, fecha_t, horiz), g in grouped:
        row: dict = {
            "Empresa": company,
            "Fecha_Datos": fecha_t,
            "Fecha_Objetivo": g["fecha_t5"].iloc[0] if "fecha_t5" in g.columns else "",
            "close_t": g["close_t"].iloc[0],
            "close_t5": g["close_t5"].iloc[0],
        }
        for family in ("classical", "quantum"):
            fam = g.loc[g["model_family"] == family]
            suffix = "Clasico" if family == "classical" else "Cuantico"
            if fam.empty:
                row[f"Pred_{suffix}"] = "—"
                row[f"Proba_{suffix}"] = "—"
                row[f"Estado_{suffix}"] = "—"
            else:
                row[f"Pred_{suffix}"] = fam["y_pred_label"].iloc[0]
                proba = fam["proba_up"].iloc[0]
                row[f"Proba_{suffix}"] = f"{float(proba)*100:.0f}%" if pd.notna(proba) else "—"
                row[f"Estado_{suffix}"] = fam["status"].iloc[0]

        statuses = []
        for family in ("classical", "quantum"):
            fam = g.loc[g["model_family"] == family]
            if not fam.empty:
                statuses.append((family, fam["status"].iloc[0]))
        aciertos = [f for f, s in statuses if s == "ACIERTO"]
        if len(aciertos) == 2:
            row["Mejor_Modelo"] = "Ambos"
        elif len(aciertos) == 1:
            row["Mejor_Modelo"] = "Clasico" if aciertos[0] == "classical" else "Cuantico"
        else:
            resolved = [s for _, s in statuses if s in ("ACIERTO", "FALLO")]
            row["Mejor_Modelo"] = "—" if not resolved else "Ninguno"

        rows.append(row)

    result = pd.DataFrame(rows)
    for col in ("close_t", "close_t5"):
        result[col] = result[col].apply(lambda x: f"${float(x):.2f}" if pd.notna(x) else "—")
    return result


def show_traceability_table(log_df: pd.DataFrame) -> None:
    section_header("📊", "Bitácora de Trazabilidad", "#64748b")
    if log_df.empty:
        st.info("La bitácora está vacía.")
        return

    display_df = _pivot_traceability(log_df)
    if display_df.empty:
        st.info("No hay datos para mostrar.")
        return

    display_df["_sort"] = pd.to_datetime(display_df["Fecha_Objetivo"], errors="coerce")
    display_df = display_df.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    styled = display_df.style.set_table_attributes('class="compact-table"')
    st.dataframe(styled, use_container_width=True, hide_index=True)
