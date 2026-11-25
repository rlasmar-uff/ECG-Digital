"""Interface interativa local (Streamlit) — Suporte Computacional para Interpretação de ECG.

Universidade Federal Fluminense (UFF) – Escola de Engenharia
Disciplina: Sinais e Sistemas – Prof. André Pinho (2º Semestre de 2026)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    COURSE_INFO,
    DATA_DIR,
    DEFAULT_FS,
    INSTITUTION,
    LITERATURE_BENCHMARKS,
    MITBIH_RECORDS,
    NOTCH_FREQ_HZ,
    NOTCH_Q,
    OUTPUTS_DIR,
    PROJECT_TITLE,
    REFRACTORY_PERIOD_MS,
    TEAM_INLINE,
    TEAM_MEMBERS,
    VALIDATION_TOLERANCE_MS,
)
from src.ecg_pipeline import (
    ECGData,
    PipelineResult,
    compute_cascade_frequency_response,
    compute_clinical_metrics,
    inspect_pdf_exam,
    load_mitbih_record,
    load_physionet_upload,
    load_tabular_file,
    process_ecg,
)
from src.plotting import (
    find_arrhythmia_bigeminy_window,
    find_best_noisy_window,
    find_bradycardia_window,
    find_fastest_tachycardia_window,
    plot_ecg_comparison,
)
from src.validation import match_beats_one_to_one

# ---------------------------------------------------------------------------
# Configuração de Página e Estilo Clínico
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ECG Digital — Sinais e Sistemas (UFF)",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

.header-bar {
    background: #FFFFFF;
    border-bottom: 2px solid #0B2545;
    padding: 1rem 1.25rem 0.9rem 1.25rem;
    margin-bottom: 1rem;
    border-radius: 6px;
}

.header-title {
    font-size: 1.38rem;
    font-weight: 700;
    color: #0B2545;
    margin: 0 0 0.25rem 0;
    line-height: 1.25;
}

.header-meta {
    font-size: 0.86rem;
    color: #475569;
    margin: 0;
}

.header-team {
    font-size: 0.80rem;
    color: #334155;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px solid #E2E8F0;
}

.triage-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.2rem;
    border-radius: 6px;
    margin-bottom: 0.9rem;
    border-left: 5px solid #0F766E;
    background: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
}

.triage-alert {
    border-left-color: #C81E1E !important;
    background: #FEF2F2 !important;
}

.triage-warn {
    border-left-color: #B45309 !important;
    background: #FFFBEB !important;
}

.triage-label {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0B2545;
}

.triage-sub {
    font-size: 0.88rem;
    color: #334155;
}

.kpi-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.75rem 0.95rem;
}

.kpi-title {
    font-size: 0.80rem;
    color: #475569;
    margin-bottom: 0.2rem;
}

.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.42rem;
    font-weight: 600;
    color: #0B2545;
    line-height: 1.2;
}

.kpi-detail {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #0F766E;
    margin-top: 0.2rem;
}

.pdf-alert-box {
    background: #FFF7ED;
    border: 1.5px solid #C2410C;
    border-left: 6px solid #C2410C;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0 1rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Funções com Cache para Carregamento Rápido
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Carregando registro real da base MIT-BIH (PhysioNet)...")
def cached_load_mitbih(record_id: str) -> ECGData:
    return load_mitbih_record(record_id=record_id, data_dir=DATA_DIR)


def build_ecg_plotly_figure(
    result: PipelineResult,
    start_s: float,
    duration_s: float,
    show_atr: bool = False,
) -> go.Figure:
    """Constrói gráfico interativo Antes x Depois em grade de papel milimetrado ECG."""
    fs = result.ecg.fs
    s_idx = max(0, int(round(start_s * fs)))
    e_idx = min(result.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
    raw_seg = result.ecg.signal[s_idx:e_idx]
    filt_seg = result.filtered_signal[s_idx:e_idx]

    win_peaks = result.r_peaks[(result.r_peaks >= s_idx) & (result.r_peaks < e_idx)]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        subplot_titles=(
            f"ANTES — Sinal Bruto ({result.ecg.source_name} · fs = {fs:.0f} Hz)",
            "DEPOIS — Sinal Filtrado (Sistemas LIT Fase Zero) + Picos R Detectados",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=raw_seg,
            mode="lines",
            name="Sinal Bruto x[n]",
            line=dict(color="#4A5568", width=1.25),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=filt_seg,
            mode="lines",
            name="ECG Filtrado (Fase Zero)",
            line=dict(color="#0B2545", width=1.6),
        ),
        row=2,
        col=1,
    )

    if len(win_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=win_peaks / fs,
                y=result.filtered_signal[win_peaks],
                mode="markers",
                name=f"Picos R detectados ({len(win_peaks)} na janela)",
                marker=dict(color="#C81E1E", size=8, line=dict(color="white", width=1)),
            ),
            row=2,
            col=1,
        )

    if show_atr and result.ecg.annotation_samples is not None:
        atr_in = result.ecg.annotation_samples[
            (result.ecg.annotation_samples >= s_idx) & (result.ecg.annotation_samples < e_idx)
        ]
        if len(atr_in) > 0:
            fig.add_trace(
                go.Scatter(
                    x=atr_in / fs,
                    y=result.filtered_signal[atr_in],
                    mode="markers",
                    name="Anotação MIT-BIH (.atr)",
                    marker=dict(
                        color="#0F766E",
                        size=11,
                        symbol="circle-open",
                        line=dict(width=2, color="#0F766E"),
                    ),
                ),
                row=2,
                col=1,
            )

    # Grade de papel milimetrado clínico (0,20 s / 0,04 s e 0,5 mV / 0,1 mV)
    for row_idx in (1, 2):
        fig.update_xaxes(
            row=row_idx,
            col=1,
            range=[float(t[0]), float(t[-1])],
            dtick=0.20,
            showgrid=True,
            gridcolor="#E5989B",
            gridwidth=1.0,
            minor=dict(dtick=0.04, showgrid=True, gridcolor="#F5D0CE", gridwidth=0.5),
            tickformat=".1f",
            zeroline=False,
        )
        fig.update_yaxes(
            row=row_idx,
            col=1,
            title_text="mV",
            dtick=0.5,
            showgrid=True,
            gridcolor="#E5989B",
            gridwidth=1.0,
            minor=dict(dtick=0.1, showgrid=True, gridcolor="#F5D0CE", gridwidth=0.5),
            zeroline=True,
            zerolinecolor="#CBD5E1",
        )

    fig.update_xaxes(
        title_text="Tempo (s) — Grade padrão ECG (0,04 s menor / 0,20 s maior · 0,1 mV / 0,5 mV)",
        row=2,
        col=1,
    )
    fig.update_layout(
        height=640,
        plot_bgcolor="#FFFDFB",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=55, r=25, t=48, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    return fig


def build_fft_plotly_figure(result: PipelineResult) -> go.Figure:
    """Constrói gráfico interativo do espectro FFT (Antes x Depois) até 100 Hz."""
    sp = result.spectral
    freqs = sp.freqs_hz
    mask = (freqs > 0.0) & (freqs <= min(100.0, result.ecg.fs / 2.0))
    f_sel = freqs[mask]
    raw_p = sp.mag_raw[mask] ** 2
    flt_p = sp.mag_filtered[mask] ** 2

    # Agrupamento fino em bins de 0,15 Hz para renderização suave no navegador
    edges = np.arange(0.05, float(f_sel[-1]) + 0.15, 0.15)
    bidx = np.digitize(f_sel, edges) - 1
    valid = (bidx >= 0) & (bidx < len(edges) - 1)
    fc = 0.5 * (edges[:-1] + edges[1:])
    rb = np.zeros_like(fc)
    fb = np.zeros_like(fc)
    ct = np.zeros_like(fc)
    np.add.at(rb, bidx[valid], raw_p[valid])
    np.add.at(fb, bidx[valid], flt_p[valid])
    np.add.at(ct, bidx[valid], 1.0)
    nz = ct > 0
    rb[nz] /= ct[nz]
    fb[nz] /= ct[nz]

    eps = 1e-13
    raw_db = 10.0 * np.log10(np.maximum(rb[nz], eps))
    flt_db = 10.0 * np.log10(np.maximum(fb[nz], eps))

    fig = go.Figure()
    fig.add_vrect(x0=0.0, x1=0.5, fillcolor="#FCA5A5", opacity=0.28, line_width=0, annotation_text="<0,5 Hz", annotation_position="top left")
    fig.add_vrect(x0=0.5, x1=40.0, fillcolor="#BBF7D0", opacity=0.18, line_width=0, annotation_text="Banda ECG (0,5–40 Hz)", annotation_position="top")
    fig.add_vrect(x0=40.0, x1=float(fc[nz][-1]), fillcolor="#FDE68A", opacity=0.22, line_width=0, annotation_text=">40 Hz (EMG)", annotation_position="top right")
    fig.add_vline(x=60.0, line_dash="dash", line_color="#C81E1E", line_width=1.6, annotation_text="Notch 60 Hz", annotation_position="top right")

    fig.add_trace(
        go.Scatter(
            x=fc[nz],
            y=raw_db,
            mode="lines",
            name="FFT Antes (Sinal Bruto)",
            line=dict(color="#64748B", width=1.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc[nz],
            y=flt_db,
            mode="lines",
            name="FFT Depois (Sinal Filtrado)",
            line=dict(color="#0B2545", width=2.0),
        )
    )

    fig.update_layout(
        height=430,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Frequência (Hz)",
        yaxis_title="Densidade Espectral de Potência (dB re mV²)",
        margin=dict(l=55, r=25, t=35, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    return fig


def build_rr_plotly_figure(result: PipelineResult) -> go.Figure:
    """Visualiza a periodicidade discreta N[k] (amostras) e a frequência cardíaca instantânea."""
    m = result.metrics
    if len(m.rr_intervals_ms) == 0:
        return go.Figure()

    beat_times_s = result.r_peaks[1:] / result.ecg.fs
    n_samples_seq = np.diff(result.r_peaks)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "Período Discreto N[k] entre Picos R Sucessivos (x[n] ≈ x[n + N])",
            "Frequência Cardíaca Instantânea (BPM) com Faixa Clínica (60–120 BPM)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=beat_times_s,
            y=n_samples_seq,
            mode="lines+markers",
            name="Período N[k] (amostras)",
            line=dict(color="#0F766E", width=1.4),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=m.mean_n_samples,
        line_dash="dot",
        line_color="#0B2545",
        annotation_text=f"N̄ = {m.mean_n_samples:.1f} amostras",
        row=1,
        col=1,
    )

    fig.add_hrect(y0=60.0, y1=120.0, fillcolor="#BBF7D0", opacity=0.22, line_width=0, row=2, col=1)
    fig.add_hline(y=60.0, line_dash="dash", line_color="#C81E1E", row=2, col=1)
    fig.add_hline(y=120.0, line_dash="dash", line_color="#C81E1E", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=beat_times_s,
            y=m.instant_bpm,
            mode="lines",
            name="BPM instantâneo",
            line=dict(color="#0B2545", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Amostras (N)", row=1, col=1)
    fig.update_yaxes(title_text="BPM", row=2, col=1)
    fig.update_xaxes(title_text="Tempo no exame (s)", row=2, col=1)
    fig.update_layout(
        height=490,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=55, r=25, t=42, b=45),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Cabeçalho Principal
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="header-bar">
        <div class="header-title">{PROJECT_TITLE}</div>
        <div class="header-meta">{INSTITUTION} · {COURSE_INFO}</div>
        <div class="header-team"><strong>Equipe:</strong> {TEAM_INLINE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra Lateral: Aquisição de Dados e Configuração dos Filtros LIT
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Aquisição do Sinal de ECG")
    source_mode = st.radio(
        "Fonte dos dados:",
        options=[
            "Base MIT-BIH (PhysioNet)",
            "Importar arquivo (.csv, .txt, .dat/.hea, .pdf)",
        ],
        index=0,
    )

    ecg_data: ECGData | None = None
    pdf_notice: dict | None = None
    suggested_start_s: float | None = None

    if source_mode == "Base MIT-BIH (PhysioNet)":
        rec_labels = {
            f"{r.record_id} — {r.category}": r for r in MITBIH_RECORDS
        }
        chosen_label = st.selectbox("Selecione o registro clínico:", list(rec_labels.keys()), index=1)
        chosen_rec = rec_labels[chosen_label]
        st.caption(chosen_rec.description)

        ecg_data = cached_load_mitbih(chosen_rec.record_id)
        suggested_start_s = chosen_rec.default_window_start_s

        # Atalho rápido para episódios clínicos marcantes na apresentação ao vivo
        if chosen_rec.record_id == "105":
            if st.button("Ir para trecho ruidoso destacado (Reg. 105)", use_container_width=True):
                st.session_state["force_start_s"] = "auto_noisy"
                st.session_state["jump_nonce"] = st.session_state.get("jump_nonce", 0) + 1
        elif chosen_rec.record_id == "231":
            if st.button("Ir para episódio de Bradicardia < 60 BPM (Reg. 231)", use_container_width=True):
                st.session_state["force_start_s"] = "auto_brady"
                st.session_state["jump_nonce"] = st.session_state.get("jump_nonce", 0) + 1
        elif chosen_rec.record_id == "119":
            if st.button("Ir para trecho de Arritmia / PVCs (Reg. 119)", use_container_width=True):
                st.session_state["force_start_s"] = "auto_bigeminy"
                st.session_state["jump_nonce"] = st.session_state.get("jump_nonce", 0) + 1
        elif chosen_rec.record_id == "209":
            if st.button("Ir para episódio de Taquicardia > 120 BPM (Reg. 209)", use_container_width=True):
                st.session_state["force_start_s"] = "auto_tachy"
                st.session_state["jump_nonce"] = st.session_state.get("jump_nonce", 0) + 1
    else:
        st.caption(
            "Envie um sinal bruto (.csv, .txt), par PhysioNet (.dat + .hea) ou um exame .pdf "
            "para verificar a detecção de formato."
        )
        uploaded_files = st.file_uploader(
            "Arraste ou selecione os arquivos:",
            type=["csv", "txt", "dat", "hea", "atr", "pdf"],
            accept_multiple_files=True,
        )
        fallback_fs = st.number_input(
            "Frequência de amostragem padrão (Hz, usada se não houver coluna de tempo):",
            min_value=50.0,
            max_value=2000.0,
            value=float(DEFAULT_FS),
            step=10.0,
        )

        if uploaded_files:
            by_ext = {Path(f.name).suffix.lower(): f for f in uploaded_files}
            if ".pdf" in by_ext:
                pdf_file = by_ext[".pdf"]
                pdf_notice = inspect_pdf_exam(pdf_file.getvalue())
            elif ".hea" in by_ext and ".dat" in by_ext:
                hea_f = by_ext[".hea"]
                dat_f = by_ext[".dat"]
                atr_f = by_ext.get(".atr")
                try:
                    ecg_data = load_physionet_upload(
                        hea_bytes=hea_f.getvalue(),
                        dat_bytes=dat_f.getvalue(),
                        record_stem=hea_f.name,
                        atr_bytes=atr_f.getvalue() if atr_f else None,
                    )
                except Exception as exc:
                    st.error(f"Erro ao ler arquivos PhysioNet: {exc}")
            elif ".dat" in by_ext or ".hea" in by_ext:
                st.warning(
                    "Para importar registros no formato PhysioNet, envie juntos os arquivos "
                    "`.hea` e `.dat` (e opcionalmente `.atr`)."
                )
            else:
                tab_f = by_ext.get(".csv") or by_ext.get(".txt")
                if tab_f is not None:
                    try:
                        ecg_data = load_tabular_file(
                            raw_bytes=tab_f.getvalue(),
                            filename=tab_f.name,
                            fallback_fs=float(fallback_fs),
                        )
                    except Exception as exc:
                        st.error(f"Erro ao ler arquivo tabular: {exc}")

    st.divider()
    st.header("2. Sistemas LIT de Fase Zero")
    apply_notch = st.checkbox("Filtro Notch 60 Hz (rede elétrica)", value=True)
    notch_q = st.slider("Fator de qualidade Q (Notch 60 Hz)", 10.0, 50.0, float(NOTCH_Q), 1.0, disabled=not apply_notch)

    apply_bp = st.checkbox("Passa-Faixa Butterworth (0,5–40 Hz)", value=True)
    bp_low = st.slider("Corte inferior (Hz — linha de base)", 0.1, 2.0, float(BANDPASS_LOW_HZ), 0.1, disabled=not apply_bp)
    bp_high = st.slider("Corte superior (Hz — tremor muscular)", 20.0, 60.0, float(BANDPASS_HIGH_HZ), 5.0, disabled=not apply_bp)
    bp_order = st.slider("Ordem do filtro Butterworth", 2, 6, int(BANDPASS_ORDER), 1, disabled=not apply_bp)

    refractory_ms = st.slider("Período refratário mínimo (ms)", 200, 320, int(REFRACTORY_PERIOD_MS), 10)

# ---------------------------------------------------------------------------
# Tratamento de Upload de PDF Clínico Pronto
# ---------------------------------------------------------------------------

if pdf_notice is not None:
    st.markdown(
        f"""
        <div class="pdf-alert-box">
            <strong>⚠️ {pdf_notice['warning_title']}</strong><br/><br/>
            {pdf_notice['warning_body']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pdf_notice["preview_png"]:
        st.subheader(f"Pré-visualização do exame enviado (Página 1 de {pdf_notice['num_pages']})")
        st.image(pdf_notice["preview_png"], use_container_width=True)
    st.stop()

if ecg_data is None:
    st.info(
        "Selecione um registro real da base MIT-BIH na barra lateral ou envie um arquivo "
        "`.csv`, `.txt` ou PhysioNet (`.dat` + `.hea`). Dica para demonstração: utilize "
        "`outputs/exemplo_ruidoso_mitbih.csv`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Execução do Pipeline de Processamento Digital de Sinais
# ---------------------------------------------------------------------------

result = process_ecg(
    ecg=ecg_data,
    apply_notch=apply_notch,
    notch_freq=NOTCH_FREQ_HZ,
    notch_q=float(notch_q),
    apply_bandpass=apply_bp,
    lowcut=float(bp_low),
    highcut=float(bp_high),
    order=int(bp_order),
    refractory_ms=float(refractory_ms),
)

# Controles de janela temporal na barra lateral (dependem da duração real do sinal)
with st.sidebar:
    st.divider()
    st.header("3. Janela de Visualização")
    max_dur = min(30.0, max(2.0, result.ecg.duration_s))
    dur_options = [d for d in (5.0, 10.0, 15.0, 30.0) if d <= result.ecg.duration_s] or [round(result.ecg.duration_s, 1)]
    default_dur_idx = dur_options.index(10.0) if 10.0 in dur_options else 0
    win_dur = st.selectbox("Duração da janela (s):", dur_options, index=default_dur_idx)

    max_start = max(0.0, result.ecg.duration_s - float(win_dur))
    default_start = min(max_start, suggested_start_s or 0.0)

    force_mode = st.session_state.pop("force_start_s", None)
    if force_mode == "auto_noisy":
        default_start = min(max_start, find_best_noisy_window(result, float(win_dur)))
    elif force_mode == "auto_brady":
        default_start = min(max_start, find_bradycardia_window(result, float(win_dur)))
    elif force_mode == "auto_bigeminy":
        default_start = min(max_start, find_arrhythmia_bigeminy_window(result, float(win_dur)))
    elif force_mode == "auto_tachy":
        default_start = min(max_start, find_fastest_tachycardia_window(result, float(win_dur)))
    elif result.ecg.record_id == "105":
        default_start = min(max_start, find_best_noisy_window(result, float(win_dur)))
    elif result.ecg.record_id == "231":
        default_start = min(max_start, find_bradycardia_window(result, float(win_dur)))
    elif result.ecg.record_id == "119":
        default_start = min(max_start, find_arrhythmia_bigeminy_window(result, float(win_dur)))
    elif result.ecg.record_id == "209":
        default_start = min(max_start, find_fastest_tachycardia_window(result, float(win_dur)))

    slider_key = f"win_start_{result.ecg.source_name}_{int(win_dur)}_{st.session_state.get('jump_nonce', 0)}"
    win_start = (
        st.slider("Início da janela (s):", 0.0, float(max_start), float(round(default_start, 0)), 1.0, key=slider_key)
        if max_start > 0
        else 0.0
    )
    show_atr_overlay = False
    if result.ecg.annotation_samples is not None:
        show_atr_overlay = st.checkbox("Sobrepor anotações de cardiologistas (.atr)", value=False)
    scope_metrics = st.radio(
        "Escopo das métricas no painel:",
        ["Janela visível", "Exame completo"],
        index=0 if result.ecg.duration_s > 60 else 1,
    )

# Métricas exibidas (janela visível ou exame completo)
s_idx = max(0, int(round(win_start * result.ecg.fs)))
e_idx = min(result.ecg.n_samples, s_idx + int(round(win_dur * result.ecg.fs)))
win_peaks = result.r_peaks[(result.r_peaks >= s_idx) & (result.r_peaks < e_idx)]
win_metrics = compute_clinical_metrics(win_peaks, result.ecg.fs)
active_m = win_metrics if scope_metrics == "Janela visível" else result.metrics

# ---------------------------------------------------------------------------
# Faixa de Triagem Clínica + KPIs Principais
# ---------------------------------------------------------------------------

triage_cls = (
    "triage-strip triage-alert"
    if active_m.classification_level == "alert"
    else ("triage-strip triage-warn" if active_m.classification_level == "warning" else "triage-strip")
)
st.markdown(
    f"""
    <div class="{triage_cls}">
        <div>
            <div class="triage-label">Classificação Clínica ({scope_metrics}): {active_m.classification}</div>
            <div class="triage-sub">
                Critério de triagem: &lt; 60 BPM (Bradicardia) · 60 a 120 BPM (Ritmo Normal) · &gt; 120 BPM (Taquicardia)
                &nbsp;|&nbsp; Regularidade: <strong>{active_m.rhythm_regularity}</strong>
            </div>
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #334155; text-align: right;">
            Fonte: {result.ecg.source_name}<br/>
            fs = {result.ecg.fs:.1f} Hz · {result.ecg.duration_s:.1f} s · {active_m.num_beats} batimentos
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="kpi-box">
            <div class="kpi-title">Frequência Cardíaca Média</div>
            <div class="kpi-val">{active_m.bpm_mean:.1f} BPM</div>
            <div class="kpi-detail">Faixa inst.: {active_m.bpm_min:.0f}–{active_m.bpm_max:.0f} BPM</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="kpi-box">
            <div class="kpi-title">Intervalo R-R Médio / Período Discreto</div>
            <div class="kpi-val">{active_m.mean_rr_ms:.1f} ms</div>
            <div class="kpi-detail">N̄ = {active_m.mean_n_samples:.1f} amostras</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="kpi-box">
            <div class="kpi-title">SDNN (Variabilidade / Periodicidade)</div>
            <div class="kpi-val">{active_m.sdnn_ms:.1f} ms</div>
            <div class="kpi-detail">RMSSD = {active_m.rmssd_ms:.1f} ms</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="kpi-box">
            <div class="kpi-title">Energia de Ruído Removida (Parseval)</div>
            <div class="kpi-val">{result.spectral.noise_removed_total_pct:.1f}%</div>
            <div class="kpi-detail">Banda AC (f &gt; 0): {result.spectral.noise_removed_ac_pct:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------------
# Abas Funcionais do Produto
# ---------------------------------------------------------------------------

tab_ecg, tab_fft, tab_rr, tab_val = st.tabs(
    [
        "📈 Traçado ECG (Antes × Depois)",
        "📊 Análise Espectral (FFT & Parseval)",
        "⏱️ Periodicidade Discreta & VFC",
        "✅ Validação MIT-BIH vs. Literatura",
    ]
)

with tab_ecg:
    fig_ecg = build_ecg_plotly_figure(
        result=result,
        start_s=float(win_start),
        duration_s=float(win_dur),
        show_atr=show_atr_overlay,
    )
    st.plotly_chart(fig_ecg, use_container_width=True)

    col_exp1, col_exp2, col_exp3 = st.columns([1.2, 1.2, 2.0])
    with col_exp1:
        # Gera PNG 300 DPI em memória para download imediato
        buf_png = io.BytesIO()
        tmp_png_path = OUTPUTS_DIR / "_preview_export.png"
        plot_ecg_comparison(result, tmp_png_path, start_s=float(win_start), duration_s=float(win_dur), dpi=300)
        png_bytes = tmp_png_path.read_bytes()
        tmp_png_path.unlink(missing_ok=True)
        st.download_button(
            "Baixar figura Antes × Depois (300 DPI)",
            data=png_bytes,
            file_name=f"ecg_comparacao_{result.ecg.record_id or 'importado'}.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_exp2:
        df_win = pd.DataFrame(
            {
                "tempo_s": np.round(np.arange(e_idx - s_idx) / result.ecg.fs, 6),
                "ecg_bruto_mV": np.round(result.ecg.signal[s_idx:e_idx], 6),
                "ecg_filtrado_mV": np.round(result.filtered_signal[s_idx:e_idx], 6),
            }
        )
        st.download_button(
            "Exportar janela atual (.csv)",
            data=df_win.to_csv(index=False).encode("utf-8"),
            file_name=f"ecg_janela_{result.ecg.record_id or 'importado'}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp3:
        st.caption(
            f"Estabilidade BIBO: max |pᵢ| = {result.filter_info.max_pole_radius:.5f} < 1 "
            f"({'Estável' if result.filter_info.is_bibo_stable else 'Instável'}) · "
            "Filtragem bidirecional de fase zero (`filtfilt`) preservando o pico R e o segmento ST."
        )

with tab_fft:
    st.plotly_chart(build_fft_plotly_figure(result), use_container_width=True)

    sp = result.spectral
    col_p1, col_p2 = st.columns([1.15, 1.85])
    with col_p1:
        st.subheader("Teorema de Parseval")
        st.latex(r"P_d = \frac{1}{M}\sum_{n=0}^{M-1} |x[n]|^2 = \sum_{k=0}^{M-1} |c_k|^2")
        parseval_df = pd.DataFrame(
            [
                {
                    "Sinal": "Antes (Bruto)",
                    "P Tempo (mV²)": f"{sp.power_time_raw:.6f}",
                    "P Parseval (mV²)": f"{sp.power_parseval_raw:.6f}",
                    "Erro Relativo": f"{sp.parseval_rel_error_raw:.2e}",
                },
                {
                    "Sinal": "Depois (Filtrado)",
                    "P Tempo (mV²)": f"{sp.power_time_filtered:.6f}",
                    "P Parseval (mV²)": f"{sp.power_parseval_filtered:.6f}",
                    "Erro Relativo": f"{sp.parseval_rel_error_filtered:.2e}",
                },
            ]
        )
        st.dataframe(parseval_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Energia total removida: **{sp.noise_removed_total_pct:.2f}%** | "
            f"Energia dinâmica AC removida (f > 0 Hz): **{sp.noise_removed_ac_pct:.2f}%**."
        )

    with col_p2:
        st.subheader("Atenuação de Potência por Faixa Espectral")
        band_rows = []
        for bname in sp.band_power_raw:
            band_rows.append(
                {
                    "Faixa Espectral": bname,
                    "Antes (mV²)": f"{sp.band_power_raw[bname]:.6f}",
                    "Depois (mV²)": f"{sp.band_power_filtered[bname]:.6f}",
                    "Atenuação (%)": f"{sp.band_attenuation_pct[bname]:.2f}%",
                    "Ganho (dB)": f"{sp.band_attenuation_db[bname]:.1f} dB",
                }
            )
        st.dataframe(pd.DataFrame(band_rows), hide_index=True, use_container_width=True)

with tab_rr:
    st.plotly_chart(build_rr_plotly_figure(result), use_container_width=True)
    st.markdown(
        f"""
        **Interpretação em Sinais e Sistemas:** Um ritmo sinusal regular comporta-se como um sinal
        quase-periódico discreto $x[n] \\approx x[n + N]$. Neste exame completo, o período discreto médio
        encontrado foi **$\\bar{{N}} = {result.metrics.mean_n_samples:.1f}$ amostras** (equivalente a
        **{result.metrics.mean_rr_ms:.1f} ms** a $f_s = {result.ecg.fs:.0f}$ Hz), com desvio padrão
        **SDNN = {result.metrics.sdnn_ms:.1f} ms** ({result.metrics.rhythm_regularity}).
        """
    )

with tab_val:
    if result.ecg.annotation_samples is not None and len(result.ecg.annotation_samples) > 0:
        tp, fp, fn = match_beats_one_to_one(
            result.r_peaks,
            result.ecg.annotation_samples,
            fs=result.ecg.fs,
            tolerance_ms=VALIDATION_TOLERANCE_MS,
        )
        se = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
        ppv = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
        f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0

        st.subheader(f"Validação instantânea do registro carregado ({result.ecg.source_name})")
        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        vc1.metric("Batimentos Reais (.atr)", f"{len(result.ecg.annotation_samples)}")
        vc2.metric("TP / FP / FN", f"{tp} / {fp} / {fn}")
        vc3.metric("Sensibilidade", f"{se:.2f}%")
        vc4.metric("Precisão (+P)", f"{ppv:.2f}%")
        vc5.metric("F1-Score", f"{f1:.2f}%")

    csv_val_path = OUTPUTS_DIR / "tabela_validacao_mitbih.csv"
    st.subheader("Tabela Consolidada de Validação — Base MIT-BIH (±150 ms)")
    if csv_val_path.exists():
        df_val = pd.read_csv(csv_val_path)
        st.dataframe(df_val, hide_index=True, use_container_width=True)
    else:
        st.warning("Execute `python run_validation.py` para gerar a tabela consolidada em `outputs/tabela_validacao_mitbih.csv`.")

    st.subheader("Comparativo com a Literatura de Referência (Relatório 1)")
    st.dataframe(pd.DataFrame(LITERATURE_BENCHMARKS), hide_index=True, use_container_width=True)
