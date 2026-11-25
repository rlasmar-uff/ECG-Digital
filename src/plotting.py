"""Geração de gráficos científicos (300 DPI) em papel milimetrado e espectros FFT/Parseval."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from src.config import COURSE_INFO, INSTITUTION, TEAM_INLINE
from src.ecg_pipeline import (
    PipelineResult,
    compute_cascade_frequency_response,
    compute_clinical_metrics,
)

# Paleta clínica sóbria e de alto contraste (papel milimetrado de ECG autêntico)
ECG_PAPER_BG = "#FFFDFB"
ECG_GRID_MINOR = "#F5D0CE"
ECG_GRID_MAJOR = "#E5989B"
TRACE_RAW_COLOR = "#4A5568"
TRACE_FILT_COLOR = "#0B2545"
PEAK_COLOR = "#D62828"
ACCENT_TEAL = "#0F766E"
ACCENT_AMBER = "#B45309"


def _configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 0.9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def apply_ecg_millimeter_grid(ax: plt.Axes, t_start: float, t_end: float) -> None:
    """Aplica grade padrão de papel milimetrado de ECG (0,04 s / 0,20 s e 0,1 mV / 0,5 mV)."""
    ax.set_facecolor(ECG_PAPER_BG)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Reduz rótulos numéricos no eixo X para leitura limpa a cada 1 segundo
    ax.set_xticks(np.arange(np.ceil(t_start), np.floor(t_end) + 0.001, 1.0))
    ax.grid(which="major", color=ECG_GRID_MAJOR, linestyle="-", linewidth=0.75, alpha=0.85)
    ax.grid(which="minor", color=ECG_GRID_MINOR, linestyle="-", linewidth=0.40, alpha=0.65)
    ax.set_axisbelow(True)


def find_best_noisy_window(
    result: PipelineResult,
    window_duration_s: float = 10.0,
) -> float:
    """Localiza automaticamente uma janela de 10 s com contraste claro e detecção fiel de picos R."""
    fs = result.ecg.fs
    raw = result.ecg.signal
    filt = result.filtered_signal
    win_len = int(round(window_duration_s * fs))
    step = int(round(2.0 * fs))

    best_start = 0
    best_score = -1.0
    n = len(raw)
    ann = result.ecg.annotation_samples

    for s in range(0, max(1, n - win_len), step):
        e = s + win_len
        r_seg = raw[s:e]
        f_seg = filt[s:e]
        noise_diff = r_seg - f_seg
        baseline_drift = float(np.ptp(noise_diff))
        filt_amp = float(np.percentile(f_seg, 99) - np.percentile(f_seg, 1))
        peaks_in = result.r_peaks[(result.r_peaks >= s) & (result.r_peaks < e)]

        if len(peaks_in) < 10 or filt_amp < 1.0 or filt_amp > 4.2:
            continue
        if baseline_drift < 1.0 or baseline_drift > 3.8:
            continue

        # Se houver anotações .atr, exige 100% de correspondência na janela exibida
        if ann is not None:
            ann_in = ann[(ann >= s + int(0.15 * fs)) & (ann < e - int(0.15 * fs))]
            det_in = peaks_in[(peaks_in >= s + int(0.15 * fs)) & (peaks_in < e - int(0.15 * fs))]
            if abs(len(det_in) - len(ann_in)) > 0:
                continue

        hf_raw = float(np.std(np.diff(r_seg)))
        hf_filt = float(np.std(np.diff(f_seg)))
        hf_gain = max(0.0, hf_raw - hf_filt)
        score = baseline_drift * 0.9 + hf_gain * 24.0
        if score > best_score:
            best_score = score
            best_start = s

    return float(best_start / fs)


def find_bradycardia_window(
    result: PipelineResult,
    window_duration_s: float = 10.0,
) -> float:
    """Localiza uma janela de 10 s com bradicardia clara (< 55 BPM) e traçado limpo."""
    fs = result.ecg.fs
    win_len = int(round(window_duration_s * fs))
    step = int(round(2.0 * fs))
    best_start = 0.0
    best_score = -1.0

    for s in range(0, max(1, result.ecg.n_samples - win_len), step):
        e = s + win_len
        p = result.r_peaks[(result.r_peaks >= s) & (result.r_peaks < e)]
        if 4 <= len(p) <= 9:
            m = compute_clinical_metrics(p, fs)
            if 32.0 <= m.bpm_mean <= 52.0:
                # Prefere bradicardia nítida (~38–46 BPM) sem pausas artificiais
                score = 60.0 - m.bpm_mean
                if score > best_score:
                    best_score = score
                    best_start = s / fs
    return float(best_start)


def find_arrhythmia_bigeminy_window(
    result: PipelineResult,
    window_duration_s: float = 10.0,
) -> float:
    """Localiza janela de 10 s com múltiplos batimentos ectópicos (alto SDNN) no registro 119."""
    fs = result.ecg.fs
    win_len = int(round(window_duration_s * fs))
    step = int(round(2.0 * fs))
    best_start = 60.0
    best_sdnn = -1.0

    for s in range(0, max(1, result.ecg.n_samples - win_len), step):
        e = s + win_len
        p = result.r_peaks[(result.r_peaks >= s) & (result.r_peaks < e)]
        if len(p) >= 9:
            m = compute_clinical_metrics(p, fs)
            if 180.0 <= m.sdnn_ms <= 360.0 and m.sdnn_ms > best_sdnn:
                best_sdnn = m.sdnn_ms
                best_start = s / fs
    return float(best_start)


def plot_ecg_comparison(
    result: PipelineResult,
    output_path: Path,
    start_s: float | None = None,
    duration_s: float = 10.0,
    dpi: int = 300,
) -> float:
    """Gera comparacao_ecg.png (300 DPI): Antes (bruto) vs Depois (filtrado + picos R + painel)."""
    _configure_matplotlib_style()
    fs = result.ecg.fs
    if start_s is None:
        start_s = find_best_noisy_window(result, window_duration_s=duration_s)

    s_idx = max(0, int(round(start_s * fs)))
    e_idx = min(result.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
    raw_seg = result.ecg.signal[s_idx:e_idx]
    filt_seg = result.filtered_signal[s_idx:e_idx]

    win_peaks = result.r_peaks[(result.r_peaks >= s_idx) & (result.r_peaks < e_idx)]
    win_metrics = compute_clinical_metrics(win_peaks, fs)
    full_metrics = result.metrics

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.8, 8.8), sharex=True, gridspec_kw={"hspace": 0.28}
    )

    # Painel superior: Sinal Bruto
    apply_ecg_millimeter_grid(ax1, t[0], t[-1])
    ax1.plot(t, raw_seg, color=TRACE_RAW_COLOR, linewidth=1.05, label="Sinal bruto x[n]")
    raw_pad = max(0.35, 0.14 * float(np.ptp(raw_seg)))
    ax1.set_ylim(float(np.min(raw_seg)) - raw_pad, float(np.max(raw_seg)) + raw_pad)
    ax1.set_ylabel("Amplitude (mV)", fontweight="bold")
    ax1.set_title(
        f"ANTES — Sinal Bruto do MIT-BIH (Registro {result.ecg.record_id or 'ECG'} · "
        f"Derivação {result.ecg.lead_name} · fs = {fs:.0f} Hz)",
        loc="left",
        pad=8,
        color="#1E293B",
    )
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    # Painel inferior: Sinal Filtrado + Picos R
    apply_ecg_millimeter_grid(ax2, t[0], t[-1])
    ax2.plot(
        t,
        filt_seg,
        color=TRACE_FILT_COLOR,
        linewidth=1.3,
        label="Sinal filtrado (Notch 60 Hz Q=30 + Butterworth 0,5–40 Hz, Fase Zero)",
    )
    peak_max_y = float(np.max(filt_seg))
    if len(win_peaks) > 0:
        peak_t = win_peaks / fs
        peak_v = result.filtered_signal[win_peaks]
        peak_max_y = float(max(peak_max_y, np.max(peak_v)))
        ax2.scatter(
            peak_t,
            peak_v,
            color=PEAK_COLOR,
            s=46,
            zorder=5,
            edgecolors="white",
            linewidths=0.7,
            label=f"Picos R detectados (n = {len(win_peaks)} na janela)",
        )

        # Destaca um intervalo R-R livre no centro-direita com o período discreto N e ms
        if len(win_peaks) >= 4:
            k = len(win_peaks) // 2
            p0, p1 = int(win_peaks[k]), int(win_peaks[k + 1])
            t0, t1 = p0 / fs, p1 / fs
            y_bar = float(max(result.filtered_signal[p0], result.filtered_signal[p1])) + 0.24
            ax2.annotate(
                "",
                xy=(t1, y_bar),
                xytext=(t0, y_bar),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_TEAL, lw=1.4),
                zorder=6,
            )
            rr_sample = p1 - p0
            rr_ms_one = (rr_sample / fs) * 1000.0
            ax2.text(
                0.5 * (t0 + t1),
                y_bar + 0.08,
                f"N = {rr_sample} amostras ({rr_ms_one:.0f} ms)",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
                color=ACCENT_TEAL,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ACCENT_TEAL, alpha=0.95),
                zorder=7,
            )

    filt_span = max(1.2, float(np.ptp(filt_seg)))
    # Reserva folga limpa acima dos picos R para o painel de triagem e seta R-R sem nenhuma sobreposição
    ax2.set_ylim(float(np.min(filt_seg)) - 0.35 * filt_span, peak_max_y + 0.62 * filt_span)
    ax2.set_ylabel("Amplitude (mV)", fontweight="bold")
    ax2.set_xlabel("Tempo (s)  —  Grade padrão ECG: 0,04 s / 0,20 s (horizontal) · 0,1 mV / 0,5 mV (vertical)", fontweight="bold")
    ax2.set_title(
        "DEPOIS — Sinal Filtrado por Sistemas LIT de Fase Zero + Detecção de Picos R (Pan-Tompkins / Hamilton-Tompkins)",
        loc="left",
        pad=8,
        color="#0B2545",
    )
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.7)

    # Faixa horizontal de triagem no topo do painel inferior (acima de todos os picos R, sem obstrução)
    diag_color = "#15803D" if full_metrics.classification == "Ritmo Normal" else "#B91C1C"
    metrics_strip = (
        f"PAINEL DE TRIAGEM (Reg. {result.ecg.record_id or 'ECG'} · 30 min):   "
        f"BPM Médio: {full_metrics.bpm_mean:.1f} BPM (janela: {win_metrics.bpm_mean:.1f})   |   "
        f"R-R Médio: {full_metrics.mean_rr_ms:.1f} ms (N̄ = {full_metrics.mean_n_samples:.1f} amostras)   |   "
        f"SDNN: {full_metrics.sdnn_ms:.1f} ms   |   "
        f"Diagnóstico (60–120 BPM): {full_metrics.classification.upper()}"
    )
    ax2.text(
        0.5,
        0.965,
        metrics_strip,
        transform=ax2.transAxes,
        va="top",
        ha="center",
        fontsize=8.35,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor="#FFFFFF",
            edgecolor=diag_color,
            linewidth=1.45,
            alpha=0.97,
        ),
        zorder=8,
    )

    fig.suptitle(
        "Processamento Digital de ECG: Filtragem LIT de Fase Zero e Detecção Automática de Picos R",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        f"{INSTITUTION} · {COURSE_INFO}\nEquipe: {TEAM_INLINE}",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#475569",
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.115)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return start_s


def plot_fft_parseval(
    result: PipelineResult,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Gera espectro_fft.png (300 DPI): FFT Antes x Depois + destaques (<0,5 Hz, >40 Hz, 60 Hz) + Parseval."""
    _configure_matplotlib_style()
    sp = result.spectral
    freqs = sp.freqs_hz

    # Limita a visualização principal até 100 Hz (Nyquist = 180 Hz) para máxima legibilidade
    mask_main = (freqs > 0.0) & (freqs <= 100.0)
    f_plot = freqs[mask_main]
    # Suavização visual leve apenas em bins densos (registro 30 min tem 325k bins FFT)
    # Fazemos binning por envoltória máxima/média a cada 0,1 Hz para preservar o vale exato em 60 Hz!
    bin_step = 0.10
    bin_edges = np.arange(0.05, 100.05, bin_step)
    bin_idx = np.digitize(f_plot, bin_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(bin_edges) - 1)

    f_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    raw_pwr = sp.mag_raw[mask_main] ** 2
    flt_pwr = sp.mag_filtered[mask_main] ** 2

    raw_binned = np.zeros(len(f_centers), dtype=np.float64)
    flt_binned = np.zeros(len(f_centers), dtype=np.float64)
    counts = np.zeros(len(f_centers), dtype=np.float64)
    np.add.at(raw_binned, bin_idx[valid], raw_pwr[valid])
    np.add.at(flt_binned, bin_idx[valid], flt_pwr[valid])
    np.add.at(counts, bin_idx[valid], 1.0)
    nz = counts > 0
    raw_binned[nz] /= counts[nz]
    flt_binned[nz] /= counts[nz]

    eps = 1e-13
    raw_db = 10.0 * np.log10(np.maximum(raw_binned, eps))
    flt_db = 10.0 * np.log10(np.maximum(flt_binned, eps))

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.34, wspace=0.22)

    ax_main = fig.add_subplot(gs[0, :])
    ax_low = fig.add_subplot(gs[1, 0])
    ax_notch = fig.add_subplot(gs[1, 1])

    # --- Painel principal (0,1 a 100 Hz) ---
    ax_main.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.30, label="Rejeição < 0,5 Hz (respiração / linha de base)")
    ax_main.axvspan(0.5, 40.0, color="#BBF7D0", alpha=0.20, label="Banda passante ECG (0,5–40 Hz)")
    ax_main.axvspan(40.0, 100.0, color="#FDE68A", alpha=0.25, label="Rejeição > 40 Hz (tremor muscular / EMG)")
    ax_main.axvline(60.0, color="#DC2626", linestyle="--", linewidth=1.4, label="Filtro Notch 60 Hz (rede elétrica, Q=30)")

    ax_main.plot(f_centers[nz], raw_db[nz], color="#718096", lw=1.15, alpha=0.90, label="FFT Antes (Sinal Bruto)")
    ax_main.plot(f_centers[nz], flt_db[nz], color=TRACE_FILT_COLOR, lw=1.55, label="FFT Depois (Sinal Filtrado)")

    ax_main.set_xlim(0, 100)
    ax_main.set_ylim(-125, max(float(np.max(raw_db[nz])) + 6, -15))
    ax_main.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_main.set_ylabel("Densidade Espectral de Potência (dB re mV²)", fontweight="bold")
    ax_main.set_title(
        f"Comparativo Espectral (FFT) — MIT-BIH Registro {result.ecg.record_id or 'ECG'} "
        f"(fs = {result.ecg.fs:.0f} Hz)",
        loc="left",
        color="#0F172A",
    )
    ax_main.grid(True, linestyle=":", alpha=0.55)
    ax_main.legend(loc="upper right", ncol=2, fontsize=8.3, frameon=True, facecolor="white", framealpha=0.95)

    # Caixa do Teorema de Parseval
    parseval_text = (
        "TEOREMA DE PARSEVAL  (P = (1/M) Σ|x[n]|² = Σ|cₖ|²)\n"
        f"• Potência Total Antes:  {sp.power_parseval_raw:.5f} mV²  (erro id.: {sp.parseval_rel_error_raw:.1e})\n"
        f"• Potência Total Depois: {sp.power_parseval_filtered:.5f} mV²  (erro id.: {sp.parseval_rel_error_filtered:.1e})\n"
        f"• Energia Total Removida (incl. DC/linha de base): {sp.noise_removed_total_pct:.2f}%\n"
        f"• Energia Dinâmica AC Removida (f > 0 Hz): {sp.noise_removed_ac_pct:.2f}%\n"
        f"• Atenuação por banda: <0,5 Hz: {sp.band_attenuation_pct['Linha de Base (< 0,5 Hz)']:.1f}%  |  "
        f"60 Hz: {sp.band_attenuation_pct['Rede Elétrica (59–61 Hz)']:.1f}%  |  "
        f">40 Hz: {sp.band_attenuation_pct['Alta Frequência (> 40 Hz)']:.1f}%"
    )
    ax_main.text(
        0.012,
        0.05,
        parseval_text,
        transform=ax_main.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.3,
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=ACCENT_TEAL, lw=1.35, alpha=0.95),
    )

    # --- Zoom inferior esquerdo: Baixa Frequência (0 a 5 Hz) ---
    mask_low = (freqs > 0.0) & (freqs <= 5.0)
    # Reagrupa em passo fino de 0,02 Hz
    edges_l = np.arange(0.01, 5.01, 0.025)
    idx_l = np.digitize(freqs[mask_low], edges_l) - 1
    v_l = (idx_l >= 0) & (idx_l < len(edges_l) - 1)
    fc_l = 0.5 * (edges_l[:-1] + edges_l[1:])
    rb_l = np.zeros_like(fc_l)
    fb_l = np.zeros_like(fc_l)
    ct_l = np.zeros_like(fc_l)
    np.add.at(rb_l, idx_l[v_l], (sp.mag_raw[mask_low] ** 2)[v_l])
    np.add.at(fb_l, idx_l[v_l], (sp.mag_filtered[mask_low] ** 2)[v_l])
    np.add.at(ct_l, idx_l[v_l], 1.0)
    nz_l = ct_l > 0
    rb_l[nz_l] /= ct_l[nz_l]
    fb_l[nz_l] /= ct_l[nz_l]

    ax_low.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.35)
    ax_low.axvline(0.5, color="#DC2626", linestyle="--", lw=1.1)
    ax_low.plot(fc_l[nz_l], 10 * np.log10(np.maximum(rb_l[nz_l], eps)), color=TRACE_RAW_COLOR, lw=1.1, label="Antes")
    ax_low.plot(fc_l[nz_l], 10 * np.log10(np.maximum(fb_l[nz_l], eps)), color=TRACE_FILT_COLOR, lw=1.35, label="Depois")
    ax_low.set_xlim(0.0, 5.0)
    ax_low.set_title("Zoom: Atenuação da Linha de Base (< 0,5 Hz)", loc="left", fontsize=10.5)
    ax_low.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_low.set_ylabel("Potência (dB re mV²)", fontweight="bold")
    ax_low.grid(True, linestyle=":", alpha=0.55)
    ax_low.legend(loc="lower right", fontsize=8.5)

    # --- Zoom inferior direito: Rede Elétrica (55 a 65 Hz) ---
    mask_n = (freqs >= 55.0) & (freqs <= 65.0)
    edges_n = np.arange(55.0, 65.05, 0.05)
    idx_n = np.digitize(freqs[mask_n], edges_n) - 1
    v_n = (idx_n >= 0) & (idx_n < len(edges_n) - 1)
    fc_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    rb_n = np.zeros_like(fc_n)
    fb_n = np.zeros_like(fc_n)
    ct_n = np.zeros_like(fc_n)
    np.add.at(rb_n, idx_n[v_n], (sp.mag_raw[mask_n] ** 2)[v_n])
    np.add.at(fb_n, idx_n[v_n], (sp.mag_filtered[mask_n] ** 2)[v_n])
    np.add.at(ct_n, idx_n[v_n], 1.0)
    nz_n = ct_n > 0
    rb_n[nz_n] /= ct_n[nz_n]
    fb_n[nz_n] /= ct_n[nz_n]

    ax_notch.axvspan(59.0, 61.0, color="#FCA5A5", alpha=0.30)
    ax_notch.axvline(60.0, color="#DC2626", linestyle="--", lw=1.2)
    ax_notch.plot(fc_n[nz_n], 10 * np.log10(np.maximum(rb_n[nz_n], eps)), color=TRACE_RAW_COLOR, lw=1.1, label="Antes")
    ax_notch.plot(fc_n[nz_n], 10 * np.log10(np.maximum(fb_n[nz_n], eps)), color=TRACE_FILT_COLOR, lw=1.35, label="Depois")
    ax_notch.set_xlim(55.0, 65.0)
    ax_notch.set_title("Zoom: Rejeição da Rede Elétrica (Notch 60 Hz, Q = 30)", loc="left", fontsize=10.5)
    ax_notch.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_notch.set_ylabel("Potência (dB re mV²)", fontweight="bold")
    ax_notch.grid(True, linestyle=":", alpha=0.55)
    ax_notch.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Análise Espectral (Transformada de Fourier) e Conservação/Atenuação de Energia (Parseval)",
        fontsize=13.2,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        f"{INSTITUTION} · {COURSE_INFO}\nEquipe: {TEAM_INLINE}",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#475569",
    )

    fig.subplots_adjust(left=0.075, right=0.975, top=0.91, bottom=0.115)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def find_fastest_tachycardia_window(
    result: PipelineResult,
    window_duration_s: float = 10.0,
) -> float:
    """Localiza a janela de 10 s com maior frequência cardíaca sustentada (> 120 BPM)."""
    fs = result.ecg.fs
    win_len = int(round(window_duration_s * fs))
    step = int(round(2.0 * fs))
    best_start = 0.0
    best_bpm = -1.0

    for s in range(0, max(1, result.ecg.n_samples - win_len), step):
        e = s + win_len
        p = result.r_peaks[(result.r_peaks >= s) & (result.r_peaks < e)]
        if len(p) >= 12:
            m = compute_clinical_metrics(p, fs)
            if m.bpm_mean > best_bpm and m.bpm_mean < 240.0:
                best_bpm = m.bpm_mean
                best_start = s / fs
    return float(best_start)


def plot_clinical_anomalies_panel(
    cases: list[tuple[str, PipelineResult, float, float]],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Gera figura 300 DPI com casos reais de anomalias (Bradicardia, Arritmia/PVCs e Taquicardia)."""
    _configure_matplotlib_style()
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(13.5, 3.1 * n_cases + 1.1), gridspec_kw={"hspace": 0.42})
    if n_cases == 1:
        axes = [axes]

    for ax, (title, res, start_s, dur_s) in zip(axes, cases):
        fs = res.ecg.fs
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = min(res.ecg.n_samples, s_idx + int(round(dur_s * fs)))
        t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
        sig = res.filtered_signal[s_idx:e_idx]
        peaks = res.r_peaks[(res.r_peaks >= s_idx) & (res.r_peaks < e_idx)]
        m = compute_clinical_metrics(peaks, fs)

        apply_ecg_millimeter_grid(ax, t[0], t[-1])
        ax.plot(t, sig, color=TRACE_FILT_COLOR, lw=1.25)
        if len(peaks) > 0:
            ax.scatter(
                peaks / fs,
                res.filtered_signal[peaks],
                color=PEAK_COLOR,
                s=38,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )

        pad = max(0.45, 0.28 * float(np.ptp(sig)))
        ax.set_ylim(float(np.min(sig)) - pad * 0.65, float(np.max(sig)) + pad * 1.45)
        ax.set_ylabel("mV", fontweight="bold")
        ax.set_title(
            f"{title}  —  MIT-BIH Registro {res.ecg.record_id} ({start_s:.0f}s–{start_s + dur_s:.0f}s)",
            loc="left",
            fontsize=10.8,
            color="#0F172A",
        )

        badge_color = "#15803D" if (m.classification == "Ritmo Normal" and m.sdnn_ms <= 100.0) else (
            "#B45309" if m.classification == "Ritmo Normal" else "#B91C1C"
        )
        extra_flag = " · ALTA VARIABILIDADE (ARRITMIA)" if (m.classification == "Ritmo Normal" and m.sdnn_ms > 100.0) else ""
        ax.text(
            0.992,
            0.94,
            f"BPM: {m.bpm_mean:.1f}  |  R-R: {m.mean_rr_ms:.0f} ms (N̄={m.mean_n_samples:.0f})  |  "
            f"SDNN: {m.sdnn_ms:.1f} ms  |  {m.classification.upper()}{extra_flag}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            color=badge_color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=badge_color, lw=1.35, alpha=0.96),
        )

    axes[-1].set_xlabel("Tempo (s) — Escala padrão de papel milimetrado ECG", fontweight="bold")
    fig.suptitle(
        "Detecção e Triagem em Anomalias Reais do MIT-BIH: Bradicardia, Arritmia Ventricular e Taquicardia",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        f"{INSTITUTION} · {COURSE_INFO}\nEquipe: {TEAM_INLINE}",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#475569",
    )
    fig.subplots_adjust(left=0.065, right=0.98, top=0.92, bottom=0.09)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_lti_filter_response(output_path: Path, fs: float = 360.0, dpi: int = 300) -> None:
    """Gera diagrama de resposta em frequência |H(e^{jw})| e plano Z (estabilidade BIBO)."""
    _configure_matplotlib_style()
    resp = compute_cascade_frequency_response(fs=fs)
    freqs = resp["freqs_hz"]
    poles = resp["poles"]
    diag = resp["diagnostics"]

    fig, (ax_mag, ax_z) = plt.subplots(
        1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.24}
    )

    ax_mag.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.30)
    ax_mag.axvspan(0.5, 40.0, color="#BBF7D0", alpha=0.22)
    ax_mag.axvspan(40.0, 100.0, color="#FDE68A", alpha=0.25)
    ax_mag.axvline(60.0, color="#DC2626", linestyle="--", lw=1.2)
    ax_mag.axhline(-3.0, color="#64748B", linestyle=":", lw=1.0, label="Corte −3 dB")

    ax_mag.plot(freqs, resp["mag_notch_db"], color="#DC2626", lw=1.1, alpha=0.75, label="Notch 60 Hz (Q = 30)")
    ax_mag.plot(freqs, resp["mag_bp_db"], color=ACCENT_TEAL, lw=1.2, alpha=0.80, label="Butterworth 0,5–40 Hz (ordem 4)")
    ax_mag.plot(
        freqs,
        resp["mag_cascade_zero_phase_db"],
        color=TRACE_FILT_COLOR,
        lw=2.0,
        label=r"Cascata Bidirecional Fase Zero $|H(e^{j\omega})|^2$",
    )
    ax_mag.set_xlim(0, 100)
    ax_mag.set_ylim(-85, 5)
    ax_mag.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_mag.set_ylabel("Magnitude (dB)", fontweight="bold")
    ax_mag.set_title("Resposta em Frequência da Cascata LIT (Fase Zero: ∠H = 0°)", loc="left")
    ax_mag.grid(True, linestyle=":", alpha=0.55)
    ax_mag.legend(loc="lower left", fontsize=8.3, frameon=True, facecolor="white")

    # Diagrama de polos no Plano Z (critério de estabilidade BIBO: |p_i| < 1)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax_z.plot(np.cos(theta), np.sin(theta), color="#334155", linestyle="--", lw=1.2, label="Círculo Unitário |z| = 1")
    ax_z.scatter(np.real(poles), np.imag(poles), marker="x", s=70, color="#DC2626", lw=1.8, label="Polos do sistema LIT")
    ax_z.axhline(0, color="#94A3B8", lw=0.7)
    ax_z.axvline(0, color="#94A3B8", lw=0.7)
    ax_z.set_aspect("equal", adjustable="box")
    ax_z.set_xlim(-1.25, 1.25)
    ax_z.set_ylim(-1.25, 1.25)
    ax_z.set_xlabel("Re(z)", fontweight="bold")
    ax_z.set_ylabel("Im(z)", fontweight="bold")
    ax_z.set_title("Estabilidade BIBO no Plano Z", loc="left")
    ax_z.grid(True, linestyle=":", alpha=0.55)
    ax_z.legend(loc="lower left", fontsize=8.2, frameon=True, facecolor="white")
    ax_z.text(
        0.04,
        0.95,
        f"max |pᵢ| = {diag.max_pole_radius:.5f} < 1\nSistema BIBO Estável",
        transform=ax_z.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color="#15803D",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#15803D", alpha=0.92),
    )

    fig.suptitle(
        "Caracterização dos Filtros Digitais LIT Projetados (Domínio da Frequência e Plano Z)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        f"{INSTITUTION} · {COURSE_INFO}\nEquipe: {TEAM_INLINE}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.065, right=0.975, top=0.87, bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
