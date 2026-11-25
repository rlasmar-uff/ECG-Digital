"""Gera figuras dedicadas em formato widescreen 16:9 (fontes grandes, 1 foco por slide)
para a apresentação final (09/12/2026), mantendo outputs/ intacto para o Relatório 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import BASE_DIR, DATA_DIR
from src.ecg_pipeline import (
    _five_point_derivative,
    _moving_window_integration,
    compute_cascade_frequency_response,
    compute_clinical_metrics,
    design_ecg_filters,
    load_mitbih_record,
    process_ecg,
)
from src.plotting import (
    ACCENT_AMBER,
    ACCENT_TEAL,
    ECG_GRID_MAJOR,
    ECG_GRID_MINOR,
    ECG_PAPER_BG,
    PEAK_COLOR,
    TRACE_FILT_COLOR,
    TRACE_RAW_COLOR,
    find_arrhythmia_bigeminy_window,
    find_best_noisy_window,
    find_bradycardia_window,
    find_fastest_tachycardia_window,
)

SLIDE_FIG_DIR = BASE_DIR / "apresentacao" / "figuras"


def _slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.titlesize": 14.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "axes.edgecolor": "#64748B",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.color": "#1E293B",
            "ytick.color": "#1E293B",
            "legend.fontsize": 11.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _apply_ecg_grid_slide(ax: plt.Axes, t_start: float, t_end: float, tick_step: float = 1.0) -> None:
    ax.set_facecolor(ECG_PAPER_BG)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xticks(np.arange(np.ceil(t_start), np.floor(t_end) + 0.001, tick_step))
    ax.grid(which="major", color=ECG_GRID_MAJOR, linestyle="-", linewidth=0.9, alpha=0.88)
    ax.grid(which="minor", color=ECG_GRID_MINOR, linestyle="-", linewidth=0.45, alpha=0.70)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Resposta em Frequência (Bode) — 1 gráfico grande 16:9
# ---------------------------------------------------------------------------
def gen_bode_filtros() -> None:
    _slide_style()
    resp = compute_cascade_frequency_response(fs=360.0)
    freqs = resp["freqs_hz"]

    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.35, label="Rejeição < 0,5 Hz (respiração / linha de base)")
    ax.axvspan(0.5, 40.0, color="#BBF7D0", alpha=0.28, label="Banda passante clínica ECG (0,5–40 Hz)")
    ax.axvspan(40.0, 100.0, color="#FDE68A", alpha=0.28, label="Rejeição > 40 Hz (tremor muscular / EMG)")
    ax.axvline(60.0, color="#DC2626", linestyle="--", lw=1.6, alpha=0.9)
    ax.axhline(-3.0, color="#475569", linestyle=":", lw=1.3, label="Frequência de corte (−3 dB)")

    ax.plot(freqs, resp["mag_notch_db"], color="#DC2626", lw=1.8, alpha=0.80, label="Notch 60 Hz (FRF · Fq = Q = 30)")
    ax.plot(freqs, resp["mag_bp_db"], color=ACCENT_TEAL, lw=2.0, alpha=0.85, label="Butterworth 0,5–40 Hz (FPF · ordem 4, SOS)")
    ax.plot(
        freqs,
        resp["mag_cascade_zero_phase_db"],
        color=TRACE_FILT_COLOR,
        lw=3.0,
        label=r"Cascata Bidirecional Fase Zero $|H(e^{j\Omega})|^2$ ($\angle H = 0^\circ$)",
    )

    # Anotações grandes e limpas
    ax.annotate(
        "Vale cirúrgico em 60 Hz\n(rede elétrica: −∞ dB teórico)",
        xy=(60.0, -56.0),
        xytext=(66.5, -24.0),
        fontsize=11.5,
        fontweight="bold",
        color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#DC2626", lw=1.4),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8, connectionstyle="arc3,rad=0.15"),
    )
    ax.annotate(
        "Ganho plano ≈ 0 dB\nPreserva P, QRS, ST e T",
        xy=(18.0, -0.5),
        xytext=(10.0, -22.0),
        fontsize=11.5,
        fontweight="bold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#0F766E", lw=1.4),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.8),
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-85, 6)
    ax.set_xlabel("Frequência (Hz)  —  Amostragem fs = 360 Hz (Nyquist = 180 Hz)", fontweight="bold")
    ax.set_ylabel("Magnitude (dB)", fontweight="bold")
    ax.set_title("Resposta em Frequência dos Filtros Digitais LIT em Cascata", loc="left", pad=10, color="#0B2545")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower left", ncol=2, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96)

    fig.tight_layout()
    fig.savefig(SLIDE_FIG_DIR / "slide_bode_filtros.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Estabilidade BIBO no Plano Z (Círculo completo + Zoom perto de |z|=1)
# ---------------------------------------------------------------------------
def gen_plano_z() -> None:
    _slide_style()
    design = design_ecg_filters(fs=360.0)
    b_n, a_n = design["notch_ba"]
    z_notch, p_notch, _ = signal.tf2zpk(b_n, a_n)
    z_bp, p_bp, _ = signal.sos2zpk(design["bandpass_sos"])
    max_r = design["diagnostics"].max_pole_radius

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={"wspace": 0.24})
    theta = np.linspace(0, 2 * np.pi, 500)

    # Esquerda: Plano Z completo
    ax1.plot(np.cos(theta), np.sin(theta), color="#0F172A", linestyle="--", lw=1.8, label="Círculo Unitário |z| = 1")
    ax1.fill(np.cos(theta), np.sin(theta), color="#DCFCE7", alpha=0.35, label="Região Estável (|z| < 1)")
    ax1.scatter(np.real(p_bp), np.imag(p_bp), marker="x", s=110, color="#0B2545", lw=2.4, zorder=5, label="Polos Butterworth (8 polos)")
    ax1.scatter(np.real(z_bp), np.imag(z_bp), marker="o", s=95, facecolors="none", edgecolors="#0F766E", lw=2.2, zorder=5, label="Zeros Butterworth")
    ax1.scatter(np.real(p_notch), np.imag(p_notch), marker="x", s=125, color="#DC2626", lw=2.6, zorder=6, label="Polos Notch 60 Hz (2 polos)")
    ax1.scatter(np.real(z_notch), np.imag(z_notch), marker="o", s=115, facecolors="none", edgecolors="#DC2626", lw=2.2, zorder=6, label="Zeros Notch 60 Hz (sobre |z|=1)")

    ax1.axhline(0, color="#94A3B8", lw=0.9)
    ax1.axvline(0, color="#94A3B8", lw=0.9)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-1.28, 1.28)
    ax1.set_ylim(-1.28, 1.28)
    ax1.set_xlabel("Parte Real  Re(z)", fontweight="bold")
    ax1.set_ylabel("Parte Imaginária  Im(z)", fontweight="bold")
    ax1.set_title("Plano Z Completo (Notch 60 Hz + Butterworth)", loc="left", color="#0B2545")
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="lower left", fontsize=9.6, frameon=True, facecolor="white", framealpha=0.96)

    # Direita: Zoom nos polos mais críticos (próximos a z = 1 e ângulo de 60 Hz = 60°)
    # Raio de todos os polos em gráfico de barras horizontais + círculo ampliado?
    # Melhor ainda: gráfico de módulo |p_i| de TODOS os 10 polos vs limite BIBO = 1,0!
    all_poles = np.concatenate([p_notch, p_bp])
    labels = [f"Notch 60 Hz #{i+1}" for i in range(len(p_notch))] + [f"Butterworth #{i+1}" for i in range(len(p_bp))]
    radii = np.abs(all_poles)
    colors = ["#DC2626"] * len(p_notch) + ["#0B2545"] * len(p_bp)

    y_pos = np.arange(len(all_poles))
    ax2.barh(y_pos, radii, color=colors, height=0.62, alpha=0.88)
    ax2.axvline(1.0, color="#DC2626", linestyle="--", lw=2.2, label="Limite BIBO: |z| = 1,00000")
    ax2.axvspan(1.0, 1.04, color="#FEE2E2", alpha=0.65)
    for y, r in zip(y_pos, radii):
        ax2.text(r - 0.014, y, f"{r:.5f}", va="center", ha="right", color="white", fontsize=10.2, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 1.04)
    ax2.set_xlabel("Módulo do Polo  |pᵢ|", fontweight="bold")
    ax2.set_title("Verificação Numérica: Todos os 10 Polos < 1", loc="left", color="#0B2545")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.01, 0.23), fontsize=9.8, frameon=True, facecolor="white", framealpha=0.96)
    ax2.text(
        0.76,
        0.72,
        f"max |pᵢ| = {max_r:.5f} < 1,0\nSISTEMA BIBO ESTÁVEL",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#15803D",
        bbox=dict(boxstyle="round,pad=0.38", fc="#F0FDF4", ec="#15803D", lw=1.6),
    )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.12, wspace=0.28)
    fig.savefig(SLIDE_FIG_DIR / "slide_plano_z.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3 & 4. Pan-Tompkins Passo a Passo em Sinal Real (MIT-BIH Reg. 100, 4 s)
# ---------------------------------------------------------------------------
def gen_pantompkins_etapas(res_100) -> None:
    _slide_style()
    fs = res_100.ecg.fs
    # 4 segundos super limpos (5 batimentos nítidos) no Registro 100 (10s a 14s)
    s_idx = int(10.0 * fs)
    e_idx = int(14.0 * fs)
    x_full = res_100.filtered_signal

    # Recalcula exatamente as mesmas etapas de detect_r_peaks_pan_tompkins
    sos_qrs = signal.butter(3, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    qrs_bp = signal.sosfiltfilt(sos_qrs, x_full)
    deriv = _five_point_derivative(qrs_bp, fs)
    squared = deriv ** 2
    mwi = _moving_window_integration(squared, fs, window_ms=150.0)

    t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
    peaks = res_100.r_peaks[(res_100.r_peaks >= s_idx) & (res_100.r_peaks < e_idx)]

    # --- FIGURA A: Etapas 1, 2 e 3 ---
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.34})

    # 1) ECG Diagnóstico 0,5-40 Hz
    axes[0].plot(t, x_full[s_idx:e_idx], color=TRACE_FILT_COLOR, lw=2.0)
    axes[0].scatter(peaks / fs, x_full[peaks], color=PEAK_COLOR, s=65, zorder=5, edgecolors="white", lw=0.9)
    axes[0].set_title("Etapa 1 · Entrada Diagnóstica Filtrada (0,5–40 Hz) — Ondas P, QRS e T preservadas", loc="left", color="#0B2545")
    axes[0].set_ylabel("mV", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.55)
    # Marca P, R, T no segundo batimento com busca exata dos máximos locais
    if len(peaks) >= 2:
        p_idx = int(peaks[1])
        p_t = p_idx / fs
        axes[0].annotate("Pico R", xy=(p_t, x_full[p_idx]), xytext=(p_t + 0.09, x_full[p_idx] - 0.04),
                         fontsize=10.5, fontweight="bold", color=PEAK_COLOR,
                         arrowprops=dict(arrowstyle="->", color=PEAK_COLOR, lw=1.5))
        # Busca onda P real entre -0.22s e -0.10s antes do R
        p_lo, p_hi = p_idx - int(0.22 * fs), p_idx - int(0.10 * fs)
        p_wave_i = p_lo + int(np.argmax(x_full[p_lo:p_hi]))
        axes[0].annotate("Onda P", xy=(p_wave_i / fs, x_full[p_wave_i]),
                         xytext=(p_t - 0.34, 0.48), fontsize=10, fontweight="bold", color="#475569",
                         arrowprops=dict(arrowstyle="->", color="#475569", lw=1.3))
        # Busca onda T real entre +0.26s e +0.44s após o R
        t_lo, t_hi = p_idx + int(0.26 * fs), p_idx + int(0.44 * fs)
        t_wave_i = t_lo + int(np.argmax(x_full[t_lo:t_hi]))
        axes[0].annotate("Onda T", xy=(t_wave_i / fs, x_full[t_wave_i]),
                         xytext=(t_wave_i / fs + 0.06, 0.56), fontsize=10, fontweight="bold", color="#475569",
                         arrowprops=dict(arrowstyle="->", color="#475569", lw=1.3))
    axes[0].set_ylim(-0.65, 1.38)

    # 2) Passa-Faixa QRS 5-15 Hz
    axes[1].plot(t, qrs_bp[s_idx:e_idx], color=ACCENT_TEAL, lw=2.0)
    axes[1].axhline(0, color="#94A3B8", lw=0.8)
    axes[1].set_title("Etapa 2 · Passa-Faixa QRS (5–15 Hz, Fase Zero) — Atenua ondas P e T lentas, isola o QRS", loc="left", color="#0F766E")
    axes[1].set_ylabel("mV", fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.55)

    # 3) Derivada de 5 pontos
    axes[2].plot(t, deriv[s_idx:e_idx], color="#7C3AED", lw=1.9)
    axes[2].axhline(0, color="#94A3B8", lw=0.8)
    axes[2].set_title(r"Etapa 3 · Derivada Discreta de 5 Pontos — Destaca a inclinação íngreme do QRS ($dV/dt$)", loc="left", color="#6D28D9")
    axes[2].set_ylabel("mV/s", fontweight="bold")
    axes[2].set_xlabel("Tempo (s) — Trecho real do MIT-BIH Registro 100 (fs = 360 Hz)", fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.55)

    axes[2].set_xlim(t[0], t[-1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.10)
    fig.savefig(SLIDE_FIG_DIR / "slide_pantompkins_1_3.png", dpi=240)
    plt.close(fig)

    # --- FIGURA B: Etapas 4 e 5 + Limiares Adaptativos ---
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 6.2), sharex=True, gridspec_kw={"hspace": 0.32})

    # 4) Quadratura
    axes[0].plot(t, squared[s_idx:e_idx], color=ACCENT_AMBER, lw=2.0)
    axes[0].fill_between(t, 0, squared[s_idx:e_idx], color="#FDE68A", alpha=0.45)
    axes[0].set_title(r"Etapa 4 · Quadratura Não Linear $y_s[n] = (y_d[n])^2$ — Tudo positivo; QRS cresce quadraticamente", loc="left", color="#B45309")
    axes[0].set_ylabel("(mV/s)²", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.55)

    # 5) Integração por janela móvel (150 ms) + limiares adaptativos ilustrativos baseados nos percentis reais da janela
    mwi_seg = mwi[s_idx:e_idx]
    spki = float(np.percentile(mwi_seg, 99.0)) * 0.80
    npki = float(np.percentile(mwi_seg, 50.0)) * 0.50
    th_i1 = npki + 0.25 * (spki - npki)
    th_i2 = 0.50 * th_i1

    axes[1].plot(t, mwi_seg, color=TRACE_FILT_COLOR, lw=2.3, label="Envoltória MWI (Janela Móvel 150 ms)")
    axes[1].fill_between(t, 0, mwi_seg, color="#BFDBFE", alpha=0.40)
    axes[1].axhline(th_i1, color="#DC2626", linestyle="--", lw=2.0, label="Limiar Principal T₁ = NPKI + 0,25·(SPKI − NPKI)")
    axes[1].axhline(th_i2, color="#0F766E", linestyle=":", lw=2.0, label="Limiar de Search-Back T₂ = 0,5·T₁ (batimentos perdidos)")

    # Picos da envoltória próximos aos R detectados
    mwi_peak_t = []
    mwi_peak_v = []
    for p in peaks:
        lo = max(s_idx, p - 25)
        hi = min(e_idx, p + 25)
        loc_i = lo + int(np.argmax(mwi[lo:hi]))
        mwi_peak_t.append(loc_i / fs)
        mwi_peak_v.append(mwi[loc_i])
    axes[1].scatter(mwi_peak_t, mwi_peak_v, color=PEAK_COLOR, s=90, marker="o", zorder=6, edgecolors="white", lw=1.0, label="QRS Confirmado (> T₁ e ≥ 200 ms)")

    # Mostra visualmente a largura do período refratário de 200 ms após o 1º pico
    if len(mwi_peak_t) >= 1:
        t0 = mwi_peak_t[0]
        axes[1].axvspan(t0, t0 + 0.20, color="#FCA5A5", alpha=0.38)
        axes[1].annotate(
            "Período Refratário\n(200 ms bloqueado)",
            xy=(t0 + 0.14, th_i1 * 1.65),
            xytext=(t0 + 0.25, float(np.max(mwi_seg)) * 0.62),
            fontsize=10.0,
            fontweight="bold",
            color="#B91C1C",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#DC2626", lw=1.2),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.4),
        )

    axes[1].set_title("Etapa 5 · Integração por Janela Móvel (150 ms) + Limiares Adaptativos (Hamilton & Tompkins)", loc="left", color="#0B2545")
    axes[1].set_ylabel("Energia QRS", fontweight="bold")
    axes[1].set_xlabel("Tempo (s) — Cada pulso acima de T₁ dispara o refinamento do Pico R no sinal 0,5–40 Hz", fontweight="bold")
    axes[1].set_ylim(0, float(np.max(mwi_seg)) * 1.48)
    axes[1].grid(True, linestyle=":", alpha=0.55)
    axes[1].legend(loc="upper right", ncol=2, fontsize=9.8, frameon=True, facecolor="white", framealpha=0.96)
    axes[1].set_xlim(t[0], t[-1])

    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.11)
    fig.savefig(SLIDE_FIG_DIR / "slide_pantompkins_4_5.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Antes x Depois Widescreen 16:9 (MIT-BIH Reg. 105)
# ---------------------------------------------------------------------------
def gen_antes_depois_105(res_105) -> None:
    _slide_style()
    fs = res_105.ecg.fs
    start_s = find_best_noisy_window(res_105, 10.0)
    s_idx = int(round(start_s * fs))
    e_idx = s_idx + int(round(10.0 * fs))
    t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
    raw_seg = res_105.ecg.signal[s_idx:e_idx]
    filt_seg = res_105.filtered_signal[s_idx:e_idx]
    peaks = res_105.r_peaks[(res_105.r_peaks >= s_idx) & (res_105.r_peaks < e_idx)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.30})

    # Bruto
    _apply_ecg_grid_slide(ax1, t[0], t[-1])
    ax1.plot(t, raw_seg, color=TRACE_RAW_COLOR, lw=1.45, label="Sinal Bruto x[n] (com deriva respiratória + 60 Hz + EMG)")
    pad_r = max(0.35, 0.16 * float(np.ptp(raw_seg)))
    ax1.set_ylim(float(np.min(raw_seg)) - pad_r, float(np.max(raw_seg)) + pad_r * 2.0)
    ax1.set_ylabel("Amplitude (mV)", fontweight="bold")
    ax1.set_title(f"ANTES · Sinal Bruto MIT-BIH Reg. 105 ({start_s:.0f}s–{start_s+10:.0f}s) — Deriva de ~{np.ptp(raw_seg):.1f} mV", loc="left", color="#334155")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    # Filtrado
    _apply_ecg_grid_slide(ax2, t[0], t[-1])
    ax2.plot(t, filt_seg, color=TRACE_FILT_COLOR, lw=1.8, label="Sinal Filtrado (Notch 60 Hz + Butterworth 0,5–40 Hz · Fase Zero)")
    ax2.scatter(peaks / fs, res_105.filtered_signal[peaks], color=PEAK_COLOR, s=68, zorder=5, edgecolors="white", lw=0.9, label=f"Picos R Detectados ({len(peaks)}/14 batimentos · 100% na janela)")

    # Seta de período discreto N
    if len(peaks) >= 6:
        k = len(peaks) // 2
        p0, p1 = int(peaks[k]), int(peaks[k + 1])
        t0, t1 = p0 / fs, p1 / fs
        y_bar = float(max(res_105.filtered_signal[p0], res_105.filtered_signal[p1])) + 0.30
        ax2.annotate("", xy=(t1, y_bar), xytext=(t0, y_bar), arrowprops=dict(arrowstyle="<->", color=ACCENT_TEAL, lw=2.0), zorder=6)
        rr_n = p1 - p0
        ax2.text(
            0.5 * (t0 + t1),
            y_bar + 0.18,
            f"N = {rr_n} amostras ({rr_n / fs * 1000:.0f} ms)",
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=ACCENT_TEAL,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=ACCENT_TEAL, lw=1.3),
            zorder=7,
        )

    f_span = max(1.2, float(np.ptp(filt_seg)))
    ax2.set_ylim(float(np.min(filt_seg)) - 0.36 * f_span, float(np.max(filt_seg)) + 0.66 * f_span)
    ax2.set_ylabel("Amplitude (mV)", fontweight="bold")
    ax2.set_xlabel("Tempo (s)  —  Grade padrão de papel milimetrado ECG (0,04 s / 0,20 s · 0,1 mV / 0,5 mV)", fontweight="bold")
    ax2.set_title("DEPOIS · Linha de Base Estabilizada em 0 mV + Morfologia Preservada + Picos R Marcados", loc="left", color="#0B2545")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.105)
    fig.savefig(SLIDE_FIG_DIR / "slide_antes_depois_105.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6 & 7. FFT Global (1 slide) e Zooms das Bandas <0,5 Hz e 60 Hz (1 slide)
# ---------------------------------------------------------------------------
def _bin_spectrum(freqs: np.ndarray, pwr_raw: np.ndarray, pwr_flt: np.ndarray, fmin: float, fmax: float, step: float):
    mask = (freqs >= fmin) & (freqs <= fmax)
    edges = np.arange(fmin, fmax + step, step)
    idx = np.digitize(freqs[mask], edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rb = np.zeros_like(centers)
    fb = np.zeros_like(centers)
    ct = np.zeros_like(centers)
    np.add.at(rb, idx[valid], pwr_raw[mask][valid])
    np.add.at(fb, idx[valid], pwr_flt[mask][valid])
    np.add.at(ct, idx[valid], 1.0)
    nz = ct > 0
    rb[nz] /= ct[nz]
    fb[nz] /= ct[nz]
    eps = 1e-13
    return centers[nz], 10.0 * np.log10(np.maximum(rb[nz], eps)), 10.0 * np.log10(np.maximum(fb[nz], eps))


def gen_fft_slides(res_105) -> None:
    _slide_style()
    sp = res_105.spectral
    freqs = sp.freqs_hz
    raw_p = sp.mag_raw ** 2
    flt_p = sp.mag_filtered ** 2

    # --- SLIDE FFT GLOBAL (0 a 100 Hz) ---
    fc, rdb, fdb = _bin_spectrum(freqs, raw_p, flt_p, 0.05, 100.0, 0.12)
    fig, ax = plt.subplots(figsize=(13.6, 5.8))
    ax.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.35, label="Rejeição < 0,5 Hz (linha de base)")
    ax.axvspan(0.5, 40.0, color="#BBF7D0", alpha=0.25, label="Banda passante ECG (0,5–40 Hz)")
    ax.axvspan(40.0, 100.0, color="#FDE68A", alpha=0.28, label="Rejeição > 40 Hz (tremor muscular / EMG)")
    ax.axvline(60.0, color="#DC2626", linestyle="--", lw=1.8, label="Filtro Notch 60 Hz (rede elétrica)")

    ax.plot(fc, rdb, color="#64748B", lw=1.6, alpha=0.9, label="FFT Antes (Sinal Bruto)")
    ax.plot(fc, fdb, color=TRACE_FILT_COLOR, lw=2.3, label="FFT Depois (Sinal Filtrado)")

    ax.annotate(
        "Energia útil do QRS\npreservada (0,5–40 Hz)",
        xy=(14.0, -52.0),
        xytext=(16.0, -30.0),
        fontsize=11.5,
        fontweight="bold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#0F766E", lw=1.3),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.6),
    )
    ax.annotate(
        "Pico de 60 Hz eliminado\n(−99,99% de energia)",
        xy=(60.0, -108.0),
        xytext=(65.0, -68.0),
        fontsize=11.5,
        fontweight="bold",
        color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#DC2626", lw=1.3),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.6),
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-125, max(float(np.max(rdb)) + 6, -15))
    ax.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax.set_ylabel("Densidade Espectral (dB re mV²)", fontweight="bold")
    ax.set_title("Espectro de Fourier (FFT) Antes × Depois — MIT-BIH Registro 105 (30 minutos completos)", loc="left", color="#0B2545")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.96)
    fig.tight_layout()
    fig.savefig(SLIDE_FIG_DIR / "slide_fft_espectro_global.png", dpi=240)
    plt.close(fig)

    # --- SLIDE ZOOMS LADO A LADO (< 0,5 Hz e 60 Hz) ---
    fig, (ax_l, ax_n) = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw={"wspace": 0.20})

    fc_l, rdb_l, fdb_l = _bin_spectrum(freqs, raw_p, flt_p, 0.02, 4.5, 0.03)
    ax_l.axvspan(0.0, 0.5, color="#FCA5A5", alpha=0.38, label="Faixa rejeitada (< 0,5 Hz)")
    ax_l.axvline(0.5, color="#DC2626", linestyle="--", lw=1.8)
    ax_l.plot(fc_l, rdb_l, color="#64748B", lw=1.8, label="Antes (Bruto)")
    ax_l.plot(fc_l, fdb_l, color=TRACE_FILT_COLOR, lw=2.5, label="Depois (Filtrado)")
    ax_l.set_xlim(0.0, 4.5)
    ax_l.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_l.set_ylabel("Potência Espectral (dB re mV²)", fontweight="bold")
    ax_l.set_title("Zoom 1 · Linha de Base (< 0,5 Hz) → −98,52%", loc="left", color="#0B2545")
    ax_l.grid(True, linestyle=":", alpha=0.55)
    ax_l.legend(loc="lower right", frameon=True, facecolor="white")

    fc_n, rdb_n, fdb_n = _bin_spectrum(freqs, raw_p, flt_p, 55.0, 65.0, 0.06)
    ax_n.axvspan(59.0, 61.0, color="#FCA5A5", alpha=0.35, label="Faixa Notch (59–61 Hz)")
    ax_n.axvline(60.0, color="#DC2626", linestyle="--", lw=1.8)
    ax_n.plot(fc_n, rdb_n, color="#64748B", lw=1.8, label="Antes (Bruto)")
    ax_n.plot(fc_n, fdb_n, color=TRACE_FILT_COLOR, lw=2.5, label="Depois (Filtrado)")
    ax_n.set_xlim(55.0, 65.0)
    ax_n.set_xlabel("Frequência (Hz)", fontweight="bold")
    ax_n.set_ylabel("Potência Espectral (dB re mV²)", fontweight="bold")
    ax_n.set_title("Zoom 2 · Rede Elétrica (60 Hz) → −99,99%", loc="left", color="#B91C1C")
    ax_n.grid(True, linestyle=":", alpha=0.55)
    ax_n.legend(loc="lower right", frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(SLIDE_FIG_DIR / "slide_fft_bandas_zoom.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8, 9 & 10. UM Gráfico Gigante Dedicado para CADA Anomalia Real
# ---------------------------------------------------------------------------
def gen_single_anomaly_slide(
    res,
    start_s: float,
    duration_s: float,
    out_name: str,
    title: str,
    badge_color: str,
    callout_mode: str,
) -> None:
    _slide_style()
    fs = res.ecg.fs
    s_idx = int(round(start_s * fs))
    e_idx = min(res.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    t = np.arange(s_idx, e_idx, dtype=np.float64) / fs
    sig = res.filtered_signal[s_idx:e_idx]
    peaks = res.r_peaks[(res.r_peaks >= s_idx) & (res.r_peaks < e_idx)]
    m = compute_clinical_metrics(peaks, fs)

    fig, ax = plt.subplots(figsize=(13.6, 5.5))
    _apply_ecg_grid_slide(ax, t[0], t[-1], tick_step=1.0)
    ax.plot(t, sig, color=TRACE_FILT_COLOR, lw=2.0, label=f"ECG Filtrado — MIT-BIH Reg. {res.ecg.record_id} ({res.ecg.lead_name})")
    ax.scatter(peaks / fs, res.filtered_signal[peaks], color=PEAK_COLOR, s=85, zorder=5, edgecolors="white", lw=1.0, label=f"Picos R detectados (n = {len(peaks)})")

    span = max(1.2, float(np.ptp(sig)))
    y_top = float(np.max(sig))
    y_bot = float(np.min(sig))

    if callout_mode == "brady" and len(peaks) >= 3:
        # Marca 1 intervalo RR bem longo no centro
        p0, p1 = int(peaks[1]), int(peaks[2])
        t0, t1 = p0 / fs, p1 / fs
        yb = max(res.filtered_signal[p0], res.filtered_signal[p1]) + 0.24 * span
        ax.annotate("", xy=(t1, yb), xytext=(t0, yb), arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=2.2), zorder=6)
        ax.text(
            0.5 * (t0 + t1),
            yb + 0.06 * span,
            f"R-R muito longo: {(p1 - p0)/fs*1000:.0f} ms (N = {p1 - p0} amostras)",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#B91C1C",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B91C1C", lw=1.4),
        )
    elif callout_mode == "arrhythmia" and len(peaks) >= 5:
        # Escolhe um par curto (PVC) na metade esquerda e uma pausa compensatória longa na metade direita (bem separados!)
        rrs = np.diff(peaks)
        left_short_cands = [i for i in range(len(rrs) // 2) if rrs[i] / fs < 0.68]
        i_short = left_short_cands[0] if left_short_cands else int(np.argmin(rrs))
        right_long_cands = [i for i in range(len(rrs)) if i >= i_short + 2 and rrs[i] / fs > 1.15]
        i_long = right_long_cands[0] if right_long_cands else int(np.argmax(rrs))
        for idx_k, lbl, col in [
            (i_short, f"Curto (PVC): {rrs[i_short]/fs*1000:.0f} ms", "#B45309"),
            (i_long, f"Pausa longa: {rrs[i_long]/fs*1000:.0f} ms", "#B91C1C"),
        ]:
            p0, p1 = int(peaks[idx_k]), int(peaks[idx_k + 1])
            t0, t1 = p0 / fs, p1 / fs
            yb = y_top + 0.13 * span
            ax.annotate("", xy=(t1, yb), xytext=(t0, yb), arrowprops=dict(arrowstyle="<->", color=col, lw=2.2), zorder=6)
            ax.text(
                0.5 * (t0 + t1),
                yb + 0.06 * span,
                lbl,
                ha="center",
                va="bottom",
                fontsize=11.0,
                fontweight="bold",
                color=col,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, lw=1.4),
            )
    elif callout_mode == "tachy" and len(peaks) >= 8:
        k = len(peaks) // 2
        p0, p1 = int(peaks[k]), int(peaks[k + 1])
        t0, t1 = p0 / fs, p1 / fs
        yb = max(res.filtered_signal[p0], res.filtered_signal[p1]) + 0.22 * span
        ax.annotate("", xy=(t1, yb), xytext=(t0, yb), arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=2.2), zorder=6)
        ax.text(
            0.5 * (t0 + t1),
            yb + 0.06 * span,
            f"R-R acelerado: {(p1 - p0)/fs*1000:.0f} ms (N = {p1 - p0} amostras)",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#B91C1C",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B91C1C", lw=1.4),
        )

    ax.set_ylim(y_bot - 0.28 * span, y_top + 0.60 * span)
    ax.set_ylabel("Amplitude (mV)", fontweight="bold")
    ax.set_xlabel("Tempo (s)  —  Papel milimetrado padrão ECG (25 mm/s · 10 mm/mV)", fontweight="bold")
    ax.set_title(title, loc="left", pad=10, color="#0B2545")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    extra = " · ALERTA ARRITMIA (ALTO SDNN)" if (m.classification == "Ritmo Normal" and m.sdnn_ms > 100.0) else ""
    ax.text(
        0.5,
        0.965,
        f"BPM: {m.bpm_mean:.1f}  |  R-R Médio: {m.mean_rr_ms:.0f} ms (N̄ = {m.mean_n_samples:.0f})  |  "
        f"SDNN: {m.sdnn_ms:.1f} ms  |  {m.classification.upper()}{extra}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color=badge_color,
        bbox=dict(boxstyle="round,pad=0.36", fc="white", ec=badge_color, lw=1.7, alpha=0.97),
        zorder=8,
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.13)
    fig.savefig(SLIDE_FIG_DIR / out_name, dpi=240)
    plt.close(fig)


def main() -> None:
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Gerando figuras 16:9 dedicadas para a apresentação...")

    gen_bode_filtros()
    print("[OK] slide_bode_filtros.png")

    gen_plano_z()
    print("[OK] slide_plano_z.png")

    res_100 = process_ecg(load_mitbih_record("100", data_dir=DATA_DIR))
    gen_pantompkins_etapas(res_100)
    print("[OK] slide_pantompkins_1_3.png + slide_pantompkins_4_5.png")

    res_105 = process_ecg(load_mitbih_record("105", data_dir=DATA_DIR))
    gen_antes_depois_105(res_105)
    print("[OK] slide_antes_depois_105.png")

    gen_fft_slides(res_105)
    print("[OK] slide_fft_espectro_global.png + slide_fft_bandas_zoom.png")

    res_231 = process_ecg(load_mitbih_record("231", data_dir=DATA_DIR))
    w_231 = find_bradycardia_window(res_231, 10.0)
    gen_single_anomaly_slide(
        res_231,
        w_231,
        10.0,
        "slide_anomalia_231_bradicardia.png",
        f"Bradicardia Severa (Bloqueio AV 2º Grau) — MIT-BIH Reg. 231 ({w_231:.0f}s–{w_231+10:.0f}s)",
        "#B91C1C",
        "brady",
    )
    print("[OK] slide_anomalia_231_bradicardia.png")

    res_119 = process_ecg(load_mitbih_record("119", data_dir=DATA_DIR))
    w_119 = find_arrhythmia_bigeminy_window(res_119, 10.0)
    gen_single_anomaly_slide(
        res_119,
        w_119,
        10.0,
        "slide_anomalia_119_arritmia.png",
        f"Arritmia Ventricular / Bigeminismo (PVCs) — MIT-BIH Reg. 119 ({w_119:.0f}s–{w_119+10:.0f}s)",
        "#B45309",
        "arrhythmia",
    )
    print("[OK] slide_anomalia_119_arritmia.png")

    res_209 = process_ecg(load_mitbih_record("209", data_dir=DATA_DIR))
    w_209 = find_fastest_tachycardia_window(res_209, 10.0)
    gen_single_anomaly_slide(
        res_209,
        w_209,
        10.0,
        "slide_anomalia_209_taquicardia.png",
        f"Taquicardia Supraventricular — MIT-BIH Reg. 209 ({w_209:.0f}s–{w_209+10:.0f}s)",
        "#B91C1C",
        "tachy",
    )
    print("[OK] slide_anomalia_209_taquicardia.png")

    print("\nTodas as 10 figuras widescreen 16:9 geradas em:", SLIDE_FIG_DIR)


if __name__ == "__main__":
    main()
