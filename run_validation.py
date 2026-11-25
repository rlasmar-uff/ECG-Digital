"""Script principal: baixa registros MIT-BIH, executa validação e gera artefatos finais."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite execução direta via `python run_validation.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_DIR,
    MITBIH_RECORDS,
    OUTPUTS_DIR,
    RELATORIO4_FIG_DIR,
)
from src.ecg_pipeline import compute_clinical_metrics
from src.plotting import (
    find_arrhythmia_bigeminy_window,
    find_best_noisy_window,
    find_bradycardia_window,
    find_fastest_tachycardia_window,
    plot_clinical_anomalies_panel,
    plot_ecg_comparison,
    plot_fft_parseval,
    plot_lti_filter_response,
)
from src.validation import run_full_validation


def export_noisy_demo_csv(
    result,
    output_path: Path,
    start_s: float,
    duration_s: float = 30.0,
) -> Path:
    """Exporta trecho real ruidoso do MIT-BIH (30 s) em CSV pronto para demonstração ao vivo."""
    fs = result.ecg.fs
    s_idx = max(0, int(round(start_s * fs)))
    e_idx = min(result.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    t_local = np.arange(e_idx - s_idx, dtype=np.float64) / fs
    df_csv = pd.DataFrame(
        {
            "tempo_s": np.round(t_local, 6),
            "ecg_bruto_mV": np.round(result.ecg.signal[s_idx:e_idx], 6),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    print("=" * 88)
    print("PIPELINE DE VALIDAÇÃO E GERAÇÃO DE ARTEFATOS – ECG (SINAIS E SISTEMAS · UFF 2026/2)")
    print("=" * 88)
    print(f"Baixando/carregando {len(MITBIH_RECORDS)} registros reais da base MIT-BIH (PhysioNet, 360 Hz)...")

    df_val, results = run_full_validation(
        records=MITBIH_RECORDS,
        data_dir=DATA_DIR,
        outputs_dir=OUTPUTS_DIR,
    )

    print("\n--- RESULTADOS DA VALIDAÇÃO COMPARATIVA (.atr ±150 ms) ---")
    cols_show = [
        "Registro",
        "Batimentos_Reais",
        "Batimentos_Detectados",
        "TP",
        "FP",
        "FN",
        "Sensibilidade_%",
        "Precisao_%",
        "F1_Score_%",
        "BPM_Medio",
        "SDNN_ms",
        "Classificacao",
    ]
    print(df_val[cols_show].to_string(index=False))

    # Registro 105 (ruído evidente) para as figuras principais e CSV de demonstração ao vivo
    res_105 = results["105"]
    best_win_105 = find_best_noisy_window(res_105, window_duration_s=10.0)
    print(f"\nMelhor janela ruidosa selecionada no registro 105: {best_win_105:.1f}s – {best_win_105 + 10.0:.1f}s")

    # 1. outputs/comparacao_ecg.png (300 DPI)
    comp_path = OUTPUTS_DIR / "comparacao_ecg.png"
    plot_ecg_comparison(res_105, comp_path, start_s=best_win_105, duration_s=10.0, dpi=300)
    print(f"[OK] Gerado: {comp_path}")

    # 2. outputs/espectro_fft.png (300 DPI)
    fft_path = OUTPUTS_DIR / "espectro_fft.png"
    plot_fft_parseval(res_105, fft_path, dpi=300)
    sp105 = res_105.spectral
    print(f"[OK] Gerado: {fft_path}")
    print(
        f"     Parseval (Reg. 105) -> P_antes={sp105.power_parseval_raw:.5f} mV² "
        f"(erro id. {sp105.parseval_rel_error_raw:.2e}) | "
        f"P_depois={sp105.power_parseval_filtered:.5f} mV² "
        f"(erro id. {sp105.parseval_rel_error_filtered:.2e})"
    )
    print(
        f"     Energia removida: Total={sp105.noise_removed_total_pct:.2f}% | "
        f"AC={sp105.noise_removed_ac_pct:.2f}% | "
        f"<0,5 Hz={sp105.band_attenuation_pct['Linha de Base (< 0,5 Hz)']:.2f}% | "
        f"60 Hz={sp105.band_attenuation_pct['Rede Elétrica (59–61 Hz)']:.2f}% | "
        f">40 Hz={sp105.band_attenuation_pct['Alta Frequência (> 40 Hz)']:.2f}%"
    )

    # 3. outputs/exemplo_ruidoso_mitbih.csv (30 s reais do registro 105 centrados na janela ruidosa)
    demo_start = max(0.0, best_win_105 - 5.0)
    demo_csv = OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv"
    export_noisy_demo_csv(res_105, demo_csv, start_s=demo_start, duration_s=30.0)
    print(f"[OK] Gerado: {demo_csv} ({demo_start:.1f}s–{demo_start + 30.0:.1f}s do registro 105)")

    # 4. Figuras extras de alta qualidade para o Relatório 4 (anomalias reais + filtros LIT/Plano Z)
    res_231 = results["231"]
    res_119 = results["119"]
    res_209 = results["209"]
    brady_start_231 = find_bradycardia_window(res_231, window_duration_s=10.0)
    bigeminy_start_119 = find_arrhythmia_bigeminy_window(res_119, window_duration_s=10.0)
    tachy_start_209 = find_fastest_tachycardia_window(res_209, window_duration_s=10.0)

    for lbl, r_obj, st_s in [
        ("Bradicardia (231)", res_231, brady_start_231),
        ("Arritmia/PVCs (119)", res_119, bigeminy_start_119),
        ("Taquicardia (209)", res_209, tachy_start_209),
    ]:
        s_i = int(round(st_s * r_obj.ecg.fs))
        e_i = s_i + int(round(10.0 * r_obj.ecg.fs))
        p_i = r_obj.r_peaks[(r_obj.r_peaks >= s_i) & (r_obj.r_peaks < e_i)]
        m_i = compute_clinical_metrics(p_i, r_obj.ecg.fs)
        print(
            f"Janela clínica [{lbl}] ({st_s:.1f}s–{st_s + 10.0:.1f}s): "
            f"{m_i.bpm_mean:.1f} BPM | SDNN={m_i.sdnn_ms:.1f} ms -> {m_i.classification}"
        )

    anomalies_path = OUTPUTS_DIR / "anomalias_clinicas.png"
    plot_clinical_anomalies_panel(
        cases=[
            ("Caso 1 — Bradicardia (< 60 BPM) com Bloqueio AV de 2º Grau", res_231, brady_start_231, 10.0),
            ("Caso 2 — Arritmia Ventricular (Bigeminismo / PVCs, SDNN elevado)", res_119, bigeminy_start_119, 10.0),
            ("Caso 3 — Episódio de Taquicardia Supraventricular (> 120 BPM)", res_209, tachy_start_209, 10.0),
        ],
        output_path=anomalies_path,
        dpi=300,
    )
    print(f"[OK] Gerado: {anomalies_path}")

    lti_path = OUTPUTS_DIR / "resposta_filtros_lit.png"
    plot_lti_filter_response(lti_path, fs=360.0, dpi=300)
    print(f"[OK] Gerado: {lti_path}")

    # Copia as figuras 300 DPI para relatorio4/figuras/ (pacote pronto para Overleaf)
    RELATORIO4_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fig_file in [comp_path, fft_path, anomalies_path, lti_path]:
        shutil.copy2(fig_file, RELATORIO4_FIG_DIR / fig_file.name)
    print(f"[OK] Figuras copiadas para: {RELATORIO4_FIG_DIR}")
    print("\nConcluído com sucesso!")


if __name__ == "__main__":
    main()
