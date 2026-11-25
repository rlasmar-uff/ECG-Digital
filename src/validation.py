"""Validação comparativa da detecção de picos R contra as anotações oficiais (.atr) do MIT-BIH."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COURSE_INFO,
    DATA_DIR,
    INSTITUTION,
    LITERATURE_BENCHMARKS,
    MITBIH_RECORDS,
    MitBihRecordInfo,
    OUTPUTS_DIR,
    TEAM_MEMBERS,
    VALIDATION_TOLERANCE_MS,
)
from src.ecg_pipeline import PipelineResult, load_mitbih_record, process_ecg


@dataclass
class ValidationResult:
    """Métricas de validação de um registro contra o padrão-ouro MIT-BIH (.atr)."""

    record_id: str
    category: str
    duration_min: float
    total_real: int
    total_detected: int
    tp: int
    fp: int
    fn: int
    sensitivity_pct: float
    precision_pct: float
    f1_score_pct: float
    bpm_mean: float
    mean_rr_ms: float
    mean_n_samples: float
    sdnn_ms: float
    classification: str
    parseval_noise_removed_total_pct: float
    parseval_noise_removed_ac_pct: float
    attenuation_baseline_pct: float
    attenuation_60hz_pct: float
    attenuation_highfreq_pct: float


def match_beats_one_to_one(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: float,
    tolerance_ms: float = VALIDATION_TOLERANCE_MS,
) -> tuple[int, int, int]:
    """Emparelhamento ótimo um-para-um entre picos detectados e anotações .atr (±150 ms)."""
    det = np.sort(np.asarray(detected_peaks, dtype=np.int64))
    ref = np.sort(np.asarray(reference_peaks, dtype=np.int64))
    tol = int(round((tolerance_ms / 1000.0) * fs))

    i = 0
    j = 0
    tp = 0
    n_det = len(det)
    n_ref = len(ref)

    while i < n_det and j < n_ref:
        diff = int(det[i] - ref[j])
        if abs(diff) <= tol:
            # Verifica se o próximo pico detectado ou referência está ainda mais próximo
            if i + 1 < n_det and abs(int(det[i + 1] - ref[j])) < abs(diff):
                # det[i] é falso positivo antes do melhor casamento det[i+1] <-> ref[j]
                i += 1
                continue
            if j + 1 < n_ref and abs(int(det[i] - ref[j + 1])) < abs(diff):
                # ref[j] foi perdido antes do melhor casamento det[i] <-> ref[j+1]
                j += 1
                continue
            tp += 1
            i += 1
            j += 1
        elif diff < -tol:
            i += 1
        else:
            j += 1

    fp = n_det - tp
    fn = n_ref - tp
    return tp, fp, fn


def evaluate_record(
    record_info: MitBihRecordInfo,
    data_dir: Path = DATA_DIR,
    tolerance_ms: float = VALIDATION_TOLERANCE_MS,
) -> tuple[ValidationResult, PipelineResult]:
    """Baixa, processa e valida um registro completo (30 min) da base MIT-BIH."""
    ecg = load_mitbih_record(record_info.record_id, data_dir=data_dir)
    res = process_ecg(ecg)

    assert ecg.annotation_samples is not None
    tp, fp, fn = match_beats_one_to_one(
        detected_peaks=res.r_peaks,
        reference_peaks=ecg.annotation_samples,
        fs=ecg.fs,
        tolerance_ms=tolerance_ms,
    )

    se = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    ppv = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * se * ppv / (se + ppv)) if (se + ppv) > 0 else 0.0

    val = ValidationResult(
        record_id=record_info.record_id,
        category=record_info.category,
        duration_min=ecg.duration_s / 60.0,
        total_real=int(len(ecg.annotation_samples)),
        total_detected=int(len(res.r_peaks)),
        tp=tp,
        fp=fp,
        fn=fn,
        sensitivity_pct=se,
        precision_pct=ppv,
        f1_score_pct=f1,
        bpm_mean=res.metrics.bpm_mean,
        mean_rr_ms=res.metrics.mean_rr_ms,
        mean_n_samples=res.metrics.mean_n_samples,
        sdnn_ms=res.metrics.sdnn_ms,
        classification=res.metrics.classification,
        parseval_noise_removed_total_pct=res.spectral.noise_removed_total_pct,
        parseval_noise_removed_ac_pct=res.spectral.noise_removed_ac_pct,
        attenuation_baseline_pct=res.spectral.band_attenuation_pct["Linha de Base (< 0,5 Hz)"],
        attenuation_60hz_pct=res.spectral.band_attenuation_pct["Rede Elétrica (59–61 Hz)"],
        attenuation_highfreq_pct=res.spectral.band_attenuation_pct["Alta Frequência (> 40 Hz)"],
    )
    return val, res


def run_full_validation(
    records: list[MitBihRecordInfo] = MITBIH_RECORDS,
    data_dir: Path = DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    tolerance_ms: float = VALIDATION_TOLERANCE_MS,
) -> tuple[pd.DataFrame, dict[str, PipelineResult]]:
    """Executa validação em todos os registros selecionados e salva CSV + relatório TXT."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pipeline_results: dict[str, PipelineResult] = {}

    tot_real = tot_det = tot_tp = tot_fp = tot_fn = 0

    for rec_info in records:
        val, pres = evaluate_record(rec_info, data_dir=data_dir, tolerance_ms=tolerance_ms)
        pipeline_results[rec_info.record_id] = pres

        tot_real += val.total_real
        tot_det += val.total_detected
        tot_tp += val.tp
        tot_fp += val.fp
        tot_fn += val.fn

        rows.append(
            {
                "Registro": val.record_id,
                "Perfil_Clinico": val.category,
                "Duracao_min": round(val.duration_min, 2),
                "Batimentos_Reais": val.total_real,
                "Batimentos_Detectados": val.total_detected,
                "TP": val.tp,
                "FP": val.fp,
                "FN": val.fn,
                "Sensibilidade_%": round(val.sensitivity_pct, 2),
                "Precisao_%": round(val.precision_pct, 2),
                "F1_Score_%": round(val.f1_score_pct, 2),
                "BPM_Medio": round(val.bpm_mean, 2),
                "RR_Medio_ms": round(val.mean_rr_ms, 2),
                "Periodo_N_amostras": round(val.mean_n_samples, 2),
                "SDNN_ms": round(val.sdnn_ms, 2),
                "Classificacao": val.classification,
                "Reducao_Potencia_Total_%": round(val.parseval_noise_removed_total_pct, 2),
                "Reducao_Potencia_AC_%": round(val.parseval_noise_removed_ac_pct, 2),
                "Atenuacao_LinhaBase_%": round(val.attenuation_baseline_pct, 2),
                "Atenuacao_60Hz_%": round(val.attenuation_60hz_pct, 2),
                "Atenuacao_AltaFreq_%": round(val.attenuation_highfreq_pct, 2),
            }
        )

    # Linha agregada (Total / Global)
    g_se = (tot_tp / (tot_tp + tot_fn) * 100.0) if (tot_tp + tot_fn) > 0 else 0.0
    g_ppv = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else 0.0
    g_f1 = (2.0 * g_se * g_ppv / (g_se + g_ppv)) if (g_se + g_ppv) > 0 else 0.0

    df_records = pd.DataFrame(rows)
    total_row = {
        "Registro": "TOTAL_GLOBAL",
        "Perfil_Clinico": f"Agregado ({len(records)} registros MIT-BIH)",
        "Duracao_min": round(float(df_records["Duracao_min"].sum()), 2),
        "Batimentos_Reais": tot_real,
        "Batimentos_Detectados": tot_det,
        "TP": tot_tp,
        "FP": tot_fp,
        "FN": tot_fn,
        "Sensibilidade_%": round(g_se, 2),
        "Precisao_%": round(g_ppv, 2),
        "F1_Score_%": round(g_f1, 2),
        "BPM_Medio": round(float(df_records["BPM_Medio"].mean()), 2),
        "RR_Medio_ms": round(float(df_records["RR_Medio_ms"].mean()), 2),
        "Periodo_N_amostras": round(float(df_records["Periodo_N_amostras"].mean()), 2),
        "SDNN_ms": round(float(df_records["SDNN_ms"].mean()), 2),
        "Classificacao": "—",
        "Reducao_Potencia_Total_%": round(float(df_records["Reducao_Potencia_Total_%"].mean()), 2),
        "Reducao_Potencia_AC_%": round(float(df_records["Reducao_Potencia_AC_%"].mean()), 2),
        "Atenuacao_LinhaBase_%": round(float(df_records["Atenuacao_LinhaBase_%"].mean()), 2),
        "Atenuacao_60Hz_%": round(float(df_records["Atenuacao_60Hz_%"].mean()), 2),
        "Atenuacao_AltaFreq_%": round(float(df_records["Atenuacao_AltaFreq_%"].mean()), 2),
    }

    df = pd.concat([df_records, pd.DataFrame([total_row])], ignore_index=True)

    csv_path = outputs_dir / "tabela_validacao_mitbih.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Relatório textual formatado (metricas_validacao.txt)
    txt_lines = [
        "=" * 92,
        INSTITUTION,
        COURSE_INFO,
        "VALIDAÇÃO COMPARATIVA – MIT-BIH ARRHYTHMIA DATABASE (PHYSIONET, fs = 360 Hz)",
        "=" * 92,
        "Equipe (ordem alfabética):",
    ]
    for name, mat in TEAM_MEMBERS:
        txt_lines.append(f"  - {name} ({mat})")

    txt_lines.extend(
        [
            "-" * 92,
            f"Tolerância de casamento (AAMI/ANSI EC57): ±{tolerance_ms:.0f} ms",
            "Pipeline LIT Fase Zero: Notch 60 Hz (Q=30) + Butterworth 0,5–40 Hz (ordem 4, filtfilt)",
            "Detector QRS: Pan & Tompkins (1985) + Hamilton & Tompkins (1986) (refratário >= 200 ms)",
            "Limiares clínicos: < 60 BPM (Bradicardia) | 60–120 BPM (Ritmo Normal) | > 120 BPM (Taquicardia)",
            "-" * 92,
            "",
            df[
                [
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
            ].to_string(index=False),
            "",
            "-" * 92,
            "COMPARAÇÃO COM A LITERATURA DE REFERÊNCIA (RELATÓRIO 1):",
            "-" * 92,
            pd.DataFrame(LITERATURE_BENCHMARKS).to_string(index=False),
            "",
            f"Resultado Global Obtido (nos {len(records)} registros, {tot_real} batimentos reais):",
            f"  TP = {tot_tp} | FP = {tot_fp} | FN = {tot_fn}",
            f"  Sensibilidade = {g_se:.2f}% | Precisão (+P) = {g_ppv:.2f}% | F1-Score = {g_f1:.2f}%",
            "=" * 92,
        ]
    )

    txt_path = outputs_dir / "metricas_validacao.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    return df, pipeline_results
