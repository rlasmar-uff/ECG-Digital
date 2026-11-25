"""Módulo central de Processamento Digital de Sinais (Sinais e Sistemas) para ECG.

Implementa:
- Aquisição de dados reais do MIT-BIH Arrhythmia Database (PhysioNet via WFDB)
  e importação de arquivos locais (.csv, .txt, .dat/.hea, detecção de .pdf);
- Sistemas LIT de fase zero (Notch 60 Hz Q=30 + Butterworth passa-faixa 0,5–40 Hz ordem 4)
  com verificação de estabilidade BIBO via polos no plano Z;
- Detecção de picos R inspirada em Pan & Tompkins (1985) e Hamilton & Tompkins (1986)
  (derivada discreta de 5 pontos, quadratura, integração por janela móvel de 150 ms,
  limiares adaptativos, período refratário >= 200 ms e search-back);
- Extração de periodicidade discreta (N médio em amostras), intervalo R-R médio (ms),
  BPM médio, SDNN (ms) e classificação clínica (<60, 60–120, >120 BPM);
- Análise espectral (FFT) e verificação numérica do Teorema de Parseval.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

from src.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    BPM_BRADYCARDIA_MAX,
    BPM_TACHYCARDIA_MIN,
    DATA_DIR,
    DEFAULT_FS,
    MITBIH_BEAT_SYMBOLS,
    NOTCH_FREQ_HZ,
    NOTCH_Q,
    REFRACTORY_PERIOD_MS,
)


@dataclass
class ECGData:
    """Estrutura padronizada para sinais discretos de ECG x[n]."""

    signal: np.ndarray
    fs: float
    source_name: str
    lead_name: str = "MLII"
    units: str = "mV"
    annotation_samples: np.ndarray | None = None
    annotation_symbols: list[str] | None = None
    record_id: str | None = None

    @property
    def n_samples(self) -> int:
        return int(len(self.signal))

    @property
    def duration_s(self) -> float:
        return float(len(self.signal) / self.fs) if self.fs > 0 else 0.0

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(len(self.signal), dtype=np.float64) / self.fs


@dataclass
class FilterDiagnostics:
    """Diagnóstico teórico-matemático dos filtros LIT aplicados (plano Z e resposta)."""

    notch_applied: bool
    bandpass_applied: bool
    notch_freq_hz: float
    notch_q: float
    lowcut_hz: float
    highcut_hz: float
    order: int
    max_pole_radius: float
    is_bibo_stable: bool
    zero_phase: bool = True


@dataclass
class ClinicalMetrics:
    """Métricas de periodicidade discreta e triagem clínica extraídas do ECG."""

    num_beats: int
    mean_n_samples: float
    mean_rr_ms: float
    median_rr_ms: float
    sdnn_ms: float
    rmssd_ms: float
    bpm_mean: float
    bpm_median: float
    bpm_min: float
    bpm_max: float
    classification: str
    classification_level: str  # "normal" | "warning" | "alert"
    rhythm_regularity: str
    rr_intervals_ms: np.ndarray
    instant_bpm: np.ndarray


@dataclass
class SpectralAnalysis:
    """Resultado da FFT e do Teorema de Parseval antes e depois da filtragem."""

    freqs_hz: np.ndarray
    mag_raw: np.ndarray
    mag_filtered: np.ndarray
    psd_raw_db: np.ndarray
    psd_filtered_db: np.ndarray
    # Potência total (incluindo DC)
    power_time_raw: float
    power_parseval_raw: float
    power_time_filtered: float
    power_parseval_filtered: float
    parseval_rel_error_raw: float
    parseval_rel_error_filtered: float
    noise_removed_total_pct: float
    # Potência AC (sinal centrado em zero, sem offset DC)
    power_ac_raw: float
    power_ac_filtered: float
    noise_removed_ac_pct: float
    # Decomposição de potência por banda (mV^2)
    band_power_raw: dict[str, float]
    band_power_filtered: dict[str, float]
    band_attenuation_pct: dict[str, float]
    band_attenuation_db: dict[str, float]


@dataclass
class PipelineResult:
    """Saída completa do processamento de um sinal de ECG."""

    ecg: ECGData
    filtered_signal: np.ndarray
    filter_info: FilterDiagnostics
    r_peaks: np.ndarray
    metrics: ClinicalMetrics
    spectral: SpectralAnalysis
    qrs_mwi: np.ndarray


# ---------------------------------------------------------------------------
# 1. Aquisição de Dados Reais (PhysioNet MIT-BIH + Upload Local)
# ---------------------------------------------------------------------------


def ensure_mitbih_record(record_id: str, data_dir: Path = DATA_DIR) -> Path:
    """Garante que os arquivos .dat, .hea e .atr de um registro MIT-BIH estejam em disco."""
    import urllib.request

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    required = [f"{record_id}.dat", f"{record_id}.hea", f"{record_id}.atr"]
    missing = [
        fname
        for fname in required
        if not (data_dir / fname).exists() or (data_dir / fname).stat().st_size == 0
    ]
    if missing:
        for fname in missing:
            p = data_dir / fname
            if p.exists():
                p.unlink()
        try:
            wfdb.dl_files(db="mitdb", dl_dir=str(data_dir), files=missing, overwrite=True)
        except Exception:
            base_url = "https://physionet.org/files/mitdb/1.0.0"
            for fname in missing:
                dest = data_dir / fname
                if not dest.exists() or dest.stat().st_size == 0:
                    urllib.request.urlretrieve(f"{base_url}/{fname}", str(dest))
    return data_dir / record_id


def load_mitbih_record(
    record_id: str,
    data_dir: Path = DATA_DIR,
    channel: int = 0,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> ECGData:
    """Carrega um registro real do MIT-BIH Arrhythmia Database e suas anotações .atr."""
    record_path = ensure_mitbih_record(record_id, data_dir=data_dir)
    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    total_samples = int(header.sig_len)

    sampfrom = int(round((start_s or 0.0) * fs))
    sampfrom = max(0, min(sampfrom, total_samples - 1))
    if duration_s is not None:
        sampto = min(total_samples, sampfrom + int(round(duration_s * fs)))
    else:
        sampto = total_samples

    record = wfdb.rdrecord(
        str(record_path),
        sampfrom=sampfrom,
        sampto=sampto,
        channels=[channel],
    )
    sig = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if np.isnan(sig).any():
        # Interpola eventuais NaNs pontuais sem alterar a base temporal
        idx = np.arange(len(sig))
        valid = ~np.isnan(sig)
        sig = np.interp(idx, idx[valid], sig[valid])

    lead_name = record.sig_name[0] if record.sig_name else f"CH{channel}"
    units = record.units[0] if record.units else "mV"

    ann = wfdb.rdann(str(record_path), "atr", sampfrom=sampfrom, sampto=sampto)
    beat_samples: list[int] = []
    beat_symbols: list[str] = []
    for sample_idx, sym in zip(ann.sample, ann.symbol):
        if sym in MITBIH_BEAT_SYMBOLS:
            local_idx = int(sample_idx) - sampfrom
            if 0 <= local_idx < len(sig):
                beat_samples.append(local_idx)
                beat_symbols.append(str(sym))

    label = f"MIT-BIH Registro {record_id} ({lead_name})"
    if duration_s is not None:
        label += f" [{sampfrom / fs:.0f}s–{sampto / fs:.0f}s]"

    return ECGData(
        signal=sig,
        fs=fs,
        source_name=label,
        lead_name=lead_name,
        units=units,
        annotation_samples=np.asarray(beat_samples, dtype=np.int64),
        annotation_symbols=beat_symbols,
        record_id=record_id,
    )


def load_tabular_file(
    raw_bytes: bytes,
    filename: str,
    fallback_fs: float = DEFAULT_FS,
) -> ECGData:
    """Importa sinais reais de ECG a partir de arquivos .csv ou .txt."""
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("O arquivo enviado está vazio.")

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Nenhuma linha de dados numéricos encontrada no arquivo.")

    # Detecta separador automaticamente
    sample_line = lines[0]
    sep: str | None
    if ";" in sample_line:
        sep = ";"
    elif "," in sample_line:
        sep = ","
    elif "\t" in sample_line:
        sep = "\t"
    else:
        sep = r"\s+"

    # Verifica se a primeira linha é cabeçalho textual
    first_tokens = (
        sample_line.split(sep) if sep != r"\s+" else sample_line.split()
    )
    has_header = False
    for tok in first_tokens:
        try:
            float(tok.strip().replace(",", "."))
        except ValueError:
            has_header = True
            break

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=sep,
        engine="python" if sep == r"\s+" else "c",
        header=0 if has_header else None,
    )
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if df.empty:
        raise ValueError("Não foi possível extrair amostras numéricas do arquivo.")

    # Converte colunas numéricas
    numeric_cols: list[tuple[str, np.ndarray]] = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() > 0.9:
            numeric_cols.append((str(col), series.interpolate().to_numpy(dtype=np.float64)))

    if not numeric_cols:
        raise ValueError("Nenhuma coluna numérica válida encontrada no arquivo.")

    fs = float(fallback_fs)
    sig: np.ndarray
    lead_name = "ECG"

    if len(numeric_cols) == 1:
        lead_name, sig = numeric_cols[0]
    else:
        # Procura coluna de tempo e coluna de amplitude de ECG
        time_idx: int | None = None
        for idx, (cname, arr) in enumerate(numeric_cols):
            cname_low = cname.lower()
            diffs = np.diff(arr)
            is_monotonic = len(diffs) > 5 and np.all(diffs > 0)
            if any(k in cname_low for k in ("time", "tempo", "t_s", "sec", "seg")) and is_monotonic:
                time_idx = idx
                break
        if time_idx is None:
            # Se primeira coluna é estritamente crescente com passo constante, trata como tempo ou índice
            diffs0 = np.diff(numeric_cols[0][1])
            if len(diffs0) > 10 and np.all(diffs0 > 0) and np.std(diffs0) / np.mean(diffs0) < 0.05:
                dt = float(np.median(diffs0))
                if 1e-4 <= dt <= 0.1:  # entre 10 Hz e 10 kHz em segundos
                    time_idx = 0

        if time_idx is not None:
            t_arr = numeric_cols[time_idx][1]
            dt = float((t_arr[-1] - t_arr[0]) / max(1, len(t_arr) - 1))
            if dt <= 0:
                dt = float(np.median(np.diff(t_arr)))
            if dt > 0:
                inferred_fs = 1.0 / dt
                if 50.0 <= inferred_fs <= 5000.0:
                    # Arredonda para inteiro quando estiver a < 0,05 Hz de uma taxa inteira padrão
                    nearest_int = round(inferred_fs)
                    fs = float(nearest_int) if abs(inferred_fs - nearest_int) < 0.05 else round(inferred_fs, 2)
            # Escolhe a primeira coluna de sinal que não seja tempo nem coluna "filtrado"
            sig_candidates = [
                (cname, arr)
                for idx, (cname, arr) in enumerate(numeric_cols)
                if idx != time_idx
            ]
            # Prefere coluna com "brut", "raw", "ecg", "mlii" se houver mais de uma
            chosen = sig_candidates[0]
            for cname, arr in sig_candidates:
                if any(k in cname.lower() for k in ("brut", "raw", "ecg", "mlii", "sinal")):
                    chosen = (cname, arr)
                    break
            lead_name, sig = chosen
        else:
            lead_name, sig = numeric_cols[0]

    if len(sig) < int(fs * 1.5):
        raise ValueError(
            f"Sinal muito curto ({len(sig)} amostras a {fs:.1f} Hz). "
            "Envie pelo menos 2 segundos de ECG."
        )

    return ECGData(
        signal=np.asarray(sig, dtype=np.float64),
        fs=fs,
        source_name=f"Arquivo importado: {filename}",
        lead_name=lead_name,
        units="mV",
    )


def load_physionet_upload(
    hea_bytes: bytes,
    dat_bytes: bytes,
    record_stem: str,
    atr_bytes: bytes | None = None,
    channel: int = 0,
) -> ECGData:
    """Carrega arquivos PhysioNet (.hea + .dat e opcionalmente .atr) enviados pelo usuário."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        stem = Path(record_stem).stem
        (tmp / f"{stem}.hea").write_bytes(hea_bytes)
        (tmp / f"{stem}.dat").write_bytes(dat_bytes)
        if atr_bytes is not None:
            (tmp / f"{stem}.atr").write_bytes(atr_bytes)

        rec = wfdb.rdrecord(str(tmp / stem), channels=[channel])
        sig = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
        fs = float(rec.fs)
        lead_name = rec.sig_name[0] if rec.sig_name else f"CH{channel}"
        units = rec.units[0] if rec.units else "mV"

        ann_samples = None
        ann_symbols = None
        if atr_bytes is not None:
            ann = wfdb.rdann(str(tmp / stem), "atr")
            beats = [
                (int(s), str(sym))
                for s, sym in zip(ann.sample, ann.symbol)
                if sym in MITBIH_BEAT_SYMBOLS
            ]
            ann_samples = np.asarray([b[0] for b in beats], dtype=np.int64)
            ann_symbols = [b[1] for b in beats]

    return ECGData(
        signal=sig,
        fs=fs,
        source_name=f"PhysioNet importado: {stem}",
        lead_name=lead_name,
        units=units,
        annotation_samples=ann_samples,
        annotation_symbols=ann_symbols,
        record_id=stem,
    )


def inspect_pdf_exam(pdf_bytes: bytes) -> dict[str, Any]:
    """Analisa PDF clínico enviado, gera pré-visualização da 1ª página e aviso técnico."""
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    preview_png = b""
    if num_pages > 0:
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        preview_png = pix.tobytes("png")
    doc.close()

    return {
        "num_pages": num_pages,
        "preview_png": preview_png,
        "warning_title": "Formato PDF/Imagem detectado — Exame já pré-filtrado pelo aparelho",
        "warning_body": (
            "Exames de ECG em PDF ou imagem são renderizações prontas já submetidas "
            "aos filtros internos do eletrocardiógrafo hospitalar e não contêm a sequência "
            "discreta bruta x[n] amostrada em milivolts. Para aplicar os filtros LIT de "
            "fase zero, extrair métricas de periodicidade (BPM/SDNN) e calcular a FFT/"
            "Parseval sobre o sinal original, envie dados brutos em .csv, .txt ou "
            "arquivos PhysioNet (.dat + .hea), ou selecione diretamente um registro "
            "real da base MIT-BIH."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Filtragem Digital (Sistemas LIT de Fase Zero + Estabilidade BIBO)
# ---------------------------------------------------------------------------


def design_ecg_filters(
    fs: float,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ_HZ,
    notch_q: float = NOTCH_Q,
    apply_bandpass: bool = True,
    lowcut: float = BANDPASS_LOW_HZ,
    highcut: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> dict[str, Any]:
    """Projeta a cascata de filtros digitais LIT (Notch 60 Hz + Butterworth 0,5–40 Hz)."""
    nyq = fs / 2.0
    notch_ba: tuple[np.ndarray, np.ndarray] | None = None
    bandpass_sos: np.ndarray | None = None
    poles: list[np.ndarray] = []

    use_notch = bool(apply_notch and 0.0 < notch_freq < nyq * 0.98)
    if use_notch:
        b_n, a_n = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
        notch_ba = (b_n, a_n)
        _, p_n, _ = signal.tf2zpk(b_n, a_n)
        poles.append(p_n)

    eff_high = min(highcut, nyq * 0.95)
    use_bp = bool(apply_bandpass and 0.0 < lowcut < eff_high)
    if use_bp:
        bandpass_sos = signal.butter(
            N=order,
            Wn=[lowcut, eff_high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        _, p_bp, _ = signal.sos2zpk(bandpass_sos)
        poles.append(p_bp)

    if poles:
        all_poles = np.concatenate(poles)
        max_pole_radius = float(np.max(np.abs(all_poles)))
    else:
        all_poles = np.array([], dtype=np.complex128)
        max_pole_radius = 0.0

    diag = FilterDiagnostics(
        notch_applied=use_notch,
        bandpass_applied=use_bp,
        notch_freq_hz=notch_freq,
        notch_q=notch_q,
        lowcut_hz=lowcut,
        highcut_hz=eff_high,
        order=order,
        max_pole_radius=max_pole_radius,
        is_bibo_stable=bool(max_pole_radius < 1.0),
        zero_phase=True,
    )

    return {
        "notch_ba": notch_ba,
        "bandpass_sos": bandpass_sos,
        "poles": all_poles,
        "diagnostics": diag,
    }


def apply_zero_phase_filters(
    raw_signal: np.ndarray,
    fs: float,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ_HZ,
    notch_q: float = NOTCH_Q,
    apply_bandpass: bool = True,
    lowcut: float = BANDPASS_LOW_HZ,
    highcut: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> tuple[np.ndarray, FilterDiagnostics]:
    """Aplica filtros LIT bidirecionais de fase zero (filtfilt / sosfiltfilt)."""
    design = design_ecg_filters(
        fs=fs,
        apply_notch=apply_notch,
        notch_freq=notch_freq,
        notch_q=notch_q,
        apply_bandpass=apply_bandpass,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )

    y = np.asarray(raw_signal, dtype=np.float64).copy()
    if design["notch_ba"] is not None:
        b_n, a_n = design["notch_ba"]
        y = signal.filtfilt(b_n, a_n, y)
    if design["bandpass_sos"] is not None:
        # sosfiltfilt executa filtragem bidirecional de fase zero sobre o Butterworth
        # com máxima estabilidade numérica (equivalente ao filtfilt em cascata de 2ª ordem).
        y = signal.sosfiltfilt(design["bandpass_sos"], y)

    return y, design["diagnostics"]


def compute_cascade_frequency_response(
    fs: float,
    worN: int = 4096,
    notch_freq: float = NOTCH_FREQ_HZ,
    notch_q: float = NOTCH_Q,
    lowcut: float = BANDPASS_LOW_HZ,
    highcut: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> dict[str, Any]:
    """Calcula a resposta em frequência |H(e^{jw})| do sistema LIT em cascata."""
    design = design_ecg_filters(
        fs=fs,
        apply_notch=True,
        notch_freq=notch_freq,
        notch_q=notch_q,
        apply_bandpass=True,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
    b_n, a_n = design["notch_ba"]
    freqs, h_notch = signal.freqz(b_n, a_n, worN=worN, fs=fs)
    _, h_bp = signal.sosfreqz(design["bandpass_sos"], worN=worN, fs=fs)

    # Em filtragem bidirecional (filtfilt), a magnitude efetiva é |H(e^{jw})|^2 e a fase é 0°
    h_single = h_notch * h_bp
    mag_single = np.abs(h_single)
    mag_zero_phase = mag_single ** 2

    eps = 1e-12
    return {
        "freqs_hz": freqs,
        "mag_notch_db": 20.0 * np.log10(np.maximum(np.abs(h_notch), eps)),
        "mag_bp_db": 20.0 * np.log10(np.maximum(np.abs(h_bp), eps)),
        "mag_cascade_zero_phase_db": 20.0 * np.log10(np.maximum(mag_zero_phase, eps)),
        "poles": design["poles"],
        "diagnostics": design["diagnostics"],
    }


# ---------------------------------------------------------------------------
# 3. Detecção de Picos R (Pan & Tompkins 1985 + Hamilton & Tompkins 1986)
# ---------------------------------------------------------------------------


def _five_point_derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """Operador derivativo discreto de 5 pontos (Pan & Tompkins, 1985) sem atraso líquido.

    Versão causal clássica: y[n] = (1/8)*(2x[n] + x[n-1] - x[n-3] - 2x[n-4])
    (atraso de grupo constante de 2 amostras). Aplicando correlação simétrica
    centrada preservamos o alinhamento exato com o complexo QRS no tempo discreto.
    """
    kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float64) * (fs / 8.0)
    # Em modo 'same', h = [2, 1, 0, -1, -2] (k = -2..2) calcula 2x[n+2]+x[n+1]-x[n-1]-2x[n-2]
    return np.convolve(x, kernel, mode="same")


def _moving_window_integration(x_sq: np.ndarray, fs: float, window_ms: float = 150.0) -> np.ndarray:
    """Integração por janela móvel de largura W = 150 ms (soma cumulativa finita)."""
    win_len = max(3, int(round((window_ms / 1000.0) * fs)))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len, dtype=np.float64) / float(win_len)
    return np.convolve(x_sq, kernel, mode="same")


def detect_r_peaks_pan_tompkins(
    filtered_ecg: np.ndarray,
    fs: float,
    refractory_ms: float = REFRACTORY_PERIOD_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta picos R combinando Pan & Tompkins (1985) e Hamilton & Tompkins (1986).

    Etapas:
      1. Passa-faixa dedicado ao complexo QRS (5–15 Hz, fase zero)
      2. Derivada discreta de 5 pontos (diferença finita)
      3. Quadratura não linear y[n] = (x_d[n])^2
      4. Integração por janela móvel (150 ms)
      5. Regras de decisão adaptativas (SPKI/NPKI + SPKF/NPKF), período
         refratário mínimo (>= 200 ms), rejeição de onda T (200–360 ms)
         e busca reversa (search-back quando RR > 1,55 * RR_médio)
      6. Refinamento do instante exato do pico R no sinal diagnóstico filtrado (0,5–40 Hz)
    """
    x = np.asarray(filtered_ecg, dtype=np.float64)
    n = len(x)
    if n < int(fs):
        return np.array([], dtype=np.int64), np.zeros_like(x)

    # 1) Passa-faixa QRS 5–15 Hz (fase zero)
    nyq = fs / 2.0
    hi_qrs = min(15.0, nyq * 0.90)
    lo_qrs = min(5.0, hi_qrs * 0.5)
    sos_qrs = signal.butter(3, [lo_qrs, hi_qrs], btype="bandpass", fs=fs, output="sos")
    qrs_bp = signal.sosfiltfilt(sos_qrs, x)
    qrs_env = np.abs(qrs_bp)

    # 2) Derivada discreta de 5 pontos
    deriv = _five_point_derivative(qrs_bp, fs)

    # 3) Quadratura
    squared = deriv ** 2

    # 4) Integração por janela móvel (150 ms)
    mwi = _moving_window_integration(squared, fs, window_ms=150.0)

    # 5) Candidatos locais na saída do integrador (mínima distância = período refratário)
    refractory_samples = max(1, int(round((refractory_ms / 1000.0) * fs)))
    twave_samples = int(round(0.360 * fs))
    # Usamos distância ligeiramente menor (150 ms) na pré-candidatura para que o
    # loop de decisão aplique explicitamente o período refratário e a regra de onda T
    min_cand_dist = max(1, int(round(0.150 * fs)))
    cand_indices, _ = signal.find_peaks(mwi, distance=min_cand_dist)
    if len(cand_indices) == 0:
        return np.array([], dtype=np.int64), mwi

    # Inicialização robusta dos níveis de sinal e ruído nos primeiros 2 a 5 segundos
    init_end = min(n, int(round(4.0 * fs)))
    init_mwi = mwi[:init_end]
    init_env = qrs_env[:init_end]
    # Percentil 99 evita que um artefato isolado enorme no início trave o limiar
    spki = float(np.percentile(init_mwi, 99.0)) * 0.65
    npki = float(np.percentile(init_mwi, 50.0)) * 0.50
    spkf = float(np.percentile(init_env, 99.0)) * 0.65
    npkf = float(np.percentile(init_env, 50.0)) * 0.50

    if spki <= 0:
        spki = float(np.max(mwi)) * 0.5
    if spkf <= 0:
        spkf = float(np.max(qrs_env)) * 0.5

    th_i1 = npki + 0.25 * (spki - npki)
    th_i2 = 0.50 * th_i1
    th_f1 = npkf + 0.25 * (spkf - npkf)
    th_f2 = 0.50 * th_f1

    qrs_peaks: list[int] = []
    qrs_slopes: list[float] = []
    rr_history: list[int] = []
    rr_regular_history: list[int] = []

    half_win = max(2, int(round(0.075 * fs)))

    def _local_peak_features(idx: int) -> tuple[float, float, float]:
        lo = max(0, idx - half_win)
        hi = min(n, idx + half_win + 1)
        val_i = float(mwi[idx])
        val_f = float(np.max(qrs_env[lo:hi]))
        max_slope = float(np.max(np.abs(deriv[lo:hi])))
        return val_i, val_f, max_slope

    def _update_rr(new_rr: int) -> None:
        rr_history.append(new_rr)
        if len(rr_history) > 8:
            rr_history.pop(0)
        rr_avg1 = float(np.mean(rr_history))
        if 0.92 * rr_avg1 <= new_rr <= 1.16 * rr_avg1:
            rr_regular_history.append(new_rr)
            if len(rr_regular_history) > 8:
                rr_regular_history.pop(0)

    def _current_rr_avg() -> float:
        if len(rr_regular_history) >= 4:
            return float(np.mean(rr_regular_history))
        if rr_history:
            return float(np.mean(rr_history))
        return 0.80 * fs  # ~75 BPM inicial

    # Pré-computa características dos candidatos para acesso rápido no search-back
    cand_features = [_local_peak_features(int(c)) for c in cand_indices]

    # Limite anti-saturação para picos anômalos de artefato não elevarem SPKI infinitamente
    global_mwi_cap = float(np.percentile(mwi, 99.5)) * 2.5
    global_env_cap = float(np.percentile(qrs_env, 99.5)) * 2.5

    for ci, idx in enumerate(cand_indices):
        idx = int(idx)
        val_i, val_f, max_slope = cand_features[ci]

        # Se já houve QRS recente dentro do período refratário (< 200 ms), mantém o maior
        if qrs_peaks and (idx - qrs_peaks[-1]) < refractory_samples:
            prev_idx = qrs_peaks[-1]
            if val_i > mwi[prev_idx] * 1.15 and val_f > qrs_env[prev_idx]:
                qrs_peaks[-1] = idx
                qrs_slopes[-1] = max_slope
            continue

        # Search-back (Hamilton & Tompkins): lacuna > 1,55 * RR médio
        if qrs_peaks:
            rr_avg = _current_rr_avg()
            rr_missed_limit = int(round(1.55 * rr_avg))
            gap = idx - qrs_peaks[-1]
            if gap > rr_missed_limit:
                # Procura o melhor candidato perdido entre o último QRS + refratário e o atual
                best_sb_idx: int | None = None
                best_sb_val = -1.0
                best_sb_feat: tuple[float, float, float] | None = None
                for prev_ci in range(ci - 1, -1, -1):
                    p_idx = int(cand_indices[prev_ci])
                    if p_idx <= qrs_peaks[-1] + refractory_samples:
                        break
                    if idx - p_idx < refractory_samples:
                        continue
                    p_vi, p_vf, p_sl = cand_features[prev_ci]
                    if p_vi > th_i2 and p_vf > th_f2 and p_vi > best_sb_val:
                        best_sb_val = p_vi
                        best_sb_idx = p_idx
                        best_sb_feat = (p_vi, p_vf, p_sl)

                if best_sb_idx is not None and best_sb_feat is not None:
                    sb_vi, sb_vf, sb_sl = best_sb_feat
                    _update_rr(best_sb_idx - qrs_peaks[-1])
                    qrs_peaks.append(best_sb_idx)
                    qrs_slopes.append(sb_sl)
                    spki = 0.25 * min(sb_vi, global_mwi_cap) + 0.75 * spki
                    spkf = 0.25 * min(sb_vf, global_env_cap) + 0.75 * spkf
                    th_i1 = npki + 0.25 * (spki - npki)
                    th_i2 = 0.50 * th_i1
                    th_f1 = npkf + 0.25 * (spkf - npkf)
                    th_f2 = 0.50 * th_f1

        # Decisão principal (limiar duplo MWI + passa-faixa QRS)
        is_qrs = (val_i >= th_i1) and (val_f >= th_f1)

        if is_qrs and qrs_peaks:
            delta = idx - qrs_peaks[-1]
            # Regra de discriminação de onda T hiperaguda (200 ms <= delta < 360 ms)
            if refractory_samples <= delta < twave_samples:
                if max_slope < 0.50 * qrs_slopes[-1]:
                    is_qrs = False

        if is_qrs:
            if qrs_peaks:
                _update_rr(idx - qrs_peaks[-1])
            qrs_peaks.append(idx)
            qrs_slopes.append(max_slope)
            spki = 0.125 * min(val_i, global_mwi_cap) + 0.875 * spki
            spkf = 0.125 * min(val_f, global_env_cap) + 0.875 * spkf
        else:
            npki = 0.125 * min(val_i, global_mwi_cap) + 0.875 * npki
            npkf = 0.125 * min(val_f, global_env_cap) + 0.875 * npkf

        th_i1 = npki + 0.25 * (spki - npki)
        th_i2 = 0.50 * th_i1
        th_f1 = npkf + 0.25 * (spkf - npkf)
        th_f2 = 0.50 * th_f1

    if not qrs_peaks:
        return np.array([], dtype=np.int64), mwi

    # 6) Refinamento do pico R exato no sinal diagnóstico filtrado (0,5–40 Hz)
    # Determina a polaridade dominante dos complexos QRS no registro
    search_rad = max(2, int(round(0.080 * fs)))
    signed_extrema = []
    for p in qrs_peaks[: min(30, len(qrs_peaks))]:
        seg = x[max(0, p - search_rad) : min(n, p + search_rad + 1)]
        if len(seg) > 0:
            max_v = float(np.max(seg))
            min_v = float(np.min(seg))
            signed_extrema.append(max_v if abs(max_v) >= abs(min_v) else min_v)
    dominant_positive = (np.median(signed_extrema) >= 0.0) if signed_extrema else True

    refined: list[int] = []
    for p in qrs_peaks:
        lo = max(0, p - search_rad)
        hi = min(n, p + search_rad + 1)
        seg = x[lo:hi]
        if len(seg) == 0:
            continue
        idx_max = int(np.argmax(seg))
        idx_min = int(np.argmin(seg))
        val_max = float(seg[idx_max])
        val_min = float(seg[idx_min])

        # Se o batimento tiver polaridade dominante forte (ex.: R positivo), escolhe o pico R;
        # mas se for um complexo claramente invertido (ex.: PVC ou QRS invertido em 108/119),
        # captura o extremo absoluto dominante.
        if dominant_positive:
            if val_max >= 0.65 * abs(val_min):
                best = lo + idx_max
            else:
                best = lo + idx_min
        else:
            if abs(val_min) >= 0.65 * val_max:
                best = lo + idx_min
            else:
                best = lo + idx_max

        if refined and (best - refined[-1]) < refractory_samples:
            # Mantém o complexo de maior amplitude absoluta dentro do período refratário
            if abs(x[best]) > abs(x[refined[-1]]):
                refined[-1] = best
        else:
            refined.append(best)

    return np.asarray(refined, dtype=np.int64), mwi


# ---------------------------------------------------------------------------
# 4. Métricas de Periodicidade Discreta e Classificação Clínica
# ---------------------------------------------------------------------------


def classify_bpm(bpm: float) -> tuple[str, str]:
    """Classifica frequência cardíaca segundo os limiares da entrevista (60–120 BPM)."""
    if bpm < BPM_BRADYCARDIA_MAX:
        return "Bradicardia", "alert"
    if bpm > BPM_TACHYCARDIA_MIN:
        return "Taquicardia", "alert"
    return "Ritmo Normal", "normal"


def compute_clinical_metrics(r_peaks: np.ndarray, fs: float) -> ClinicalMetrics:
    """Calcula periodicidade discreta N̄, intervalo R-R médio, BPM médio, SDNN e diagnóstico."""
    num_beats = int(len(r_peaks))
    if num_beats < 2:
        return ClinicalMetrics(
            num_beats=num_beats,
            mean_n_samples=0.0,
            mean_rr_ms=0.0,
            median_rr_ms=0.0,
            sdnn_ms=0.0,
            rmssd_ms=0.0,
            bpm_mean=0.0,
            bpm_median=0.0,
            bpm_min=0.0,
            bpm_max=0.0,
            classification="Indeterminado (batimentos insuficientes)",
            classification_level="warning",
            rhythm_regularity="Indeterminado",
            rr_intervals_ms=np.array([], dtype=np.float64),
            instant_bpm=np.array([], dtype=np.float64),
        )

    rr_samples = np.diff(r_peaks).astype(np.float64)
    rr_ms = (rr_samples / fs) * 1000.0
    instant_bpm = 60000.0 / rr_ms

    mean_n = float(np.mean(rr_samples))
    mean_rr = float(np.mean(rr_ms))
    median_rr = float(np.median(rr_ms))
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0

    bpm_mean = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    bpm_median = 60000.0 / median_rr if median_rr > 0 else 0.0
    bpm_min = float(np.min(instant_bpm))
    bpm_max = float(np.max(instant_bpm))

    classification, level = classify_bpm(bpm_mean)

    # Indicador complementar de periodicidade discreta x[n] = x[n+N] via SDNN
    if sdnn < 50.0:
        regularity = "Ritmo Regular (quase-periódico)"
    elif sdnn <= 100.0:
        regularity = "Variabilidade Moderada"
    else:
        regularity = "Alta Variabilidade (sugestivo de arritmia)"

    return ClinicalMetrics(
        num_beats=num_beats,
        mean_n_samples=mean_n,
        mean_rr_ms=mean_rr,
        median_rr_ms=median_rr,
        sdnn_ms=sdnn,
        rmssd_ms=rmssd,
        bpm_mean=bpm_mean,
        bpm_median=bpm_median,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        classification=classification,
        classification_level=level,
        rhythm_regularity=regularity,
        rr_intervals_ms=rr_ms,
        instant_bpm=instant_bpm,
    )


# ---------------------------------------------------------------------------
# 5. Análise Espectral (FFT e Teorema de Parseval)
# ---------------------------------------------------------------------------


def _one_sided_power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Calcula FFT unilateral e verifica exatamente o Teorema de Parseval P = sum(|c_k|^2)."""
    x = np.asarray(x, dtype=np.float64)
    m = len(x)
    # Potência média no domínio do tempo (Aula 1, slide 16): P_d = (1/M) * sum |x[n]|^2
    power_time = float(np.mean(x ** 2))

    # Coeficientes da série discreta de Fourier: c_k = X[k] / M
    # Pelo Teorema de Parseval: (1/M) * sum |x[n]|^2 == sum_{k=0}^{M-1} |c_k|^2
    rfft_vals = np.fft.rfft(x) / float(m)
    freqs = np.fft.rfftfreq(m, d=1.0 / fs)
    power_one_sided = np.abs(rfft_vals) ** 2
    if m % 2 == 0:
        power_one_sided[1:-1] *= 2.0
    else:
        power_one_sided[1:] *= 2.0

    power_parseval = float(np.sum(power_one_sided))
    mag_one_sided = np.sqrt(power_one_sided)
    return freqs, mag_one_sided, power_time, power_parseval


def compute_spectral_analysis(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    fs: float,
) -> SpectralAnalysis:
    """Compara FFT antes/depois e quantifica energia de ruído removida via Parseval."""
    raw = np.asarray(raw_signal, dtype=np.float64)
    filt = np.asarray(filtered_signal, dtype=np.float64)

    freqs, mag_raw, p_time_raw, p_pars_raw = _one_sided_power_spectrum(raw, fs)
    _, mag_filt, p_time_filt, p_pars_filt = _one_sided_power_spectrum(filt, fs)

    rel_err_raw = abs(p_time_raw - p_pars_raw) / max(p_time_raw, 1e-15)
    rel_err_filt = abs(p_time_filt - p_pars_filt) / max(p_time_filt, 1e-15)

    p_spec_raw = mag_raw ** 2
    p_spec_filt = mag_filt ** 2

    # Potência AC (excluindo DC f = 0 Hz para evidenciar remoção de ruído dinâmico)
    p_ac_raw = float(np.sum(p_spec_raw[1:]))
    p_ac_filt = float(np.sum(p_spec_filt[1:]))

    noise_removed_total_pct = (
        max(0.0, (p_pars_raw - p_pars_filt) / p_pars_raw) * 100.0
        if p_pars_raw > 0
        else 0.0
    )
    noise_removed_ac_pct = (
        max(0.0, (p_ac_raw - p_ac_filt) / p_ac_raw) * 100.0
        if p_ac_raw > 0
        else 0.0
    )

    # Decomposição por faixas de interesse em Sinais e Sistemas
    bands = {
        "DC (0 Hz)": freqs == 0.0,
        "Linha de Base (< 0,5 Hz)": (freqs > 0.0) & (freqs < 0.5),
        "Banda ECG (0,5–40 Hz)": (freqs >= 0.5) & (freqs <= 40.0),
        "Alta Frequência (> 40 Hz)": freqs > 40.0,
        "Rede Elétrica (59–61 Hz)": (freqs >= 59.0) & (freqs <= 61.0),
    }

    band_power_raw: dict[str, float] = {}
    band_power_filt: dict[str, float] = {}
    band_att_pct: dict[str, float] = {}
    band_att_db: dict[str, float] = {}

    for bname, mask in bands.items():
        pr = float(np.sum(p_spec_raw[mask]))
        pf = float(np.sum(p_spec_filt[mask]))
        band_power_raw[bname] = pr
        band_power_filt[bname] = pf
        if pr > 1e-15:
            band_att_pct[bname] = (1.0 - pf / pr) * 100.0
            band_att_db[bname] = 10.0 * np.log10(max(pf, 1e-18) / pr)
        else:
            band_att_pct[bname] = 0.0
            band_att_db[bname] = 0.0

    # Espectro em dB (referência em mV^2) com piso numérico para visualização limpa
    eps = 1e-14
    psd_raw_db = 10.0 * np.log10(np.maximum(p_spec_raw, eps))
    psd_filt_db = 10.0 * np.log10(np.maximum(p_spec_filt, eps))

    return SpectralAnalysis(
        freqs_hz=freqs,
        mag_raw=mag_raw,
        mag_filtered=mag_filt,
        psd_raw_db=psd_raw_db,
        psd_filtered_db=psd_filt_db,
        power_time_raw=p_time_raw,
        power_parseval_raw=p_pars_raw,
        power_time_filtered=p_time_filt,
        power_parseval_filtered=p_pars_filt,
        parseval_rel_error_raw=rel_err_raw,
        parseval_rel_error_filtered=rel_err_filt,
        noise_removed_total_pct=noise_removed_total_pct,
        power_ac_raw=p_ac_raw,
        power_ac_filtered=p_ac_filt,
        noise_removed_ac_pct=noise_removed_ac_pct,
        band_power_raw=band_power_raw,
        band_power_filtered=band_power_filt,
        band_attenuation_pct=band_att_pct,
        band_attenuation_db=band_att_db,
    )


# ---------------------------------------------------------------------------
# 6. Orquestração Completa do Pipeline
# ---------------------------------------------------------------------------


def process_ecg(
    ecg: ECGData,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ_HZ,
    notch_q: float = NOTCH_Q,
    apply_bandpass: bool = True,
    lowcut: float = BANDPASS_LOW_HZ,
    highcut: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
    refractory_ms: float = REFRACTORY_PERIOD_MS,
) -> PipelineResult:
    """Executa o pipeline completo: filtragem LIT fase zero -> picos R -> métricas -> FFT/Parseval."""
    filtered, f_diag = apply_zero_phase_filters(
        raw_signal=ecg.signal,
        fs=ecg.fs,
        apply_notch=apply_notch,
        notch_freq=notch_freq,
        notch_q=notch_q,
        apply_bandpass=apply_bandpass,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )

    r_peaks, mwi = detect_r_peaks_pan_tompkins(
        filtered_ecg=filtered,
        fs=ecg.fs,
        refractory_ms=refractory_ms,
    )
    metrics = compute_clinical_metrics(r_peaks=r_peaks, fs=ecg.fs)
    spectral = compute_spectral_analysis(
        raw_signal=ecg.signal,
        filtered_signal=filtered,
        fs=ecg.fs,
    )

    return PipelineResult(
        ecg=ecg,
        filtered_signal=filtered,
        filter_info=f_diag,
        r_peaks=r_peaks,
        metrics=metrics,
        spectral=spectral,
        qrs_mwi=mwi,
    )
