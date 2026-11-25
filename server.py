r"""Servidor Web FastAPI de Alta Performance — Suporte Computacional para ECG.

Universidade Federal Fluminense (UFF) – Escola de Engenharia
Disciplina: Sinais e Sistemas – Prof. André Pinho (2º Semestre de 2026)

Execução local:
    .\.venv\Scripts\python.exe server.py
Acesse no navegador a URL exibida no terminal (padrão: http://localhost:8050).
"""

from __future__ import annotations

import base64
import io
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymupdf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    TEAM_MEMBERS,
    VALIDATION_TOLERANCE_MS,
)
from src.ecg_pipeline import (
    ECGData,
    PipelineResult,
    compute_cascade_frequency_response,
    compute_clinical_metrics,
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

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(
    title="CardioSignal DSP — UFF Sinais e Sistemas",
    description="Backend FastAPI para filtragem LIT fase zero, Pan-Tompkins, FFT e Parseval em ECG.",
    version="2.0.0",
)

# Armazena sessões ativas em memória (últimas 12 sessões para evitar crescimento ilimitado)
SESSIONS: dict[str, PipelineResult] = {}
SESSION_ORDER: list[str] = []
MAX_SESSIONS = 12


def _store_session(result: PipelineResult) -> str:
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = result
    SESSION_ORDER.append(sid)
    while len(SESSION_ORDER) > MAX_SESSIONS:
        old = SESSION_ORDER.pop(0)
        SESSIONS.pop(old, None)
    return sid


@lru_cache(maxsize=16)
def _cached_mitbih(record_id: str) -> ECGData:
    return load_mitbih_record(record_id=record_id, data_dir=DATA_DIR)


def _bin_spectrum(freqs: np.ndarray, p_raw: np.ndarray, p_flt: np.ndarray, f_min: float, f_max: float, step: float) -> dict[str, list[float]]:
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_sub = freqs[mask]
    r_sub = p_raw[mask]
    fl_sub = p_flt[mask]
    eps = 1e-13

    if len(f_sub) == 0:
        return {"freqs": [], "raw_db": [], "filt_db": []}

    if len(f_sub) <= 650:
        return {
            "freqs": np.round(f_sub, 4).tolist(),
            "raw_db": np.round(10.0 * np.log10(np.maximum(r_sub, eps)), 2).tolist(),
            "filt_db": np.round(10.0 * np.log10(np.maximum(fl_sub, eps)), 2).tolist(),
        }

    edges = np.arange(f_min, f_max + step, step)
    idx = np.digitize(f_sub, edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    fc = 0.5 * (edges[:-1] + edges[1:])
    rb = np.zeros_like(fc)
    fb = np.zeros_like(fc)
    ct = np.zeros_like(fc)
    np.add.at(rb, idx[valid], r_sub[valid])
    np.add.at(fb, idx[valid], fl_sub[valid])
    np.add.at(ct, idx[valid], 1.0)
    nz = ct > 0
    return {
        "freqs": np.round(fc[nz], 3).tolist(),
        "raw_db": np.round(10.0 * np.log10(np.maximum(rb[nz] / ct[nz], eps)), 2).tolist(),
        "filt_db": np.round(10.0 * np.log10(np.maximum(fb[nz] / ct[nz], eps)), 2).tolist(),
    }


def _suggested_window(result: PipelineResult) -> tuple[float, str]:
    rid = result.ecg.record_id
    dur = result.ecg.duration_s
    win_dur = min(10.0, dur)
    if rid == "105":
        return find_best_noisy_window(result, win_dur), "Janela ruidosa destacada (Reg. 105)"
    if rid == "231":
        return find_bradycardia_window(result, win_dur), "Episódio de Bradicardia < 60 BPM (Bloqueio AV)"
    if rid == "119":
        return find_arrhythmia_bigeminy_window(result, win_dur), "Trecho de Arritmia Ventricular / Bigeminismo"
    if rid == "209":
        return find_fastest_tachycardia_window(result, win_dur), "Episódio de Taquicardia Supraventricular > 120 BPM"
    if rid == "108":
        return min(120.0, max(0.0, dur - win_dur)), "Trecho com deriva severa e QRS atípico"
    return min(10.0, max(0.0, dur - win_dur)), "Janela inicial do exame"


def _serialize_metrics(m) -> dict[str, Any]:
    return {
        "num_beats": m.num_beats,
        "mean_n_samples": round(m.mean_n_samples, 2),
        "mean_rr_ms": round(m.mean_rr_ms, 2),
        "median_rr_ms": round(m.median_rr_ms, 2),
        "sdnn_ms": round(m.sdnn_ms, 2),
        "rmssd_ms": round(m.rmssd_ms, 2),
        "bpm_mean": round(m.bpm_mean, 2),
        "bpm_median": round(m.bpm_median, 2),
        "bpm_min": round(m.bpm_min, 2),
        "bpm_max": round(m.bpm_max, 2),
        "classification": m.classification,
        "classification_level": m.classification_level,
        "rhythm_regularity": m.rhythm_regularity,
    }


def _serialize_session_summary(sid: str, result: PipelineResult) -> dict[str, Any]:
    sp = result.spectral
    fi = result.filter_info
    fs = result.ecg.fs
    f_max = min(100.0, fs / 2.0)

    p_raw = sp.mag_raw ** 2
    p_flt = sp.mag_filtered ** 2

    fft_main = _bin_spectrum(sp.freqs_hz, p_raw, p_flt, 0.05, f_max, 0.16)
    fft_low = _bin_spectrum(sp.freqs_hz, p_raw, p_flt, 0.02, min(5.0, f_max), 0.03)
    fft_notch = _bin_spectrum(sp.freqs_hz, p_raw, p_flt, 55.0, min(65.0, f_max), 0.06) if f_max >= 62.0 else {"freqs": [], "raw_db": [], "filt_db": []}

    # Resposta em frequência e polos no plano Z com os parâmetros atuais
    resp = compute_cascade_frequency_response(
        fs=fs,
        worN=1024,
        notch_freq=fi.notch_freq_hz,
        notch_q=fi.notch_q,
        lowcut=fi.lowcut_hz,
        highcut=fi.highcut_hz,
        order=fi.order,
    )
    rf = resp["freqs_hz"]
    rmask = rf <= f_max
    # Amostra até 400 pontos para gráfico Bode suave
    step_r = max(1, int(np.sum(rmask)) // 400)
    bode = {
        "freqs": np.round(rf[rmask][::step_r], 3).tolist(),
        "cascade_db": np.round(resp["mag_cascade_zero_phase_db"][rmask][::step_r], 2).tolist(),
        "notch_db": np.round(resp["mag_notch_db"][rmask][::step_r], 2).tolist(),
        "bp_db": np.round(resp["mag_bp_db"][rmask][::step_r], 2).tolist(),
        "poles_re": np.round(np.real(resp["poles"]), 6).tolist(),
        "poles_im": np.round(np.imag(resp["poles"]), 6).tolist(),
    }

    # Validação contra .atr se disponível
    live_validation = None
    if result.ecg.annotation_samples is not None and len(result.ecg.annotation_samples) > 0:
        tp, fp, fn = match_beats_one_to_one(
            result.r_peaks, result.ecg.annotation_samples, fs=fs, tolerance_ms=VALIDATION_TOLERANCE_MS
        )
        se = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0
        ppv = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
        f1 = (2.0 * se * ppv / (se + ppv)) if (se + ppv) else 0.0
        live_validation = {
            "total_real": int(len(result.ecg.annotation_samples)),
            "total_detected": int(len(result.r_peaks)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity_pct": round(se, 2),
            "precision_pct": round(ppv, 2),
            "f1_score_pct": round(f1, 2),
        }

    # Série temporal de R-R / BPM instantâneo (downsampled se > 1200 batimentos para render rápido)
    rr_times = (result.r_peaks[1:] / fs) if len(result.r_peaks) > 1 else np.array([])
    rr_n_samples = np.diff(result.r_peaks) if len(result.r_peaks) > 1 else np.array([])
    inst_bpm = result.metrics.instant_bpm
    stride_rr = max(1, len(rr_times) // 1000)

    sug_start, sug_label = _suggested_window(result)

    # Envia uma fatia inicial generosa (60 s centrados na janela sugerida) já na 1ª resposta
    buf_start = max(0.0, sug_start - 15.0)
    buf_dur = min(60.0, result.ecg.duration_s - buf_start)
    initial_slice = _extract_window_slice(result, buf_start, buf_dur)

    band_table = [
        {
            "band": bname,
            "power_raw": round(sp.band_power_raw[bname], 6),
            "power_filtered": round(sp.band_power_filtered[bname], 6),
            "attenuation_pct": round(sp.band_attenuation_pct[bname], 2),
            "attenuation_db": round(sp.band_attenuation_db[bname], 2),
        }
        for bname in sp.band_power_raw
    ]

    return {
        "kind": "ecg",
        "session_id": sid,
        "ecg": {
            "source_name": result.ecg.source_name,
            "lead_name": result.ecg.lead_name,
            "units": result.ecg.units,
            "fs": round(fs, 2),
            "n_samples": result.ecg.n_samples,
            "duration_s": round(result.ecg.duration_s, 3),
            "record_id": result.ecg.record_id,
            "has_annotations": result.ecg.annotation_samples is not None,
        },
        "suggested_window": {
            "start_s": round(sug_start, 2),
            "duration_s": round(min(10.0, result.ecg.duration_s), 2),
            "label": sug_label,
        },
        "filter_info": {
            "notch_applied": fi.notch_applied,
            "bandpass_applied": fi.bandpass_applied,
            "notch_freq_hz": fi.notch_freq_hz,
            "notch_q": fi.notch_q,
            "lowcut_hz": fi.lowcut_hz,
            "highcut_hz": fi.highcut_hz,
            "order": fi.order,
            "max_pole_radius": round(fi.max_pole_radius, 5),
            "is_bibo_stable": fi.is_bibo_stable,
            "zero_phase": fi.zero_phase,
        },
        "metrics_full": _serialize_metrics(result.metrics),
        "r_peaks_all": result.r_peaks.tolist(),
        "spectral": {
            "power_time_raw": round(sp.power_time_raw, 6),
            "power_parseval_raw": round(sp.power_parseval_raw, 6),
            "power_time_filtered": round(sp.power_time_filtered, 6),
            "power_parseval_filtered": round(sp.power_parseval_filtered, 6),
            "parseval_rel_error_raw": sp.parseval_rel_error_raw,
            "parseval_rel_error_filtered": sp.parseval_rel_error_filtered,
            "noise_removed_total_pct": round(sp.noise_removed_total_pct, 2),
            "power_ac_raw": round(sp.power_ac_raw, 6),
            "power_ac_filtered": round(sp.power_ac_filtered, 6),
            "noise_removed_ac_pct": round(sp.noise_removed_ac_pct, 2),
            "band_table": band_table,
            "fft_main": fft_main,
            "fft_low": fft_low,
            "fft_notch": fft_notch,
        },
        "bode": bode,
        "tachogram": {
            "times_s": np.round(rr_times[::stride_rr], 2).tolist(),
            "n_samples": rr_n_samples[::stride_rr].tolist(),
            "instant_bpm": np.round(inst_bpm[::stride_rr], 2).tolist(),
        },
        "live_validation": live_validation,
        "initial_slice": initial_slice,
    }


def _extract_window_slice(result: PipelineResult, start_s: float, duration_s: float) -> dict[str, Any]:
    fs = result.ecg.fs
    s_idx = max(0, int(round(start_s * fs)))
    e_idx = min(result.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    if e_idx <= s_idx:
        e_idx = min(result.ecg.n_samples, s_idx + int(round(2.0 * fs)))

    peaks_in = result.r_peaks[(result.r_peaks >= s_idx) & (result.r_peaks < e_idx)]
    atr_in = (
        result.ecg.annotation_samples[
            (result.ecg.annotation_samples >= s_idx) & (result.ecg.annotation_samples < e_idx)
        ]
        if result.ecg.annotation_samples is not None
        else np.array([], dtype=np.int64)
    )

    return {
        "start_idx": s_idx,
        "end_idx": e_idx,
        "start_s": round(s_idx / fs, 4),
        "end_s": round(e_idx / fs, 4),
        "fs": fs,
        "raw": np.round(result.ecg.signal[s_idx:e_idx], 4).tolist(),
        "filtered": np.round(result.filtered_signal[s_idx:e_idx], 4).tolist(),
        "r_peaks": peaks_in.tolist(),
        "atr_peaks": atr_in.tolist(),
    }


# ---------------------------------------------------------------------------
# Modelos de Requisição
# ---------------------------------------------------------------------------


class FilterParams(BaseModel):
    apply_notch: bool = True
    notch_freq: float = NOTCH_FREQ_HZ
    notch_q: float = NOTCH_Q
    apply_bandpass: bool = True
    lowcut: float = BANDPASS_LOW_HZ
    highcut: float = BANDPASS_HIGH_HZ
    order: int = BANDPASS_ORDER
    refractory_ms: float = REFRACTORY_PERIOD_MS


class MitBihRequest(FilterParams):
    record_id: str = "105"


# ---------------------------------------------------------------------------
# Endpoints REST
# ---------------------------------------------------------------------------


@app.get("/api/meta")
def get_meta() -> dict[str, Any]:
    val_csv = OUTPUTS_DIR / "tabela_validacao_mitbih.csv"
    validation_table = pd.read_csv(val_csv).to_dict(orient="records") if val_csv.exists() else []
    return {
        "project_title": PROJECT_TITLE,
        "institution": INSTITUTION,
        "course_info": COURSE_INFO,
        "team": [{"name": n, "matricula": m} for n, m in TEAM_MEMBERS],
        "records": [
            {
                "record_id": r.record_id,
                "category": r.category,
                "description": r.description,
            }
            for r in MITBIH_RECORDS
        ],
        "validation_table": validation_table,
        "literature_benchmarks": LITERATURE_BENCHMARKS,
    }


@app.post("/api/analyze/mitbih")
def analyze_mitbih(req: MitBihRequest) -> dict[str, Any]:
    valid_ids = {r.record_id for r in MITBIH_RECORDS}
    if req.record_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Registro '{req.record_id}' não suportado.")
    ecg = _cached_mitbih(req.record_id)
    res = process_ecg(
        ecg=ecg,
        apply_notch=req.apply_notch,
        notch_freq=req.notch_freq,
        notch_q=req.notch_q,
        apply_bandpass=req.apply_bandpass,
        lowcut=req.lowcut,
        highcut=req.highcut,
        order=req.order,
        refractory_ms=req.refractory_ms,
    )
    sid = _store_session(res)
    return _serialize_session_summary(sid, res)


@app.post("/api/analyze/demo-csv")
def analyze_demo_csv(req: FilterParams) -> dict[str, Any]:
    demo_path = OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="exemplo_ruidoso_mitbih.csv não encontrado. Execute run_validation.py.")
    ecg = load_tabular_file(demo_path.read_bytes(), demo_path.name, fallback_fs=DEFAULT_FS)
    res = process_ecg(
        ecg=ecg,
        apply_notch=req.apply_notch,
        notch_freq=req.notch_freq,
        notch_q=req.notch_q,
        apply_bandpass=req.apply_bandpass,
        lowcut=req.lowcut,
        highcut=req.highcut,
        order=req.order,
        refractory_ms=req.refractory_ms,
    )
    sid = _store_session(res)
    return _serialize_session_summary(sid, res)


@app.post("/api/analyze/upload")
async def analyze_upload(
    files: list[UploadFile] = File(...),
    fallback_fs: float = Form(DEFAULT_FS),
    apply_notch: bool = Form(True),
    notch_q: float = Form(NOTCH_Q),
    apply_bandpass: bool = Form(True),
    lowcut: float = Form(BANDPASS_LOW_HZ),
    highcut: float = Form(BANDPASS_HIGH_HZ),
    order: int = Form(BANDPASS_ORDER),
    refractory_ms: float = Form(REFRACTORY_PERIOD_MS),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado.")

    by_ext: dict[str, tuple[str, bytes]] = {}
    for uf in files:
        fname = uf.filename or "arquivo"
        ext = Path(fname).suffix.lower()
        by_ext[ext] = (fname, await uf.read())

    # 1) PDF clínico pronto -> alerta técnico + renderização das páginas
    if ".pdf" in by_ext:
        fname, raw_pdf = by_ext[".pdf"]
        try:
            doc = pymupdf.open(stream=raw_pdf, filetype="pdf")
            pages_b64: list[str] = []
            for p_idx in range(min(3, len(doc))):
                pix = doc[p_idx].get_pixmap(dpi=150)
                pages_b64.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
            num_pages = len(doc)
            doc.close()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Não foi possível abrir o PDF: {exc}") from exc

        return JSONResponse(
            {
                "kind": "pdf_warning",
                "filename": fname,
                "num_pages": num_pages,
                "pages_b64": pages_b64,
                "warning_title": "Formato PDF detectado — Exame já filtrado pelo aparelho hospitalar",
                "warning_body": (
                    "Exames de ECG em PDF são laudos/renderizações já processados pelos filtros internos "
                    "do eletrocardiógrafo hospitalar e não contêm a sequência discreta bruta x[n] em milivolts. "
                    "Para aplicar a filtragem LIT de fase zero, detectar picos R e calcular FFT/Parseval sobre "
                    "os dados reais, envie um arquivo de dados brutos (.csv, .txt ou par PhysioNet .dat + .hea) "
                    "ou clique em 'Testar CSV Ruidoso Real' / selecione um registro do MIT-BIH."
                ),
            }
        )

    # 2) Imagem de ECG (.png/.jpg/.jpeg/.webp) -> mesmo aviso técnico + visualização
    for img_ext in (".png", ".jpg", ".jpeg", ".webp"):
        if img_ext in by_ext:
            fname, raw_img = by_ext[img_ext]
            mime = "image/png" if img_ext == ".png" else ("image/webp" if img_ext == ".webp" else "image/jpeg")
            return JSONResponse(
                {
                    "kind": "image_warning",
                    "filename": fname,
                    "mime": mime,
                    "image_b64": base64.b64encode(raw_img).decode("ascii"),
                    "warning_title": "Imagem de exame detectada — Traçado já filtrado e sem amostras brutas x[n]",
                    "warning_body": (
                        "Fotografias ou capturas de tela de ECG em papel milimetrado já passaram pelos filtros "
                        "do equipamento clínico e não fornecem o vetor temporal discreto x[n]. Envie dados brutos "
                        "(.csv, .txt ou .dat + .hea do PhysioNet) para executar o pipeline completo de Sinais e Sistemas."
                    ),
                }
            )

    # 3) Par PhysioNet (.hea + .dat + opcional .atr)
    ecg: ECGData | None = None
    try:
        if ".hea" in by_ext and ".dat" in by_ext:
            hea_name, hea_bytes = by_ext[".hea"]
            _, dat_bytes = by_ext[".dat"]
            atr_bytes = by_ext[".atr"][1] if ".atr" in by_ext else None
            ecg = load_physionet_upload(
                hea_bytes=hea_bytes,
                dat_bytes=dat_bytes,
                record_stem=hea_name,
                atr_bytes=atr_bytes,
            )
        elif ".dat" in by_ext or ".hea" in by_ext:
            raise HTTPException(
                status_code=400,
                detail="Para registros PhysioNet, envie juntos os arquivos .hea e .dat (e opcionalmente .atr).",
            )
        else:
            tab_pair = by_ext.get(".csv") or by_ext.get(".txt")
            if tab_pair is None:
                raise HTTPException(
                    status_code=400,
                    detail="Formato não suportado. Envie .csv, .txt, .dat+.hea (PhysioNet) ou .pdf.",
                )
            fname, raw_bytes = tab_pair
            ecg = load_tabular_file(raw_bytes=raw_bytes, filename=fname, fallback_fs=fallback_fs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao interpretar arquivo de ECG: {exc}") from exc

    res = process_ecg(
        ecg=ecg,
        apply_notch=apply_notch,
        notch_freq=NOTCH_FREQ_HZ,
        notch_q=notch_q,
        apply_bandpass=apply_bandpass,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        refractory_ms=refractory_ms,
    )
    sid = _store_session(res)
    return JSONResponse(_serialize_session_summary(sid, res))


@app.post("/api/reprocess/{session_id}")
def reprocess_session(session_id: str, req: FilterParams) -> dict[str, Any]:
    prev = SESSIONS.get(session_id)
    if prev is None:
        raise HTTPException(status_code=404, detail="Sessão expirada ou inexistente.")
    res = process_ecg(
        ecg=prev.ecg,
        apply_notch=req.apply_notch,
        notch_freq=req.notch_freq,
        notch_q=req.notch_q,
        apply_bandpass=req.apply_bandpass,
        lowcut=req.lowcut,
        highcut=req.highcut,
        order=req.order,
        refractory_ms=req.refractory_ms,
    )
    SESSIONS[session_id] = res
    return _serialize_session_summary(session_id, res)


@app.get("/api/window/{session_id}")
def get_window(
    session_id: str,
    start_s: float = Query(0.0, ge=0.0),
    duration_s: float = Query(30.0, gt=0.5, le=120.0),
) -> dict[str, Any]:
    res = SESSIONS.get(session_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return _extract_window_slice(res, start_s=start_s, duration_s=duration_s)


@app.get("/api/export/{session_id}/png")
def export_window_png(
    session_id: str,
    start_s: float = Query(0.0, ge=0.0),
    duration_s: float = Query(10.0, gt=1.0, le=30.0),
) -> Response:
    res = SESSIONS.get(session_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "ecg_export.png"
        plot_ecg_comparison(res, out_png, start_s=start_s, duration_s=duration_s, dpi=300)
        png_bytes = out_png.read_bytes()
    fname = f"ecg_comparacao_{res.ecg.record_id or 'importado'}_{int(start_s)}s.png"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/export/{session_id}/csv")
def export_window_csv(
    session_id: str,
    start_s: float = Query(0.0, ge=0.0),
    duration_s: float = Query(30.0, gt=1.0, le=300.0),
) -> Response:
    res = SESSIONS.get(session_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    fs = res.ecg.fs
    s_idx = max(0, int(round(start_s * fs)))
    e_idx = min(res.ecg.n_samples, s_idx + int(round(duration_s * fs)))
    df = pd.DataFrame(
        {
            "tempo_s": np.round(np.arange(e_idx - s_idx, dtype=np.float64) / fs, 6),
            "ecg_bruto_mV": np.round(res.ecg.signal[s_idx:e_idx], 6),
            "ecg_filtrado_mV": np.round(res.filtered_signal[s_idx:e_idx], 6),
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    fname = f"ecg_processado_{res.ecg.record_id or 'importado'}_{int(start_s)}s.csv"
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/export/demo-csv")
def download_demo_csv() -> FileResponse:
    demo = OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv"
    if not demo.exists():
        raise HTTPException(status_code=404, detail="Arquivo de demonstração não encontrado.")
    return FileResponse(demo, media_type="text/csv", filename="exemplo_ruidoso_mitbih.csv")


# Serve frontend estático (tanto em /static quanto relativo à raiz para compatibilidade com GitHub Pages)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/styles.css")
def serve_styles() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
def serve_app_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/dsp_engine.js")
def serve_dsp_js() -> FileResponse:
    return FileResponse(WEB_DIR / "dsp_engine.js", media_type="application/javascript")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _pick_available_port(preferred: int = 8050) -> int:
    import os
    import socket

    start_port = int(os.environ.get("ECG_PORT", str(preferred)))
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    chosen_port = _pick_available_port(8050)
    print("=" * 76)
    print("CardioSignal DSP (Web App FastAPI + Canvas 60 FPS) — UFF Sinais e Sistemas")
    print(f"Acesse no navegador: http://localhost:{chosen_port}")
    print("=" * 76)
    uvicorn.run("server:app", host="127.0.0.1", port=chosen_port, reload=False)
