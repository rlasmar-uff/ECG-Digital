"""Testes unitários e de integração do pipeline de ECG (Sinais e Sistemas)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from src.config import DATA_DIR, OUTPUTS_DIR
from src.ecg_pipeline import (
    ECGData,
    apply_zero_phase_filters,
    classify_bpm,
    compute_clinical_metrics,
    compute_spectral_analysis,
    detect_r_peaks_pan_tompkins,
    inspect_pdf_exam,
    load_mitbih_record,
    load_physionet_upload,
    load_tabular_file,
    process_ecg,
)
from src.validation import match_beats_one_to_one


def _make_synthetic_ecg(fs: float = 360.0, duration_s: float = 20.0, bpm: float = 75.0) -> tuple[np.ndarray, np.ndarray]:
    """Gera ECG sintético periódico controlado com ruído de linha de base, 60 Hz e EMG."""
    n = int(round(fs * duration_s))
    t = np.arange(n, dtype=np.float64) / fs
    rr_s = 60.0 / bpm
    beat_times = np.arange(0.6, duration_s - 0.6, rr_s)
    beat_samples = np.round(beat_times * fs).astype(np.int64)

    clean = np.zeros(n, dtype=np.float64)
    for bt in beat_times:
        clean += 0.12 * np.exp(-0.5 * ((t - (bt - 0.16)) / 0.025) ** 2)
        clean -= 0.15 * np.exp(-0.5 * ((t - (bt - 0.03)) / 0.010) ** 2)
        clean += 1.30 * np.exp(-0.5 * ((t - bt) / 0.012) ** 2)
        clean -= 0.25 * np.exp(-0.5 * ((t - (bt + 0.03)) / 0.012) ** 2)
        clean += 0.28 * np.exp(-0.5 * ((t - (bt + 0.24)) / 0.045) ** 2)

    rng = np.random.default_rng(42)
    baseline = 0.45 * np.sin(2.0 * np.pi * 0.20 * t) + 0.30
    powerline = 0.25 * np.sin(2.0 * np.pi * 60.0 * t)
    emg = 0.08 * np.sin(2.0 * np.pi * 85.0 * t) + 0.02 * rng.standard_normal(n)
    return clean + baseline + powerline + emg, beat_samples


def test_zero_phase_filters_attenuate_noise_and_stay_bibo_stable() -> None:
    fs = 360.0
    raw, _ = _make_synthetic_ecg(fs=fs, duration_s=20.0, bpm=75.0)
    filt, diag = apply_zero_phase_filters(raw, fs=fs)

    assert diag.is_bibo_stable is True
    assert diag.max_pole_radius < 1.0
    assert diag.zero_phase is True

    spec = compute_spectral_analysis(raw, filt, fs=fs)
    assert spec.parseval_rel_error_raw < 1e-12
    assert spec.parseval_rel_error_filtered < 1e-12
    assert spec.band_attenuation_pct["Linha de Base (< 0,5 Hz)"] > 95.0
    assert spec.band_attenuation_pct["Rede Elétrica (59–61 Hz)"] > 99.0
    assert spec.band_attenuation_pct["Alta Frequência (> 40 Hz)"] > 95.0


@pytest.mark.parametrize(
    ("target_bpm", "expected_label"),
    [
        (48.0, "Bradicardia"),
        (75.0, "Ritmo Normal"),
        (140.0, "Taquicardia"),
    ],
)
def test_pan_tompkins_detects_all_clinical_bpm_regimes(target_bpm: float, expected_label: str) -> None:
    fs = 360.0
    raw, ref_peaks = _make_synthetic_ecg(fs=fs, duration_s=20.0, bpm=target_bpm)
    ecg = ECGData(signal=raw, fs=fs, source_name="synthetic")
    res = process_ecg(ecg)

    tp, fp, fn = match_beats_one_to_one(res.r_peaks, ref_peaks, fs=fs, tolerance_ms=150.0)
    assert tp >= len(ref_peaks) - 1
    assert fp == 0
    assert abs(res.metrics.bpm_mean - target_bpm) < 2.0
    assert res.metrics.classification == expected_label


def test_classify_bpm_thresholds_match_interview_spec() -> None:
    assert classify_bpm(59.9)[0] == "Bradicardia"
    assert classify_bpm(60.0)[0] == "Ritmo Normal"
    assert classify_bpm(105.0)[0] == "Ritmo Normal"
    assert classify_bpm(120.0)[0] == "Ritmo Normal"
    assert classify_bpm(120.1)[0] == "Taquicardia"


def test_load_tabular_csv_infers_fs_and_parses() -> None:
    demo_csv = OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv"
    assert demo_csv.exists(), "exemplo_ruidoso_mitbih.csv deve existir em outputs/"
    ecg = load_tabular_file(demo_csv.read_bytes(), demo_csv.name, fallback_fs=250.0)
    assert abs(ecg.fs - 360.0) < 0.5
    assert ecg.duration_s == pytest.approx(30.0, abs=0.2)
    res = process_ecg(ecg)
    assert res.metrics.num_beats >= 30


def test_load_physionet_upload_from_local_mitbih_bytes() -> None:
    hea = (DATA_DIR / "100.hea").read_bytes()
    dat = (DATA_DIR / "100.dat").read_bytes()
    atr = (DATA_DIR / "100.atr").read_bytes()
    ecg = load_physionet_upload(hea, dat, "100.hea", atr_bytes=atr)
    assert ecg.fs == 360.0
    assert ecg.annotation_samples is not None
    assert len(ecg.annotation_samples) == 2273


def test_pdf_exam_detection_warns_and_renders_preview() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Exame Clinico ECG (PDF ja filtrado)")
    pdf_bytes = doc.tobytes()
    doc.close()

    info = inspect_pdf_exam(pdf_bytes)
    assert info["num_pages"] == 1
    assert len(info["preview_png"]) > 100
    assert "PDF" in info["warning_title"]
    assert "filtros" in info["warning_body"]


def test_refractory_period_enforced() -> None:
    fs = 360.0
    raw, _ = _make_synthetic_ecg(fs=fs, duration_s=15.0, bpm=90.0)
    filt, _ = apply_zero_phase_filters(raw, fs=fs)
    peaks, _ = detect_r_peaks_pan_tompkins(filt, fs=fs, refractory_ms=200.0)
    if len(peaks) >= 2:
        min_rr_ms = float(np.min(np.diff(peaks)) / fs * 1000.0)
        assert min_rr_ms >= 200.0 - (1000.0 / fs)
    metrics = compute_clinical_metrics(peaks, fs=fs)
    assert metrics.sdnn_ms < 25.0


def test_real_mitbih_bradycardia_and_normal_records() -> None:
    ecg_100 = load_mitbih_record("100", data_dir=DATA_DIR, start_s=0.0, duration_s=60.0)
    res_100 = process_ecg(ecg_100)
    assert res_100.metrics.classification == "Ritmo Normal"
    assert res_100.spectral.parseval_rel_error_raw < 1e-12

    ecg_231 = load_mitbih_record("231", data_dir=DATA_DIR, start_s=466.0, duration_s=10.0)
    res_231 = process_ecg(ecg_231)
    assert res_231.metrics.classification == "Bradicardia"
    assert res_231.metrics.bpm_mean < 60.0

    ecg_209 = load_mitbih_record("209", data_dir=DATA_DIR, start_s=608.0, duration_s=10.0)
    res_209 = process_ecg(ecg_209)
    assert res_209.metrics.classification == "Taquicardia"
    assert res_209.metrics.bpm_mean > 120.0


def test_no_health_professional_names_in_project_artifacts() -> None:
    """Verifica a restrição obrigatória: nenhum nome/CRM de médico em código, saídas ou UI."""
    import re

    forbidden_patterns = [
        r"\bcrm\b",
        r"\bdra?\.\s+[a-záéíóúâêôãõç]+",
        r"\bfiocruz\b",
    ]
    files_to_check = [
        Path("app.py"),
        Path("server.py"),
        Path("run_validation.py"),
        Path("README.md"),
        *Path("src").glob("*.py"),
        *Path("web").glob("*.*"),
        *Path("docs").glob("*.*"),
        *Path("outputs").glob("*.csv"),
        *Path("outputs").glob("*.txt"),
        Path("relatorio4/main.tex"),
    ]
    for fpath in files_to_check:
        if fpath.exists():
            content_low = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            for pat in forbidden_patterns:
                assert not re.search(pat, content_low), f"Padrão proibido '{pat}' encontrado em {fpath}"


def test_fastapi_webapp_endpoints_and_uploads() -> None:
    """Valida todos os endpoints do app web FastAPI: página, MIT-BIH, upload CSV/PDF, reprocess e export."""
    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app)

    # 1) HTML, favicon, assets relativos e metadados
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "ecgCanvas" in r_idx.text
    assert "🫀" in r_idx.text
    assert client.get("/styles.css").status_code == 200
    assert client.get("/dsp_engine.js").status_code == 200
    assert client.get("/app.js").status_code == 200

    r_meta = client.get("/api/meta")
    assert r_meta.status_code == 200
    meta = r_meta.json()
    assert len(meta["team"]) == 5
    assert meta["team"][0]["name"].startswith("Cauã")
    assert len(meta["records"]) == 6

    # 2) Análise MIT-BIH (Bradicardia 231)
    r_231 = client.post("/api/analyze/mitbih", json={"record_id": "231"})
    assert r_231.status_code == 200
    d231 = r_231.json()
    assert d231["kind"] == "ecg"
    assert d231["metrics_full"]["classification"] == "Bradicardia"
    assert d231["live_validation"]["sensitivity_pct"] == 100.0
    sid = d231["session_id"]

    # 3) Window slicing e reprocessamento de filtros
    r_win = client.get(f"/api/window/{sid}", params={"start_s": 466.0, "duration_s": 10.0})
    assert r_win.status_code == 200
    assert len(r_win.json()["filtered"]) == 3600

    r_reproc = client.post(f"/api/reprocess/{sid}", json={"apply_notch": False, "apply_bandpass": True})
    assert r_reproc.status_code == 200
    assert r_reproc.json()["filter_info"]["notch_applied"] is False

    # 4) Upload real de CSV ruidoso
    demo_csv = (OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv").read_bytes()
    r_csv = client.post(
        "/api/analyze/upload",
        files=[("files", ("meu_ecg.csv", demo_csv, "text/csv"))],
        data={"fallback_fs": "360"},
    )
    assert r_csv.status_code == 200
    dcsv = r_csv.json()
    assert dcsv["kind"] == "ecg"
    assert dcsv["ecg"]["fs"] == 360.0
    assert dcsv["metrics_full"]["num_beats"] >= 30

    # 5) Upload de PDF clínico -> aviso técnico + renderização
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "ECG Hospitalar em PDF")
    pdf_bytes = doc.tobytes()
    doc.close()

    r_pdf = client.post(
        "/api/analyze/upload",
        files=[("files", ("exame_hospital.pdf", pdf_bytes, "application/pdf"))],
    )
    assert r_pdf.status_code == 200
    dpdf = r_pdf.json()
    assert dpdf["kind"] == "pdf_warning"
    assert dpdf["num_pages"] == 1
    assert len(dpdf["pages_b64"]) == 1

    # 6) Exportação PNG 300 DPI e CSV processado
    r_png = client.get(f"/api/export/{dcsv['session_id']}/png", params={"start_s": 0.0, "duration_s": 10.0})
    assert r_png.status_code == 200
    assert r_png.headers["content-type"] == "image/png"
    assert len(r_png.content) > 10_000


def test_streamlit_app_renders_without_exceptions() -> None:
    """Executa app.py via Streamlit AppTest para garantir inicialização limpa sem exceções."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)
    at.run()
    assert not at.exception, f"Exceção no Streamlit app.py: {at.exception}"
