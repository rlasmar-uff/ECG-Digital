"""Gera a pasta estática docs/ 100% pronta para publicação no GitHub Pages (sem backend).

Copia o frontend web/ (com motor DSP completo em JavaScript puro) e empacota
trechos clínicos reais de 60 segundos dos 6 registros do MIT-BIH (com anotações .atr),
a tabela completa de validação dos 30 minutos e o CSV ruidoso de demonstração.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    BASE_DIR,
    COURSE_INFO,
    DATA_DIR,
    INSTITUTION,
    LITERATURE_BENCHMARKS,
    MITBIH_RECORDS,
    OUTPUTS_DIR,
    PROJECT_TITLE,
    TEAM_MEMBERS,
)
from src.ecg_pipeline import load_mitbih_record, process_ecg
from src.plotting import (
    find_arrhythmia_bigeminy_window,
    find_best_noisy_window,
    find_bradycardia_window,
    find_fastest_tachycardia_window,
)

WEB_DIR = BASE_DIR / "web"
DOCS_DIR = BASE_DIR / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Copia arquivos estáticos do frontend
    for fname in ("index.html", "styles.css", "app.js", "dsp_engine.js"):
        shutil.copy2(WEB_DIR / fname, DOCS_DIR / fname)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # 2) Gera data/meta.json (mesmo schema de GET /api/meta)
    val_csv = OUTPUTS_DIR / "tabela_validacao_mitbih.csv"
    val_rows = pd.read_csv(val_csv).to_dict(orient="records") if val_csv.exists() else []
    meta = {
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
        "validation_table": val_rows,
        "literature_benchmarks": LITERATURE_BENCHMARKS,
    }
    (DOCS_DATA_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # 3) Copia exemplo_ruidoso_mitbih.csv para 1-clique no GitHub Pages
    demo_csv = OUTPUTS_DIR / "exemplo_ruidoso_mitbih.csv"
    if demo_csv.exists():
        shutil.copy2(demo_csv, DOCS_DATA_DIR / "exemplo_ruidoso_mitbih.csv")

    # 4) Exporta 60 s reais de cada registro MIT-BIH centrados no destaque clínico (+ anotações .atr)
    excerpt_dur_s = 60.0
    total_kb = 0.0
    for rec in MITBIH_RECORDS:
        full_ecg = load_mitbih_record(rec.record_id, data_dir=DATA_DIR)
        full_res = process_ecg(full_ecg)

        if rec.record_id == "105":
            hl = find_best_noisy_window(full_res, 10.0)
            hl_label = "Trecho ruidoso destacado (Reg. 105)"
        elif rec.record_id == "231":
            hl = find_bradycardia_window(full_res, 10.0)
            hl_label = "Episódio de Bradicardia < 60 BPM (Bloqueio AV)"
        elif rec.record_id == "119":
            hl = find_arrhythmia_bigeminy_window(full_res, 10.0)
            hl_label = "Trecho de Arritmia Ventricular / Bigeminismo"
        elif rec.record_id == "209":
            hl = find_fastest_tachycardia_window(full_res, 10.0)
            hl_label = "Episódio de Taquicardia Supraventricular > 120 BPM"
        elif rec.record_id == "108":
            hl = 120.0
            hl_label = "Deriva severa de linha de base e QRS atípico"
        else:
            hl = 10.0
            hl_label = "Ritmo sinusal normal (Referência)"

        fs = full_ecg.fs
        win_start = max(0.0, min(full_ecg.duration_s - excerpt_dur_s, hl - 20.0))
        s_idx = int(round(win_start * fs))
        e_idx = min(full_ecg.n_samples, s_idx + int(round(excerpt_dur_s * fs)))

        sig_slice = np.round(full_ecg.signal[s_idx:e_idx], 4).tolist()
        ann_local = []
        if full_ecg.annotation_samples is not None:
            ann_in = full_ecg.annotation_samples[
                (full_ecg.annotation_samples >= s_idx) & (full_ecg.annotation_samples < e_idx)
            ]
            ann_local = (ann_in - s_idx).tolist()

        payload = {
            "record_id": rec.record_id,
            "source_name": f"MIT-BIH Registro {rec.record_id} ({full_ecg.lead_name}) [{win_start:.0f}s–{e_idx / fs:.0f}s]",
            "lead_name": full_ecg.lead_name,
            "units": full_ecg.units,
            "fs": fs,
            "suggested_start_s": round(max(0.0, hl - win_start), 2),
            "suggested_label": hl_label,
            "signal": sig_slice,
            "annotation_samples": ann_local,
        }
        out_file = DOCS_DATA_DIR / f"mitdb_{rec.record_id}.json"
        out_file.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        kb = out_file.stat().st_size / 1024.0
        total_kb += kb
        print(f"[OK] {out_file.name}: {len(sig_slice)} amostras, {len(ann_local)} anotações .atr ({kb:.1f} KB)")

    print(f"\nSite estático GitHub Pages gerado em: {DOCS_DIR} (dados MIT-BIH: {total_kb / 1024.0:.2f} MB)")


if __name__ == "__main__":
    main()
