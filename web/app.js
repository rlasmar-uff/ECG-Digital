/**
 * CardioSignal DSP — Frontend 60 FPS em HTML5 Canvas + FastAPI
 * UFF · Sinais e Sistemas · Prof. André Pinho (2º Semestre de 2026)
 */

(() => {
  "use strict";

  const STATE = {
    mode: "api", // "api" (FastAPI local) | "static" (GitHub Pages 100% client-side)
    staticRecordCache: {},
    meta: null,
    session: null,
    activeRecordId: "105",
    scope: "window", // "window" | "full"
    viewMode: "split", // "split" | "overlay"
    showAtr: false,
    // Viewport temporal atual no Canvas (segundos)
    winStartS: 0,
    winDurS: 10,
    // Buffer local de alta resolução (evita chamadas HTTP durante pan/play a 60 FPS)
    buffer: null, // { start_idx, end_idx, start_s, end_s, fs, raw, filtered, r_peaks, atr_peaks }
    fetchingWindow: false,
    // Reprodução em tempo real (monitor hospitalar)
    playing: false,
    playSpeed: 1,
    lastFrameTs: 0,
    // Interação com mouse no Canvas
    dragging: false,
    dragStartX: 0,
    dragStartWinS: 0,
    hoverX: null,
    hoverY: null,
    activeTab: "ecg",
  };

  const $ = (id) => document.getElementById(id);

  // -------------------------------------------------------------------------
  // Utilidades Numéricas e Formatação PT-BR
  // -------------------------------------------------------------------------

  const fmt = (n, dec = 1) =>
    Number.isFinite(n) ? n.toFixed(dec).replace(".", ",") : "—";

  const fmtSci = (n) => {
    if (!Number.isFinite(n)) return "—";
    if (n === 0) return "0,00";
    return n.toExponential(2).replace(".", ",");
  };

  function getFilterParams() {
    return {
      apply_notch: $("chkNotch").checked,
      notch_freq: 60.0,
      notch_q: parseFloat($("sldNotchQ").value),
      apply_bandpass: $("chkBandpass").checked,
      lowcut: parseFloat($("sldLowcut").value),
      highcut: parseFloat($("sldHighcut").value),
      order: parseInt($("sldOrder").value, 10),
      refractory_ms: parseFloat($("sldRefractory").value),
    };
  }

  function classifyBpm(bpm) {
    if (bpm < 60) return { label: "Bradicardia", level: "alert" };
    if (bpm > 120) return { label: "Taquicardia", level: "alert" };
    return { label: "Ritmo Normal", level: "normal" };
  }

  function computeMetricsFromPeaks(peaks, fs) {
    const n = peaks.length;
    if (n < 2) {
      return {
        num_beats: n,
        mean_n_samples: 0,
        mean_rr_ms: 0,
        sdnn_ms: 0,
        rmssd_ms: 0,
        bpm_mean: 0,
        bpm_min: 0,
        bpm_max: 0,
        classification: "Indeterminado",
        classification_level: "warning",
        rhythm_regularity: "Batimentos insuficientes",
      };
    }
    const rrSamples = [];
    const rrMs = [];
    const instBpm = [];
    for (let i = 1; i < n; i++) {
      const dn = peaks[i] - peaks[i - 1];
      const ms = (dn / fs) * 1000;
      rrSamples.push(dn);
      rrMs.push(ms);
      instBpm.push(60000 / ms);
    }
    const meanN = rrSamples.reduce((a, b) => a + b, 0) / rrSamples.length;
    const meanRR = rrMs.reduce((a, b) => a + b, 0) / rrMs.length;
    let sdnn = 0;
    let rmssd = 0;
    if (rrMs.length > 1) {
      const varSum = rrMs.reduce((acc, v) => acc + (v - meanRR) ** 2, 0);
      sdnn = Math.sqrt(varSum / (rrMs.length - 1));
      let diffSq = 0;
      for (let i = 1; i < rrMs.length; i++) diffSq += (rrMs[i] - rrMs[i - 1]) ** 2;
      rmssd = Math.sqrt(diffSq / (rrMs.length - 1));
    }
    const bpmMean = 60000 / meanRR;
    const cls = classifyBpm(bpmMean);
    let regularity = "Ritmo Regular (quase-periódico)";
    if (sdnn > 100) regularity = "Alta Variabilidade (sugestivo de arritmia)";
    else if (sdnn >= 50) regularity = "Variabilidade Moderada";

    return {
      num_beats: n,
      mean_n_samples: meanN,
      mean_rr_ms: meanRR,
      sdnn_ms: sdnn,
      rmssd_ms: rmssd,
      bpm_mean: bpmMean,
      bpm_min: Math.min(...instBpm),
      bpm_max: Math.max(...instBpm),
      classification: cls.label,
      classification_level: cls.level,
      rhythm_regularity: regularity,
    };
  }

  function setupHiDpiCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(300, Math.floor(rect.width));
    const h = Math.max(180, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // -------------------------------------------------------------------------
  // Comunicação com Backend FastAPI
  // -------------------------------------------------------------------------

  function setLoading(show, msg = "Processando sinal...") {
    const el = $("loadingPill");
    el.textContent = msg;
    el.classList.toggle("hidden", !show);
  }

  function setUploadStatus(text, isError = false) {
    const el = $("uploadStatus");
    if (!text) {
      el.classList.add("hidden");
      return;
    }
    el.textContent = text;
    el.classList.toggle("error", isError);
    el.classList.remove("hidden");
  }

  async function initApp() {
    // Detecta automaticamente backend FastAPI local vs hospedagem estática (GitHub Pages)
    try {
      const res = await fetch("/api/meta", { cache: "no-store" });
      if (!res.ok) throw new Error("no-api");
      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("application/json")) throw new Error("not-json");
      STATE.meta = await res.json();
      STATE.mode = "api";
    } catch {
      const resStatic = await fetch("./data/meta.json");
      STATE.meta = await resStatic.json();
      STATE.mode = "static";
      const demoLink = document.querySelector(".link-subtle");
      if (demoLink) demoLink.setAttribute("href", "./data/exemplo_ruidoso_mitbih.csv");
    }

    // Rodapé com a equipe sempre em ordem alfabética
    $("footerTeam").textContent = STATE.meta.team
      .map((m) => `${m.name} (${m.matricula})`)
      .join(" · ");

    // Renderiza chips de acesso rápido para os 6 casos reais MIT-BIH
    const presetContainer = $("presetButtons");
    presetContainer.innerHTML = "";
    const anomalyIds = new Set(["105", "108", "119", "231", "209"]);
    const shortLabels = {
      "100": "100 · Normal",
      "105": "105 · Ruído Severo",
      "108": "108 · Deriva Extrema",
      "119": "119 · Arritmia / PVCs",
      "231": "231 · Bradicardia (<60)",
      "209": "209 · Taquicardia (>120)",
    };

    for (const r of STATE.meta.records) {
      const btn = document.createElement("button");
      btn.className = "preset-chip" + (anomalyIds.has(r.record_id) ? " chip-anomaly" : "");
      btn.dataset.recordId = r.record_id;
      btn.textContent = shortLabels[r.record_id] || `${r.record_id} · ${r.category}`;
      btn.title = r.description;
      btn.addEventListener("click", () => loadMitBih(r.record_id));
      presetContainer.appendChild(btn);
    }

    renderValidationTables();
    bindEvents();
    await loadMitBih("105");
  }

  function highlightActivePreset(recordId) {
    document.querySelectorAll(".preset-chip").forEach((el) => {
      el.classList.toggle("active", el.dataset.recordId === recordId);
    });
  }

  async function loadMitBih(recordId) {
    stopPlayback();
    hidePdfNotice();
    setUploadStatus("");
    setLoading(true, `Carregando MIT-BIH ${recordId}...`);
    try {
      let data;
      if (STATE.mode === "api") {
        const body = { record_id: recordId, ...getFilterParams() };
        const res = await fetch("/api/analyze/mitbih", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Falha ao carregar registro.");
        data = await res.json();
      } else {
        if (!STATE.staticRecordCache[recordId]) {
          const r = await fetch(`./data/mitdb_${recordId}.json`);
          if (!r.ok) throw new Error(`Arquivo estático mitdb_${recordId}.json não encontrado.`);
          STATE.staticRecordCache[recordId] = await r.json();
        }
        const rec = STATE.staticRecordCache[recordId];
        data = window.CardioDSP.buildSessionPayload(
          rec,
          getFilterParams(),
          rec.suggested_start_s ?? 0,
          rec.suggested_label || rec.source_name
        );
      }
      STATE.activeRecordId = recordId;
      highlightActivePreset(recordId);
      applySessionPayload(data, true);
    } catch (err) {
      setUploadStatus(err.message, true);
    } finally {
      setLoading(false);
    }
  }

  async function loadDemoCsv() {
    stopPlayback();
    hidePdfNotice();
    setLoading(true, "Importando exemplo_ruidoso_mitbih.csv...");
    try {
      let data;
      if (STATE.mode === "api") {
        const res = await fetch("/api/analyze/demo-csv", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(getFilterParams()),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Falha ao carregar CSV.");
        data = await res.json();
      } else {
        const res = await fetch("./data/exemplo_ruidoso_mitbih.csv");
        if (!res.ok) throw new Error("Arquivo exemplo_ruidoso_mitbih.csv não encontrado.");
        const txt = await res.text();
        const ecgObj = window.CardioDSP.parseTabularText(txt, "exemplo_ruidoso_mitbih.csv", 360);
        data = window.CardioDSP.buildSessionPayload(ecgObj, getFilterParams(), 5.0, "CSV ruidoso real (Reg. 105)");
      }
      STATE.activeRecordId = null;
      highlightActivePreset(null);
      setUploadStatus("CSV real ruidoso importado (30 s · 360 Hz).");
      applySessionPayload(data, true);
    } catch (err) {
      setUploadStatus(err.message, true);
    } finally {
      setLoading(false);
    }
  }

  async function uploadFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    stopPlayback();
    setLoading(true, "Analisando arquivo enviado...");
    setUploadStatus(`Enviando ${fileList.length} arquivo(s)...`);

    const fp = getFilterParams();
    const fallbackFs = parseFloat($("inputFallbackFs").value) || 360;

    if (STATE.mode === "static") {
      try {
        const byExt = {};
        for (const f of fileList) {
          const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
          byExt[ext] = f;
        }
        if (byExt[".pdf"]) {
          const pdfFile = byExt[".pdf"];
          const blobUrl = URL.createObjectURL(pdfFile);
          showPdfOrImageNotice({
            kind: "pdf_blob_warning",
            filename: pdfFile.name,
            blob_url: blobUrl,
            warning_title: "Formato PDF detectado — Exame já filtrado pelo aparelho hospitalar",
            warning_body:
              "Exames de ECG em PDF são laudos/renderizações já processados pelos filtros internos do eletrocardiógrafo hospitalar e não contêm a sequência discreta bruta x[n] em milivolts. Envie dados brutos (.csv, .txt ou par PhysioNet .dat + .hea) ou teste um dos registros reais do MIT-BIH acima.",
          });
          setUploadStatus(`Arquivo ${pdfFile.name}: formato pré-filtrado detectado.`, true);
          return;
        }
        for (const imgExt of [".png", ".jpg", ".jpeg", ".webp"]) {
          if (byExt[imgExt]) {
            const imgFile = byExt[imgExt];
            const blobUrl = URL.createObjectURL(imgFile);
            showPdfOrImageNotice({
              kind: "image_blob_warning",
              filename: imgFile.name,
              blob_url: blobUrl,
              warning_title: "Imagem de exame detectada — Traçado já filtrado e sem amostras brutas x[n]",
              warning_body:
                "Fotografias ou capturas de tela de ECG em papel milimetrado já passaram pelos filtros do equipamento clínico e não fornecem o vetor temporal discreto x[n]. Envie dados brutos (.csv, .txt ou .dat + .hea) para executar o pipeline completo.",
            });
            setUploadStatus(`Arquivo ${imgFile.name}: imagem detectada.`, true);
            return;
          }
        }
        let ecgObj;
        if (byExt[".hea"] && byExt[".dat"]) {
          const heaText = await byExt[".hea"].text();
          const datBuf = await byExt[".dat"].arrayBuffer();
          const stem = byExt[".hea"].name.replace(/\.hea$/i, "");
          ecgObj = window.CardioDSP.parsePhysioNetHeaDat(heaText, datBuf, stem);
        } else if (byExt[".hea"] || byExt[".dat"]) {
          throw new Error("Para registros PhysioNet, envie juntos os arquivos .hea e .dat.");
        } else {
          const tabFile = byExt[".csv"] || byExt[".txt"];
          if (!tabFile) throw new Error("Formato não suportado. Envie .csv, .txt, .dat+.hea ou .pdf.");
          const txt = await tabFile.text();
          ecgObj = window.CardioDSP.parseTabularText(txt, tabFile.name, fallbackFs);
        }
        const data = window.CardioDSP.buildSessionPayload(ecgObj, fp, 0, ecgObj.source_name);
        hidePdfNotice();
        STATE.activeRecordId = null;
        highlightActivePreset(null);
        setUploadStatus(`Importado no navegador: ${data.ecg.source_name}`);
        applySessionPayload(data, true);
      } catch (err) {
        setUploadStatus(err.message, true);
      } finally {
        setLoading(false);
      }
      return;
    }

    const fd = new FormData();
    for (const f of fileList) fd.append("files", f);
    fd.append("fallback_fs", String(fallbackFs));
    fd.append("apply_notch", String(fp.apply_notch));
    fd.append("notch_q", String(fp.notch_q));
    fd.append("apply_bandpass", String(fp.apply_bandpass));
    fd.append("lowcut", String(fp.lowcut));
    fd.append("highcut", String(fp.highcut));
    fd.append("order", String(fp.order));
    fd.append("refractory_ms", String(fp.refractory_ms));

    try {
      const res = await fetch("/api/analyze/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Erro ao processar arquivo enviado.");

      if (data.kind === "pdf_warning" || data.kind === "image_warning") {
        showPdfOrImageNotice(data);
        setUploadStatus(`Arquivo ${data.filename}: formato pré-filtrado detectado.`, true);
        return;
      }

      hidePdfNotice();
      STATE.activeRecordId = null;
      highlightActivePreset(null);
      setUploadStatus(`Importado com sucesso: ${data.ecg.source_name}`);
      applySessionPayload(data, true);
    } catch (err) {
      setUploadStatus(err.message, true);
    } finally {
      setLoading(false);
    }
  }

  async function reprocessActiveSession() {
    if (!STATE.session) return;
    setLoading(true, "Recalculando filtros LIT e FFT...");
    try {
      let data;
      if (STATE.mode === "api") {
        const res = await fetch(`/api/reprocess/${STATE.session.session_id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(getFilterParams()),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Falha ao reprocessar.");
        data = await res.json();
      } else {
        data = window.CardioDSP.buildSessionPayload(
          STATE.session._raw_ecg,
          getFilterParams(),
          STATE.session.suggested_window.start_s,
          STATE.session.suggested_window.label
        );
      }
      applySessionPayload(data, false);
    } catch (err) {
      setUploadStatus(err.message, true);
    } finally {
      setLoading(false);
    }
  }

  async function ensureBufferCovers(startS, durS) {
    if (!STATE.session || STATE.mode === "static") return;
    const totalDur = STATE.session.ecg.duration_s;
    const endS = Math.min(totalDur, startS + durS);
    const b = STATE.buffer;
    const margin = Math.min(8, durS * 0.6);

    const needsFetch =
      !b ||
      startS < b.start_s + (b.start_s > 0 ? margin * 0.25 : 0) ||
      endS > b.end_s - (b.end_s < totalDur ? margin * 0.25 : 0);

    if (!needsFetch || STATE.fetchingWindow) return;

    STATE.fetchingWindow = true;
    try {
      const fetchStart = Math.max(0, startS - 20);
      const fetchDur = Math.min(80, Math.max(durS + 40, 60), totalDur - fetchStart);
      const url = `/api/window/${STATE.session.session_id}?start_s=${fetchStart.toFixed(2)}&duration_s=${fetchDur.toFixed(2)}`;
      const res = await fetch(url);
      if (res.ok) {
        STATE.buffer = await res.json();
        drawEcgCanvas();
      }
    } finally {
      STATE.fetchingWindow = false;
    }
  }

  // -------------------------------------------------------------------------
  // Atualização de Estado e UI
  // -------------------------------------------------------------------------

  function applySessionPayload(data, resetWindow) {
    STATE.session = data;
    STATE.buffer = data.initial_slice;

    const totalDur = data.ecg.duration_s;
    if (resetWindow) {
      STATE.winDurS = Math.min(10, Math.max(2, totalDur));
      STATE.winStartS = Math.min(
        Math.max(0, totalDur - STATE.winDurS),
        data.suggested_window.start_s
      );
      // Ajusta select de duração
      const selDur = $("selWindowDur");
      const validOpts = Array.from(selDur.options).map((o) => parseFloat(o.value));
      const nearest = validOpts.reduce((a, b) =>
        Math.abs(b - STATE.winDurS) < Math.abs(a - STATE.winDurS) ? b : a
      );
      selDur.value = String(nearest);
      if (totalDur < nearest) STATE.winDurS = totalDur;
    } else {
      // Mantém a janela atual, buscando o slice correspondente após refiltragem
      STATE.winStartS = Math.min(STATE.winStartS, Math.max(0, totalDur - STATE.winDurS));
      if (
        STATE.winStartS < STATE.buffer.start_s ||
        STATE.winStartS + STATE.winDurS > STATE.buffer.end_s
      ) {
        ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      }
    }

    // Metadados laterais
    $("metaSource").textContent = data.ecg.source_name;
    $("metaLead").textContent = `${data.ecg.lead_name} (${data.ecg.units})`;
    $("metaFs").textContent = `${data.ecg.fs} Hz`;
    $("metaDur").textContent = `${fmt(totalDur, 1)} s (${data.ecg.n_samples.toLocaleString("pt-BR")} pts)`;
    $("metaPoles").textContent = `max |pᵢ| = ${data.filter_info.max_pole_radius.toFixed(5)} < 1`;
    $("biboBadge").textContent = data.filter_info.is_bibo_stable ? "BIBO Estável" : "Verificar";

    $("wrapShowAtr").classList.toggle("hidden", !data.ecg.has_annotations);
    if (!data.ecg.has_annotations) {
      STATE.showAtr = false;
      $("chkShowAtr").checked = false;
    }

    // Configura timeline slider
    const slider = $("timelineSlider");
    slider.min = "0";
    slider.max = String(Math.max(0, totalDur - STATE.winDurS).toFixed(2));
    slider.value = String(STATE.winStartS.toFixed(2));

    updateTimelineLabels();
    updateMetricsUI();
    renderParsevalTables();
    renderLiveValidation();
    drawActiveTabCanvases();
  }

  function updateTimelineLabels() {
    if (!STATE.session) return;
    const endS = Math.min(STATE.session.ecg.duration_s, STATE.winStartS + STATE.winDurS);
    $("timeStartLbl").textContent = `${fmt(STATE.winStartS, 1)} s`;
    $("timeEndLbl").textContent = `${fmt(endS, 1)} s`;
    $("timelineSlider").value = String(STATE.winStartS.toFixed(2));
  }

  function updateMetricsUI() {
    if (!STATE.session) return;
    const fs = STATE.session.ecg.fs;
    let m;
    if (STATE.scope === "full") {
      m = STATE.session.metrics_full;
    } else {
      const sIdx = Math.round(STATE.winStartS * fs);
      const eIdx = Math.round((STATE.winStartS + STATE.winDurS) * fs);
      const winPeaks = STATE.session.r_peaks_all.filter((p) => p >= sIdx && p < eIdx);
      m = computeMetricsFromPeaks(winPeaks, fs);
    }

    const banner = $("triageBanner");
    banner.classList.remove("status-normal", "status-alert", "status-warning");
    if (m.classification_level === "alert") banner.classList.add("status-alert");
    else if (m.classification_level === "warning" || m.sdnn_ms > 100) banner.classList.add("status-warning");
    else banner.classList.add("status-normal");

    $("diagBadge").textContent = m.classification.toUpperCase();
    $("regularityBadge").textContent = m.rhythm_regularity;
    $("kpiBpmHero").textContent = m.bpm_mean > 0 ? fmt(m.bpm_mean, 1) : "--,-";

    $("kpiRR").textContent = m.mean_rr_ms > 0 ? `${fmt(m.mean_rr_ms, 1)} ms` : "—";
    $("kpiN").textContent = `N̄ = ${fmt(m.mean_n_samples, 1)} amostras`;
    $("kpiSdnn").textContent = `${fmt(m.sdnn_ms, 1)} ms`;
    $("kpiRmssd").textContent = `RMSSD = ${fmt(m.rmssd_ms, 1)} ms`;
    $("kpiBeats").textContent = `${m.num_beats.toLocaleString("pt-BR")}`;
    $("kpiBpmRange").textContent =
      m.bpm_max > 0 ? `Faixa: ${fmt(m.bpm_min, 0)}–${fmt(m.bpm_max, 0)} BPM` : "Faixa: —";

    const sp = STATE.session.spectral;
    $("kpiParseval").textContent = `${fmt(sp.noise_removed_total_pct, 1)}%`;
    $("kpiParsevalAc").textContent = `Banda AC (f > 0): ${fmt(sp.noise_removed_ac_pct, 1)}%`;
  }

  function showPdfOrImageNotice(data) {
    const sec = $("pdfNoticeSection");
    $("pdfWarningTitle").textContent = `⚠️ ${data.warning_title}`;
    $("pdfWarningBody").textContent = data.warning_body;
    const grid = $("pdfPreviewGrid");
    grid.innerHTML = "";

    if (data.kind === "pdf_warning" && data.pages_b64) {
      data.pages_b64.forEach((b64, i) => {
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${b64}`;
        img.alt = `Pré-visualização página ${i + 1} de ${data.filename}`;
        grid.appendChild(img);
      });
    } else if (data.kind === "pdf_blob_warning" && data.blob_url) {
      const embed = document.createElement("iframe");
      embed.src = data.blob_url;
      embed.title = `Pré-visualização de ${data.filename}`;
      embed.style.width = "100%";
      embed.style.height = "520px";
      embed.style.border = "1px solid #cbd5e1";
      embed.style.borderRadius = "6px";
      embed.style.background = "#ffffff";
      grid.appendChild(embed);
    } else if (data.kind === "image_warning" && data.image_b64) {
      const img = document.createElement("img");
      img.src = `data:${data.mime};base64,${data.image_b64}`;
      img.alt = `Pré-visualização de ${data.filename}`;
      grid.appendChild(img);
    } else if (data.kind === "image_blob_warning" && data.blob_url) {
      const img = document.createElement("img");
      img.src = data.blob_url;
      img.alt = `Pré-visualização de ${data.filename}`;
      grid.appendChild(img);
    }

    sec.classList.remove("hidden");
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function hidePdfNotice() {
    $("pdfNoticeSection").classList.add("hidden");
  }

  // -------------------------------------------------------------------------
  // Renderização 60 FPS do Monitor ECG em Papel Milimetrado
  // -------------------------------------------------------------------------

  function drawMillimeterGrid(ctx, x0, y0, pw, ph, tStart, tEnd, yMin, yMax) {
    ctx.fillStyle = "#FFFDFB";
    ctx.fillRect(x0, y0, pw, ph);

    const tSpan = tEnd - tStart;
    const ySpan = yMax - yMin;
    const xOf = (t) => x0 + ((t - tStart) / tSpan) * pw;
    const yOf = (v) => y0 + ph - ((v - yMin) / ySpan) * ph;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, pw, ph);
    ctx.clip();

    // Grade menor (0,04 s × 0,1 mV) — oculta se zoom muito aberto para manter nitidez
    if (tSpan <= 18) {
      ctx.strokeStyle = "#F5D0CE";
      ctx.lineWidth = 0.65;
      ctx.beginPath();
      const t0Minor = Math.ceil(tStart / 0.04) * 0.04;
      for (let t = t0Minor; t <= tEnd; t += 0.04) {
        const x = Math.round(xOf(t)) + 0.5;
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y0 + ph);
      }
      const y0Minor = Math.ceil(yMin / 0.1) * 0.1;
      for (let v = y0Minor; v <= yMax; v += 0.1) {
        const y = Math.round(yOf(v)) + 0.5;
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + pw, y);
      }
      ctx.stroke();
    }

    // Grade maior (0,20 s × 0,5 mV)
    ctx.strokeStyle = "#E5989B";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    const tStepMajor = tSpan > 22 ? 0.4 : 0.2;
    const t0Maj = Math.ceil(tStart / tStepMajor) * tStepMajor;
    for (let t = t0Maj; t <= tEnd; t += tStepMajor) {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y0 + ph);
    }
    const y0Maj = Math.ceil(yMin / 0.5) * 0.5;
    for (let v = y0Maj; v <= yMax; v += 0.5) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + pw, y);
    }
    ctx.stroke();

    // Linha de referência 0 mV
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = "rgba(100, 116, 139, 0.35)";
      ctx.lineWidth = 1;
      const yz = Math.round(yOf(0)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x0, yz);
      ctx.lineTo(x0 + pw, yz);
      ctx.stroke();
    }

    ctx.restore();

    // Borda e rótulos dos eixos
    ctx.strokeStyle = "#94A3B8";
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);

    ctx.fillStyle = "#334155";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = y0Maj; v <= yMax + 1e-6; v += 0.5) {
      const y = yOf(v);
      if (y >= y0 + 8 && y <= y0 + ph - 8) {
        ctx.fillText(v.toFixed(1), x0 - 6, y);
      }
    }

    return { xOf, yOf };
  }

  function drawEcgCanvas() {
    const canvas = $("ecgCanvas");
    if (!canvas || !STATE.session || !STATE.buffer) return;
    const { ctx, w, h } = setupHiDpiCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const b = STATE.buffer;
    const fs = b.fs;
    const tStart = STATE.winStartS;
    const tEnd = Math.min(STATE.session.ecg.duration_s, tStart + STATE.winDurS);

    const idxStart = Math.max(0, Math.floor(tStart * fs) - b.start_idx);
    const idxEnd = Math.min(b.raw.length - 1, Math.ceil(tEnd * fs) - b.start_idx);
    if (idxEnd <= idxStart + 2) return;

    const padL = 52;
    const padR = 16;
    const padT = 28;
    const padB = 34;
    const pw = w - padL - padR;

    const rawSlice = b.raw.slice(idxStart, idxEnd + 1);
    const filtSlice = b.filtered.slice(idxStart, idxEnd + 1);

    const calcBounds = (arr, extraTop = 0.22) => {
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
      }
      const span = Math.max(0.8, mx - mn);
      return [mn - span * 0.16, mx + span * extraTop];
    };

    const drawTrace = (arr, xOf, yOf, color, lineWidth) => {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      const step = Math.max(1, Math.floor(arr.length / (pw * 2)));
      for (let i = 0; i < arr.length; i += step) {
        const t = (b.start_idx + idxStart + i) / fs;
        const x = xOf(t);
        const y = yOf(arr[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    const drawPeaksAndCaliper = (xOf, yOf, yTopBox) => {
      const sAbs = b.start_idx + idxStart;
      const eAbs = b.start_idx + idxEnd;
      const peaks = b.r_peaks.filter((p) => p >= sAbs && p <= eAbs);

      // Opcional: anotações oficiais MIT-BIH (.atr) em verde-azulado
      if (STATE.showAtr && b.atr_peaks && b.atr_peaks.length > 0) {
        ctx.strokeStyle = "#0F766E";
        ctx.lineWidth = 2;
        for (const ap of b.atr_peaks) {
          if (ap < sAbs || ap > eAbs) continue;
          const rel = ap - b.start_idx;
          if (rel < 0 || rel >= b.filtered.length) continue;
          const x = xOf(ap / fs);
          const y = yOf(b.filtered[rel]);
          ctx.beginPath();
          ctx.arc(x, y, 7, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Picos R detectados em vermelho clínico
      for (const p of peaks) {
        const rel = p - b.start_idx;
        if (rel < 0 || rel >= b.filtered.length) continue;
        const x = xOf(p / fs);
        const y = yOf(b.filtered[rel]);
        ctx.beginPath();
        ctx.fillStyle = "#FFFFFF";
        ctx.arc(x, y, 5.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.fillStyle = "#DC2626";
        ctx.arc(x, y, 3.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Compasso de periodicidade discreta N sobre o par central (ou par próximo ao mouse)
      if (peaks.length >= 2) {
        let pairIdx = Math.floor((peaks.length - 1) / 2);
        if (STATE.hoverX !== null) {
          const hoverT = tStart + ((STATE.hoverX - padL) / pw) * (tEnd - tStart);
          const hoverSample = hoverT * fs;
          for (let i = 0; i < peaks.length - 1; i++) {
            if (hoverSample >= peaks[i] && hoverSample <= peaks[i + 1]) {
              pairIdx = i;
              break;
            }
          }
        }
        const p0 = peaks[pairIdx];
        const p1 = peaks[pairIdx + 1];
        const x0 = xOf(p0 / fs);
        const x1 = xOf(p1 / fs);
        const yBar = yTopBox + 22;
        const dn = p1 - p0;
        const rrMs = (dn / fs) * 1000;
        const instBpm = 60000 / rrMs;

        ctx.strokeStyle = "#0F766E";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(x0, yBar);
        ctx.lineTo(x1, yBar);
        ctx.moveTo(x0, yBar - 5);
        ctx.lineTo(x0, yBar + 5);
        ctx.moveTo(x1, yBar - 5);
        ctx.lineTo(x1, yBar + 5);
        ctx.stroke();

        const lbl = `N = ${dn} amostras · ${Math.round(rrMs)} ms (${fmt(instBpm, 0)} BPM)`;
        ctx.font = "600 11px 'IBM Plex Mono', monospace";
        const tw = ctx.measureText(lbl).width + 12;
        const cx = Math.min(Math.max((x0 + x1) / 2, padL + tw / 2 + 4), padL + pw - tw / 2 - 4);
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.strokeStyle = "#0F766E";
        ctx.lineWidth = 1.1;
        ctx.fillRect(cx - tw / 2, yBar - 19, tw, 17);
        ctx.strokeRect(cx - tw / 2, yBar - 19, tw, 17);
        ctx.fillStyle = "#0F766E";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(lbl, cx, yBar - 10);
      }
    };

    const drawXAxisTicks = (yAxis) => {
      ctx.fillStyle = "#334155";
      ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const tSpan = tEnd - tStart;
      const step = tSpan <= 6 ? 0.5 : tSpan <= 16 ? 1.0 : 2.0;
      const t0 = Math.ceil(tStart / step) * step;
      for (let t = t0; t <= tEnd + 1e-6; t += step) {
        const x = padL + ((t - tStart) / tSpan) * pw;
        ctx.fillText(`${t.toFixed(step < 1 ? 1 : 0)}s`, x, yAxis + 5);
      }
      ctx.font = "500 11px 'IBM Plex Sans', sans-serif";
      ctx.fillStyle = "#475569";
      ctx.fillText(
        "Tempo (s) — Escala clínica de papel milimetrado (0,04 s / 0,20 s horizontal · 0,1 mV / 0,5 mV vertical)",
        padL + pw / 2,
        yAxis + 19
      );
    };

    if (STATE.viewMode === "split") {
      const gap = 34;
      const trackH = Math.floor((h - padT - padB - gap) / 2);
      const yTop1 = padT;
      const yTop2 = padT + trackH + gap;

      // Título faixa 1
      ctx.fillStyle = "#334155";
      ctx.font = "700 12px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(
        `ANTES — Sinal Bruto x[n] (${STATE.session.ecg.source_name} · fs = ${fs} Hz)`,
        padL,
        yTop1 - 6
      );

      const [rMin, rMax] = calcBounds(rawSlice, 0.18);
      const g1 = drawMillimeterGrid(ctx, padL, yTop1, pw, trackH, tStart, tEnd, rMin, rMax);
      drawTrace(rawSlice, g1.xOf, g1.yOf, "#52606D", 1.3);

      // Título faixa 2
      ctx.fillStyle = "#0B2545";
      ctx.font = "700 12px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(
        "DEPOIS — Sinal Filtrado (Notch 60 Hz + Butterworth Fase Zero) + Picos R em vermelho",
        padL,
        yTop2 - 6
      );

      const [fMin, fMax] = calcBounds(filtSlice, 0.34);
      const g2 = drawMillimeterGrid(ctx, padL, yTop2, pw, trackH, tStart, tEnd, fMin, fMax);
      drawTrace(filtSlice, g2.xOf, g2.yOf, "#0B2545", 1.85);
      drawPeaksAndCaliper(g2.xOf, g2.yOf, yTop2);
      drawXAxisTicks(yTop2 + trackH);
    } else {
      // Modo Sobreposição Direta (evidencia fase zero exata!)
      const trackH = h - padT - padB;
      ctx.fillStyle = "#0B2545";
      ctx.font = "700 12px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(
        "SOBREPOSIÇÃO DIRETA — Cinza: Sinal Bruto x[n]  |  Azul Marinho: Filtrado Fase Zero (sem atraso temporal)",
        padL,
        padT - 6
      );

      const combined = rawSlice.concat(filtSlice);
      const [cMin, cMax] = calcBounds(combined, 0.26);
      const g = drawMillimeterGrid(ctx, padL, padT, pw, trackH, tStart, tEnd, cMin, cMax);
      drawTrace(rawSlice, g.xOf, g.yOf, "rgba(100, 116, 139, 0.55)", 1.25);
      drawTrace(filtSlice, g.xOf, g.yOf, "#0B2545", 2.0);
      drawPeaksAndCaliper(g.xOf, g.yOf, padT);
      drawXAxisTicks(padT + trackH);
    }

    // Linha vertical de cursor / hover crosshair
    if (STATE.hoverX !== null && STATE.hoverX >= padL && STATE.hoverX <= padL + pw) {
      const hoverT = tStart + ((STATE.hoverX - padL) / pw) * (tEnd - tStart);
      const sampleAbs = Math.round(hoverT * fs);
      const rel = sampleAbs - b.start_idx;

      ctx.strokeStyle = "rgba(11, 37, 69, 0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(STATE.hoverX, padT);
      ctx.lineTo(STATE.hoverX, h - padB);
      ctx.stroke();
      ctx.setLineDash([]);

      if (rel >= 0 && rel < b.raw.length) {
        const rv = b.raw[rel];
        const fv = b.filtered[rel];
        const el = $("hoverReadout");
        el.textContent = `t = ${hoverT.toFixed(3)} s (n=${sampleAbs}) | Bruto: ${rv.toFixed(3)} mV | Filtrado: ${fv.toFixed(3)} mV | ΔRuído: ${(rv - fv).toFixed(3)} mV`;
        el.classList.remove("hidden");
      }
    } else {
      $("hoverReadout").classList.add("hidden");
    }
  }

  // -------------------------------------------------------------------------
  // Renderização dos Gráficos FFT, Bode, Plano Z e Tacograma R-R
  // -------------------------------------------------------------------------

  function drawFftSeries(canvasId, series, xMin, xMax, opts = {}) {
    const canvas = $(canvasId);
    if (!canvas || !series || !series.freqs || series.freqs.length === 0) return;
    const { ctx, w, h } = setupHiDpiCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const padL = 56;
    const padR = 16;
    const padT = 18;
    const padB = 34;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    const allY = series.raw_db.concat(series.filt_db);
    const yMax = Math.max(-15, Math.ceil(Math.max(...allY) / 10) * 10 + 5);
    const yMin = opts.yMin ?? -125;

    const xOf = (f) => padL + ((f - xMin) / (xMax - xMin)) * pw;
    const yOf = (db) => padT + ph - ((Math.max(yMin, Math.min(yMax, db)) - yMin) / (yMax - yMin)) * ph;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(padL, padT, pw, ph);

    // Bandas sombreadas
    if (opts.showBands) {
      const band = (a, b, color) => {
        const xa = xOf(Math.max(xMin, a));
        const xb = xOf(Math.min(xMax, b));
        if (xb > xa) {
          ctx.fillStyle = color;
          ctx.fillRect(xa, padT, xb - xa, ph);
        }
      };
      band(0, 0.5, "rgba(252, 165, 165, 0.32)");
      band(0.5, 40, "rgba(187, 247, 208, 0.24)");
      band(40, xMax, "rgba(253, 230, 138, 0.26)");
    } else if (opts.highlightRange) {
      const [a, b] = opts.highlightRange;
      ctx.fillStyle = "rgba(252, 165, 165, 0.30)";
      ctx.fillRect(xOf(a), padT, xOf(b) - xOf(a), ph);
    }

    // Grade
    ctx.strokeStyle = "#E2E8F0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let db = Math.ceil(yMin / 20) * 20; db <= yMax; db += 20) {
      const y = Math.round(yOf(db)) + 0.5;
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
    }
    const xStep = xMax - xMin <= 12 ? 1 : 10;
    for (let f = Math.ceil(xMin / xStep) * xStep; f <= xMax; f += xStep) {
      const x = Math.round(xOf(f)) + 0.5;
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + ph);
    }
    ctx.stroke();

    // Linha vertical de referência (0,5 Hz ou 60 Hz)
    if (opts.vLine && opts.vLine >= xMin && opts.vLine <= xMax) {
      ctx.strokeStyle = "#DC2626";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 4]);
      const xv = xOf(opts.vLine);
      ctx.beginPath();
      ctx.moveTo(xv, padT);
      ctx.lineTo(xv, padT + ph);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const plotLine = (arr, color, lw) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < series.freqs.length; i++) {
        const x = xOf(series.freqs[i]);
        const y = yOf(arr[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    plotLine(series.raw_db, "#64748B", 1.35);
    plotLine(series.filt_db, "#0B2545", 2.0);

    ctx.strokeStyle = "#94A3B8";
    ctx.strokeRect(padL + 0.5, padT + 0.5, pw - 1, ph - 1);

    // Rótulos
    ctx.fillStyle = "#334155";
    ctx.font = "10.5px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let db = Math.ceil(yMin / 20) * 20; db <= yMax; db += 20) {
      ctx.fillText(`${db} dB`, padL - 6, yOf(db));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let f = Math.ceil(xMin / xStep) * xStep; f <= xMax; f += xStep) {
      ctx.fillText(`${f} Hz`, xOf(f), padT + ph + 6);
    }

    // Legenda compacta
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(padL + pw - 210, padT + 6, 202, 22);
    ctx.strokeStyle = "#CBD5E1";
    ctx.strokeRect(padL + pw - 210, padT + 6, 202, 22);
    ctx.font = "600 10.5px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#64748B";
    ctx.fillRect(padL + pw - 202, padT + 15, 14, 3);
    ctx.fillText("Antes (Bruto)", padL + pw - 183, padT + 17);
    ctx.fillStyle = "#0B2545";
    ctx.fillRect(padL + pw - 102, padT + 15, 14, 3);
    ctx.fillText("Depois (LIT)", padL + pw - 83, padT + 17);
  }

  function drawBodeCanvas() {
    const canvas = $("bodeCanvas");
    if (!canvas || !STATE.session) return;
    const b = STATE.session.bode;
    const { ctx, w, h } = setupHiDpiCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const padL = 52, padR = 14, padT = 14, padB = 32;
    const pw = w - padL - padR, ph = h - padT - padB;
    const xMax = b.freqs[b.freqs.length - 1] || 100;
    const yMin = -85, yMax = 5;
    const xOf = (f) => padL + (f / xMax) * pw;
    const yOf = (db) => padT + ph - ((Math.max(yMin, Math.min(yMax, db)) - yMin) / (yMax - yMin)) * ph;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(padL, padT, pw, ph);
    ctx.fillStyle = "rgba(187,247,208,0.25)";
    ctx.fillRect(xOf(0.5), padT, xOf(Math.min(40, xMax)) - xOf(0.5), ph);

    ctx.strokeStyle = "#E2E8F0";
    ctx.beginPath();
    for (let db = -80; db <= 0; db += 20) {
      const y = Math.round(yOf(db)) + 0.5;
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
    }
    for (let f = 0; f <= xMax; f += 10) {
      const x = Math.round(xOf(f)) + 0.5;
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + ph);
    }
    ctx.stroke();

    // Linha -3 dB
    ctx.strokeStyle = "#94A3B8";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, yOf(-3));
    ctx.lineTo(padL + pw, yOf(-3));
    ctx.stroke();
    ctx.setLineDash([]);

    const trace = (arr, color, lw) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < b.freqs.length; i++) {
        const x = xOf(b.freqs[i]);
        const y = yOf(arr[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    trace(b.notch_db, "rgba(220,38,38,0.65)", 1.2);
    trace(b.bp_db, "rgba(15,118,110,0.75)", 1.3);
    trace(b.cascade_db, "#0B2545", 2.2);

    ctx.strokeStyle = "#94A3B8";
    ctx.strokeRect(padL + 0.5, padT + 0.5, pw - 1, ph - 1);

    ctx.fillStyle = "#334155";
    ctx.font = "10.5px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let db = -80; db <= 0; db += 20) ctx.fillText(`${db} dB`, padL - 6, yOf(db));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let f = 0; f <= xMax; f += 20) ctx.fillText(`${f} Hz`, xOf(f), padT + ph + 6);
  }

  function drawZPlaneCanvas() {
    const canvas = $("zplaneCanvas");
    if (!canvas || !STATE.session) return;
    const b = STATE.session.bode;
    const fi = STATE.session.filter_info;
    $("zPlaneSubtitle").textContent = `max |pᵢ| = ${fi.max_pole_radius.toFixed(5)} < 1 (${fi.is_bibo_stable ? "BIBO Estável" : "Instável"})`;

    const { ctx, w, h } = setupHiDpiCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.37;

    // Eixos Re(z) e Im(z)
    ctx.strokeStyle = "#CBD5E1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - radius * 1.25, cy);
    ctx.lineTo(cx + radius * 1.25, cy);
    ctx.moveTo(cx, cy - radius * 1.25);
    ctx.lineTo(cx, cy + radius * 1.25);
    ctx.stroke();

    // Círculo Unitário |z| = 1
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Polos em X vermelho
    ctx.strokeStyle = "#DC2626";
    ctx.lineWidth = 2.1;
    for (let i = 0; i < b.poles_re.length; i++) {
      const px = cx + b.poles_re[i] * radius;
      const py = cy - b.poles_im[i] * radius;
      const s = 5.5;
      ctx.beginPath();
      ctx.moveTo(px - s, py - s);
      ctx.lineTo(px + s, py + s);
      ctx.moveTo(px + s, py - s);
      ctx.lineTo(px - s, py + s);
      ctx.stroke();
    }

    ctx.fillStyle = "#475569";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("Re(z)", cx + radius * 1.05, cy - 6);
    ctx.fillText("Im(z)", cx + 6, cy - radius * 1.12);
    ctx.fillStyle = "#15803D";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.fillText("|z| = 1", cx - radius * 0.95, cy - radius * 0.85);
  }

  function drawRrCanvas() {
    const canvas = $("rrCanvas");
    if (!canvas || !STATE.session) return;
    const tg = STATE.session.tachogram;
    const { ctx, w, h } = setupHiDpiCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    if (!tg.times_s || tg.times_s.length < 2) return;

    const padL = 54, padR = 16, padT = 16, padB = 34;
    const pw = w - padL - padR, ph = h - padT - padB;
    const xMax = STATE.session.ecg.duration_s;
    const yMin = Math.max(20, Math.min(45, Math.floor(Math.min(...tg.instant_bpm) / 10) * 10 - 5));
    const yMax = Math.min(220, Math.max(145, Math.ceil(Math.max(...tg.instant_bpm) / 10) * 10 + 5));

    const xOf = (t) => padL + (t / xMax) * pw;
    const yOf = (v) => padT + ph - ((Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * ph;

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(padL, padT, pw, ph);

    // Faixa verde 60–120 BPM
    ctx.fillStyle = "rgba(187, 247, 208, 0.30)";
    ctx.fillRect(padL, yOf(120), pw, yOf(60) - yOf(120));

    // Limiares 60 e 120 BPM
    ctx.strokeStyle = "#DC2626";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    [60, 120].forEach((lim) => {
      const y = yOf(lim);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Linha de BPM instantâneo
    ctx.strokeStyle = "#0F766E";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < tg.times_s.length; i++) {
      const x = xOf(tg.times_s[i]);
      const y = yOf(tg.instant_bpm[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = "#94A3B8";
    ctx.strokeRect(padL + 0.5, padT + 0.5, pw - 1, ph - 1);

    ctx.fillStyle = "#334155";
    ctx.font = "10.5px 'IBM Plex Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = 40; v <= yMax; v += 20) {
      if (v >= yMin) ctx.fillText(`${v}`, padL - 6, yOf(v));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xStep = xMax > 300 ? 300 : xMax > 60 ? 30 : 5;
    for (let t = 0; t <= xMax; t += xStep) ctx.fillText(`${t}s`, xOf(t), padT + ph + 6);
  }

  function drawActiveTabCanvases() {
    if (!STATE.session) return;
    if (STATE.activeTab === "ecg") {
      drawEcgCanvas();
    } else if (STATE.activeTab === "fft") {
      const sp = STATE.session.spectral;
      const fMax = Math.min(100, STATE.session.ecg.fs / 2);
      drawFftSeries("fftMainCanvas", sp.fft_main, 0, fMax, { showBands: true, vLine: 60 });
      drawFftSeries("fftLowCanvas", sp.fft_low, 0, Math.min(5, fMax), { highlightRange: [0, 0.5], vLine: 0.5 });
      if (sp.fft_notch.freqs.length > 0) {
        drawFftSeries("fftNotchCanvas", sp.fft_notch, 55, 65, { highlightRange: [59, 61], vLine: 60 });
      }
    } else if (STATE.activeTab === "lti") {
      drawBodeCanvas();
      drawZPlaneCanvas();
      drawRrCanvas();
    }
  }

  // -------------------------------------------------------------------------
  // Tabelas (Parseval, Validação MIT-BIH e Literatura)
  // -------------------------------------------------------------------------

  function renderParsevalTables() {
    const sp = STATE.session.spectral;
    $("parsevalSummaryTable").innerHTML = `
      <thead>
        <tr><th>Sinal</th><th>P Tempo (mV²)</th><th>P Parseval Σ|cₖ|²</th><th>Erro Relativo</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Antes (Bruto)</td>
          <td class="mono">${sp.power_time_raw.toFixed(6)}</td>
          <td class="mono">${sp.power_parseval_raw.toFixed(6)}</td>
          <td class="mono">${fmtSci(sp.parseval_rel_error_raw)}</td>
        </tr>
        <tr>
          <td>Depois (Filtrado)</td>
          <td class="mono">${sp.power_time_filtered.toFixed(6)}</td>
          <td class="mono">${sp.power_parseval_filtered.toFixed(6)}</td>
          <td class="mono">${fmtSci(sp.parseval_rel_error_filtered)}</td>
        </tr>
        <tr class="row-total">
          <td>Energia Removida</td>
          <td class="mono" colspan="3">Total: ${fmt(sp.noise_removed_total_pct, 2)}% · Banda AC (f &gt; 0 Hz): ${fmt(sp.noise_removed_ac_pct, 2)}%</td>
        </tr>
      </tbody>
    `;

    const rows = sp.band_table
      .map(
        (b) => `<tr>
          <td>${b.band}</td>
          <td class="mono">${b.power_raw.toFixed(6)}</td>
          <td class="mono">${b.power_filtered.toFixed(6)}</td>
          <td class="mono">${fmt(b.attenuation_pct, 2)}%</td>
          <td class="mono">${fmt(b.attenuation_db, 1)} dB</td>
        </tr>`
      )
      .join("");
    $("parsevalBandTable").innerHTML = `
      <thead>
        <tr><th>Banda</th><th>Antes (mV²)</th><th>Depois (mV²)</th><th>Atenuação</th><th>Ganho</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    `;
  }

  function renderLiveValidation() {
    const card = $("liveValCard");
    const lv = STATE.session.live_validation;
    if (!lv) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    $("liveValTitle").textContent = `Validação em Tempo Real (.atr ±150 ms) — ${STATE.session.ecg.source_name}`;
    $("liveValMetrics").innerHTML = `
      <div class="lv-item"><span>Reais (.atr)</span><strong>${lv.total_real.toLocaleString("pt-BR")}</strong></div>
      <div class="lv-item"><span>Detectados</span><strong>${lv.total_detected.toLocaleString("pt-BR")}</strong></div>
      <div class="lv-item"><span>TP / FP / FN</span><strong>${lv.tp} / ${lv.fp} / ${lv.fn}</strong></div>
      <div class="lv-item"><span>Sensibilidade</span><strong>${fmt(lv.sensitivity_pct, 2)}%</strong></div>
      <div class="lv-item"><span>Precisão (+P)</span><strong>${fmt(lv.precision_pct, 2)}%</strong></div>
      <div class="lv-item"><span>F1-Score</span><strong>${fmt(lv.f1_score_pct, 2)}%</strong></div>
    `;
  }

  function renderValidationTables() {
    const rows = STATE.meta.validation_table || [];
    if (rows.length > 0) {
      const bodyHtml = rows
        .map((r) => {
          const isTot = r.Registro === "TOTAL_GLOBAL";
          return `<tr class="${isTot ? "row-total" : ""}">
            <td class="mono">${isTot ? "TOTAL GLOBAL" : r.Registro}</td>
            <td>${r.Perfil_Clinico}</td>
            <td class="mono">${r.Batimentos_Reais}</td>
            <td class="mono">${r.TP} / ${r.FP} / ${r.FN}</td>
            <td class="mono">${fmt(r["Sensibilidade_%"], 2)}%</td>
            <td class="mono">${fmt(r["Precisao_%"], 2)}%</td>
            <td class="mono">${fmt(r["F1_Score_%"], 2)}%</td>
            <td class="mono">${fmt(r.BPM_Medio, 1)}</td>
            <td class="mono">${fmt(r.SDNN_ms, 1)} ms</td>
            <td>${r.Classificacao}</td>
          </tr>`;
        })
        .join("");
      $("mitbihValidationTable").innerHTML = `
        <thead>
          <tr>
            <th>Reg.</th><th>Perfil Clínico</th><th>Reais</th><th>TP / FP / FN</th>
            <th>Sensib.</th><th>Precisão (+P)</th><th>F1-Score</th><th>BPM</th><th>SDNN</th><th>Classificação</th>
          </tr>
        </thead>
        <tbody>${bodyHtml}</tbody>
      `;
    }

    const litRows = STATE.meta.literature_benchmarks
      .map(
        (b) => `<tr>
          <td><strong>${b["Estudo"]}</strong></td>
          <td>${b["Método Principal"]}</td>
          <td class="mono">${b["Sensibilidade (%)"]}%</td>
          <td class="mono">${b["Precisão / +P (%)"]}%</td>
          <td>${b["Observação"]}</td>
        </tr>`
      )
      .join("");
    $("literatureTable").innerHTML = `
      <thead>
        <tr><th>Estudo</th><th>Método Principal</th><th>Sensibilidade</th><th>Precisão (+P)</th><th>Observação</th></tr>
      </thead>
      <tbody>${litRows}</tbody>
    `;
  }

  // -------------------------------------------------------------------------
  // Reprodução Fluida 60 FPS (Monitor Cardíaco em Tempo Real)
  // -------------------------------------------------------------------------

  function startPlayback() {
    if (!STATE.session) return;
    STATE.playing = true;
    STATE.lastFrameTs = performance.now();
    $("btnPlayPause").textContent = "⏸ Pausar";
    requestAnimationFrame(playbackTick);
  }

  function stopPlayback() {
    STATE.playing = false;
    $("btnPlayPause").textContent = "▶ Reproduzir (25 mm/s)";
  }

  function playbackTick(ts) {
    if (!STATE.playing || !STATE.session) return;
    const dtSec = Math.min(0.1, (ts - STATE.lastFrameTs) / 1000) * STATE.playSpeed;
    STATE.lastFrameTs = ts;

    const maxStart = Math.max(0, STATE.session.ecg.duration_s - STATE.winDurS);
    STATE.winStartS += dtSec;
    if (STATE.winStartS >= maxStart) {
      STATE.winStartS = maxStart;
      stopPlayback();
    }

    ensureBufferCovers(STATE.winStartS, STATE.winDurS);
    updateTimelineLabels();
    if (STATE.scope === "window") updateMetricsUI();
    drawEcgCanvas();

    if (STATE.playing) requestAnimationFrame(playbackTick);
  }

  // -------------------------------------------------------------------------
  // Eventos de UI, Upload, Drag-and-Drop e Canvas Interativo
  // -------------------------------------------------------------------------

  function bindEvents() {
    // Upload via clique ou botão principal
    const triggerPick = () => $("fileInput").click();
    $("btnOpenUpload").addEventListener("click", triggerPick);
    $("uploadDropzone").addEventListener("click", triggerPick);
    $("uploadDropzone").addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") triggerPick();
    });
    $("fileInput").addEventListener("change", (e) => {
      uploadFiles(e.target.files);
      e.target.value = "";
    });
    $("btnQuickDemoCsv").addEventListener("click", loadDemoCsv);
    $("btnClosePdfNotice").addEventListener("click", hidePdfNotice);

    // Drag & Drop em qualquer lugar da janela
    let dragCounter = 0;
    window.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragCounter++;
      $("globalDropOverlay").classList.remove("hidden");
    });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("dragleave", () => {
      dragCounter = Math.max(0, dragCounter - 1);
      if (dragCounter === 0) $("globalDropOverlay").classList.add("hidden");
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      dragCounter = 0;
      $("globalDropOverlay").classList.add("hidden");
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    });

    // Sliders dos filtros LIT
    $("sldNotchQ").addEventListener("input", (e) => ($("valNotchQ").textContent = e.target.value));
    $("sldLowcut").addEventListener("input", (e) => ($("valLowcut").textContent = `${e.target.value.replace(".", ",")} Hz`));
    $("sldHighcut").addEventListener("input", (e) => ($("valHighcut").textContent = `${e.target.value} Hz`));
    $("sldOrder").addEventListener("input", (e) => ($("valOrder").textContent = e.target.value));
    $("sldRefractory").addEventListener("input", (e) => ($("valRefractory").textContent = `${e.target.value} ms`));
    $("btnApplyFilters").addEventListener("click", reprocessActiveSession);
    $("chkNotch").addEventListener("change", reprocessActiveSession);
    $("chkBandpass").addEventListener("change", reprocessActiveSession);

    // Escopo das métricas
    document.querySelectorAll(".scope-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.scope = btn.dataset.scope;
        document.querySelectorAll(".scope-btn").forEach((b) => b.classList.toggle("active", b === btn));
        updateMetricsUI();
      });
    });

    // Abas
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.activeTab = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${STATE.activeTab}`));
        requestAnimationFrame(drawActiveTabCanvases);
      });
    });

    // Controles do Monitor ECG
    $("btnPlayPause").addEventListener("click", () => (STATE.playing ? stopPlayback() : startPlayback()));
    $("selPlaySpeed").addEventListener("change", (e) => (STATE.playSpeed = parseFloat(e.target.value)));
    $("btnJumpHighlight").addEventListener("click", () => {
      if (!STATE.session) return;
      stopPlayback();
      STATE.winStartS = Math.min(
        Math.max(0, STATE.session.ecg.duration_s - STATE.winDurS),
        STATE.session.suggested_window.start_s
      );
      ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      updateTimelineLabels();
      updateMetricsUI();
      drawEcgCanvas();
    });

    $("selWindowDur").addEventListener("change", (e) => {
      if (!STATE.session) return;
      STATE.winDurS = Math.min(parseFloat(e.target.value), STATE.session.ecg.duration_s);
      const maxStart = Math.max(0, STATE.session.ecg.duration_s - STATE.winDurS);
      STATE.winStartS = Math.min(STATE.winStartS, maxStart);
      $("timelineSlider").max = String(maxStart.toFixed(2));
      ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      updateTimelineLabels();
      updateMetricsUI();
      drawEcgCanvas();
    });

    $("selViewMode").addEventListener("change", (e) => {
      STATE.viewMode = e.target.value;
      drawEcgCanvas();
    });
    $("chkShowAtr").addEventListener("change", (e) => {
      STATE.showAtr = e.target.checked;
      drawEcgCanvas();
    });

    $("timelineSlider").addEventListener("input", (e) => {
      if (!STATE.session) return;
      stopPlayback();
      STATE.winStartS = parseFloat(e.target.value);
      ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      updateTimelineLabels();
      if (STATE.scope === "window") updateMetricsUI();
      drawEcgCanvas();
    });

    // Exportar PNG e CSV (funciona tanto no FastAPI quanto 100% offline no GitHub Pages)
    $("btnExportPng").addEventListener("click", () => {
      if (!STATE.session) return;
      if (STATE.mode === "api") {
        window.location.href = `/api/export/${STATE.session.session_id}/png?start_s=${STATE.winStartS.toFixed(2)}&duration_s=${Math.min(30, STATE.winDurS).toFixed(2)}`;
      } else {
        const link = document.createElement("a");
        link.download = `ecg_${STATE.session.ecg.record_id || "importado"}_${Math.round(STATE.winStartS)}s.png`;
        link.href = $("ecgCanvas").toDataURL("image/png");
        link.click();
      }
    });
    $("btnExportCsv").addEventListener("click", () => {
      if (!STATE.session) return;
      if (STATE.mode === "api") {
        window.location.href = `/api/export/${STATE.session.session_id}/csv?start_s=${STATE.winStartS.toFixed(2)}&duration_s=${STATE.winDurS.toFixed(2)}`;
      } else {
        const b = STATE.buffer;
        const fs = b.fs;
        const sIdx = Math.max(0, Math.floor(STATE.winStartS * fs) - b.start_idx);
        const eIdx = Math.min(b.raw.length, Math.ceil((STATE.winStartS + STATE.winDurS) * fs) - b.start_idx);
        const lines = ["tempo_s,ecg_bruto_mV,ecg_filtrado_mV"];
        for (let i = sIdx; i < eIdx; i++) {
          lines.push(`${((i - sIdx) / fs).toFixed(6)},${b.raw[i].toFixed(6)},${b.filtered[i].toFixed(6)}`);
        }
        const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `ecg_processado_${STATE.session.ecg.record_id || "importado"}_${Math.round(STATE.winStartS)}s.csv`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });

    // Pan & Zoom 60 FPS no Canvas ECG
    const canvas = $("ecgCanvas");
    canvas.addEventListener("mousedown", (e) => {
      if (!STATE.session) return;
      stopPlayback();
      STATE.dragging = true;
      STATE.dragStartX = e.clientX;
      STATE.dragStartWinS = STATE.winStartS;
    });
    window.addEventListener("mouseup", () => {
      if (STATE.dragging) {
        STATE.dragging = false;
        ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      }
    });
    window.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
        STATE.hoverX = e.clientX - rect.left;
        STATE.hoverY = e.clientY - rect.top;
      } else {
        STATE.hoverX = null;
        STATE.hoverY = null;
      }

      if (STATE.dragging && STATE.session) {
        const dx = e.clientX - STATE.dragStartX;
        const plotW = Math.max(200, rect.width - 68);
        const dt = -(dx / plotW) * STATE.winDurS;
        const maxStart = Math.max(0, STATE.session.ecg.duration_s - STATE.winDurS);
        STATE.winStartS = Math.max(0, Math.min(maxStart, STATE.dragStartWinS + dt));
        updateTimelineLabels();
        if (STATE.scope === "window") updateMetricsUI();
        ensureBufferCovers(STATE.winStartS, STATE.winDurS);
      }
      if (STATE.activeTab === "ecg") drawEcgCanvas();
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        if (!STATE.session) return;
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1.18 : 0.85;
        const total = STATE.session.ecg.duration_s;
        const newDur = Math.max(3, Math.min(Math.min(30, total), STATE.winDurS * factor));
        const center = STATE.winStartS + STATE.winDurS / 2;
        STATE.winDurS = newDur;
        const maxStart = Math.max(0, total - newDur);
        STATE.winStartS = Math.max(0, Math.min(maxStart, center - newDur / 2));
        $("timelineSlider").max = String(maxStart.toFixed(2));
        ensureBufferCovers(STATE.winStartS, STATE.winDurS);
        updateTimelineLabels();
        if (STATE.scope === "window") updateMetricsUI();
        drawEcgCanvas();
      },
      { passive: false }
    );

    window.addEventListener("resize", () => requestAnimationFrame(drawActiveTabCanvases));
  }

  document.addEventListener("DOMContentLoaded", initApp);
})();
