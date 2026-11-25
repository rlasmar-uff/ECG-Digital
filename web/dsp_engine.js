/**
 * CardioSignal DSP Engine (100% Client-Side JavaScript para GitHub Pages)
 *
 * Implementa matematicamente no navegador (sem backend):
 *  - Filtro Notch IIR 60 Hz (Q=30) + Butterworth Passa-Faixa (ordem 2..6) em SOS
 *  - Filtragem bidirecional de fase zero (sosfiltfilt) + polos no Plano Z (BIBO)
 *  - Detector QRS Pan & Tompkins (1985) + Hamilton & Tompkins (1986)
 *  - Periodicidade discreta N̄, R-R médio, BPM, SDNN, RMSSD e limiares 60–120 BPM
 *  - FFT arbitrária (Bluestein / Radix-2) + Teorema de Parseval P = (1/M)Σ|x|² = Σ|c_k|²
 *  - Parser de arquivos .csv, .txt e PhysioNet (.hea + .dat formato 212/16 + .atr)
 */

(function (globalScope) {
  "use strict";

  // -------------------------------------------------------------------------
  // Aritmética Complexa Auxiliar
  // -------------------------------------------------------------------------
  const cAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const cSub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const cMul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const cDiv = (a, b) => {
    const d = b[0] * b[0] + b[1] * b[1] || 1e-30;
    return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
  };
  const cSqrt = (a) => {
    const r = Math.hypot(a[0], a[1]);
    const theta = Math.atan2(a[1], a[0]) / 2;
    const sr = Math.sqrt(r);
    return [sr * Math.cos(theta), sr * Math.sin(theta)];
  };
  const cAbs = (a) => Math.hypot(a[0], a[1]);

  // -------------------------------------------------------------------------
  // 1. Projeto de Filtros LIT (Notch 60 Hz + Butterworth Passa-Faixa SOS)
  // -------------------------------------------------------------------------

  function designNotchSos(f0, Q, fs) {
    // Equivalente exato ao scipy.signal.iirnotch(w0=f0, Q=Q, fs=fs)
    const w0 = (2 * Math.PI * f0) / fs;
    const bw = w0 / Q;
    const gb = Math.SQRT1_2; // -3 dB
    const beta = (Math.sqrt(1 - gb * gb) / gb) * Math.tan(bw / 2);
    const gain = 1 / (1 + beta);
    const b0 = gain;
    const b1 = -2 * gain * Math.cos(w0);
    const b2 = gain;
    const a1 = b1;
    const a2 = 2 * gain - 1;

    // Polos de 1 + a1 z^-1 + a2 z^-2 = 0
    const disc = a1 * a1 - 4 * a2;
    let poles;
    if (disc >= 0) {
      const sd = Math.sqrt(disc);
      poles = [
        [(-a1 + sd) / 2, 0],
        [(-a1 - sd) / 2, 0],
      ];
    } else {
      const sd = Math.sqrt(-disc);
      poles = [
        [-a1 / 2, sd / 2],
        [-a1 / 2, -sd / 2],
      ];
    }

    return {
      sos: [[b0, b1, b2, 1.0, a1, a2]],
      poles,
    };
  }

  function designButterBandpassSos(order, lowcut, highcut, fs) {
    // Projeto analógico Butterworth -> LP-to-BP -> Bilinear -> SOS (idêntico ao SciPy)
    const fs2 = 2.0 * fs;
    const w1 = fs2 * Math.tan((Math.PI * lowcut) / fs);
    const w2 = fs2 * Math.tan((Math.PI * highcut) / fs);
    const bw = w2 - w1;
    const w0Sq = w1 * w2;

    // Polos protótipo Butterworth passa-baixas de ordem N no semiplano esquerdo
    const lpPoles = [];
    for (let k = 0; k < order; k++) {
      const ang = (Math.PI * (2 * k + order + 1)) / (2 * order);
      lpPoles.push([Math.cos(ang), Math.sin(ang)]);
    }

    // Transformação LP -> BP: cada polo p_k gera um par de polos s = (bw*p ± sqrt((bw*p)^2 - 4*w0^2)) / 2
    const bpPolesS = [];
    for (const p of lpPoles) {
      const bwp = [p[0] * bw, p[1] * bw];
      const rad = cSqrt(cSub(cMul(bwp, bwp), [4 * w0Sq, 0]));
      bpPolesS.push([(bwp[0] + rad[0]) / 2, (bwp[1] + rad[1]) / 2]);
      bpPolesS.push([(bwp[0] - rad[0]) / 2, (bwp[1] - rad[1]) / 2]);
    }

    // Transformação bilinear s -> z = (1 + s/fs2) / (1 - s/fs2)
    // Zeros digitais do passa-faixa de ordem N: N zeros em z = +1 e N zeros em z = -1
    const zPoles = bpPolesS.map((s) => {
      const num = [1 + s[0] / fs2, s[1] / fs2];
      const den = [1 - s[0] / fs2, -s[1] / fs2];
      return cDiv(num, den);
    });

    // Agrupa polos em pares conjugados (ou pares reais) para montar N seções SOS
    const used = new Array(zPoles.length).fill(false);
    const pairs = [];
    for (let i = 0; i < zPoles.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      const p = zPoles[i];
      if (Math.abs(p[1]) > 1e-9) {
        let bestJ = -1;
        let bestDist = Infinity;
        for (let j = i + 1; j < zPoles.length; j++) {
          if (used[j]) continue;
          const d = Math.hypot(zPoles[j][0] - p[0], zPoles[j][1] + p[1]);
          if (d < bestDist) {
            bestDist = d;
            bestJ = j;
          }
        }
        if (bestJ >= 0) {
          used[bestJ] = true;
          pairs.push([p, zPoles[bestJ]]);
        }
      } else {
        // Polo real: emparelha com outro polo real
        let bestJ = -1;
        for (let j = i + 1; j < zPoles.length; j++) {
          if (!used[j] && Math.abs(zPoles[j][1]) <= 1e-9) {
            bestJ = j;
            break;
          }
        }
        if (bestJ >= 0) {
          used[bestJ] = true;
          pairs.push([p, zPoles[bestJ]]);
        }
      }
    }

    // Cada seção recebe um par de zeros (+1, -1) => numerador (1 - z^-2) = [1, 0, -1]
    const sos = pairs.map(([p1, p2]) => {
      const a1 = -(p1[0] + p2[0]);
      const a2 = p1[0] * p2[0] - p1[1] * p2[1];
      return [1.0, 0.0, -1.0, 1.0, a1, a2];
    });

    // Normaliza ganho unitário (0 dB) na frequência central geométrica w_center
    const fCenter = Math.sqrt(lowcut * highcut);
    const wCenter = (2 * Math.PI * fCenter) / fs;
    const hCenter = evalSosFreq(sos, wCenter);
    const gCenter = cAbs(hCenter) || 1.0;
    const perSecGain = Math.pow(1.0 / gCenter, 1.0 / sos.length);
    for (const sec of sos) {
      sec[0] *= perSecGain;
      sec[1] *= perSecGain;
      sec[2] *= perSecGain;
    }

    return { sos, poles: zPoles };
  }

  function evalSosFreq(sos, omega) {
    const ejw1 = [Math.cos(-omega), Math.sin(-omega)];
    const ejw2 = [Math.cos(-2 * omega), Math.sin(-2 * omega)];
    let h = [1.0, 0.0];
    for (const [b0, b1, b2, a0, a1, a2] of sos) {
      const num = [
        b0 + b1 * ejw1[0] + b2 * ejw2[0],
        b1 * ejw1[1] + b2 * ejw2[1],
      ];
      const den = [
        a0 + a1 * ejw1[0] + a2 * ejw2[0],
        a1 * ejw1[1] + a2 * ejw2[1],
      ];
      h = cMul(h, cDiv(num, den));
    }
    return h;
  }

  function sosfiltForward(sos, x) {
    let cur = new Float64Array(x);
    const n = cur.length;
    for (const [b0, b1, b2, _a0, a1, a2] of sos) {
      const out = new Float64Array(n);
      // Estado inicial em regime permanente para x[0] reduz transiente de borda
      const denom = 1 + a1 + a2;
      const steady = Math.abs(denom) > 1e-12 ? (b0 + b1 + b2) / denom : 0;
      let z1 = (steady - b0) * cur[0];
      let z2 = (b2 - a2 * steady) * cur[0];
      // Para filtro passa-faixa (b0+b1+b2=0), steady=0 -> z1 = -b0*x[0], z2 = b2*x[0]
      for (let i = 0; i < n; i++) {
        const xi = cur[i];
        const yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        out[i] = yi;
      }
      cur = out;
    }
    return cur;
  }

  function sosfiltfilt(sos, x) {
    if (!sos || sos.length === 0) return Float64Array.from(x);
    const n = x.length;
    const padLen = Math.min(n - 1, Math.max(15, 3 * (2 * sos.length + 1)));
    const ext = new Float64Array(n + 2 * padLen);

    // Reflexão ímpar nas bordas (mesmo esquema do scipy.signal.sosfiltfilt)
    const x0 = x[0];
    const xEnd = x[n - 1];
    for (let i = 0; i < padLen; i++) {
      ext[i] = 2 * x0 - x[padLen - i];
    }
    for (let i = 0; i < n; i++) {
      ext[padLen + i] = x[i];
    }
    for (let i = 0; i < padLen; i++) {
      ext[padLen + n + i] = 2 * xEnd - x[n - 2 - i];
    }

    const fwd = sosfiltForward(sos, ext);
    const revIn = new Float64Array(fwd.length);
    for (let i = 0; i < fwd.length; i++) revIn[i] = fwd[fwd.length - 1 - i];
    const revOut = sosfiltForward(sos, revIn);

    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = revOut[revOut.length - 1 - (padLen + i)];
    }
    return y;
  }

  function applyZeroPhaseFilters(rawSignal, fs, params) {
    const nyq = fs / 2;
    const applyNotch = Boolean(params.apply_notch && params.notch_freq > 0 && params.notch_freq < nyq * 0.98);
    const effHigh = Math.min(params.highcut, nyq * 0.95);
    const applyBp = Boolean(params.apply_bandpass && params.lowcut > 0 && params.lowcut < effHigh);

    let y = Float64Array.from(rawSignal);
    const allPoles = [];
    let notchSos = [];
    let bpSos = [];

    if (applyNotch) {
      const nDes = designNotchSos(params.notch_freq, params.notch_q, fs);
      notchSos = nDes.sos;
      allPoles.push(...nDes.poles);
      y = sosfiltfilt(notchSos, y);
    }
    if (applyBp) {
      const bDes = designButterBandpassSos(params.order, params.lowcut, effHigh, fs);
      bpSos = bDes.sos;
      allPoles.push(...bDes.poles);
      y = sosfiltfilt(bpSos, y);
    }

    let maxPole = 0;
    for (const p of allPoles) {
      const r = cAbs(p);
      if (r > maxPole) maxPole = r;
    }

    // Resposta Bode sempre calculada sobre o par nominal para visualização na aba LIT
    const nRef = designNotchSos(params.notch_freq || 60, params.notch_q || 30, fs);
    const bRef = designButterBandpassSos(params.order || 4, params.lowcut || 0.5, effHigh, fs);
    const fMax = Math.min(100, nyq);
    const nPts = 320;
    const bodeFreqs = [];
    const notchDb = [];
    const bpDb = [];
    const cascadeDb = [];
    const eps = 1e-12;
    for (let i = 0; i <= nPts; i++) {
      const f = (fMax * i) / nPts;
      const w = (2 * Math.PI * f) / fs;
      const hn = cAbs(evalSosFreq(nRef.sos, w));
      const hb = cAbs(evalSosFreq(bRef.sos, w));
      const hc2 = (hn * hb) ** 2;
      bodeFreqs.push(Number(f.toFixed(3)));
      notchDb.push(Number((20 * Math.log10(Math.max(hn, eps))).toFixed(2)));
      bpDb.push(Number((20 * Math.log10(Math.max(hb, eps))).toFixed(2)));
      cascadeDb.push(Number((20 * Math.log10(Math.max(hc2, eps))).toFixed(2)));
    }
    const refPoles = [...nRef.poles, ...bRef.poles];

    return {
      filtered: y,
      filter_info: {
        notch_applied: applyNotch,
        bandpass_applied: applyBp,
        notch_freq_hz: params.notch_freq || 60,
        notch_q: params.notch_q || 30,
        lowcut_hz: params.lowcut || 0.5,
        highcut_hz: effHigh,
        order: params.order || 4,
        max_pole_radius: Number(maxPole.toFixed(5)),
        is_bibo_stable: maxPole < 1.0,
        zero_phase: true,
      },
      bode: {
        freqs: bodeFreqs,
        notch_db: notchDb,
        bp_db: bpDb,
        cascade_db: cascadeDb,
        poles_re: refPoles.map((p) => Number(p[0].toFixed(6))),
        poles_im: refPoles.map((p) => Number(p[1].toFixed(6))),
      },
    };
  }

  // -------------------------------------------------------------------------
  // 2. Detecção de Picos R (Pan & Tompkins 1985 + Hamilton & Tompkins 1986)
  // -------------------------------------------------------------------------

  function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const copy = Array.from(arr).sort((a, b) => a - b);
    const idx = (p / 100) * (copy.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return copy[lo];
    return copy[lo] + (copy[hi] - copy[lo]) * (idx - lo);
  }

  function findLocalPeaksWithDistance(arr, minDist) {
    const rawPeaks = [];
    for (let i = 1; i < arr.length - 1; i++) {
      if (arr[i] > arr[i - 1] && arr[i] >= arr[i + 1]) {
        rawPeaks.push(i);
      }
    }
    if (minDist <= 1 || rawPeaks.length <= 1) return rawPeaks;
    // Mesmo comportamento guloso por altura do scipy.signal.find_peaks(distance=...)
    const order = rawPeaks.slice().sort((a, b) => arr[b] - arr[a]);
    const keep = new Uint8Array(arr.length);
    const blocked = new Uint8Array(arr.length);
    for (const idx of order) {
      if (blocked[idx]) continue;
      keep[idx] = 1;
      const lo = Math.max(0, idx - minDist + 1);
      const hi = Math.min(arr.length - 1, idx + minDist - 1);
      for (let k = lo; k <= hi; k++) blocked[k] = 1;
    }
    const out = [];
    for (let i = 0; i < arr.length; i++) if (keep[i]) out.push(i);
    return out;
  }

  function detectRPeaksPanTompkins(filteredEcg, fs, refractoryMs = 200) {
    const n = filteredEcg.length;
    if (n < Math.floor(fs)) return [];

    // 1) Passa-faixa QRS 5–15 Hz (ordem 3, fase zero)
    const nyq = fs / 2;
    const hiQrs = Math.min(15.0, nyq * 0.9);
    const loQrs = Math.min(5.0, hiQrs * 0.5);
    const qrsDes = designButterBandpassSos(3, loQrs, hiQrs, fs);
    const qrsBp = sosfiltfilt(qrsDes.sos, filteredEcg);
    const qrsEnv = new Float64Array(n);
    for (let i = 0; i < n; i++) qrsEnv[i] = Math.abs(qrsBp[i]);

    // 2) Derivada discreta centrada de 5 pontos
    const deriv = new Float64Array(n);
    const scaleD = fs / 8.0;
    for (let i = 2; i < n - 2; i++) {
      deriv[i] =
        scaleD * (2 * qrsBp[i + 2] + qrsBp[i + 1] - qrsBp[i - 1] - 2 * qrsBp[i - 2]);
    }

    // 3) Quadratura
    const sq = new Float64Array(n);
    for (let i = 0; i < n; i++) sq[i] = deriv[i] * deriv[i];

    // 4) Integração por janela móvel de 150 ms (soma cumulativa O(N))
    let winLen = Math.max(3, Math.round(0.15 * fs));
    if (winLen % 2 === 0) winLen += 1;
    const halfW = (winLen - 1) >> 1;
    const pref = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) pref[i + 1] = pref[i] + sq[i];
    const mwi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - halfW);
      const b = Math.min(n, i + halfW + 1);
      mwi[i] = (pref[b] - pref[a]) / winLen;
    }

    // 5) Regras adaptativas + período refratário + discriminação de onda T + search-back
    const refractorySamples = Math.max(1, Math.round((refractoryMs / 1000) * fs));
    const twaveSamples = Math.round(0.36 * fs);
    const minCandDist = Math.max(1, Math.round(0.15 * fs));
    const candIndices = findLocalPeaksWithDistance(mwi, minCandDist);
    if (candIndices.length === 0) return [];

    const initEnd = Math.min(n, Math.round(4.0 * fs));
    const initMwi = mwi.subarray(0, initEnd);
    const initEnv = qrsEnv.subarray(0, initEnd);

    let spki = percentile(initMwi, 99.0) * 0.65;
    let npki = percentile(initMwi, 50.0) * 0.5;
    let spkf = percentile(initEnv, 99.0) * 0.65;
    let npkf = percentile(initEnv, 50.0) * 0.5;
    if (spki <= 0) spki = percentile(mwi, 99.5) * 0.5;
    if (spkf <= 0) spkf = percentile(qrsEnv, 99.5) * 0.5;

    let thI1 = npki + 0.25 * (spki - npki);
    let thI2 = 0.5 * thI1;
    let thF1 = npkf + 0.25 * (spkf - npkf);
    let thF2 = 0.5 * thF1;

    const halfWin = Math.max(2, Math.round(0.075 * fs));
    const candFeats = candIndices.map((idx) => {
      const lo = Math.max(0, idx - halfWin);
      const hi = Math.min(n, idx + halfWin + 1);
      let maxEnv = 0;
      let maxSl = 0;
      for (let k = lo; k < hi; k++) {
        if (qrsEnv[k] > maxEnv) maxEnv = qrsEnv[k];
        const ad = Math.abs(deriv[k]);
        if (ad > maxSl) maxSl = ad;
      }
      return [mwi[idx], maxEnv, maxSl];
    });

    const mwiCap = percentile(mwi, 99.5) * 2.5 || 1e9;
    const envCap = percentile(qrsEnv, 99.5) * 2.5 || 1e9;

    const qrsPeaks = [];
    const qrsSlopes = [];
    const rrHist = [];
    const rrRegHist = [];

    const updateRR = (rr) => {
      rrHist.push(rr);
      if (rrHist.length > 8) rrHist.shift();
      const avg1 = rrHist.reduce((a, b) => a + b, 0) / rrHist.length;
      if (rr >= 0.92 * avg1 && rr <= 1.16 * avg1) {
        rrRegHist.push(rr);
        if (rrRegHist.length > 8) rrRegHist.shift();
      }
    };
    const currentRRAvg = () => {
      if (rrRegHist.length >= 4) return rrRegHist.reduce((a, b) => a + b, 0) / rrRegHist.length;
      if (rrHist.length > 0) return rrHist.reduce((a, b) => a + b, 0) / rrHist.length;
      return 0.8 * fs;
    };

    for (let ci = 0; ci < candIndices.length; ci++) {
      const idx = candIndices[ci];
      const [valI, valF, maxSl] = candFeats[ci];

      if (qrsPeaks.length > 0 && idx - qrsPeaks[qrsPeaks.length - 1] < refractorySamples) {
        const prev = qrsPeaks[qrsPeaks.length - 1];
        if (valI > mwi[prev] * 1.15 && valF > qrsEnv[prev]) {
          qrsPeaks[qrsPeaks.length - 1] = idx;
          qrsSlopes[qrsSlopes.length - 1] = maxSl;
        }
        continue;
      }

      // Search-back
      if (qrsPeaks.length > 0) {
        const rrAvg = currentRRAvg();
        const gap = idx - qrsPeaks[qrsPeaks.length - 1];
        if (gap > Math.round(1.55 * rrAvg)) {
          let bestIdx = -1;
          let bestVal = -1;
          let bestFeat = null;
          for (let pci = ci - 1; pci >= 0; pci--) {
            const pIdx = candIndices[pci];
            if (pIdx <= qrsPeaks[qrsPeaks.length - 1] + refractorySamples) break;
            if (idx - pIdx < refractorySamples) continue;
            const [pvi, pvf, psl] = candFeats[pci];
            if (pvi > thI2 && pvf > thF2 && pvi > bestVal) {
              bestVal = pvi;
              bestIdx = pIdx;
              bestFeat = [pvi, pvf, psl];
            }
          }
          if (bestIdx >= 0 && bestFeat) {
            updateRR(bestIdx - qrsPeaks[qrsPeaks.length - 1]);
            qrsPeaks.push(bestIdx);
            qrsSlopes.push(bestFeat[2]);
            spki = 0.25 * Math.min(bestFeat[0], mwiCap) + 0.75 * spki;
            spkf = 0.25 * Math.min(bestFeat[1], envCap) + 0.75 * spkf;
            thI1 = npki + 0.25 * (spki - npki);
            thI2 = 0.5 * thI1;
            thF1 = npkf + 0.25 * (spkf - npkf);
            thF2 = 0.5 * thF1;
          }
        }
      }

      let isQrs = valI >= thI1 && valF >= thF1;
      if (isQrs && qrsPeaks.length > 0) {
        const delta = idx - qrsPeaks[qrsPeaks.length - 1];
        if (delta >= refractorySamples && delta < twaveSamples) {
          if (maxSl < 0.5 * qrsSlopes[qrsSlopes.length - 1]) isQrs = false;
        }
      }

      if (isQrs) {
        if (qrsPeaks.length > 0) updateRR(idx - qrsPeaks[qrsPeaks.length - 1]);
        qrsPeaks.push(idx);
        qrsSlopes.push(maxSl);
        spki = 0.125 * Math.min(valI, mwiCap) + 0.875 * spki;
        spkf = 0.125 * Math.min(valF, envCap) + 0.875 * spkf;
      } else {
        npki = 0.125 * Math.min(valI, mwiCap) + 0.875 * npki;
        npkf = 0.125 * Math.min(valF, envCap) + 0.875 * npkf;
      }
      thI1 = npki + 0.25 * (spki - npki);
      thI2 = 0.5 * thI1;
      thF1 = npkf + 0.25 * (spkf - npkf);
      thF2 = 0.5 * thF1;
    }

    if (qrsPeaks.length === 0) return [];

    // 6) Refinamento do pico R no sinal diagnóstico filtrado (0,5–40 Hz)
    const searchRad = Math.max(2, Math.round(0.08 * fs));
    const sampleExt = [];
    for (let i = 0; i < Math.min(30, qrsPeaks.length); i++) {
      const p = qrsPeaks[i];
      const lo = Math.max(0, p - searchRad);
      const hi = Math.min(n, p + searchRad + 1);
      let mx = -Infinity;
      let mn = Infinity;
      for (let k = lo; k < hi; k++) {
        if (filteredEcg[k] > mx) mx = filteredEcg[k];
        if (filteredEcg[k] < mn) mn = filteredEcg[k];
      }
      sampleExt.push(Math.abs(mx) >= Math.abs(mn) ? mx : mn);
    }
    const domPos = percentile(sampleExt, 50) >= 0;

    const refined = [];
    for (const p of qrsPeaks) {
      const lo = Math.max(0, p - searchRad);
      const hi = Math.min(n, p + searchRad + 1);
      let idxMax = lo;
      let idxMin = lo;
      for (let k = lo; k < hi; k++) {
        if (filteredEcg[k] > filteredEcg[idxMax]) idxMax = k;
        if (filteredEcg[k] < filteredEcg[idxMin]) idxMin = k;
      }
      const vMax = filteredEcg[idxMax];
      const vMin = filteredEcg[idxMin];
      let best;
      if (domPos) {
        best = vMax >= 0.65 * Math.abs(vMin) ? idxMax : idxMin;
      } else {
        best = Math.abs(vMin) >= 0.65 * vMax ? idxMin : idxMax;
      }
      if (refined.length > 0 && best - refined[refined.length - 1] < refractorySamples) {
        if (Math.abs(filteredEcg[best]) > Math.abs(filteredEcg[refined[refined.length - 1]])) {
          refined[refined.length - 1] = best;
        }
      } else {
        refined.push(best);
      }
    }
    return refined;
  }

  // -------------------------------------------------------------------------
  // 3. FFT Exata (Radix-2 + Bluestein para comprimento arbitrário) & Parseval
  // -------------------------------------------------------------------------

  function fftRadix2(re, im, inverse = false) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    const sign = inverse ? 1 : -1;
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (sign * 2 * Math.PI) / len;
      const wLenR = Math.cos(ang);
      const wLenI = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let wR = 1;
        let wI = 0;
        for (let j = 0; j < half; j++) {
          const uR = re[i + j];
          const uI = im[i + j];
          const vR = re[i + j + half] * wR - im[i + j + half] * wI;
          const vI = re[i + j + half] * wI + im[i + j + half] * wR;
          re[i + j] = uR + vR;
          im[i + j] = uI + vI;
          re[i + j + half] = uR - vR;
          im[i + j + half] = uI - vI;
          const nxtR = wR * wLenR - wI * wLenI;
          wI = wR * wLenI + wI * wLenR;
          wR = nxtR;
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
    }
  }

  function dftExact(xReal) {
    // Algoritmo de Bluestein: calcula DFT exata de QUALQUER comprimento M em O(M log M)
    const n = xReal.length;
    if ((n & (n - 1)) === 0) {
      const re = Float64Array.from(xReal);
      const im = new Float64Array(n);
      fftRadix2(re, im, false);
      return { re, im };
    }
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;

    const cosT = new Float64Array(n);
    const sinT = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const ang = (Math.PI * ((k * k) % (2 * n))) / n;
      cosT[k] = Math.cos(ang);
      sinT[k] = Math.sin(ang);
    }

    const aRe = new Float64Array(m);
    const aIm = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      aRe[k] = xReal[k] * cosT[k];
      aIm[k] = -xReal[k] * sinT[k];
    }

    const bRe = new Float64Array(m);
    const bIm = new Float64Array(m);
    bRe[0] = cosT[0];
    bIm[0] = sinT[0];
    for (let k = 1; k < n; k++) {
      bRe[k] = cosT[k];
      bIm[k] = sinT[k];
      bRe[m - k] = cosT[k];
      bIm[m - k] = sinT[k];
    }

    fftRadix2(aRe, aIm, false);
    fftRadix2(bRe, bIm, false);
    for (let i = 0; i < m; i++) {
      const nr = aRe[i] * bRe[i] - aIm[i] * bIm[i];
      const ni = aRe[i] * bIm[i] + aIm[i] * bRe[i];
      aRe[i] = nr;
      aIm[i] = ni;
    }
    fftRadix2(aRe, aIm, true);

    const outRe = new Float64Array(n);
    const outIm = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      outRe[k] = aRe[k] * cosT[k] + aIm[k] * sinT[k];
      outIm[k] = aIm[k] * cosT[k] - aRe[k] * sinT[k];
    }
    return { re: outRe, im: outIm };
  }

  function oneSidedPowerSpectrum(x, fs) {
    // Limita a janela espectral a 32768 pontos (91 s a 360 Hz) para resposta < 25 ms no browser
    const m = Math.min(x.length, 32768);
    const seg = x.length > m ? x.subarray(0, m) : x;
    let sumSq = 0;
    for (let i = 0; i < m; i++) sumSq += seg[i] * seg[i];
    const powerTime = sumSq / m;

    const { re, im } = dftExact(seg);
    const nHalf = Math.floor(m / 2) + 1;
    const freqs = new Float64Array(nHalf);
    const pOne = new Float64Array(nHalf);
    const invM2 = 1 / (m * m);

    let powerParseval = 0;
    for (let k = 0; k < nHalf; k++) {
      freqs[k] = (k * fs) / m;
      let pk = (re[k] * re[k] + im[k] * im[k]) * invM2;
      if (k > 0 && (m % 2 !== 0 || k < nHalf - 1)) pk *= 2;
      pOne[k] = pk;
      powerParseval += pk;
    }
    return { freqs, pOne, powerTime, powerParseval };
  }

  function binSpectrum(freqs, pRaw, pFlt, fMin, fMax, step) {
    const eps = 1e-13;
    const fOut = [];
    const rOut = [];
    const flOut = [];
    let count = 0;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= fMin && freqs[i] <= fMax) count++;
    }
    if (count === 0) return { freqs: [], raw_db: [], filt_db: [] };

    if (count <= 600) {
      for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] >= fMin && freqs[i] <= fMax) {
          fOut.push(Number(freqs[i].toFixed(4)));
          rOut.push(Number((10 * Math.log10(Math.max(pRaw[i], eps))).toFixed(2)));
          flOut.push(Number((10 * Math.log10(Math.max(pFlt[i], eps))).toFixed(2)));
        }
      }
      return { freqs: fOut, raw_db: rOut, filt_db: flOut };
    }

    const nBins = Math.max(1, Math.ceil((fMax - fMin) / step));
    const rb = new Float64Array(nBins);
    const fb = new Float64Array(nBins);
    const ct = new Float64Array(nBins);
    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      if (f < fMin || f > fMax) continue;
      const bi = Math.min(nBins - 1, Math.floor((f - fMin) / step));
      rb[bi] += pRaw[i];
      fb[bi] += pFlt[i];
      ct[bi] += 1;
    }
    for (let b = 0; b < nBins; b++) {
      if (ct[b] > 0) {
        fOut.push(Number((fMin + (b + 0.5) * step).toFixed(3)));
        rOut.push(Number((10 * Math.log10(Math.max(rb[b] / ct[b], eps))).toFixed(2)));
        flOut.push(Number((10 * Math.log10(Math.max(fb[b] / ct[b], eps))).toFixed(2)));
      }
    }
    return { freqs: fOut, raw_db: rOut, filt_db: flOut };
  }

  function computeSpectralAnalysis(raw, filtered, fs) {
    const sRaw = oneSidedPowerSpectrum(raw, fs);
    const sFlt = oneSidedPowerSpectrum(filtered, fs);
    const freqs = sRaw.freqs;
    const pRaw = sRaw.pOne;
    const pFlt = sFlt.pOne;

    const relErrRaw = Math.abs(sRaw.powerTime - sRaw.powerParseval) / Math.max(sRaw.powerTime, 1e-15);
    const relErrFlt = Math.abs(sFlt.powerTime - sFlt.powerParseval) / Math.max(sFlt.powerTime, 1e-15);

    let pAcRaw = 0;
    let pAcFlt = 0;
    for (let i = 1; i < freqs.length; i++) {
      pAcRaw += pRaw[i];
      pAcFlt += pFlt[i];
    }

    const remTotal =
      sRaw.powerParseval > 0
        ? Math.max(0, ((sRaw.powerParseval - sFlt.powerParseval) / sRaw.powerParseval) * 100)
        : 0;
    const remAc =
      pAcRaw > 0 ? Math.max(0, ((pAcRaw - pAcFlt) / pAcRaw) * 100) : 0;

    const bandDefs = [
      ["DC (0 Hz)", (f) => f === 0],
      ["Linha de Base (< 0,5 Hz)", (f) => f > 0 && f < 0.5],
      ["Banda ECG (0,5–40 Hz)", (f) => f >= 0.5 && f <= 40.0],
      ["Alta Frequência (> 40 Hz)", (f) => f > 40.0],
      ["Rede Elétrica (59–61 Hz)", (f) => f >= 59.0 && f <= 61.0],
    ];

    const bandTable = bandDefs.map(([name, pred]) => {
      let pr = 0;
      let pf = 0;
      for (let i = 0; i < freqs.length; i++) {
        if (pred(freqs[i])) {
          pr += pRaw[i];
          pf += pFlt[i];
        }
      }
      const attPct = pr > 1e-15 ? (1 - pf / pr) * 100 : 0;
      const attDb = pr > 1e-15 ? 10 * Math.log10(Math.max(pf, 1e-18) / pr) : 0;
      return {
        band: name,
        power_raw: Number(pr.toFixed(6)),
        power_filtered: Number(pf.toFixed(6)),
        attenuation_pct: Number(attPct.toFixed(2)),
        attenuation_db: Number(attDb.toFixed(2)),
      };
    });

    const fMax = Math.min(100, fs / 2);
    return {
      power_time_raw: Number(sRaw.powerTime.toFixed(6)),
      power_parseval_raw: Number(sRaw.powerParseval.toFixed(6)),
      power_time_filtered: Number(sFlt.powerTime.toFixed(6)),
      power_parseval_filtered: Number(sFlt.powerParseval.toFixed(6)),
      parseval_rel_error_raw: relErrRaw,
      parseval_rel_error_filtered: relErrFlt,
      noise_removed_total_pct: Number(remTotal.toFixed(2)),
      power_ac_raw: Number(pAcRaw.toFixed(6)),
      power_ac_filtered: Number(pAcFlt.toFixed(6)),
      noise_removed_ac_pct: Number(remAc.toFixed(2)),
      band_table: bandTable,
      fft_main: binSpectrum(freqs, pRaw, pFlt, 0.05, fMax, 0.16),
      fft_low: binSpectrum(freqs, pRaw, pFlt, 0.02, Math.min(5, fMax), 0.03),
      fft_notch: fMax >= 62 ? binSpectrum(freqs, pRaw, pFlt, 55, Math.min(65, fMax), 0.06) : { freqs: [], raw_db: [], filt_db: [] },
    };
  }

  // -------------------------------------------------------------------------
  // 4. Métricas Clínicas & Validação ±150 ms contra .atr
  // -------------------------------------------------------------------------

  function computeClinicalMetrics(peaks, fs) {
    const n = peaks.length;
    if (n < 2) {
      return {
        num_beats: n,
        mean_n_samples: 0,
        mean_rr_ms: 0,
        median_rr_ms: 0,
        sdnn_ms: 0,
        rmssd_ms: 0,
        bpm_mean: 0,
        bpm_median: 0,
        bpm_min: 0,
        bpm_max: 0,
        classification: "Indeterminado",
        classification_level: "warning",
        rhythm_regularity: "Batimentos insuficientes",
        instant_bpm: [],
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
    const medRR = percentile(rrMs, 50);
    let sdnn = 0;
    let rmssd = 0;
    if (rrMs.length > 1) {
      sdnn = Math.sqrt(rrMs.reduce((acc, v) => acc + (v - meanRR) ** 2, 0) / (rrMs.length - 1));
      let dSq = 0;
      for (let i = 1; i < rrMs.length; i++) dSq += (rrMs[i] - rrMs[i - 1]) ** 2;
      rmssd = Math.sqrt(dSq / (rrMs.length - 1));
    }
    const bpmMean = 60000 / meanRR;
    const bpmMed = 60000 / medRR;
    let cls = "Ritmo Normal";
    let lvl = "normal";
    if (bpmMean < 60) {
      cls = "Bradicardia";
      lvl = "alert";
    } else if (bpmMean > 120) {
      cls = "Taquicardia";
      lvl = "alert";
    }
    let reg = "Ritmo Regular (quase-periódico)";
    if (sdnn > 100) reg = "Alta Variabilidade (sugestivo de arritmia)";
    else if (sdnn >= 50) reg = "Variabilidade Moderada";

    return {
      num_beats: n,
      mean_n_samples: Number(meanN.toFixed(2)),
      mean_rr_ms: Number(meanRR.toFixed(2)),
      median_rr_ms: Number(medRR.toFixed(2)),
      sdnn_ms: Number(sdnn.toFixed(2)),
      rmssd_ms: Number(rmssd.toFixed(2)),
      bpm_mean: Number(bpmMean.toFixed(2)),
      bpm_median: Number(bpmMed.toFixed(2)),
      bpm_min: Number(Math.min(...instBpm).toFixed(2)),
      bpm_max: Number(Math.max(...instBpm).toFixed(2)),
      classification: cls,
      classification_level: lvl,
      rhythm_regularity: reg,
      instant_bpm: instBpm,
    };
  }

  function matchBeatsOneToOne(detPeaks, refPeaks, fs, tolMs = 150) {
    const det = detPeaks.slice().sort((a, b) => a - b);
    const ref = refPeaks.slice().sort((a, b) => a - b);
    const tol = Math.round((tolMs / 1000) * fs);
    let i = 0;
    let j = 0;
    let tp = 0;
    while (i < det.length && j < ref.length) {
      const diff = det[i] - ref[j];
      if (Math.abs(diff) <= tol) {
        if (i + 1 < det.length && Math.abs(det[i + 1] - ref[j]) < Math.abs(diff)) {
          i++;
          continue;
        }
        if (j + 1 < ref.length && Math.abs(det[i] - ref[j + 1]) < Math.abs(diff)) {
          j++;
          continue;
        }
        tp++;
        i++;
        j++;
      } else if (diff < -tol) {
        i++;
      } else {
        j++;
      }
    }
    const fp = det.length - tp;
    const fn = ref.length - tp;
    const se = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const ppv = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const f1 = se + ppv > 0 ? (2 * se * ppv) / (se + ppv) : 0;
    return {
      total_real: ref.length,
      total_detected: det.length,
      tp,
      fp,
      fn,
      sensitivity_pct: Number(se.toFixed(2)),
      precision_pct: Number(ppv.toFixed(2)),
      f1_score_pct: Number(f1.toFixed(2)),
    };
  }

  // -------------------------------------------------------------------------
  // 5. Parsers de Arquivos no Navegador (.csv, .txt, PhysioNet .hea + .dat)
  // -------------------------------------------------------------------------

  function parseTabularText(text, filename, fallbackFs = 360) {
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0 && !l.startsWith("#"));
    if (lines.length < 10) {
      throw new Error("Arquivo tabular vazio ou com amostras insuficientes.");
    }
    const first = lines[0];
    const sep = first.includes(";") ? ";" : first.includes(",") ? "," : first.includes("\t") ? "\t" : null;
    const splitLine = (ln) => (sep ? ln.split(sep) : ln.split(/\s+/));

    const firstToks = splitLine(first);
    let hasHeader = false;
    for (const tok of firstToks) {
      if (Number.isNaN(parseFloat(tok.replace(",", ".")))) {
        hasHeader = true;
        break;
      }
    }
    const headers = hasHeader ? firstToks.map((s) => s.trim()) : firstToks.map((_, i) => `col_${i}`);
    const startRow = hasHeader ? 1 : 0;
    const cols = headers.map(() => []);

    for (let r = startRow; r < lines.length; r++) {
      const toks = splitLine(lines[r]);
      if (toks.length < headers.length) continue;
      for (let c = 0; c < headers.length; c++) {
        const v = parseFloat(toks[c].trim().replace(",", "."));
        cols[c].push(Number.isFinite(v) ? v : NaN);
      }
    }

    const validCols = [];
    for (let c = 0; c < headers.length; c++) {
      const arr = cols[c].filter((v) => Number.isFinite(v));
      if (arr.length > 0.9 * cols[c].length && arr.length > 10) {
        validCols.push({ name: headers[c], data: Float64Array.from(arr) });
      }
    }
    if (validCols.length === 0) throw new Error("Nenhuma coluna numérica válida encontrada.");

    let fs = fallbackFs;
    let leadName = validCols[0].name;
    let sig = validCols[0].data;

    if (validCols.length >= 2) {
      let timeIdx = -1;
      for (let i = 0; i < validCols.length; i++) {
        const nm = validCols[i].name.toLowerCase();
        const d = validCols[i].data;
        if (/(time|tempo|t_s|sec|seg)/.test(nm) && d[d.length - 1] > d[0]) {
          timeIdx = i;
          break;
        }
      }
      if (timeIdx === -1) {
        const d0 = validCols[0].data;
        const dt0 = (d0[d0.length - 1] - d0[0]) / (d0.length - 1);
        if (dt0 >= 1e-4 && dt0 <= 0.1 && d0[1] > d0[0]) timeIdx = 0;
      }
      if (timeIdx >= 0) {
        const tArr = validCols[timeIdx].data;
        const dt = (tArr[tArr.length - 1] - tArr[0]) / Math.max(1, tArr.length - 1);
        if (dt > 0) {
          const inf = 1 / dt;
          if (inf >= 50 && inf <= 5000) {
            const nearest = Math.round(inf);
            fs = Math.abs(inf - nearest) < 0.05 ? nearest : Number(inf.toFixed(2));
          }
        }
        const candidates = validCols.filter((_, idx) => idx !== timeIdx);
        let chosen = candidates[0];
        for (const c of candidates) {
          if (/(brut|raw|ecg|mlii|sinal)/i.test(c.name)) {
            chosen = c;
            break;
          }
        }
        leadName = chosen.name;
        sig = chosen.data;
      }
    }

    if (sig.length < Math.floor(fs * 1.5)) {
      throw new Error(`Sinal muito curto (${sig.length} amostras a ${fs} Hz). Envie ao menos 2 segundos.`);
    }
    return {
      signal: sig,
      fs,
      source_name: `Arquivo importado: ${filename}`,
      lead_name: leadName,
      units: "mV",
      record_id: null,
      annotation_samples: null,
    };
  }

  function parsePhysioNetHeaDat(heaText, datBuf, recordStem) {
    const lines = heaText
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0 && !l.startsWith("#"));
    if (lines.length < 2) throw new Error("Cabeçalho .hea inválido.");

    const recTokens = lines[0].split(/\s+/);
    const nSig = parseInt(recTokens[1], 10) || 1;
    const fs = parseFloat((recTokens[2] || "360").split("/")[0]) || 360;

    const sigTokens = lines[1].split(/\s+/);
    const fmtCode = parseInt((sigTokens[1] || "212").split("+")[0], 10);
    const gainStr = (sigTokens[2] || "200").split("(")[0].split("/")[0];
    const gain = parseFloat(gainStr) || 200;
    const baselineMatch = (sigTokens[2] || "").match(/\((-?\d+)\)/);
    const adcZero = parseInt(sigTokens[4] || "1024", 10);
    const baseline = baselineMatch ? parseInt(baselineMatch[1], 10) : adcZero;
    const leadName = sigTokens[8] || "MLII";

    const bytes = new Uint8Array(datBuf);
    let ch0 = [];

    if (fmtCode === 212) {
      // Formato 212 (padrão MIT-BIH: 2 amostras de 12 bits em cada 3 bytes)
      const totalPairs = Math.floor(bytes.length / 3);
      const allSamples = new Int16Array(totalPairs * 2);
      for (let p = 0; p < totalPairs; p++) {
        const b0 = bytes[p * 3];
        const b1 = bytes[p * 3 + 1];
        const b2 = bytes[p * 3 + 2];
        let s1 = b0 | ((b1 & 0x0f) << 8);
        let s2 = b2 | ((b1 & 0xf0) << 4);
        if (s1 >= 2048) s1 -= 4096;
        if (s2 >= 2048) s2 -= 4096;
        allSamples[p * 2] = s1;
        allSamples[p * 2 + 1] = s2;
      }
      const nFrames = Math.floor(allSamples.length / nSig);
      // Limita a 120s (43.200 amostras) no navegador para desempenho instantâneo
      const maxFrames = Math.min(nFrames, Math.round(fs * 120));
      ch0 = new Float64Array(maxFrames);
      for (let i = 0; i < maxFrames; i++) {
        ch0[i] = (allSamples[i * nSig] - baseline) / gain;
      }
    } else if (fmtCode === 16) {
      const view = new DataView(datBuf);
      const totalSamples = Math.floor(bytes.length / 2);
      const nFrames = Math.floor(totalSamples / nSig);
      const maxFrames = Math.min(nFrames, Math.round(fs * 120));
      ch0 = new Float64Array(maxFrames);
      for (let i = 0; i < maxFrames; i++) {
        const rawInt = view.getInt16(i * nSig * 2, true);
        ch0[i] = (rawInt - baseline) / gain;
      }
    } else {
      throw new Error(`Formato WFDB ${fmtCode} não suportado no modo estático (suporta formatos 212 e 16).`);
    }

    return {
      signal: ch0,
      fs,
      source_name: `PhysioNet importado: ${recordStem}`,
      lead_name: leadName,
      units: "mV",
      record_id: recordStem,
      annotation_samples: null,
    };
  }

  // -------------------------------------------------------------------------
  // 6. Orquestração Completa no Mesmo Formato JSON do Backend FastAPI
  // -------------------------------------------------------------------------

  function buildSessionPayload(ecgObj, filterParams, suggestedStartS = null, suggestedLabel = null) {
    const raw = Float64Array.from(ecgObj.signal);
    const fs = ecgObj.fs;
    const n = raw.length;
    const durS = n / fs;

    const { filtered, filter_info, bode } = applyZeroPhaseFilters(raw, fs, filterParams);
    const rPeaks = detectRPeaksPanTompkins(filtered, fs, filterParams.refractory_ms || 200);
    const metrics = computeClinicalMetrics(rPeaks, fs);
    const spectral = computeSpectralAnalysis(raw, filtered, fs);

    let liveVal = null;
    if (ecgObj.annotation_samples && ecgObj.annotation_samples.length > 0) {
      liveVal = matchBeatsOneToOne(rPeaks, ecgObj.annotation_samples, fs, 150);
    }

    const rrTimes = [];
    const rrNSamples = [];
    const stride = Math.max(1, Math.floor(rPeaks.length / 1000));
    for (let i = 1; i < rPeaks.length; i += stride) {
      rrTimes.push(Number((rPeaks[i] / fs).toFixed(2)));
      rrNSamples.push(rPeaks[i] - rPeaks[i - 1]);
    }
    const instBpmOut = [];
    for (let i = 0; i < metrics.instant_bpm.length; i += stride) {
      instBpmOut.push(Number(metrics.instant_bpm[i].toFixed(2)));
    }

    const sugStart = suggestedStartS !== null ? Math.min(Math.max(0, durS - 10), suggestedStartS) : 0;

    return {
      kind: "ecg",
      session_id: "static_" + Math.random().toString(36).slice(2, 10),
      _raw_ecg: ecgObj,
      ecg: {
        source_name: ecgObj.source_name,
        lead_name: ecgObj.lead_name || "MLII",
        units: ecgObj.units || "mV",
        fs: Number(fs.toFixed(2)),
        n_samples: n,
        duration_s: Number(durS.toFixed(3)),
        record_id: ecgObj.record_id || null,
        has_annotations: Boolean(ecgObj.annotation_samples && ecgObj.annotation_samples.length > 0),
      },
      suggested_window: {
        start_s: Number(sugStart.toFixed(2)),
        duration_s: Number(Math.min(10, durS).toFixed(2)),
        label: suggestedLabel || "Janela clínica",
      },
      filter_info,
      metrics_full: metrics,
      r_peaks_all: rPeaks,
      spectral,
      bode,
      tachogram: {
        times_s: rrTimes,
        n_samples: rrNSamples,
        instant_bpm: instBpmOut,
      },
      live_validation: liveVal,
      initial_slice: {
        start_idx: 0,
        end_idx: n,
        start_s: 0,
        end_s: Number(durS.toFixed(4)),
        fs,
        raw: Array.from(raw),
        filtered: Array.from(filtered),
        r_peaks: rPeaks,
        atr_peaks: ecgObj.annotation_samples || [],
      },
    };
  }

  globalScope.CardioDSP = {
    applyZeroPhaseFilters,
    detectRPeaksPanTompkins,
    computeClinicalMetrics,
    computeSpectralAnalysis,
    matchBeatsOneToOne,
    parseTabularText,
    parsePhysioNetHeaDat,
    buildSessionPayload,
  };
})(typeof window !== "undefined" ? window : globalThis);
