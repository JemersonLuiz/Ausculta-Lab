"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const el = Object.fromEntries([
    "siteHeader", "versionText", "modelState", "modelStateText", "themeToggle", "globalMessage", "heroUploadCta",
    "wsTabAnalyze", "wsTabTests", "wsPanelAnalyze", "wsPanelTests", "testAudioList",
    "dropzone", "uploadButton", "uploadButtonText", "audioFile", "uploadLimits",
    "quickExamples", "exampleSelect", "loadExample", "loadAndAnalyze", "audioWorkspace",
    "fileName", "fileMeta", "clearFile", "waveform", "waveformTooltip",
    "waveformMid", "waveformEnd", "resetFullCycle",
    "playSegment", "playSegmentIcon", "playSegmentText",
    "loopToggle", "filterToggle", "gainToggle", "speedToggle", "detectCyclesBtn", "suggestedCycles",
    "audioPlayer", "startTime", "endTime", "segmentDuration", "segmentError",
    "referenceNote", "cycleConfirmed", "analyzeButton", "analyzeButtonText", "analyzeHelp",
    "resultStateText", "resultEmpty", "analysisProgress", "resultContent",
    "predictionCard", "predictionLabel", "predictionMeta", "confidenceStat", "marginStat",
    "referenceComparison", "crackleProb", "crackleBar", "wheezeProb", "wheezeBar",
    "scoreList", "acousticNote", "spectrogramImage", "spectrogramUnavailable",
    "contrastImage", "contrastUnavailable", "contrastTabBtn", "zoomEvidenceBtn",
    "imageLightbox", "lightboxTitle", "lightboxDownload", "lightboxClose", "lightboxImage", "lightboxCaption",
    "techTelemetry", "qualityMessage",
    "downloadResult", "copySummary", "sessionHistoryWrap", "sessionHistory", "clearHistory",
    "statAccuracy", "statMacroF1", "statIcbhi", "statCycles",
    "metricSwitcher", "perfSubtitle", "modelComparisonBars", "ciNote",
    "confusionMatrix", "perClassTable", "datasetSummary", "limitationsList",
    "deviceSection", "deviceGrid", "toastNotice", "backToTop"
  ].map((id) => [id, byId(id)]));

  const LABELS = ["Sem sibilos/estertores", "Estertores", "Sibilos", "Ambos"];
  const SHORT_LABELS = ["Sem eventos", "Estertores", "Sibilos", "Ambos"];
  const ACOUSTIC_NOTES = {
    "Sem sibilos/estertores": "Padrão compatível com som respiratório basal, sem detecção predominante de estertores ou sibilos neste recorte. Ausência desses padrões acústicos não exclui doença clínica.",
    "Estertores": "Padrão compatível com estertores: sons descontínuos, breves e impulsivos (5 a 20 ms), frequentemente visíveis como traços verticais curtos no espectrograma.",
    "Sibilos": "Padrão compatível com sibilos: sons musicais contínuos e tonais (acima de 100 ms), visíveis como faixas horizontais estreitas no espectrograma e no mapa de contraste.",
    "Ambos": "Padrão compatível com presença combinada de estertores (impulsos curtos) e sibilos (faixas tonais sustentadas) dentro do mesmo ciclo respiratório."
  };
  const VOICE_MSG = "Arquivo OGG ou Opus. Para analisar, escolha uma gravação de ausculta em WAV.";

  const state = {
    status: null,
    file: null,
    decoded: null,
    audioUrl: null,
    duration: 0,
    sampleRate: 0,
    channels: 1,
    activeExample: null,
    busy: false,
    loading: false,
    revision: 0,
    controller: null,
    examples: [],
    result: null,
    history: [],
    selectedMetric: "accuracy",
    maxUploadMb: 20,
    maxDuration: 120,
    maxSegmentDuration: 20,
    // Studio Web Audio playback state
    loop: false,
    filterOn: false,
    gain2x: false,
    slowSpeed: false,
    audioCtx: null,
    sourceNode: null,
    hpNode: null,
    lpNode: null,
    gainNode: null,
    isPlayingStudio: false,
    studioPlayStartWall: 0,
    studioSegStart: 0,
    studioSegLen: 0,
    studioCursorBase: 0,
    rafId: null,
    playheadTime: null,
    // Waveform interaction state
    dragMode: null,
    dragAnchor: 0,
    activeTab: "spec",
    lightboxKind: "spec",
    toastTimer: null,
    copyTimer: null,
  };

  const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hide = (node, yes = true) => node && node.classList.toggle("hidden", yes);
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const intFmt = (v) => new Intl.NumberFormat("pt-BR").format(v);
  const dec = (v, digits = 2) => new Intl.NumberFormat("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
  const decAuto = (v, digits = 2) => new Intl.NumberFormat("pt-BR", {
    maximumFractionDigits: digits,
  }).format(v);
  const pct = (v, digits = 1) => `${decAuto(v * 100, digits)}%`;
  const shortModelName = (name) => String(name || "").replace(/\s*\([^)]*\)\s*$/, "").trim();

  function resolveExampleLabel(example) {
    if (!example) return null;
    if (typeof example.label === "number") return LABELS[example.label] || null;
    return typeof example.label === "string" && example.label ? example.label : null;
  }

  function pcmWavInfo(buffer) {
    if (buffer.byteLength < 44) return null;
    const view = new DataView(buffer);
    const txt = (off, len) => String.fromCharCode(...new Uint8Array(buffer, off, len));
    if (txt(0, 4) !== "RIFF" || txt(8, 4) !== "WAVE") return null;
    let format = null;
    let dataBytes = 0;
    for (let off = 12; off + 8 <= buffer.byteLength;) {
      const id = txt(off, 4);
      const len = view.getUint32(off + 4, true);
      const body = off + 8;
      if (body + len > buffer.byteLength) return null;
      if (id === "fmt " && len >= 16) {
        format = {
          code: view.getUint16(body, true),
          channels: view.getUint16(body + 2, true),
          rate: view.getUint32(body + 4, true),
          alignment: view.getUint16(body + 12, true),
        };
      }
      if (id === "data") dataBytes += len;
      off = body + len + (len % 2);
    }
    if (!format || ![1, 3].includes(format.code) || !format.rate || !format.alignment || !dataBytes || dataBytes % format.alignment) {
      return null;
    }
    return {
      duration: dataBytes / format.alignment / format.rate,
      sampleRate: format.rate,
      channels: format.channels,
    };
  }

  function showMessage(msg) {
    if (!el.globalMessage) return;
    el.globalMessage.textContent = msg;
    hide(el.globalMessage, !msg);
    if (msg) {
      el.globalMessage.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "nearest" });
    }
  }

  function showToast(msg) {
    if (!el.toastNotice || !msg) return;
    el.toastNotice.textContent = msg;
    hide(el.toastNotice, false);
    if (state.toastTimer) clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => hide(el.toastNotice, true), 2400);
  }

  function syncQuickExamples() {
    if (!el.quickExamples) return;
    const activeCat = resolveExampleLabel(state.activeExample);
    el.quickExamples.querySelectorAll(".seg-option").forEach((btn) => {
      const isActive = Boolean(activeCat && btn.dataset.category === activeCat);
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-pressed", String(isActive));
    });
  }

  /* ======================== STATUS & PERFORMANCE DASHBOARD ======================== */
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) throw new Error("Servidor indisponível");
      state.status = await res.json();
      const limits = state.status.limits || {};
      if (isNum(limits.max_upload_mb)) state.maxUploadMb = limits.max_upload_mb;
      if (isNum(limits.max_duration_s)) state.maxDuration = limits.max_duration_s;
      if (isNum(limits.max_segment_s)) state.maxSegmentDuration = limits.max_segment_s;

      if (el.uploadLimits) {
        el.uploadLimits.textContent = `WAV mono ou estéreo, até ${decAuto(state.maxUploadMb, 0)} MB e ${decAuto(state.maxDuration, 0)} s`;
      }
      if (el.versionText && state.status.version) {
        el.versionText.textContent = `v${state.status.version}`;
      }

      const ready = state.status.model_ready === true;
      const modelShort = shortModelName(state.status.model?.name);
      if (el.modelState) {
        el.modelState.className = `model-status ${ready ? "ready" : "unavailable"}`;
      }
      if (el.modelStateText) {
        el.modelStateText.textContent = ready
          ? (modelShort ? `Pronto · ${modelShort}` : "Pronto para análise")
          : "Modelo indisponível";
      }
      renderPerformanceDashboard();
    } catch {
      state.status = null;
      if (el.modelState) el.modelState.className = "model-status unavailable";
      if (el.modelStateText) el.modelStateText.textContent = "Sem conexão";
      showMessage("Não foi possível acessar o servidor local. Abra a aplicação e recarregue esta página.");
    }
    updateControls();
  }

  function renderPerformanceDashboard() {
    const status = state.status;
    if (!status) return;
    const metrics = status.model?.metrics || {};
    const dataset = status.dataset || {};

    if (el.statAccuracy) el.statAccuracy.textContent = isNum(metrics.accuracy) ? pct(metrics.accuracy, 1) : "-";
    if (el.statMacroF1) el.statMacroF1.textContent = isNum(metrics.macro_f1) ? dec(metrics.macro_f1, 3) : "-";
    if (el.statIcbhi) el.statIcbhi.textContent = isNum(metrics.icbhi_score) ? pct(metrics.icbhi_score, 1) : "-";
    if (el.statCycles) el.statCycles.textContent = isNum(dataset.cycles) ? `${intFmt(dataset.cycles)} ciclos` : "-";

    if (el.perfSubtitle && isNum(metrics.n_cycles)) {
      const testPatients = dataset.splits?.test?.patients;
      el.perfSubtitle.textContent = `Avaliação estrita por paciente em ${intFmt(metrics.n_cycles)} ciclos reais${testPatients ? ` de ${intFmt(testPatients)} pacientes reservados` : ""}.`;
    }

    renderModelBars();

    if (el.ciNote) {
      const ci = metrics.macro_f1_ci95;
      if (ci && isNum(ci.lower) && isNum(ci.upper)) {
        el.ciNote.textContent = `IC 95% do Macro-F1 (${ci.method || "bootstrap por paciente"}, ${ci.repetitions || 500} repetições): ${dec(ci.lower, 4)} a ${dec(ci.upper, 4)}.`;
      } else {
        el.ciNote.textContent = "";
      }
    }

    if (Array.isArray(metrics.confusion_matrix) && metrics.confusion_matrix.length === 4) {
      renderConfusionMatrix(metrics.confusion_matrix);
    }

    const tbody = el.perClassTable?.querySelector("tbody");
    if (tbody && Array.isArray(metrics.per_class)) {
      tbody.replaceChildren();
      metrics.per_class.forEach((row) => {
        const tr = document.createElement("tr");
        const cells = [
          row.label,
          intFmt(row.support || 0),
          isNum(row.sensitivity) ? pct(row.sensitivity, 1) : "-",
          isNum(row.specificity) ? pct(row.specificity, 1) : "-",
          isNum(row.precision) ? pct(row.precision, 1) : "-",
          isNum(row.f1) ? dec(row.f1, 3) : "-",
        ];
        cells.forEach((txt) => {
          const td = document.createElement("td");
          td.textContent = txt;
          tr.append(td);
        });
        tbody.append(tr);
      });
    }

    if (el.datasetSummary) {
      const parts = [];
      if (dataset.name) parts.push(dataset.name);
      if (isNum(dataset.patients)) parts.push(`${intFmt(dataset.patients)} pacientes`);
      if (isNum(dataset.recordings)) parts.push(`${intFmt(dataset.recordings)} gravações`);
      if (isNum(dataset.cycles)) parts.push(`${intFmt(dataset.cycles)} ciclos`);
      parts.push("0 sobreposição de pacientes");
      el.datasetSummary.textContent = parts.join(" · ");
    }

    if (el.limitationsList && Array.isArray(status.model?.limitations)) {
      el.limitationsList.replaceChildren();
      status.model.limitations.forEach((lim) => {
        const li = document.createElement("li");
        li.textContent = lim;
        el.limitationsList.append(li);
      });
    }

    renderDeviceGeneralization(metrics.per_device);
  }

  function renderModelBars() {
    if (!el.modelComparisonBars) return;
    const comp = state.status?.model?.comparison || {};
    const key = state.selectedMetric;
    const isPct = key !== "macro_f1";
    const entries = [];

    if (comp.majority_baseline?.test) {
      entries.push({ name: "Classe majoritária (referência)", data: comp.majority_baseline.test, active: false });
    }
    if (comp.random_forest?.test) {
      entries.push({ name: "MFCC + Random Forest (v1.0)", data: comp.random_forest.test, active: false });
    }
    if (comp.cnn_v1?.test) {
      entries.push({ name: "SmallCNN (v1.0)", data: comp.cnn_v1.test, active: false });
    }
    if (comp.cnn_v1_1?.test) {
      entries.push({ name: "BandCNN por faixas (v1.1)", data: comp.cnn_v1_1.test, active: false });
    }
    if (comp.cnn?.test) {
      const activeShort = shortModelName(state.status?.model?.name);
      const activeName = activeShort ? `${activeShort} (ativo)` : "Modelo ativo";
      entries.push({ name: activeName, data: comp.cnn.test, active: true });
    }

    el.modelComparisonBars.replaceChildren();
    entries.forEach((entry) => {
      const val = entry.data?.[key];
      if (!isNum(val)) return;
      const row = document.createElement("div");
      row.className = `bar-row${entry.active ? " is-active" : ""}`;

      const meta = document.createElement("div");
      meta.className = "bar-meta";
      const nm = document.createElement("span");
      nm.textContent = entry.name;
      const vl = document.createElement("strong");
      vl.textContent = isPct ? pct(val, 2) : dec(val, 4);
      meta.append(nm, vl);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${Math.max(2, Math.min(100, val * 100))}%`;
      track.append(fill);

      row.append(meta, track);
      el.modelComparisonBars.append(row);
    });
  }

  function renderConfusionMatrix(matrix) {
    if (!el.confusionMatrix) return;
    const grid = document.createElement("div");
    grid.className = "cm-grid";

    const corner = document.createElement("div");
    corner.className = "cm-head";
    corner.textContent = "Real \\ Prev.";
    grid.append(corner);

    SHORT_LABELS.forEach((lbl) => {
      const h = document.createElement("div");
      h.className = "cm-head";
      h.textContent = lbl;
      grid.append(h);
    });

    matrix.forEach((rowVals, rIdx) => {
      const rowTotal = rowVals.reduce((a, b) => a + b, 0) || 1;
      const rLabel = document.createElement("div");
      rLabel.className = "cm-row-label";
      rLabel.textContent = SHORT_LABELS[rIdx];
      grid.append(rLabel);

      rowVals.forEach((count, cIdx) => {
        const share = count / rowTotal;
        const cell = document.createElement("div");
        const diag = rIdx === cIdx;
        cell.className = `cm-cell${diag ? " diag" : ""}`;

        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        if (diag) {
          if (isDark) {
            const strongFill = share >= 0.55;
            const alpha = strongFill ? Math.min(0.92, 0.68 + (share - 0.55) * 0.45) : 0.14 + share * 0.36;
            cell.style.background = `rgba(46, 196, 182, ${alpha.toFixed(2)})`;
            cell.style.color = strongFill ? "#051A1C" : "#ECF3F6";
          } else {
            const deep = share >= 0.55;
            const alpha = deep ? Math.min(1, 0.88 + (share - 0.55) * 0.25) : 0.08 + share * 0.35;
            cell.style.background = `rgba(11, 110, 117, ${alpha.toFixed(2)})`;
            cell.style.color = deep ? "#FFFFFF" : "#0C1C24";
          }
        } else {
          if (isDark) {
            cell.style.background = `rgba(242, 107, 122, ${(share * 0.30).toFixed(2)})`;
            cell.style.color = "#ECF3F6";
          } else {
            cell.style.background = `rgba(184, 50, 68, ${(share * 0.24).toFixed(2)})`;
            cell.style.color = "#0C1C24";
          }
        }

        cell.title = `${LABELS[rIdx]} -> ${LABELS[cIdx]}: ${intFmt(count)} ciclos (${pct(share, 1)})`;

        const strong = document.createElement("strong");
        strong.textContent = String(count);
        const sub = document.createElement("span");
        sub.textContent = pct(share, 0);
        if (diag && share >= 0.55) {
          sub.style.color = isDark ? "#051A1C" : "#FFFFFF";
          sub.style.opacity = "0.94";
        }
        cell.append(strong, sub);
        grid.append(cell);
      });
    });

    el.confusionMatrix.replaceChildren(grid);
  }

  function renderDeviceGeneralization(perDevice) {
    if (!el.deviceSection || !el.deviceGrid) return;
    const entries = perDevice && typeof perDevice === "object" ? Object.entries(perDevice) : [];
    if (!entries.length) {
      hide(el.deviceSection, true);
      return;
    }

    el.deviceGrid.replaceChildren();
    entries
      .sort((a, b) => (b[1]?.cycles || 0) - (a[1]?.cycles || 0))
      .forEach(([dev, info]) => {
        if (!info || !isNum(info.accuracy)) return;
        const card = document.createElement("div");
        card.className = "device-item";

        const top = document.createElement("div");
        top.className = "device-top";

        const left = document.createElement("div");
        const name = document.createElement("div");
        name.className = "device-name";
        name.textContent = dev;
        const cyc = document.createElement("div");
        cyc.className = "device-cycles";
        cyc.textContent = `${intFmt(info.cycles || 0)} ciclos no teste`;
        left.append(name, cyc);
        top.append(left);

        if (isNum(info.accuracy_v1_1)) {
          const diffPp = (info.accuracy - info.accuracy_v1_1) * 100;
          const gain = document.createElement("span");
          gain.className = `device-gain ${diffPp >= 0 ? "pos" : "neg"}`;
          gain.textContent = `${diffPp >= 0 ? "+" : ""}${dec(diffPp, 1)} p.p.`;
          top.append(gain);
        }
        card.append(top);

        const addBar = (label, value, cls) => {
          const row = document.createElement("div");
          row.className = `device-bar-row ${cls}`;
          const tag = document.createElement("span");
          tag.textContent = label;
          const track = document.createElement("div");
          track.className = "device-track";
          const fill = document.createElement("div");
          fill.className = "device-fill";
          fill.style.width = `${Math.max(2, Math.min(100, value * 100))}%`;
          track.append(fill);
          const val = document.createElement("strong");
          val.textContent = pct(value, 1);
          row.append(tag, track, val);
          card.append(row);
        };

        if (isNum(info.accuracy_v1_1)) addBar("v1.1", info.accuracy_v1_1, "is-old");
        addBar("v2.0", info.accuracy, "is-cur");
        el.deviceGrid.append(card);
      });

    hide(el.deviceSection, el.deviceGrid.children.length === 0);
  }

  /* ======================== EXAMPLES ======================== */
  async function fetchExamples() {
    try {
      const res = await fetch("/api/examples", { cache: "no-store" });
      if (!res.ok) throw new Error("Exemplos indisponíveis");
      const payload = await res.json();
      state.examples = Array.isArray(payload)
        ? payload
        : (Array.isArray(payload.examples) ? payload.examples : []);

      el.exampleSelect?.replaceChildren();
      el.quickExamples?.replaceChildren();
      if (!state.examples.length) throw new Error("Nenhum exemplo");

      const groups = new Map();
      const firstByClass = new Map();
      state.examples.forEach((example, idx) => {
        const cat = resolveExampleLabel(example) || "Ausculta";
        if (!firstByClass.has(cat)) firstByClass.set(cat, idx);
        if (!groups.has(cat)) {
          const g = document.createElement("optgroup");
          g.label = cat;
          groups.set(cat, g);
          el.exampleSelect?.append(g);
        }
        const g = groups.get(cat);
        g.append(new Option(`${cat} · Exemplo ${g.children.length + 1}`, String(idx)));
      });

      LABELS.forEach((cat, clsIdx) => {
        if (!firstByClass.has(cat) || !el.quickExamples) return;
        const idx = firstByClass.get(cat);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "seg-option";
        btn.dataset.category = cat;
        btn.setAttribute("aria-pressed", "false");
        btn.textContent = SHORT_LABELS[clsIdx];
        btn.addEventListener("click", () => {
          if (el.exampleSelect) el.exampleSelect.value = String(idx);
          loadExample(idx);
        });
        el.quickExamples.append(btn);
      });

      updateControls();
    } catch {
      state.examples = [];
      if (el.exampleSelect) {
        el.exampleSelect.replaceChildren(new Option("Sem exemplos disponíveis", ""));
        el.exampleSelect.disabled = true;
      }
      if (el.loadExample) el.loadExample.disabled = true;
      if (el.loadAndAnalyze) el.loadAndAnalyze.disabled = true;
    }
  }

  async function loadExample(explicitIdx = null, autoAnalyze = false) {
    const idx = explicitIdx !== null ? explicitIdx : Number(el.exampleSelect?.value);
    const example = state.examples[idx];
    if (!example || state.busy || state.loading) return;

    state.loading = true;
    updateControls();
    if (el.loadExample) el.loadExample.textContent = "Abrindo...";
    let loadedOk = false;

    try {
      const res = await fetch(`/api/examples/${encodeURIComponent(String(example.id))}/audio`);
      if (!res.ok) throw new Error("Não foi possível carregar o exemplo de teste.");
      const blob = await res.blob();
      const rawName = String(example.name || `exemplo-${example.id}.wav`).replace(/[\\/]/g, "_");
      const wavName = /\.wav$/i.test(rawName) ? rawName : `${rawName}.wav`;
      await loadFile(new File([blob], wavName, { type: "audio/wav" }), example);
      loadedOk = Boolean(state.file && state.decoded);
      if (loadedOk) showToast(`Exemplo carregado: ${example.name || resolveExampleLabel(example) || "WAV"}`);
    } catch (err) {
      showMessage(err.message);
    } finally {
      state.loading = false;
      if (el.loadExample) el.loadExample.textContent = "Carregar";
      updateControls();
    }

    if (autoAnalyze && loadedOk && el.analyzeButton && !el.analyzeButton.disabled) {
      await analyze();
    }
  }

  /* ======================== SEGMENT & CONTROLS ======================== */
  function segment() {
    const endRaw = Number(el.endTime?.value);
    return {
      start: Number(el.startTime?.value),
      end: endRaw === Number(state.duration.toFixed(3)) ? state.duration : endRaw,
    };
  }

  function segmentError() {
    if (!state.file || !state.decoded) return "Selecione uma gravação válida.";
    if (el.startTime?.value === "" || el.endTime?.value === "") return "Informe o início e o fim do ciclo.";
    const { start, end } = segment();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return "Use valores numéricos para os limites do ciclo.";
    if (start < 0) return "O início precisa ser igual ou maior que zero.";
    if (end <= start) return "O fim precisa ser posterior ao início.";
    if (end > state.duration + 1e-6) return "O fim do trecho ultrapassa a duração da gravação.";
    if (end - start < 0.1 - 1e-9) return "Selecione um trecho de pelo menos 0,1 segundo.";
    if (end - start > state.maxSegmentDuration + 1e-9) {
      return `O ciclo selecionado pode ter no máximo ${decAuto(state.maxSegmentDuration, 0)} segundos.`;
    }
    return "";
  }

  function invalidatePredictionForEdit() {
    if (el.cycleConfirmed) el.cycleConfirmed.checked = false;
    state.activeExample = null;
    hide(el.referenceNote, true);
    stopStudioAudio();
    state.playheadTime = null;
    el.audioPlayer?.pause();
    resetResult();
    updateControls();
  }

  function updateControls() {
    const err = state.file ? segmentError() : "";
    if (el.segmentError) {
      el.segmentError.textContent = err;
      hide(el.segmentError, !err);
    }
    const { start, end } = segment();
    if (el.segmentDuration) {
      el.segmentDuration.textContent = !err && state.file ? `${dec(end - start, 2)} s` : "-";
    }

    const ready = state.status?.model_ready === true;
    const locked = state.busy || state.loading;

    if (el.analyzeButton) {
      el.analyzeButton.disabled = !ready || !state.file || Boolean(err) || !el.cycleConfirmed?.checked || locked;
    }
    if (el.playSegment) el.playSegment.disabled = Boolean(err) || state.busy || !state.file;
    if (el.startTime) el.startTime.disabled = locked;
    if (el.endTime) el.endTime.disabled = locked;
    document.querySelectorAll(".stepper-btn[data-target][data-delta]").forEach((btn) => {
      btn.disabled = locked || !state.decoded;
    });
    if (el.resetFullCycle) el.resetFullCycle.disabled = locked || !state.decoded;
    if (el.cycleConfirmed) el.cycleConfirmed.disabled = locked;
    if (el.uploadButton) el.uploadButton.disabled = locked;
    if (el.audioFile) el.audioFile.disabled = locked;
    if (el.clearFile) el.clearFile.disabled = locked;
    if (el.loadExample) el.loadExample.disabled = !state.examples.length || locked;
    if (el.loadAndAnalyze) el.loadAndAnalyze.disabled = !state.examples.length || !ready || locked;
    if (el.exampleSelect) el.exampleSelect.disabled = !state.examples.length || locked;
    if (el.dropzone) el.dropzone.classList.toggle("has-audio", Boolean(state.file));
    if (el.uploadButtonText) el.uploadButtonText.textContent = state.file ? "Trocar arquivo WAV" : "Selecionar arquivo";
    el.quickExamples?.querySelectorAll("button").forEach((b) => { b.disabled = locked; });
    syncQuickExamples();

    if (el.analyzeButtonText) {
      el.analyzeButtonText.textContent = state.busy ? "Analisando o áudio..." : "Analisar ciclo respiratório";
    }
    if (el.analyzeHelp) {
      el.analyzeHelp.textContent = state.busy
        ? "Processando o trecho no servidor local..."
        : !ready
          ? "Aguardando o modelo local ficar pronto."
          : !el.cycleConfirmed?.checked
            ? "Marque a confirmação do ciclo para habilitar a análise."
            : "Tudo pronto: pressione Enter ou clique acima para classificar.";
    }

    drawWaveform();
  }

  function resetResult() {
    state.result = null;
    hide(el.resultContent, true);
    hide(el.analysisProgress, true);
    hide(el.resultEmpty, false);
    if (el.resultStateText) {
      el.resultStateText.textContent = state.file ? "Pronto para analisar" : "Aguardando áudio";
    }
  }

  function clearFile() {
    state.revision += 1;
    state.controller?.abort();
    stopStudioAudio();
    state.file = null;
    state.decoded = null;
    state.duration = 0;
    state.sampleRate = 0;
    state.channels = 1;
    state.activeExample = null;
    state.loading = false;
    state.playheadTime = null;
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = null;
    }
    if (el.audioPlayer) {
      el.audioPlayer.pause();
      el.audioPlayer.removeAttribute("src");
      el.audioPlayer.load();
    }
    if (el.audioFile) el.audioFile.value = "";
    if (el.startTime) el.startTime.value = "0";
    if (el.endTime) el.endTime.value = "";
    if (el.cycleConfirmed) el.cycleConfirmed.checked = false;
    hide(el.audioWorkspace, true);
    hide(el.referenceNote, true);
    hide(el.suggestedCycles, true);
    el.suggestedCycles?.replaceChildren();
    resetResult();
    updateControls();
  }

  async function loadFile(file, example = null) {
    if (state.busy) return;
    clearFile();
    showMessage("");

    if (!file || !/\.wav$/i.test(file.name)) {
      showMessage(
        /\.(ogg|oga|opus)$/i.test(file?.name || "")
          ? VOICE_MSG
          : "Escolha uma gravação de ausculta em formato WAV."
      );
      return;
    }
    if (!file.size) {
      showMessage("O arquivo está vazio. Selecione outra gravação.");
      return;
    }
    if (file.size > state.maxUploadMb * 1024 * 1024) {
      showMessage(`O arquivo ultrapassa o limite de ${decAuto(state.maxUploadMb, 0)} MB.`);
      return;
    }

    const revision = state.revision;
    state.loading = true;
    updateControls();
    let decodeCtx;

    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error("Este navegador não suporta Web Audio API.");
      const buf = await file.arrayBuffer();
      const wavMeta = pcmWavInfo(buf);
      const header = String.fromCharCode(...new Uint8Array(buf, 0, Math.min(buf.byteLength, 12)));
      if (header.startsWith("OggS")) throw new Error(VOICE_MSG);
      if (!/^(RIFF|RIFX|RF64)/.test(header) || header.slice(8, 12) !== "WAVE") {
        throw new Error("O conteúdo do arquivo não é WAV válido. Escolha uma gravação de ausculta em WAV.");
      }

      decodeCtx = new AC();
      const decoded = await decodeCtx.decodeAudioData(buf.slice(0));
      if (revision !== state.revision) return;

      if (!Number.isFinite(decoded.duration) || decoded.duration <= 0) {
        throw new Error("A gravação não tem duração válida.");
      }
      const channels = wavMeta?.channels || decoded.numberOfChannels;
      if (channels < 1 || channels > 2) {
        throw new Error("A aplicação aceita áudios mono ou estéreo.");
      }
      const srcDur = wavMeta && isNum(wavMeta.duration) ? wavMeta.duration : decoded.duration;
      if (srcDur > state.maxDuration) {
        throw new Error(`A gravação ultrapassa o limite de ${decAuto(state.maxDuration, 0)} segundos.`);
      }

      state.file = file;
      state.decoded = decoded;
      state.duration = srcDur;
      state.sampleRate = wavMeta?.sampleRate || decoded.sampleRate;
      state.channels = channels;
      state.activeExample = example || null;
      state.audioUrl = URL.createObjectURL(file);
      if (el.audioPlayer) el.audioPlayer.src = state.audioUrl;

      if (el.fileName) el.fileName.textContent = example?.name || file.name;
      if (el.fileMeta) {
        el.fileMeta.textContent = `${dec(srcDur, 2)} s · ${intFmt(state.sampleRate)} Hz · ${state.channels === 2 ? "Estéreo" : "Mono"} · WAV`;
      }
      if (el.waveformMid) el.waveformMid.textContent = `${dec(srcDur / 2, 1)} s`;
      if (el.waveformEnd) el.waveformEnd.textContent = `${dec(srcDur, 2)} s`;
      if (el.startTime) el.startTime.max = String(srcDur);
      if (el.endTime) el.endTime.max = srcDur.toFixed(3);

      const propStart = example && isNum(example.start) ? example.start : 0;
      const propEnd = example && isNum(example.end)
        ? example.end
        : Math.min(srcDur, state.maxSegmentDuration);
      if (el.startTime) el.startTime.value = String(Number(Math.max(0, propStart).toFixed(3)));
      if (el.endTime) el.endTime.value = String(Number(Math.min(srcDur, propEnd).toFixed(3)));
      if (el.cycleConfirmed) el.cycleConfirmed.checked = Boolean(example);

      if (example && el.referenceNote) {
        const ref = resolveExampleLabel(example);
        el.referenceNote.textContent = `Referência anotada do conjunto de teste: ${ref || "não informada"} (não enviada ao modelo).`;
        hide(el.referenceNote, false);
      }

      hide(el.detectCyclesBtn, srcDur < 3.5);
      hide(el.audioWorkspace, false);
      resetResult();
      el.audioWorkspace?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "nearest" });
      requestAnimationFrame(drawWaveform);
    } catch (err) {
      if (revision === state.revision) {
        clearFile();
        showMessage(
          err instanceof DOMException
            ? "Não foi possível decodificar este WAV. Verifique a integridade do arquivo."
            : err.message
        );
      }
    } finally {
      if (decodeCtx && decodeCtx.state !== "closed") await decodeCtx.close().catch(() => {});
      if (revision === state.revision) {
        state.loading = false;
        updateControls();
      }
    }
  }

  /* ======================== HIGH-DPI INTERACTIVE OSCILLOSCOPE ======================== */
  function drawWaveform() {
    const canvas = el.waveform;
    if (!canvas || !state.decoded || canvas.clientWidth < 1) return;

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth);
    const height = Math.round(canvas.clientHeight || Number(canvas.getAttribute("height")) || 148);
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    // Deep Pulmonary Ink chamber background
    ctx.fillStyle = "#0C1C24";
    ctx.fillRect(0, 0, width, height);

    // Subtle calibration grid lines at 25%, 50%, 75%
    ctx.strokeStyle = "rgba(176, 224, 228, 0.08)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((f) => {
      const y = Math.round(height * f) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    });

    const samples = state.decoded.getChannelData(0);
    const { start, end } = segment();
    const x1 = Math.max(0, Math.min(width, (start / state.duration) * width));
    const x2 = Math.max(0, Math.min(width, (end / state.duration) * width));
    const validSel = Number.isFinite(x1) && Number.isFinite(x2) && x2 > x1;

    // Selected region fill
    if (validSel) {
      ctx.fillStyle = "rgba(78, 224, 198, 0.14)";
      ctx.fillRect(x1, 0, x2 - x1, height);
    }

    // Waveform peaks: bright cyan-teal inside selection, muted slate-teal outside
    const midY = height / 2;
    const amp = height * 0.43;
    const stride = Math.max(1, Math.floor((samples.length / width) * 2));
    for (let x = 0; x < width; x += 2) {
      const offset = Math.floor((x / width) * samples.length);
      const limit = Math.min(samples.length, offset + stride);
      let low = 1;
      let high = -1;
      for (let i = offset; i < limit; i++) {
        const s = samples[i];
        if (s < low) low = s;
        if (s > high) high = s;
      }
      if (low > high) {
        low = 0;
        high = 0;
      }
      const inSel = validSel && x >= x1 && x <= x2;
      ctx.strokeStyle = inSel ? "#4EE0C6" : "rgba(148, 210, 206, 0.30)";
      ctx.lineWidth = inSel ? 1.5 : 1;
      const y1 = midY + low * amp;
      const y2 = Math.max(y1 + 1, midY + high * amp);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, y1);
      ctx.lineTo(x + 0.5, y2);
      ctx.stroke();
    }

    // Draggable crisp white handles with rounded grip pill and #0C1C24 inner notch
    if (validSel) {
      [x1, x2].forEach((rawX) => {
        const sx = Math.max(4, Math.min(width - 4, rawX));
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx, height);
        ctx.stroke();

        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(sx - 4, midY - 14, 8, 28, 4);
        } else {
          ctx.rect(sx - 4, midY - 14, 8, 28);
        }
        ctx.fill();

        ctx.fillStyle = "#0C1C24";
        ctx.fillRect(sx - 1, midY - 6, 2, 12);
      });
    }

    // Live warm amber playhead
    if (isNum(state.playheadTime) && state.duration > 0) {
      const px = Math.max(0, Math.min(width, (state.playheadTime / state.duration) * width));
      ctx.strokeStyle = "#FFAE52";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();
    }
  }

  function pointerToTime(event) {
    const rect = el.waveform.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const x = event.clientX - rect.left;
    const frac = Math.max(0, Math.min(1, x / width));
    return { frac, x, width, time: frac * state.duration };
  }

  el.waveform?.addEventListener("pointerdown", (e) => {
    if (!state.decoded || state.busy || state.loading) return;
    const { x, width, time } = pointerToTime(e);
    const { start, end } = segment();
    const x1 = (start / state.duration) * width;
    const x2 = (end / state.duration) * width;
    const dStart = Math.abs(x - x1);
    const dEnd = Math.abs(x - x2);

    if (dStart <= 14 && dStart <= dEnd) {
      state.dragMode = "start";
    } else if (dEnd <= 14) {
      state.dragMode = "end";
    } else {
      state.dragMode = "new";
      state.dragAnchor = time;
    }

    el.waveform.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  });

  el.waveform?.addEventListener("pointermove", (e) => {
    if (!state.decoded) return;
    const { x, width, time } = pointerToTime(e);

    if (el.waveformTooltip) {
      el.waveformTooltip.textContent = `${dec(time, 2)} s`;
      el.waveformTooltip.style.left = `${Math.max(28, Math.min(width - 28, x))}px`;
      hide(el.waveformTooltip, false);
    }

    if (!state.dragMode) {
      const { start, end } = segment();
      const x1 = (start / state.duration) * width;
      const x2 = (end / state.duration) * width;
      el.waveform.style.cursor = (Math.abs(x - x1) <= 14 || Math.abs(x - x2) <= 14)
        ? "ew-resize"
        : "crosshair";
      return;
    }

    let { start, end } = segment();
    if (state.dragMode === "start") {
      start = Math.max(0, Math.min(time, end - 0.1));
    } else if (state.dragMode === "end") {
      end = Math.min(state.duration, Math.max(time, start + 0.1));
    } else if (state.dragMode === "new") {
      start = Math.min(state.dragAnchor, time);
      end = Math.max(state.dragAnchor, time);
      if (end - start < 0.1) {
        end = Math.min(state.duration, start + 0.1);
        if (end - start < 0.1) start = Math.max(0, end - 0.1);
      }
    }

    if (end - start > state.maxSegmentDuration) {
      if (state.dragMode === "start" || (state.dragMode === "new" && time < state.dragAnchor)) {
        start = end - state.maxSegmentDuration;
      } else {
        end = start + state.maxSegmentDuration;
      }
    }

    if (el.startTime) el.startTime.value = String(Number(start.toFixed(3)));
    if (el.endTime) el.endTime.value = String(Number(end.toFixed(3)));
    invalidatePredictionForEdit();
  });

  function endWaveformDrag(e) {
    if (!state.dragMode) return;
    state.dragMode = null;
    if (e.pointerId !== undefined) {
      try { el.waveform?.releasePointerCapture?.(e.pointerId); } catch {}
    }
  }

  el.waveform?.addEventListener("pointerup", endWaveformDrag);
  el.waveform?.addEventListener("pointercancel", endWaveformDrag);
  el.waveform?.addEventListener("pointerleave", () => {
    if (!state.dragMode) hide(el.waveformTooltip, true);
  });

  /* ======================== WEB AUDIO LISTENING STUDIO ======================== */
  function stopStudioAudio() {
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    if (state.sourceNode) {
      try {
        state.sourceNode.onended = null;
        state.sourceNode.stop();
        state.sourceNode.disconnect();
      } catch {}
      state.sourceNode = null;
    }
    state.isPlayingStudio = false;
    if (el.playSegmentIcon) el.playSegmentIcon.textContent = "▶";
    if (el.playSegmentText) el.playSegmentText.textContent = "Ouvir trecho";
  }

  function applyLiveStudioNodes(smooth = true) {
    if (!state.audioCtx) return;
    const now = state.audioCtx.currentTime;
    const hpFreq = state.filterOn ? 80 : 10;
    const lpFreq = state.filterOn ? 1600 : Math.min(20000, state.audioCtx.sampleRate / 2 - 100);
    const gainVal = state.gain2x ? 2.0 : 1.0;

    if (smooth && state.isPlayingStudio) {
      state.hpNode?.frequency.setTargetAtTime(hpFreq, now, 0.015);
      state.lpNode?.frequency.setTargetAtTime(lpFreq, now, 0.015);
      state.gainNode?.gain.setTargetAtTime(gainVal, now, 0.015);
    } else {
      if (state.hpNode) state.hpNode.frequency.value = hpFreq;
      if (state.lpNode) state.lpNode.frequency.value = lpFreq;
      if (state.gainNode) state.gainNode.gain.value = gainVal;
    }

    if (state.sourceNode) {
      state.sourceNode.playbackRate.value = state.slowSpeed ? 0.75 : 1.0;
      state.sourceNode.loop = state.loop;
    }
  }

  async function startStudioFrom(offsetOverride = null) {
    if (segmentError() || !state.decoded) return;
    stopStudioAudio();
    el.audioPlayer?.pause();

    const { start, end } = segment();
    const safeStart = Math.max(0, Math.min(start, Math.max(0, state.decoded.duration - 0.01)));
    const segLen = Math.max(0.01, Math.min(end - safeStart, state.decoded.duration - safeStart));
    const segEnd = safeStart + segLen;
    const startAt = isNum(offsetOverride) && offsetOverride >= safeStart && offsetOverride < segEnd - 0.01
      ? offsetOverride
      : safeStart;

    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!state.audioCtx || state.audioCtx.state === "closed") state.audioCtx = new AC();
      if (state.audioCtx.state === "suspended") await state.audioCtx.resume();

      const src = state.audioCtx.createBufferSource();
      src.buffer = state.decoded;
      src.playbackRate.value = state.slowSpeed ? 0.75 : 1.0;
      src.loop = state.loop;
      src.loopStart = safeStart;
      src.loopEnd = segEnd;

      const hp = state.audioCtx.createBiquadFilter();
      hp.type = "highpass";
      hp.Q.value = 0.707;
      const lp = state.audioCtx.createBiquadFilter();
      lp.type = "lowpass";
      lp.Q.value = 0.707;
      const gain = state.audioCtx.createGain();

      src.connect(hp);
      hp.connect(lp);
      lp.connect(gain);
      gain.connect(state.audioCtx.destination);

      state.sourceNode = src;
      state.hpNode = hp;
      state.lpNode = lp;
      state.gainNode = gain;
      applyLiveStudioNodes(false);

      state.isPlayingStudio = true;
      state.studioSegStart = safeStart;
      state.studioSegLen = segLen;
      state.studioCursorBase = startAt;
      state.studioPlayStartWall = state.audioCtx.currentTime;
      state.playheadTime = startAt;

      if (el.playSegmentIcon) el.playSegmentIcon.textContent = "■";
      if (el.playSegmentText) el.playSegmentText.textContent = "Parar trecho";

      src.onended = () => {
        if (state.sourceNode === src) {
          stopStudioAudio();
          state.playheadTime = null;
          drawWaveform();
        }
      };

      if (state.loop) {
        src.start(0, startAt);
      } else {
        src.start(0, startAt, Math.max(0.01, segEnd - startAt));
      }

      const tick = () => {
        if (!state.isPlayingStudio || !state.audioCtx) return;
        const rate = state.slowSpeed ? 0.75 : 1.0;
        const elapsed = (state.audioCtx.currentTime - state.studioPlayStartWall) * rate;
        if (state.loop) {
          const span = Math.max(0.01, state.studioSegLen);
          const rel = ((state.studioCursorBase - state.studioSegStart) + elapsed) % span;
          state.playheadTime = state.studioSegStart + (rel < 0 ? rel + span : rel);
        } else {
          state.playheadTime = Math.min(state.studioSegStart + state.studioSegLen, state.studioCursorBase + elapsed);
        }
        drawWaveform();
        state.rafId = requestAnimationFrame(tick);
      };
      state.rafId = requestAnimationFrame(tick);
    } catch {
      stopStudioAudio();
      showMessage("O navegador bloqueou a reprodução via Web Audio. Utilize o reprodutor nativo abaixo.");
    }
  }

  async function toggleStudioPlay() {
    if (state.isPlayingStudio) {
      stopStudioAudio();
      state.playheadTime = null;
      drawWaveform();
      return;
    }
    await startStudioFrom();
  }

  /* ======================== AUTO-SUGGEST BREATH CYCLES ======================== */
  function suggestCycles() {
    if (!state.decoded || state.duration < 3.5 || !el.suggestedCycles) return;
    const samples = state.decoded.getChannelData(0);
    const sr = state.decoded.sampleRate;
    const hop = Math.max(1, Math.round(sr * 0.05));
    const env = [];
    for (let i = 0; i < samples.length; i += hop) {
      let sum = 0;
      const limit = Math.min(samples.length, i + hop);
      for (let j = i; j < limit; j++) sum += samples[j] * samples[j];
      env.push(Math.sqrt(sum / Math.max(1, limit - i)));
    }

    // 7-frame moving average smoothing (350 ms)
    const smooth = env.map((_, idx) => {
      let total = 0;
      let count = 0;
      for (let k = Math.max(0, idx - 3); k <= Math.min(env.length - 1, idx + 3); k++) {
        total += env[k];
        count += 1;
      }
      return total / Math.max(1, count);
    });

    // Local valleys spaced at least 1.4 s (28 frames) apart
    const boundaries = [0];
    let lastIdx = 0;
    for (let i = 6; i < smooth.length - 6; i++) {
      if (i - lastIdx < 28) continue;
      let isMin = true;
      for (let k = i - 5; k <= i + 5; k++) {
        if (smooth[k] < smooth[i]) {
          isMin = false;
          break;
        }
      }
      if (isMin) {
        boundaries.push((i * hop) / sr);
        lastIdx = i;
      }
    }
    boundaries.push(state.duration);

    const cycles = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const a = boundaries[i];
      const b = boundaries[i + 1];
      if (b - a >= 0.8 && b - a <= state.maxSegmentDuration) cycles.push([a, b]);
    }

    el.suggestedCycles.replaceChildren();
    if (!cycles.length) {
      const note = document.createElement("span");
      note.className = "helper";
      note.textContent = "Nenhum vale rítmico nítido detectado; delimite manualmente na forma de onda.";
      el.suggestedCycles.append(note);
    } else {
      cycles.slice(0, 10).forEach(([a, b], idx) => {
        const option = document.createElement("button");
        option.type = "button";
        option.className = "cycle-option";
        option.textContent = `Ciclo ${idx + 1}: ${dec(a, 2)} a ${dec(b, 2)} s`;
        option.addEventListener("click", () => {
          el.suggestedCycles.querySelectorAll(".cycle-option").forEach((node) => {
            node.classList.toggle("active", node === option);
          });
          if (el.startTime) el.startTime.value = String(Number(a.toFixed(3)));
          if (el.endTime) el.endTime.value = String(Number(b.toFixed(3)));
          invalidatePredictionForEdit();
        });
        el.suggestedCycles.append(option);
      });
    }
    hide(el.suggestedCycles, false);
  }

  /* ======================== EVIDENCE TABS, LIGHTBOX & RESULTS ======================== */
  function syncZoomButton() {
    if (!el.zoomEvidenceBtn) return;
    const isImgTab = state.activeTab === "spec" || state.activeTab === "contrast";
    const imgEl = state.activeTab === "contrast" ? el.contrastImage : el.spectrogramImage;
    const hasSrc = Boolean(imgEl && imgEl.getAttribute("src"));
    hide(el.zoomEvidenceBtn, !(isImgTab && hasSrc));
  }

  function setEvidenceTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll(".tab-btn").forEach((b) => {
      const active = b.dataset.tab === tabName;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
      hide(p, p.dataset.panel !== tabName);
    });
    syncZoomButton();
  }

  function openEvidenceLightbox(kind = state.activeTab) {
    if (!el.imageLightbox) return;
    const useContrast = kind === "contrast";
    const srcImg = useContrast ? el.contrastImage : el.spectrogramImage;
    const src = srcImg?.getAttribute("src");
    if (!src) return;

    state.lightboxKind = useContrast ? "contrast" : "spec";
    if (el.lightboxTitle) {
      el.lightboxTitle.textContent = useContrast
        ? "Mapa de contraste acústico (CMN invariante a estetoscópio)"
        : "Espectrograma log-Mel filtrado (80 a 1.200 Hz)";
    }
    if (el.lightboxImage) {
      el.lightboxImage.src = src;
      el.lightboxImage.alt = srcImg.alt || el.lightboxTitle?.textContent || "Evidência acústica";
    }
    if (el.lightboxCaption) {
      el.lightboxCaption.textContent = useContrast
        ? "Realce após subtração da resposta estacionária do microfone: cristas horizontais em ciano indicam sibilos; transientes verticais em âmbar indicam estertores."
        : "Espectrograma log-Mel na banda respiratória: faixas horizontais sustentadas sugerem sibilos e impulsos verticais curtos sugerem estertores.";
    }

    if (typeof el.imageLightbox.showModal === "function") {
      if (!el.imageLightbox.open) el.imageLightbox.showModal();
    } else {
      el.imageLightbox.setAttribute("open", "");
    }
  }

  function closeEvidenceLightbox() {
    if (!el.imageLightbox) return;
    if (typeof el.imageLightbox.close === "function" && el.imageLightbox.open) {
      el.imageLightbox.close();
    } else {
      el.imageLightbox.removeAttribute("open");
    }
  }

  function renderResult(result, contextInfo = null) {
    const pred = result.prediction;
    if (!pred || typeof pred.label !== "string" || !Array.isArray(pred.scores) || !pred.scores.length) {
      throw new Error("O servidor retornou uma classificação incompleta.");
    }
    const scores = pred.scores.filter(
      (item) => typeof item.label === "string" && isNum(item.score) && item.score >= 0 && item.score <= 1
    );
    if (scores.length !== pred.scores.length) {
      throw new Error("Pontuações inválidas recebidas do servidor.");
    }

    const clsIdx = isNum(pred.index) ? pred.index : Math.max(0, LABELS.indexOf(pred.label));
    if (el.predictionCard) el.predictionCard.className = `prediction-summary cls-${clsIdx}`;
    if (el.predictionLabel) el.predictionLabel.textContent = pred.label;

    const sorted = [...scores].sort((a, b) => b.score - a.score);
    const top1 = sorted[0]?.score ?? 0;
    const top2 = sorted[1]?.score ?? 0;
    if (el.confidenceStat) el.confidenceStat.textContent = `Escore principal: ${pct(top1, 1)}`;
    if (el.marginStat) el.marginStat.textContent = `Margem +${decAuto((top1 - top2) * 100, 1)} p.p.`;

    if (el.predictionMeta) {
      const dur = result.segment?.duration_s;
      const modelName = shortModelName(result.model?.name) || "Modelo local";
      const rawVer = result.model?.version ? `v${result.model.version}` : "";
      const verSuffix = rawVer && !/v\d+\.\d+/i.test(modelName) ? ` ${rawVer}` : "";
      el.predictionMeta.textContent = `${isNum(dur) ? `Trecho de ${dec(dur, 2)} s · ` : ""}${modelName}${verSuffix}`.trim();
    }

    const refLabel = contextInfo?.referenceLabel ?? resolveExampleLabel(state.activeExample);
    if (el.referenceComparison) {
      if (refLabel) {
        const match = refLabel === pred.label;
        el.referenceComparison.className = `reference-compare ${match ? "match" : "mismatch"}`;
        el.referenceComparison.textContent = match
          ? `✓ Corresponde à referência anotada do exemplo (${refLabel})`
          : `≠ Referência anotada: ${refLabel} · Previsão do modelo: ${pred.label}`;
        hide(el.referenceComparison, false);
      } else {
        hide(el.referenceComparison, true);
      }
    }

    const byLbl = Object.fromEntries(scores.map((s) => [s.label, s.score]));
    const pCrackle = isNum(pred.events?.estertores)
      ? pred.events.estertores
      : Math.min(1, (byLbl["Estertores"] || 0) + (byLbl["Ambos"] || 0));
    const pWheeze = isNum(pred.events?.sibilos)
      ? pred.events.sibilos
      : Math.min(1, (byLbl["Sibilos"] || 0) + (byLbl["Ambos"] || 0));

    if (el.crackleProb) el.crackleProb.textContent = pct(pCrackle, 1);
    if (el.crackleBar) el.crackleBar.style.width = `${Math.max(1, pCrackle * 100)}%`;
    if (el.wheezeProb) el.wheezeProb.textContent = pct(pWheeze, 1);
    if (el.wheezeBar) el.wheezeBar.style.width = `${Math.max(1, pWheeze * 100)}%`;

    if (el.scoreList) {
      el.scoreList.replaceChildren();
      scores.forEach((item) => {
        const row = document.createElement("div");
        row.className = `score-row${item.label === pred.label ? " winner" : ""}`;
        const text = document.createElement("div");
        text.className = "score-label";
        const lbl = document.createElement("span");
        lbl.textContent = item.label;
        const val = document.createElement("span");
        val.textContent = pct(item.score, 1);
        text.append(lbl, val);

        const track = document.createElement("div");
        track.className = "score-track";
        const fill = document.createElement("div");
        fill.className = "score-fill";
        fill.style.width = `${item.score * 100}%`;
        track.append(fill);

        row.append(text, track);
        el.scoreList.append(row);
      });
    }

    if (el.acousticNote) {
      el.acousticNote.textContent = ACOUSTIC_NOTES[pred.label]
        || "Pontuações experimentais produzidas por modelo acadêmico local.";
    }

    // Spectrogram PNG
    const specImg = result.spectrogram_png;
    if (typeof specImg === "string" && specImg.startsWith("data:image/png;base64,")) {
      if (el.spectrogramImage) el.spectrogramImage.src = specImg;
      hide(el.spectrogramImage, false);
      hide(el.spectrogramUnavailable, true);
    } else {
      el.spectrogramImage?.removeAttribute("src");
      hide(el.spectrogramImage, true);
      hide(el.spectrogramUnavailable, false);
    }

    // Acoustic Contrast CMN PNG (hide tab when unavailable)
    const contrastImg = result.contrast_png;
    if (typeof contrastImg === "string" && contrastImg.startsWith("data:image/png;base64,")) {
      if (el.contrastImage) el.contrastImage.src = contrastImg;
      hide(el.contrastImage, false);
      hide(el.contrastUnavailable, true);
      hide(el.contrastTabBtn, false);
    } else {
      el.contrastImage?.removeAttribute("src");
      hide(el.contrastImage, true);
      hide(el.contrastUnavailable, false);
      hide(el.contrastTabBtn, true);
    }

    // Technical telemetry & quality warnings
    if (el.techTelemetry) {
      el.techTelemetry.replaceChildren();
      const techItems = [
        ["Taxa original", result.audio?.sample_rate ? `${intFmt(result.audio.sample_rate)} Hz` : "-"],
        ["Canais", result.audio?.channels === 2 ? "Estéreo para mono" : "Mono"],
        [
          "Trecho",
          isNum(result.segment?.start) && isNum(result.segment?.end)
            ? `${dec(result.segment.start, 2)} a ${dec(result.segment.end, 2)} s`
            : "-",
        ],
      ];
      techItems.forEach(([k, v]) => {
        const box = document.createElement("div");
        box.className = "tech-stat";
        const sp = document.createElement("span");
        sp.textContent = k;
        const st = document.createElement("strong");
        st.textContent = v;
        box.append(sp, st);
        el.techTelemetry.append(box);
      });
    }

    if (el.qualityMessage) {
      const warnings = Array.isArray(result.quality?.warnings)
        ? result.quality.warnings.filter((x) => typeof x === "string")
        : [];
      el.qualityMessage.replaceChildren();
      if (warnings.length) {
        const ul = document.createElement("ul");
        warnings.forEach((w) => {
          const li = document.createElement("li");
          li.textContent = w;
          ul.append(li);
        });
        el.qualityMessage.append(ul);
      } else {
        el.qualityMessage.textContent = "Nenhum aviso técnico foi sinalizado para este recorte.";
      }
      hide(el.qualityMessage, false);
    }

    setEvidenceTab("spec");
    state.result = result;
    hide(el.resultEmpty, true);
    hide(el.analysisProgress, true);
    hide(el.resultContent, false);
    if (el.resultStateText) el.resultStateText.textContent = "Análise concluída";
  }

  function pushHistory(result) {
    const entry = {
      id: Date.now(),
      name: state.activeExample?.name || state.file?.name || "Áudio WAV",
      referenceLabel: resolveExampleLabel(state.activeExample),
      result,
    };
    state.history.unshift(entry);
    if (state.history.length > 8) state.history.pop();
    renderHistory();
  }

  function renderHistory() {
    if (!el.sessionHistory) return;
    el.sessionHistory.replaceChildren();
    hide(el.sessionHistoryWrap, state.history.length === 0);
    state.history.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-item";
      const nm = document.createElement("span");
      nm.className = "hist-name";
      nm.textContent = `${item.name} (${dec(item.result.segment?.duration_s || 0, 2)} s)`;
      const pr = document.createElement("span");
      pr.className = "hist-pred";
      const topScore = Math.max(...(item.result.prediction?.scores || []).map((s) => s.score), 0);
      pr.textContent = `${item.result.prediction?.label} · ${pct(topScore, 0)}`;
      btn.append(nm, pr);
      btn.addEventListener("click", () => {
        renderResult(item.result, { referenceLabel: item.referenceLabel });
      });
      el.sessionHistory.append(btn);
    });
  }

  async function analyze() {
    if (!el.analyzeButton || el.analyzeButton.disabled) return;
    const err = segmentError();
    if (err) {
      showMessage(err);
      return;
    }
    const { start, end } = segment();
    const revision = state.revision;
    const form = new FormData();
    form.append("file", state.file);
    form.append("start", String(start));
    form.append("end", String(end));

    state.busy = true;
    state.controller = new AbortController();
    showMessage("");
    state.result = null;
    hide(el.resultEmpty, true);
    hide(el.resultContent, true);
    hide(el.analysisProgress, false);
    if (el.resultStateText) el.resultStateText.textContent = "Processando...";
    updateControls();

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
        signal: state.controller.signal,
      });
      let payload;
      try {
        payload = await res.json();
      } catch {
        throw new Error("O servidor não retornou uma resposta JSON válida.");
      }
      if (!res.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "A gravação não pôde ser analisada.");
      }
      if (revision === state.revision) {
        renderResult(payload);
        pushHistory(payload);
        if (window.matchMedia("(max-width: 960px)").matches) {
          el.resultStateText?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "nearest" });
        }
      }
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.revision) {
        resetResult();
        showMessage(error.message || "Falha na análise. Tente novamente.");
      }
    } finally {
      state.busy = false;
      state.controller = null;
      updateControls();
    }
  }

  /* ======================== EVENT WIRING ======================== */
  el.heroUploadCta?.addEventListener("click", () => {
    setWorkstationTab("analyze");
    byId("analysis")?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "start",
    });
    if (el.audioFile && !el.audioFile.disabled) {
      el.audioFile.click();
    }
  });

  el.uploadButton?.addEventListener("click", () => el.audioFile?.click());
  el.audioFile?.addEventListener("change", () => {
    const f = el.audioFile.files?.[0];
    if (f) loadFile(f);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    el.dropzone?.addEventListener(ev, (e) => {
      e.preventDefault();
      if (!state.busy) el.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    el.dropzone?.addEventListener(ev, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("dragover");
    });
  });
  el.dropzone?.addEventListener("drop", (e) => {
    if (state.busy || state.loading) return;
    if (e.dataTransfer.files.length !== 1) {
      showMessage("Selecione uma única gravação por análise.");
      return;
    }
    loadFile(e.dataTransfer.files[0]);
  });

  el.clearFile?.addEventListener("click", clearFile);

  [el.startTime, el.endTime].forEach((inp) => {
    inp?.addEventListener("input", invalidatePredictionForEdit);
  });

  document.querySelectorAll(".stepper-btn[data-target][data-delta]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.decoded || state.busy || state.loading) return;
      const target = byId(btn.dataset.target);
      if (!target) return;
      const delta = Number(btn.dataset.delta);
      const next = Math.max(0, Math.min(state.duration, (Number(target.value) || 0) + delta));
      target.value = String(Number(next.toFixed(3)));
      invalidatePredictionForEdit();
    });
  });

  el.resetFullCycle?.addEventListener("click", () => {
    if (!state.decoded || state.busy || state.loading) return;
    if (el.startTime) el.startTime.value = "0";
    if (el.endTime) {
      el.endTime.value = String(Number(Math.min(state.duration, state.maxSegmentDuration).toFixed(3)));
    }
    invalidatePredictionForEdit();
  });

  el.cycleConfirmed?.addEventListener("change", updateControls);
  el.playSegment?.addEventListener("click", toggleStudioPlay);

  el.loopToggle?.addEventListener("click", () => {
    const resumeAt = state.isPlayingStudio && isNum(state.playheadTime) ? state.playheadTime : null;
    state.loop = !state.loop;
    el.loopToggle.setAttribute("aria-pressed", String(state.loop));
    el.loopToggle.classList.toggle("active", state.loop);
    if (state.isPlayingStudio) {
      startStudioFrom(resumeAt);
    } else {
      applyLiveStudioNodes();
    }
  });

  el.filterToggle?.addEventListener("click", () => {
    state.filterOn = !state.filterOn;
    el.filterToggle.setAttribute("aria-pressed", String(state.filterOn));
    el.filterToggle.classList.toggle("active", state.filterOn);
    applyLiveStudioNodes();
  });

  el.gainToggle?.addEventListener("click", () => {
    state.gain2x = !state.gain2x;
    el.gainToggle.setAttribute("aria-pressed", String(state.gain2x));
    el.gainToggle.classList.toggle("active", state.gain2x);
    applyLiveStudioNodes();
  });

  el.speedToggle?.addEventListener("click", () => {
    if (state.isPlayingStudio && state.audioCtx && isNum(state.playheadTime)) {
      state.studioCursorBase = state.playheadTime;
      state.studioPlayStartWall = state.audioCtx.currentTime;
    }
    state.slowSpeed = !state.slowSpeed;
    el.speedToggle.setAttribute("aria-pressed", String(state.slowSpeed));
    el.speedToggle.classList.toggle("active", state.slowSpeed);
    applyLiveStudioNodes();
  });

  el.detectCyclesBtn?.addEventListener("click", suggestCycles);

  // Sync native audio player with waveform playhead
  el.audioPlayer?.addEventListener("play", () => {
    stopStudioAudio();
  });
  el.audioPlayer?.addEventListener("timeupdate", () => {
    if (!state.isPlayingStudio && !el.audioPlayer.paused) {
      state.playheadTime = el.audioPlayer.currentTime;
      drawWaveform();
    }
  });
  ["pause", "ended"].forEach((ev) => {
    el.audioPlayer?.addEventListener(ev, () => {
      if (!state.isPlayingStudio) {
        state.playheadTime = null;
        drawWaveform();
      }
    });
  });

  el.loadExample?.addEventListener("click", () => loadExample());
  el.loadAndAnalyze?.addEventListener("click", () => loadExample(null, true));
  el.analyzeButton?.addEventListener("click", analyze);

  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.addEventListener("click", () => setEvidenceTab(b.dataset.tab));
  });
  el.zoomEvidenceBtn?.addEventListener("click", () => openEvidenceLightbox(state.activeTab));
  el.spectrogramImage?.addEventListener("click", () => openEvidenceLightbox("spec"));
  el.contrastImage?.addEventListener("click", () => openEvidenceLightbox("contrast"));
  el.lightboxClose?.addEventListener("click", closeEvidenceLightbox);
  el.imageLightbox?.addEventListener("click", (e) => {
    if (e.target === el.imageLightbox) closeEvidenceLightbox();
  });
  el.lightboxDownload?.addEventListener("click", () => {
    const src = el.lightboxImage?.getAttribute("src");
    if (!src) return;
    const link = document.createElement("a");
    link.href = src;
    link.download = state.lightboxKind === "contrast"
      ? "ausculta-contraste-cmn.png"
      : "ausculta-espectrograma.png";
    document.body.append(link);
    link.click();
    link.remove();
    showToast("Imagem PNG salva.");
  });

  el.downloadResult?.addEventListener("click", () => {
    if (!state.result) return;
    const report = {
      exported_at: new Date().toISOString(),
      filename: state.file?.name || null,
      result: state.result,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "ausculta-resultado.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Resultado JSON exportado.");
  });

  el.copySummary?.addEventListener("click", async () => {
    if (!state.result) return;
    const pred = state.result.prediction;
    const lines = [
      "Ausculta Lab: Resumo da análise",
      `Arquivo: ${state.activeExample?.name || state.file?.name || "WAV"}`,
      `Trecho: ${dec(state.result.segment?.start || 0, 2)} a ${dec(state.result.segment?.end || 0, 2)} s (${dec(state.result.segment?.duration_s || 0, 2)} s)`,
      `Categoria prevista: ${pred.label}`,
      `Pontuações: ${pred.scores.map((s) => `${s.label}: ${pct(s.score, 1)}`).join(" | ")}`,
      "Aviso: protótipo acadêmico experimental, sem validação clínica.",
    ];
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      el.copySummary.textContent = "✓ Copiado";
      showToast("Resumo copiado para a área de transferência.");
      if (state.copyTimer) clearTimeout(state.copyTimer);
      state.copyTimer = setTimeout(() => {
        if (el.copySummary) el.copySummary.textContent = "Copiar resumo";
      }, 1800);
    } catch {
      showMessage("Não foi possível copiar para a área de transferência.");
    }
  });

  el.clearHistory?.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  el.metricSwitcher?.querySelectorAll(".seg-option, button").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!btn.dataset.metric) return;
      state.selectedMetric = btn.dataset.metric;
      el.metricSwitcher.querySelectorAll(".seg-option, button").forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-pressed", String(active));
      });
      renderModelBars();
    });
  });

  /* ======================== NAVIGATION SCROLLSPY & SCROLL ======================== */
  function setActiveNav(sectionId) {
    const activeWsTab = el.wsPanelTests && !el.wsPanelTests.classList.contains("hidden") ? "tests" : "analyze";
    document.querySelectorAll(".nav-link").forEach((item) => {
      const wsTab = item.dataset.wsTab;
      const match = sectionId === "analysis" && wsTab
        ? wsTab === activeWsTab
        : !wsTab && item.getAttribute("href") === `#${sectionId}`;
      item.classList.toggle("active", match);
      if (match) item.setAttribute("aria-current", "location");
      else item.removeAttribute("aria-current");
    });
  }

  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      if (link.dataset.wsTab) return;
      const href = link.getAttribute("href") || "";
      if (href.startsWith("#")) setActiveNav(href.slice(1));
    });
  });

  if (typeof IntersectionObserver === "function") {
    const sections = ["analysis", "performance", "help"].map((id) => byId(id)).filter(Boolean);
    const spy = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (visible.length) setActiveNav(visible[0].target.id);
    }, { rootMargin: "-25% 0px -55% 0px", threshold: [0.1, 0.35, 0.6] });
    sections.forEach((sec) => spy.observe(sec));
  }

  function syncScrollUi() {
    el.siteHeader?.classList.toggle("scrolled", window.scrollY > 12);
    hide(el.backToTop, window.scrollY < 420);
  }

  el.backToTop?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  });
  window.addEventListener("scroll", syncScrollUi, { passive: true });
  syncScrollUi();

  /* ======================== KEYBOARD SHORTCUTS ======================== */
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const active = document.activeElement;
    const tag = active?.tagName;
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag) || active?.isContentEditable) return;
    if (el.imageLightbox?.open) return;

    if (e.code === "Space" && state.file && !segmentError()) {
      e.preventDefault();
      toggleStudioPlay();
    } else if (e.code === "Enter" && el.analyzeButton && !el.analyzeButton.disabled) {
      e.preventDefault();
      analyze();
    } else if ((e.key === "[" || e.key === "]") && state.examples.length && !state.busy && !state.loading) {
      e.preventDefault();
      const curIdx = Number(el.exampleSelect?.value);
      const base = Number.isFinite(curIdx) ? curIdx : 0;
      const nextIdx = (base + (e.key === "]" ? 1 : -1) + state.examples.length) % state.examples.length;
      if (el.exampleSelect) el.exampleSelect.value = String(nextIdx);
      loadExample(nextIdx);
    }
  });

  /* ======================== MOTION & RESIZE ======================== */
  document.documentElement.classList.add("motion-ready");

  const revealTargets = document.querySelectorAll(".reveal");
  if (typeof IntersectionObserver === "function" && !prefersReducedMotion()) {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    revealTargets.forEach((node) => revealObserver.observe(node));
  } else {
    revealTargets.forEach((node) => node.classList.add("in"));
  }

  if (typeof MutationObserver === "function") {
    const popTargets = [
      el.statAccuracy, el.statMacroF1, el.statIcbhi, el.statCycles,
      el.predictionLabel, el.crackleProb, el.wheezeProb, el.resultStateText, el.modelStateText,
    ].filter(Boolean);
    popTargets.forEach((node) => {
      const observer = new MutationObserver(() => {
        if (prefersReducedMotion()) return;
        node.classList.remove("value-pop");
        void node.offsetWidth;
        node.classList.add("value-pop");
      });
      observer.observe(node, { childList: true, characterData: true, subtree: true });
    });
  }

  if (typeof ResizeObserver === "function" && el.waveform) {
    new ResizeObserver(() => drawWaveform()).observe(el.waveform);
  }
  window.addEventListener("resize", drawWaveform);

  window.addEventListener("beforeunload", () => {
    stopStudioAudio();
    state.controller?.abort();
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    if (state.audioCtx && state.audioCtx.state !== "closed") {
      state.audioCtx.close().catch(() => {});
    }
  });

  /* ======================== WORKSTATION TABS & TEST WAVS ======================== */
  function setWorkstationTab(tabName) {
    const isTests = tabName === "tests";
    if (el.wsTabAnalyze) {
      el.wsTabAnalyze.classList.toggle("active", !isTests);
      el.wsTabAnalyze.setAttribute("aria-selected", String(!isTests));
    }
    if (el.wsTabTests) {
      el.wsTabTests.classList.toggle("active", isTests);
      el.wsTabTests.setAttribute("aria-selected", String(isTests));
    }
    hide(el.wsPanelAnalyze, isTests);
    hide(el.wsPanelTests, !isTests);
    if (!isTests) drawWaveform();

    document.querySelectorAll(".nav-link[data-ws-tab]").forEach((link) => {
      const active = link.dataset.wsTab === (isTests ? "tests" : "analyze");
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  el.wsTabAnalyze?.addEventListener("click", () => setWorkstationTab("analyze"));
  el.wsTabTests?.addEventListener("click", () => setWorkstationTab("tests"));

  document.querySelectorAll(".nav-link[data-ws-tab]").forEach((link) => {
    link.addEventListener("click", () => {
      setWorkstationTab(link.dataset.wsTab || "analyze");
    });
  });

  el.testAudioList?.querySelectorAll(".test-load-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (state.busy || state.loading) return;
      const src = btn.dataset.testSrc;
      const name = btn.dataset.testName || "audio-teste.wav";
      if (!src) return;
      const originalText = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Carregando...";
      try {
        const res = await fetch(src, { cache: "no-store" });
        if (!res.ok) throw new Error("Não foi possível carregar o arquivo WAV de teste.");
        const blob = await res.blob();
        const file = new File([blob], name, { type: "audio/wav" });
        setWorkstationTab("analyze");
        await loadFile(file, null);
        if (state.file && state.decoded) {
          showToast(`${name} carregado na estação de análise.`);
        }
      } catch (err) {
        showMessage(err.message || "Falha ao carregar o áudio de teste.");
      } finally {
        btn.disabled = false;
        btn.textContent = originalText;
      }
    });
  });

  /* ======================== THEME TOGGLE ======================== */
  function getTheme() {
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function applyTheme(theme, persist = false) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    if (persist) {
      try { localStorage.setItem("ausculta-theme", next); } catch {}
    }
    if (el.themeToggle) {
      const dark = next === "dark";
      el.themeToggle.setAttribute("aria-pressed", String(dark));
      el.themeToggle.setAttribute("aria-label", dark ? "Alternar para modo claro" : "Alternar para modo escuro");
      el.themeToggle.title = dark ? "Mudar para modo claro" : "Mudar para modo escuro";
    }
    const matrix = state.status?.model?.metrics?.confusion_matrix;
    if (Array.isArray(matrix) && matrix.length === 4) {
      renderConfusionMatrix(matrix);
    }
  }

  el.themeToggle?.addEventListener("click", () => {
    applyTheme(getTheme() === "dark" ? "light" : "dark", true);
  });

  applyTheme(getTheme(), false);

  fetchStatus();
  fetchExamples();
})();
