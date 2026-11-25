"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const el = Object.fromEntries([
    "versionChip", "modelState", "modelStateText", "globalMessage",
    "kpiAccuracy", "kpiMacroF1", "kpiIcbhi", "kpiCycles",
    "dropzone", "uploadButton", "uploadButtonText", "audioFile", "uploadLimits",
    "quickExamples", "exampleSelect", "loadExample", "audioWorkspace",
    "fileName", "fileMeta", "clearFile", "waveform", "waveformTooltip",
    "waveformMid", "waveformEnd", "resetFullCycle",
    "playSegment", "playSegmentIcon", "playSegmentText",
    "loopToggle", "filterToggle", "gainToggle", "speedToggle", "detectCyclesBtn", "suggestedCycles",
    "audioPlayer", "startTime", "endTime", "segmentDuration", "segmentError",
    "referenceNote", "cycleConfirmed", "analyzeButton", "analyzeButtonText", "analyzeHelp",
    "resultTag", "resultEmpty", "analysisProgress", "resultContent",
    "predictionCard", "predictionLabel", "predictionMeta", "confidenceBadge", "marginBadge",
    "referenceComparison", "crackleProb", "crackleBar", "wheezeProb", "wheezeBar",
    "scoreList", "acousticNote", "spectrogramImage", "spectrogramUnavailable",
    "contrastImage", "contrastUnavailable", "contrastTabBtn", "zoomEvidenceBtn",
    "imageLightbox", "lightboxTitle", "lightboxDownload", "lightboxClose", "lightboxImage", "lightboxCaption",
    "techTelemetry", "qualityMessage",
    "downloadResult", "copySummary", "sessionHistoryWrap", "sessionHistory", "clearHistory",
    "metricSwitcher", "perfSubtitle", "modelComparisonBars", "ciNote",
    "confusionMatrix", "perClassTable", "datasetPills", "limitationsList",
    "deviceCard", "deviceGrid"
  ].map((id) => [id, byId(id)]));

  const LABELS = ["Sem sibilos/estertores", "Estertores", "Sibilos", "Ambos"];
  const SHORT_LABELS = ["Sem eventos", "Estertores", "Sibilos", "Ambos"];
  const ACOUSTIC_NOTES = {
    "Sem sibilos/estertores": "Padrão compatível com som respiratório basal, sem detecção predominante de estertores ou sibilos neste recorte. Ausência desses padrões acústicos não exclui doença clínica.",
    "Estertores": "Padrão compatível com estertores: sons descontínuos, breves e impulsivos (5–20 ms), frequentemente visíveis como traços verticais curtos no espectrograma.",
    "Sibilos": "Padrão compatível com sibilos: sons musicais contínuos e tonais (>100 ms), visíveis como faixas horizontais estreitas no espectrograma e no mapa de contraste.",
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
    // Studio WebAudio playback state
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
    studioOffsetStart: 0,
    studioSegLen: 0,
    rafId: null,
    playheadTime: null,
    // Waveform interaction state
    dragMode: null, // "start" | "end" | "new" | null
    dragAnchor: 0,
    activeTab: "spec",
    lightboxKind: "spec",
  };

  const hide = (node, yes = true) => node && node.classList.toggle("hidden", yes);
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const dec = (v, digits = 2) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
  const decAuto = (v, digits = 2) => new Intl.NumberFormat("pt-BR", { maximumFractionDigits: digits }).format(v);
  const pct = (v, digits = 1) => `${decAuto(v * 100, digits)}%`;

  function pcmWavInfo(buffer) {
    if (buffer.byteLength < 44) return null;
    const view = new DataView(buffer);
    const txt = (off, len) => String.fromCharCode(...new Uint8Array(buffer, off, len));
    if (txt(0, 4) !== "RIFF" || txt(8, 4) !== "WAVE") return null;
    let format = null, dataBytes = 0;
    for (let off = 12; off + 8 <= buffer.byteLength;) {
      const id = txt(off, 4), len = view.getUint32(off + 4, true), body = off + 8;
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
    if (!format || ![1, 3].includes(format.code) || !format.rate || !format.alignment || !dataBytes || dataBytes % format.alignment) return null;
    return { duration: dataBytes / format.alignment / format.rate, sampleRate: format.rate, channels: format.channels };
  }

  function showMessage(msg) {
    el.globalMessage.textContent = msg;
    hide(el.globalMessage, !msg);
    if (msg) el.globalMessage.scrollIntoView({ block: "nearest" });
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
      el.uploadLimits.textContent = `WAV mono ou estéreo · até ${decAuto(state.maxUploadMb, 0)} MB · até ${decAuto(state.maxDuration, 0)} s`;
      if (state.status.version) el.versionChip.textContent = `v${state.status.version}`;
      const ready = state.status.model_ready === true;
      el.modelStateText.textContent = ready ? (state.status.model?.name ? `Pronto · ${state.status.model.name}` : "Pronto para análise") : "Modelo indisponível";
      el.modelState.className = "status-pill " + (ready ? "ready" : "unavailable");
      renderPerformanceDashboard();
    } catch {
      state.status = null;
      el.modelState.className = "status-pill unavailable";
      el.modelStateText.textContent = "Sem conexão";
      showMessage("Não foi possível acessar o servidor local. Abra a aplicação e recarregue esta página.");
    }
    updateControls();
  }

  function renderPerformanceDashboard() {
    const status = state.status;
    if (!status) return;
    const metrics = status.model?.metrics || {};
    const comp = status.model?.comparison || {};
    const dataset = status.dataset || {};

    el.kpiAccuracy.textContent = isNum(metrics.accuracy) ? pct(metrics.accuracy, 1) : "—";
    el.kpiMacroF1.textContent = isNum(metrics.macro_f1) ? dec(metrics.macro_f1, 3) : "—";
    el.kpiIcbhi.textContent = isNum(metrics.icbhi_score) ? pct(metrics.icbhi_score, 1) : "—";
    el.kpiCycles.textContent = isNum(dataset.cycles) ? `${new Intl.NumberFormat("pt-BR").format(dataset.cycles)} ciclos` : "—";

    if (isNum(metrics.n_cycles)) {
      const testPatients = dataset.splits?.test?.patients;
      el.perfSubtitle.textContent = `Avaliação estrita por paciente em ${new Intl.NumberFormat("pt-BR").format(metrics.n_cycles)} ciclos reais${testPatients ? ` de ${testPatients} pacientes reservados` : ""}.`;
    }

    renderModelBars();

    const ci = metrics.macro_f1_ci95;
    if (ci && isNum(ci.lower) && isNum(ci.upper)) {
      el.ciNote.textContent = `IC 95% do Macro-F1 (${ci.method || "bootstrap por paciente"}, ${ci.repetitions || 500} repetições): ${dec(ci.lower, 4)} a ${dec(ci.upper, 4)}.`;
    } else {
      el.ciNote.textContent = "";
    }

    // Confusion Matrix
    if (Array.isArray(metrics.confusion_matrix) && metrics.confusion_matrix.length === 4) {
      renderConfusionMatrix(metrics.confusion_matrix);
    }

    // Per-class table
    const tbody = el.perClassTable?.querySelector("tbody");
    if (tbody && Array.isArray(metrics.per_class)) {
      tbody.replaceChildren();
      metrics.per_class.forEach((row) => {
        const tr = document.createElement("tr");
        const cells = [
          row.label,
          new Intl.NumberFormat("pt-BR").format(row.support || 0),
          isNum(row.sensitivity) ? pct(row.sensitivity, 1) : "—",
          isNum(row.specificity) ? pct(row.specificity, 1) : "—",
          isNum(row.precision) ? pct(row.precision, 1) : "—",
          isNum(row.f1) ? dec(row.f1, 3) : "—",
        ];
        cells.forEach((txt) => {
          const td = document.createElement("td");
          td.textContent = txt;
          tr.append(td);
        });
        tbody.append(tr);
      });
    }

    // Dataset pills & limitations
    if (el.datasetPills) {
      el.datasetPills.replaceChildren();
      const pills = [];
      if (dataset.name) pills.push(dataset.name);
      if (isNum(dataset.patients)) pills.push(`${dataset.patients} pacientes`);
      if (isNum(dataset.recordings)) pills.push(`${dataset.recordings} gravações`);
      if (isNum(dataset.cycles)) pills.push(`${new Intl.NumberFormat("pt-BR").format(dataset.cycles)} ciclos`);
      pills.push("0 sobreposição de pacientes");
      pills.forEach((txt) => {
        const span = document.createElement("span");
        span.className = "ds-pill";
        span.textContent = txt;
        el.datasetPills.append(span);
      });
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

  function renderDeviceGeneralization(perDevice) {
    if (!el.deviceCard || !el.deviceGrid) return;
    const entries = perDevice && typeof perDevice === "object" ? Object.entries(perDevice) : [];
    if (!entries.length) {
      hide(el.deviceCard, true);
      return;
    }
    el.deviceGrid.replaceChildren();
    entries.sort((a, b) => (b[1]?.cycles || 0) - (a[1]?.cycles || 0)).forEach(([dev, info]) => {
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
      cyc.textContent = `${new Intl.NumberFormat("pt-BR").format(info.cycles || 0)} ciclos no teste`;
      left.append(name, cyc);
      top.append(left);

      if (isNum(info.accuracy_v1_1)) {
        const diffPp = (info.accuracy - info.accuracy_v1_1) * 100;
        const badge = document.createElement("span");
        badge.className = `device-gain ${diffPp >= 0 ? "pos" : "neg"}`;
        badge.textContent = `${diffPp >= 0 ? "+" : ""}${dec(diffPp, 1)} p.p.`;
        top.append(badge);
      }
      card.append(top);

      const addBar = (label, value, cls) => {
        const row = document.createElement("div");
        row.className = `device-bar-row ${cls}`;
        const tag = document.createElement("span"); tag.textContent = label;
        const track = document.createElement("div"); track.className = "device-track";
        const fill = document.createElement("div"); fill.className = "device-fill";
        fill.style.width = `${Math.max(2, Math.min(100, value * 100))}%`;
        track.append(fill);
        const val = document.createElement("strong"); val.textContent = pct(value, 1);
        row.append(tag, track, val);
        card.append(row);
      };

      if (isNum(info.accuracy_v1_1)) addBar("v1.1", info.accuracy_v1_1, "is-old");
      addBar("v2.0", info.accuracy, "is-cur");
      el.deviceGrid.append(card);
    });
    hide(el.deviceCard, el.deviceGrid.children.length === 0);
  }

  function renderModelBars() {
    const comp = state.status?.model?.comparison || {};
    const key = state.selectedMetric;
    const isPct = key !== "macro_f1";
    const entries = [];
    if (comp.majority_baseline?.test) entries.push({ name: "Classe majoritária (referência)", data: comp.majority_baseline.test, active: false });
    if (comp.random_forest?.test) entries.push({ name: "MFCC + Random Forest (v1.0)", data: comp.random_forest.test, active: false });
    if (comp.cnn_v1?.test) entries.push({ name: "SmallCNN (v1.0)", data: comp.cnn_v1.test, active: false });
    if (comp.cnn_v1_1?.test) entries.push({ name: "BandCNN por faixas (v1.1)", data: comp.cnn_v1_1.test, active: false });
    if (comp.cnn?.test) {
      const activeName = state.status?.model?.name ? `${state.status.model.name} (ativo)` : "Modelo ativo";
      entries.push({ name: activeName, data: comp.cnn.test, active: true });
    }
    el.modelComparisonBars.replaceChildren();
    entries.forEach((entry) => {
      const val = entry.data?.[key];
      if (!isNum(val)) return;
      const row = document.createElement("div");
      row.className = "bar-row" + (entry.active ? " is-active" : "");
      const meta = document.createElement("div");
      meta.className = "bar-meta";
      const nm = document.createElement("span"); nm.textContent = entry.name;
      const vl = document.createElement("strong"); vl.textContent = isPct ? pct(val, 2) : dec(val, 4);
      meta.append(nm, vl);
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.width = `${Math.max(2, Math.min(100, val * 100))}%`;
      track.append(fill);
      row.append(meta, track);
      el.modelComparisonBars.append(row);
    });
  }

  function renderConfusionMatrix(matrix) {
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
        cell.className = "cm-cell" + (diag ? " diag" : "");
        const alpha = 0.08 + share * 0.52;
        cell.style.background = diag
          ? `rgba(125, 232, 199, ${alpha.toFixed(2)})`
          : `rgba(248, 200, 255, ${(share * 0.34).toFixed(2)})`;
        cell.title = `${LABELS[rIdx]} → ${LABELS[cIdx]}: ${count} ciclos (${pct(share, 1)})`;
        const strong = document.createElement("strong"); strong.textContent = String(count);
        const sub = document.createElement("span"); sub.textContent = pct(share, 0);
        cell.append(strong, sub);
        grid.append(cell);
      });
    });
    el.confusionMatrix.replaceChildren(grid);
  }

  /* ======================== EXAMPLES ======================== */
  async function fetchExamples() {
    try {
      const res = await fetch("/api/examples", { cache: "no-store" });
      if (!res.ok) throw new Error("Exemplos indisponíveis");
      const payload = await res.json();
      state.examples = Array.isArray(payload) ? payload : (Array.isArray(payload.examples) ? payload.examples : []);
      el.exampleSelect.replaceChildren();
      el.quickExamples.replaceChildren();
      if (!state.examples.length) throw new Error("Nenhum exemplo");

      const groups = new Map();
      const firstByClass = new Map();
      state.examples.forEach((example, idx) => {
        const cat = example.label || "Ausculta";
        if (!firstByClass.has(cat)) firstByClass.set(cat, idx);
        if (!groups.has(cat)) {
          const g = document.createElement("optgroup");
          g.label = cat;
          groups.set(cat, g);
          el.exampleSelect.append(g);
        }
        const g = groups.get(cat);
        g.append(new Option(`${cat} · Exemplo ${g.children.length + 1}`, String(idx)));
      });

      // Quick 1-click pills per category
      LABELS.forEach((cat, clsIdx) => {
        if (!firstByClass.has(cat)) return;
        const idx = firstByClass.get(cat);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quick-pill";
        const dot = document.createElement("span");
        dot.className = `dot dot-${clsIdx}`;
        btn.append(dot, document.createTextNode(SHORT_LABELS[clsIdx]));
        btn.addEventListener("click", () => {
          el.exampleSelect.value = String(idx);
          loadExample(idx);
        });
        el.quickExamples.append(btn);
      });
      updateControls();
    } catch {
      state.examples = [];
      el.exampleSelect.replaceChildren(new Option("Sem exemplos disponíveis", ""));
      el.exampleSelect.disabled = true;
      el.loadExample.disabled = true;
    }
  }

  async function loadExample(explicitIdx = null) {
    const idx = explicitIdx !== null ? explicitIdx : Number(el.exampleSelect.value);
    const example = state.examples[idx];
    if (!example || state.busy || state.loading) return;
    state.loading = true;
    updateControls();
    el.loadExample.textContent = "Abrindo…";
    try {
      const res = await fetch(`/api/examples/${encodeURIComponent(String(example.id))}/audio`);
      if (!res.ok) throw new Error("Não foi possível carregar o exemplo de teste.");
      const blob = await res.blob();
      const name = String(example.name || `exemplo-${example.id}.wav`).replace(/[\\/]/g, "_");
      await loadFile(new File([blob], /\.wav$/i.test(name) ? name : `${name}.wav`, { type: "audio/wav" }), example);
    } catch (err) {
      showMessage(err.message);
    } finally {
      state.loading = false;
      el.loadExample.textContent = "Carregar";
      updateControls();
    }
  }

  /* ======================== SEGMENT & CONTROLS ======================== */
  function segment() {
    const endRaw = Number(el.endTime.value);
    return {
      start: Number(el.startTime.value),
      end: endRaw === Number(state.duration.toFixed(3)) ? state.duration : endRaw,
    };
  }

  function segmentError() {
    if (!state.file || !state.decoded) return "Selecione uma gravação válida.";
    if (el.startTime.value === "" || el.endTime.value === "") return "Informe o início e o fim do ciclo.";
    const { start, end } = segment();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return "Use valores numéricos para os limites do ciclo.";
    if (start < 0) return "O início precisa ser igual ou maior que zero.";
    if (end <= start) return "O fim precisa ser posterior ao início.";
    if (end > state.duration + 1e-6) return "O fim do trecho ultrapassa a duração da gravação.";
    if (end - start < 0.1 - 1e-9) return "Selecione um trecho de pelo menos 0,1 segundo.";
    if (end - start > state.maxSegmentDuration + 1e-9) return `O ciclo selecionado pode ter no máximo ${decAuto(state.maxSegmentDuration, 0)} segundos.`;
    return "";
  }

  function invalidatePredictionForEdit() {
    el.cycleConfirmed.checked = false;
    state.activeExample = null;
    hide(el.referenceNote);
    stopStudioAudio();
    el.audioPlayer.pause();
    resetResult();
    updateControls();
  }

  function updateControls() {
    const err = state.file ? segmentError() : "";
    el.segmentError.textContent = err;
    hide(el.segmentError, !err);
    const { start, end } = segment();
    el.segmentDuration.textContent = !err && state.file ? `${dec(end - start, 2)} s` : "—";
    const ready = state.status?.model_ready === true;
    const locked = state.busy || state.loading;

    el.analyzeButton.disabled = !ready || !state.file || !!err || !el.cycleConfirmed.checked || locked;
    el.playSegment.disabled = !!err || state.busy || !state.file;
    el.startTime.disabled = locked;
    el.endTime.disabled = locked;
    el.cycleConfirmed.disabled = locked;
    el.uploadButton.disabled = locked;
    el.audioFile.disabled = locked;
    el.clearFile.disabled = locked;
    el.loadExample.disabled = !state.examples.length || locked;
    el.exampleSelect.disabled = !state.examples.length || locked;
    el.dropzone.classList.toggle("has-audio", Boolean(state.file));
    el.uploadButtonText.textContent = state.file ? "Trocar arquivo WAV" : "Selecionar arquivo";
    el.quickExamples.querySelectorAll("button").forEach((b) => { b.disabled = locked; });

    el.analyzeButtonText.textContent = state.busy ? "Analisando o áudio…" : "Analisar ciclo respiratório";
    el.analyzeHelp.textContent = state.busy
      ? "Processando o trecho no servidor local…"
      : !ready
        ? "Aguardando o modelo local ficar pronto."
        : !el.cycleConfirmed.checked
          ? "Marque a confirmação do ciclo para habilitar a análise."
          : "Tudo pronto — pressione Enter ou clique acima para classificar.";

    drawWaveform();
  }

  function resetResult() {
    state.result = null;
    hide(el.resultContent);
    hide(el.analysisProgress);
    hide(el.resultEmpty, false);
    el.resultTag.textContent = state.file ? "Pronto para analisar" : "Aguardando áudio";
  }

  function clearFile() {
    state.revision += 1;
    state.controller?.abort();
    stopStudioAudio();
    state.file = null;
    state.decoded = null;
    state.duration = 0;
    state.activeExample = null;
    state.loading = false;
    state.playheadTime = null;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = null;
    el.audioPlayer.pause();
    el.audioPlayer.removeAttribute("src");
    el.audioPlayer.load();
    el.audioFile.value = "";
    el.startTime.value = "0";
    el.endTime.value = "";
    el.cycleConfirmed.checked = false;
    hide(el.audioWorkspace);
    hide(el.referenceNote);
    hide(el.suggestedCycles);
    el.suggestedCycles.replaceChildren();
    resetResult();
    updateControls();
  }

  async function loadFile(file, example = null) {
    if (state.busy) return;
    clearFile();
    showMessage("");
    if (!file || !/\.wav$/i.test(file.name)) {
      showMessage(/\.(ogg|oga|opus)$/i.test(file?.name || "") ? VOICE_MSG : "Escolha uma gravação de ausculta em formato WAV.");
      return;
    }
    if (!file.size) { showMessage("O arquivo está vazio. Selecione outra gravação."); return; }
    if (file.size > state.maxUploadMb * 1024 * 1024) { showMessage(`O arquivo ultrapassa o limite de ${decAuto(state.maxUploadMb, 0)} MB.`); return; }

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
      if (!Number.isFinite(decoded.duration) || decoded.duration <= 0) throw new Error("A gravação não tem duração válida.");
      if (decoded.numberOfChannels < 1 || decoded.numberOfChannels > 2) throw new Error("A aplicação aceita áudios mono ou estéreo.");
      const srcDur = wavMeta && isNum(wavMeta.duration) ? wavMeta.duration : decoded.duration;
      if (srcDur > state.maxDuration) throw new Error(`A gravação ultrapassa o limite de ${decAuto(state.maxDuration, 0)} segundos.`);

      state.file = file;
      state.decoded = decoded;
      state.duration = srcDur;
      state.sampleRate = wavMeta?.sampleRate || decoded.sampleRate;
      state.channels = wavMeta?.channels || decoded.numberOfChannels;
      state.activeExample = example || null;
      state.audioUrl = URL.createObjectURL(file);
      el.audioPlayer.src = state.audioUrl;

      el.fileName.textContent = example?.name || file.name;
      el.fileMeta.textContent = `${dec(srcDur, 2)} s · ${new Intl.NumberFormat("pt-BR").format(state.sampleRate)} Hz · ${state.channels === 2 ? "Estéreo" : "Mono"} · WAV`;
      el.waveformMid.textContent = `${dec(srcDur / 2, 1)} s`;
      el.waveformEnd.textContent = `${dec(srcDur, 2)} s`;
      el.startTime.max = String(srcDur);
      el.endTime.max = srcDur.toFixed(3);

      const propStart = example && isNum(example.start) ? example.start : 0;
      const propEnd = example && isNum(example.end) ? example.end : Math.min(srcDur, state.maxSegmentDuration);
      el.startTime.value = String(Number(Math.max(0, propStart).toFixed(3)));
      el.endTime.value = String(Number(Math.min(srcDur, propEnd).toFixed(3)));
      el.cycleConfirmed.checked = Boolean(example);

      if (example) {
        const ref = typeof example.label === "number" ? LABELS[example.label] : example.label;
        el.referenceNote.textContent = `Referência anotada do conjunto de teste: ${ref || "não informada"} (não enviada ao modelo).`;
        hide(el.referenceNote, false);
      }
      hide(el.detectCyclesBtn, srcDur < 3.5);
      hide(el.audioWorkspace, false);
      resetResult();
      el.audioWorkspace.scrollIntoView({ block: "nearest" });
    } catch (err) {
      if (revision === state.revision) {
        clearFile();
        showMessage(err instanceof DOMException ? "Não foi possível decodificar este WAV. Verifique a integridade do arquivo." : err.message);
      }
    } finally {
      if (decodeCtx && decodeCtx.state !== "closed") await decodeCtx.close().catch(() => {});
      if (revision === state.revision) { state.loading = false; updateControls(); }
    }
  }

  /* ======================== INTERACTIVE WAVEFORM CANVAS ======================== */
  function drawWaveform() {
    const canvas = el.waveform;
    if (!state.decoded || canvas.clientWidth < 1) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth), height = 148;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);

    // Subtle horizontal grid lines
    ctx.strokeStyle = "rgba(185, 255, 251, 0.06)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((f) => {
      ctx.beginPath(); ctx.moveTo(0, height * f); ctx.lineTo(width, height * f); ctx.stroke();
    });

    const samples = state.decoded.getChannelData(0);
    const { start, end } = segment();
    const x1 = Math.max(0, Math.min(width, (start / state.duration) * width));
    const x2 = Math.max(0, Math.min(width, (end / state.duration) * width));
    const validSel = Number.isFinite(x1) && Number.isFinite(x2) && x2 > x1;

    // Selection shading
    if (validSel) {
      ctx.fillStyle = "rgba(125, 232, 199, 0.14)";
      ctx.fillRect(x1, 0, x2 - x1, height);
    }

    // Draw waveform peaks (bright inside selection, dim outside)
    const stride = Math.max(1, Math.floor(samples.length / width));
    for (let x = 0; x < width; x += 2) {
      const offset = Math.floor((x / width) * samples.length);
      let low = 1, high = -1;
      const limit = Math.min(samples.length, offset + stride * 2);
      for (let i = offset; i < limit; i++) {
        const s = samples[i];
        if (s < low) low = s;
        if (s > high) high = s;
      }
      const inSel = validSel && x >= x1 && x <= x2;
      ctx.strokeStyle = inSel ? "#83e6cc" : "rgba(131, 230, 204, 0.30)";
      ctx.lineWidth = inSel ? 1.4 : 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, height / 2 + low * height * 0.43);
      ctx.lineTo(x + 0.5, height / 2 + high * height * 0.43);
      ctx.stroke();
    }

    // Draggable handles at x1 and x2
    if (validSel) {
      [x1, x2].forEach((rawX) => {
        const sx = Math.max(2, Math.min(width - 2, rawX));
        ctx.strokeStyle = "#cbfffc";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, height); ctx.stroke();
        // Grip pill in middle
        ctx.fillStyle = "#cbfffc";
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") ctx.roundRect(sx - 4, height / 2 - 14, 8, 28, 4);
        else ctx.rect(sx - 4, height / 2 - 14, 8, 28);
        ctx.fill();
        ctx.fillStyle = "#011c1a";
        ctx.fillRect(sx - 1, height / 2 - 6, 2, 12);
      });
    }

    // Live Playhead line
    if (isNum(state.playheadTime) && state.duration > 0) {
      const px = Math.max(0, Math.min(width, (state.playheadTime / state.duration) * width));
      ctx.strokeStyle = "#f8c8ff";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, height); ctx.stroke();
    }
  }

  function pointerToTime(event) {
    const rect = el.waveform.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
    return { frac, x: event.clientX - rect.left, width: rect.width, time: frac * state.duration };
  }

  el.waveform.addEventListener("pointerdown", (e) => {
    if (!state.decoded || state.busy || state.loading) return;
    const { x, width, time } = pointerToTime(e);
    const { start, end } = segment();
    const x1 = (start / state.duration) * width;
    const x2 = (end / state.duration) * width;
    if (Math.abs(x - x1) <= 14) state.dragMode = "start";
    else if (Math.abs(x - x2) <= 14) state.dragMode = "end";
    else { state.dragMode = "new"; state.dragAnchor = time; }
    el.waveform.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  el.waveform.addEventListener("pointermove", (e) => {
    if (!state.decoded) return;
    const { x, width, time } = pointerToTime(e);
    // Update tooltip
    el.waveformTooltip.textContent = `${dec(time, 2)} s`;
    el.waveformTooltip.style.left = `${Math.max(28, Math.min(width - 28, x))}px`;
    hide(el.waveformTooltip, false);

    if (!state.dragMode) {
      const { start, end } = segment();
      const x1 = (start / state.duration) * width;
      const x2 = (end / state.duration) * width;
      el.waveform.style.cursor = (Math.abs(x - x1) <= 14 || Math.abs(x - x2) <= 14) ? "ew-resize" : "crosshair";
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
      if (end - start < 0.1) end = Math.min(state.duration, start + 0.1);
    }
    if (end - start > state.maxSegmentDuration) {
      if (state.dragMode === "start") start = end - state.maxSegmentDuration;
      else end = start + state.maxSegmentDuration;
    }
    el.startTime.value = String(Number(start.toFixed(3)));
    el.endTime.value = String(Number(end.toFixed(3)));
    invalidatePredictionForEdit();
  });

  el.waveform.addEventListener("pointerup", (e) => {
    if (state.dragMode) {
      state.dragMode = null;
      el.waveform.releasePointerCapture?.(e.pointerId);
    }
  });
  el.waveform.addEventListener("pointerleave", () => {
    if (!state.dragMode) hide(el.waveformTooltip, true);
  });

  /* ======================== WEBAUDIO STETHOSCOPE PLAYBACK ======================== */
  function stopStudioAudio() {
    if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = null; }
    if (state.sourceNode) {
      try { state.sourceNode.onended = null; state.sourceNode.stop(); state.sourceNode.disconnect(); } catch {}
      state.sourceNode = null;
    }
    state.isPlayingStudio = false;
    el.playSegmentIcon.textContent = "▶";
    el.playSegmentText.textContent = "Ouvir trecho";
  }

  function applyLiveStudioNodes() {
    if (!state.audioCtx) return;
    if (state.hpNode) state.hpNode.frequency.value = state.filterOn ? 80 : 10;
    if (state.lpNode) state.lpNode.frequency.value = state.filterOn ? 1600 : Math.min(20000, state.audioCtx.sampleRate / 2 - 100);
    if (state.gainNode) state.gainNode.gain.value = state.gain2x ? 2.0 : 1.0;
    if (state.sourceNode) {
      state.sourceNode.playbackRate.value = state.slowSpeed ? 0.75 : 1.0;
      state.sourceNode.loop = state.loop;
    }
  }

  async function toggleStudioPlay() {
    if (state.isPlayingStudio) {
      stopStudioAudio();
      state.playheadTime = null;
      drawWaveform();
      return;
    }
    if (segmentError() || !state.decoded) return;
    el.audioPlayer.pause();
    const { start, end } = segment();
    const safeStart = Math.max(0, Math.min(start, Math.max(0, state.decoded.duration - 0.01)));
    const segLen = Math.max(0.01, Math.min(end - safeStart, state.decoded.duration - safeStart));
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!state.audioCtx || state.audioCtx.state === "closed") state.audioCtx = new AC();
      if (state.audioCtx.state === "suspended") await state.audioCtx.resume();

      const src = state.audioCtx.createBufferSource();
      src.buffer = state.decoded;
      src.playbackRate.value = state.slowSpeed ? 0.75 : 1.0;
      src.loop = state.loop;
      src.loopStart = safeStart;
      src.loopEnd = safeStart + segLen;

      const hp = state.audioCtx.createBiquadFilter();
      hp.type = "highpass"; hp.Q.value = 0.707;
      const lp = state.audioCtx.createBiquadFilter();
      lp.type = "lowpass"; lp.Q.value = 0.707;
      const gain = state.audioCtx.createGain();

      src.connect(hp); hp.connect(lp); lp.connect(gain); gain.connect(state.audioCtx.destination);
      state.sourceNode = src; state.hpNode = hp; state.lpNode = lp; state.gainNode = gain;
      applyLiveStudioNodes();

      state.isPlayingStudio = true;
      state.studioPlayStartWall = state.audioCtx.currentTime;
      state.studioOffsetStart = safeStart;
      state.studioSegLen = segLen;
      el.playSegmentIcon.textContent = "■";
      el.playSegmentText.textContent = "Parar trecho";

      src.onended = () => {
        if (state.sourceNode === src) {
          stopStudioAudio();
          state.playheadTime = null;
          drawWaveform();
        }
      };

      if (state.loop) src.start(0, safeStart);
      else src.start(0, safeStart, segLen);

      const tick = () => {
        if (!state.isPlayingStudio || !state.audioCtx) return;
        const rate = state.slowSpeed ? 0.75 : 1.0;
        const elapsed = (state.audioCtx.currentTime - state.studioPlayStartWall) * rate;
        state.playheadTime = state.loop
          ? state.studioOffsetStart + (elapsed % Math.max(0.01, state.studioSegLen))
          : Math.min(end, state.studioOffsetStart + elapsed);
        drawWaveform();
        state.rafId = requestAnimationFrame(tick);
      };
      state.rafId = requestAnimationFrame(tick);
    } catch {
      showMessage("O navegador bloqueou a reprodução via Web Audio. Utilize o reprodutor nativo abaixo.");
    }
  }

  /* ======================== AUTO-SUGGEST BREATH CYCLES ======================== */
  function suggestCycles() {
    if (!state.decoded || state.duration < 3.5) return;
    const samples = state.decoded.getChannelData(0);
    const sr = state.decoded.sampleRate;
    const hop = Math.max(1, Math.round(sr * 0.05)); // 50 ms
    const env = [];
    for (let i = 0; i < samples.length; i += hop) {
      let sum = 0;
      const end = Math.min(samples.length, i + hop);
      for (let j = i; j < end; j++) sum += samples[j] * samples[j];
      env.push(Math.sqrt(sum / Math.max(1, end - i)));
    }
    // Smooth envelope (7-window moving average = 350 ms)
    const smooth = env.map((_, idx) => {
      let s = 0, c = 0;
      for (let k = Math.max(0, idx - 3); k <= Math.min(env.length - 1, idx + 3); k++) { s += env[k]; c++; }
      return s / c;
    });
    // Find local valleys spaced >= 1.4s (28 frames) apart
    const boundaries = [0];
    let lastIdx = 0;
    for (let i = 6; i < smooth.length - 6; i++) {
      if (i - lastIdx < 28) continue;
      let isMin = true;
      for (let k = i - 5; k <= i + 5; k++) { if (smooth[k] < smooth[i]) { isMin = false; break; } }
      if (isMin) {
        boundaries.push((i * hop) / sr);
        lastIdx = i;
      }
    }
    boundaries.push(state.duration);
    const cycles = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const a = boundaries[i], b = boundaries[i + 1];
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
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "cycle-chip";
        chip.textContent = `Ciclo ${idx + 1}: ${dec(a, 2)}–${dec(b, 2)} s`;
        chip.addEventListener("click", () => {
          el.startTime.value = String(Number(a.toFixed(3)));
          el.endTime.value = String(Number(b.toFixed(3)));
          invalidatePredictionForEdit();
        });
        el.suggestedCycles.append(chip);
      });
    }
    hide(el.suggestedCycles, false);
  }

  /* ======================== RENDER RESULT & HISTORY ======================== */
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
    el.lightboxTitle.textContent = useContrast
      ? "Mapa de contraste acústico (CMN invariante a estetoscópio)"
      : "Espectrograma log-Mel filtrado (80–1.200 Hz)";
    el.lightboxImage.src = src;
    el.lightboxImage.alt = srcImg.alt || el.lightboxTitle.textContent;
    el.lightboxCaption.textContent = useContrast
      ? "Realce após subtração da resposta estacionária do microfone: cristas horizontais em ciano indicam sibilos; transientes verticais em âmbar indicam estertores."
      : "Espectrograma log-Mel na banda respiratória: faixas horizontais sustentadas sugerem sibilos e impulsos verticais curtos sugerem estertores.";
    if (typeof el.imageLightbox.showModal === "function") {
      if (!el.imageLightbox.open) el.imageLightbox.showModal();
    } else {
      el.imageLightbox.setAttribute("open", "");
    }
  }

  function closeEvidenceLightbox() {
    if (!el.imageLightbox) return;
    if (typeof el.imageLightbox.close === "function" && el.imageLightbox.open) el.imageLightbox.close();
    else el.imageLightbox.removeAttribute("open");
  }

  function renderResult(result, contextInfo = null) {
    const pred = result.prediction;
    if (!pred || typeof pred.label !== "string" || !Array.isArray(pred.scores) || !pred.scores.length) {
      throw new Error("O servidor retornou uma classificação incompleta.");
    }
    const scores = pred.scores.filter((item) => typeof item.label === "string" && isNum(item.score) && item.score >= 0 && item.score <= 1);
    if (scores.length !== pred.scores.length) throw new Error("Pontuações inválidas recebidas do servidor.");

    const clsIdx = isNum(pred.index) ? pred.index : Math.max(0, LABELS.indexOf(pred.label));
    el.predictionCard.className = `prediction-card cls-${clsIdx}`;
    el.predictionLabel.textContent = pred.label;

    const sorted = [...scores].sort((a, b) => b.score - a.score);
    const top1 = sorted[0]?.score ?? 0;
    const top2 = sorted[1]?.score ?? 0;
    el.confidenceBadge.textContent = `Escore principal: ${pct(top1, 1)}`;
    el.marginBadge.textContent = `Margem +${decAuto((top1 - top2) * 100, 1)} p.p.`;

    const dur = result.segment?.duration_s;
    const modelName = result.model?.name || "Modelo local";
    const modelVer = result.model?.version ? `v${result.model.version}` : "";
    el.predictionMeta.textContent = `${isNum(dur) ? `Trecho de ${dec(dur, 2)} s · ` : ""}${modelName} ${modelVer}`.trim();

    // Reference comparison if example was analyzed unmodified
    const refLabel = contextInfo?.referenceLabel ?? (state.activeExample ? (typeof state.activeExample.label === "number" ? LABELS[state.activeExample.label] : state.activeExample.label) : null);
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

    // Independent event probabilities (Crackle = Estertores + Ambos; Wheeze = Sibilos + Ambos)
    const byLbl = Object.fromEntries(scores.map((s) => [s.label, s.score]));
    const pCrackle = isNum(pred.events?.estertores) ? pred.events.estertores : Math.min(1, (byLbl["Estertores"] || 0) + (byLbl["Ambos"] || 0));
    const pWheeze = isNum(pred.events?.sibilos) ? pred.events.sibilos : Math.min(1, (byLbl["Sibilos"] || 0) + (byLbl["Ambos"] || 0));
    el.crackleProb.textContent = pct(pCrackle, 1);
    el.crackleBar.style.width = `${Math.max(1, pCrackle * 100)}%`;
    el.wheezeProb.textContent = pct(pWheeze, 1);
    el.wheezeBar.style.width = `${Math.max(1, pWheeze * 100)}%`;

    // 4-class cards
    el.scoreList.replaceChildren();
    scores.forEach((item) => {
      const row = document.createElement("div");
      row.className = `score-row${item.label === pred.label ? " winner" : ""}`;
      const text = document.createElement("div"); text.className = "score-label";
      const lbl = document.createElement("span"); lbl.textContent = item.label;
      const val = document.createElement("span"); val.textContent = pct(item.score, 1);
      text.append(lbl, val);
      const track = document.createElement("div"); track.className = "score-track";
      const fill = document.createElement("div"); fill.className = "score-fill";
      fill.style.width = `${item.score * 100}%`;
      track.append(fill);
      row.append(text, track);
      el.scoreList.append(row);
    });

    el.acousticNote.textContent = ACOUSTIC_NOTES[pred.label] || "Pontuações experimentais produzidas por modelo acadêmico local.";

    // Spectrogram PNG
    const specImg = result.spectrogram_png;
    if (typeof specImg === "string" && specImg.startsWith("data:image/png;base64,")) {
      el.spectrogramImage.src = specImg;
      hide(el.spectrogramImage, false);
      hide(el.spectrogramUnavailable, true);
    } else {
      el.spectrogramImage.removeAttribute("src");
      hide(el.spectrogramImage, true);
      hide(el.spectrogramUnavailable, false);
    }

    // Acoustic Contrast PNG (optional v2.0 backend visual)
    const contrastImg = result.contrast_png;
    if (typeof contrastImg === "string" && contrastImg.startsWith("data:image/png;base64,")) {
      el.contrastImage.src = contrastImg;
      hide(el.contrastImage, false);
      hide(el.contrastUnavailable, true);
      hide(el.contrastTabBtn, false);
    } else {
      el.contrastImage.removeAttribute("src");
      hide(el.contrastImage, true);
      hide(el.contrastUnavailable, false);
      hide(el.contrastTabBtn, true);
    }

    // Tech telemetry
    el.techTelemetry.replaceChildren();
    const techItems = [
      ["Taxa original", result.audio?.sample_rate ? `${new Intl.NumberFormat("pt-BR").format(result.audio.sample_rate)} Hz` : "—"],
      ["Canais", result.audio?.channels === 2 ? "Estéreo → Mono" : "Mono"],
      ["Trecho", isNum(result.segment?.start) && isNum(result.segment?.end) ? `${dec(result.segment.start, 2)}–${dec(result.segment.end, 2)} s` : "—"],
    ];
    techItems.forEach(([k, v]) => {
      const box = document.createElement("div"); box.className = "tech-stat";
      const sp = document.createElement("span"); sp.textContent = k;
      const st = document.createElement("strong"); st.textContent = v;
      box.append(sp, st);
      el.techTelemetry.append(box);
    });

    const warnings = Array.isArray(result.quality?.warnings) ? result.quality.warnings.filter((x) => typeof x === "string") : [];
    el.qualityMessage.replaceChildren();
    if (warnings.length) {
      const ul = document.createElement("ul");
      warnings.forEach((w) => { const li = document.createElement("li"); li.textContent = w; ul.append(li); });
      el.qualityMessage.append(ul);
    } else {
      el.qualityMessage.textContent = "Nenhum aviso técnico foi sinalizado para este recorte.";
    }
    hide(el.qualityMessage, false);

    setEvidenceTab("spec");
    state.result = result;
    hide(el.resultEmpty, true);
    hide(el.analysisProgress, true);
    hide(el.resultContent, false);
    el.resultTag.textContent = "Análise concluída";
  }

  function pushHistory(result) {
    const entry = {
      id: Date.now(),
      name: state.activeExample?.name || state.file?.name || "Áudio WAV",
      referenceLabel: state.activeExample ? (typeof state.activeExample.label === "number" ? LABELS[state.activeExample.label] : state.activeExample.label) : null,
      result,
    };
    state.history.unshift(entry);
    if (state.history.length > 8) state.history.pop();
    renderHistory();
  }

  function renderHistory() {
    el.sessionHistory.replaceChildren();
    hide(el.sessionHistoryWrap, state.history.length === 0);
    state.history.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-item";
      const nm = document.createElement("span"); nm.className = "hist-name";
      nm.textContent = `${item.name} (${dec(item.result.segment?.duration_s || 0, 2)} s)`;
      const pr = document.createElement("span"); pr.className = "hist-pred";
      const topScore = Math.max(...(item.result.prediction?.scores || []).map((s) => s.score), 0);
      pr.textContent = `${item.result.prediction?.label} · ${pct(topScore, 0)}`;
      btn.append(nm, pr);
      btn.addEventListener("click", () => renderResult(item.result, { referenceLabel: item.referenceLabel }));
      el.sessionHistory.append(btn);
    });
  }

  async function analyze() {
    if (el.analyzeButton.disabled) return;
    const err = segmentError();
    if (err) { showMessage(err); return; }
    const { start, end } = segment(), revision = state.revision;
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
    el.resultTag.textContent = "Processando…";
    updateControls();

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form, signal: state.controller.signal });
      let payload;
      try { payload = await res.json(); } catch { throw new Error("O servidor não retornou uma resposta JSON válida."); }
      if (!res.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "A gravação não pôde ser analisada.");
      if (revision === state.revision) {
        renderResult(payload);
        pushHistory(payload);
        if (window.matchMedia("(max-width: 960px)").matches) {
          el.resultTag.scrollIntoView({ block: "nearest" });
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

  /* ======================== EVENT LISTENERS ======================== */
  el.uploadButton.addEventListener("click", () => el.audioFile.click());
  el.audioFile.addEventListener("change", () => { const f = el.audioFile.files?.[0]; if (f) loadFile(f); });
  ["dragenter", "dragover"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); if (!state.busy) el.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); }));
  el.dropzone.addEventListener("drop", (e) => {
    if (state.busy || state.loading) return;
    if (e.dataTransfer.files.length !== 1) { showMessage("Selecione uma única gravação por análise."); return; }
    loadFile(e.dataTransfer.files[0]);
  });
  el.clearFile.addEventListener("click", clearFile);

  [el.startTime, el.endTime].forEach((inp) => inp.addEventListener("input", invalidatePredictionForEdit));
  document.querySelectorAll(".nudge-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.decoded || state.busy || state.loading) return;
      const target = byId(btn.dataset.target);
      const delta = Number(btn.dataset.delta);
      const next = Math.max(0, Math.min(state.duration, (Number(target.value) || 0) + delta));
      target.value = String(Number(next.toFixed(3)));
      invalidatePredictionForEdit();
    });
  });
  el.resetFullCycle.addEventListener("click", () => {
    if (!state.decoded || state.busy) return;
    el.startTime.value = "0";
    el.endTime.value = String(Number(Math.min(state.duration, state.maxSegmentDuration).toFixed(3)));
    invalidatePredictionForEdit();
  });

  el.cycleConfirmed.addEventListener("change", updateControls);
  el.playSegment.addEventListener("click", toggleStudioPlay);
  el.loopToggle.addEventListener("click", () => {
    state.loop = !state.loop;
    el.loopToggle.setAttribute("aria-pressed", String(state.loop));
    el.loopToggle.classList.toggle("active", state.loop);
    applyLiveStudioNodes();
  });
  el.filterToggle.addEventListener("click", () => {
    state.filterOn = !state.filterOn;
    el.filterToggle.setAttribute("aria-pressed", String(state.filterOn));
    el.filterToggle.classList.toggle("active", state.filterOn);
    applyLiveStudioNodes();
  });
  el.gainToggle.addEventListener("click", () => {
    state.gain2x = !state.gain2x;
    el.gainToggle.setAttribute("aria-pressed", String(state.gain2x));
    el.gainToggle.classList.toggle("active", state.gain2x);
    applyLiveStudioNodes();
  });
  el.speedToggle.addEventListener("click", () => {
    if (state.isPlayingStudio && state.audioCtx && isNum(state.playheadTime)) {
      state.studioOffsetStart = state.playheadTime;
      state.studioPlayStartWall = state.audioCtx.currentTime;
    }
    state.slowSpeed = !state.slowSpeed;
    el.speedToggle.setAttribute("aria-pressed", String(state.slowSpeed));
    el.speedToggle.classList.toggle("active", state.slowSpeed);
    applyLiveStudioNodes();
  });
  el.detectCyclesBtn.addEventListener("click", suggestCycles);

  // Sync native audio player playhead
  el.audioPlayer.addEventListener("play", () => { stopStudioAudio(); });
  el.audioPlayer.addEventListener("timeupdate", () => {
    if (!state.isPlayingStudio && !el.audioPlayer.paused) {
      state.playheadTime = el.audioPlayer.currentTime;
      drawWaveform();
    }
  });
  el.audioPlayer.addEventListener("pause", () => {
    if (!state.isPlayingStudio) { state.playheadTime = null; drawWaveform(); }
  });

  el.loadExample.addEventListener("click", () => loadExample());
  el.analyzeButton.addEventListener("click", analyze);

  document.querySelectorAll(".tab-btn").forEach((b) => b.addEventListener("click", () => setEvidenceTab(b.dataset.tab)));
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
    link.download = state.lightboxKind === "contrast" ? "ausculta-contraste-cmn.png" : "ausculta-espectrograma.png";
    document.body.append(link);
    link.click();
    link.remove();
  });

  el.downloadResult.addEventListener("click", () => {
    if (!state.result) return;
    const report = { exported_at: new Date().toISOString(), filename: state.file?.name, result: state.result };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "ausculta-resultado.json";
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  el.copySummary.addEventListener("click", async () => {
    if (!state.result) return;
    const pred = state.result.prediction;
    const lines = [
      `Ausculta Lab — Resumo da Análise`,
      `Arquivo: ${state.activeExample?.name || state.file?.name || "WAV"}`,
      `Trecho: ${dec(state.result.segment?.start || 0, 2)} s a ${dec(state.result.segment?.end || 0, 2)} s (${dec(state.result.segment?.duration_s || 0, 2)} s)`,
      `Categoria prevista: ${pred.label}`,
      `Pontuações: ` + pred.scores.map((s) => `${s.label}: ${pct(s.score, 1)}`).join(" | "),
      `Aviso: protótipo acadêmico experimental, sem validação clínica.`,
    ];
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      el.copySummary.textContent = "✓ Copiado!";
      setTimeout(() => { el.copySummary.textContent = "Copiar resumo"; }, 1800);
    } catch {
      showMessage("Não foi possível copiar para a área de transferência.");
    }
  });

  el.clearHistory.addEventListener("click", () => { state.history = []; renderHistory(); });

  el.metricSwitcher.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedMetric = btn.dataset.metric;
      el.metricSwitcher.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      renderModelBars();
    });
  });

  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
      link.classList.add("active");
    });
  });

  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
    if (e.code === "Space" && state.file && !segmentError()) {
      e.preventDefault();
      toggleStudioPlay();
    } else if (e.code === "Enter" && !el.analyzeButton.disabled) {
      e.preventDefault();
      analyze();
    }
  });

  window.addEventListener("resize", drawWaveform);
  window.addEventListener("beforeunload", () => {
    stopStudioAudio();
    state.controller?.abort();
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  });

  fetchStatus();
  fetchExamples();
})();
