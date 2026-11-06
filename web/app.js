"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const elements = Object.fromEntries([
    "modelState", "modelStateText", "globalMessage", "dropzone", "uploadButton", "uploadButtonText", "audioFile",
    "uploadLimits", "exampleSelect", "loadExample", "audioWorkspace", "fileName", "fileMeta", "clearFile",
    "waveform", "waveformEnd", "audioPlayer", "playSegment", "startTime", "endTime", "segmentDuration",
    "segmentError", "cycleConfirmed", "referenceNote", "analyzeButton", "analyzeButtonText", "analyzeHelp",
    "resultTag", "resultEmpty", "analysisProgress", "resultContent", "predictionLabel", "predictionMeta",
    "scoreList", "spectrogramImage", "spectrogramUnavailable", "qualityMessage", "downloadResult", "resultDetails"
  ].map((id) => [id, byId(id)]));

  const state = { status: null, file: null, decoded: null, audioUrl: null, duration: 0, busy: false, loading: false,
    revision: 0, controller: null, examples: [], result: null, playEnd: null, maxUploadMb: 20, maxDuration: 120, maxSegmentDuration: 20 };
  const labels = ["Sem sibilos/estertores", "Estertores", "Sibilos", "Ambos"];
  const voiceFormatMessage = "Arquivo OGG ou Opus. Para analisar, escolha uma gravação de ausculta em WAV.";
  const hide = (element, yes = true) => element.classList.toggle("hidden", yes);
  const number = (value) => typeof value === "number" && Number.isFinite(value);
  const decimal = (value, digits = 2) => new Intl.NumberFormat("pt-BR", { maximumFractionDigits: digits }).format(value);
  const percent = (value) => `${decimal(value * 100, 1)}%`;

  function pcmWavInfo(buffer) {
    // WebAudio reamostra para a taxa do dispositivo e pode arredondar a duração.
    // Em WAV PCM/float, preservar os limites no tempo original do arquivo.
    if (buffer.byteLength < 44) return null;
    const view = new DataView(buffer);
    const text = (offset, length) => String.fromCharCode(...new Uint8Array(buffer, offset, length));
    if (text(0, 4) !== "RIFF" || text(8, 4) !== "WAVE") return null;
    let format = null, dataBytes = 0;
    for (let offset = 12; offset + 8 <= buffer.byteLength;) {
      const id = text(offset, 4), length = view.getUint32(offset + 4, true), body = offset + 8;
      if (body + length > buffer.byteLength) return null;
      if (id === "fmt " && length >= 16) format = { code: view.getUint16(body, true), channels: view.getUint16(body + 2, true), rate: view.getUint32(body + 4, true), alignment: view.getUint16(body + 12, true) };
      if (id === "data") dataBytes += length;
      offset = body + length + length % 2;
    }
    if (!format || ![1, 3].includes(format.code) || !format.rate || !format.alignment || !dataBytes || dataBytes % format.alignment) return null;
    return { duration: dataBytes / format.alignment / format.rate, sampleRate: format.rate, channels: format.channels };
  }

  function showMessage(message) {
    elements.globalMessage.textContent = message;
    hide(elements.globalMessage, !message);
    if (message) elements.globalMessage.scrollIntoView({ block: "nearest" });
  }
  async function fetchStatus() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error("Servidor local indisponível");
      state.status = await response.json();
      const limits = state.status.limits || {};
      if (number(limits.max_upload_mb)) state.maxUploadMb = limits.max_upload_mb;
      if (number(limits.max_duration_s)) state.maxDuration = limits.max_duration_s;
      if (number(limits.max_segment_s)) state.maxSegmentDuration = limits.max_segment_s;
      elements.uploadLimits.textContent = "WAV · até " + decimal(state.maxUploadMb, 0) + " MB";
      elements.modelStateText.textContent = state.status.model_ready ? "Pronto para análise" : "Modelo indisponível";
      elements.modelState.className = "status-pill " + (state.status.model_ready ? "ready" : "unavailable");
    } catch {
      state.status = null;
      elements.modelState.className = "status-pill unavailable";
      elements.modelStateText.textContent = "Sem conexão";
      showMessage("Não foi possível acessar o servidor local. Abra a aplicação e recarregue esta página.");
    }
    updateControls();
  }
  async function fetchExamples() {
    try {
      const response = await fetch("/api/examples", { cache: "no-store" });
      if (!response.ok) throw new Error("Exemplos indisponíveis");
      const payload = await response.json();
      state.examples = Array.isArray(payload) ? payload : (Array.isArray(payload.examples) ? payload.examples : []);
      elements.exampleSelect.replaceChildren();
      if (!state.examples.length) throw new Error("Nenhum exemplo registrado");
      const groups = new Map();
      state.examples.forEach((example, index) => {
        const category = example.label || "Ausculta";
        if (!groups.has(category)) {
          const group = document.createElement("optgroup");
          group.label = category;
          groups.set(category, group);
          elements.exampleSelect.append(group);
        }
        const group = groups.get(category);
        const option = new Option(`${category} · ${group.children.length + 1}`, String(index));
        group.append(option);
      });
      updateControls();
    } catch (error) {
      state.examples = []; elements.exampleSelect.replaceChildren(new Option("Sem exemplos disponíveis", ""));
      elements.exampleSelect.disabled = true; elements.loadExample.disabled = true;
    }
  }
  function segment() {
    const end = Number(elements.endTime.value);
    return {
      start: Number(elements.startTime.value),
      end: end === Number(state.duration.toFixed(3)) ? state.duration : end,
    };
  }
  function segmentError() {
    if (!state.file || !state.decoded) return "Selecione uma gravação válida.";
    if (elements.startTime.value === "" || elements.endTime.value === "") return "Informe o início e o fim do ciclo.";
    const { start, end } = segment();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return "Use valores numéricos para os limites do ciclo.";
    if (start < 0) return "O início precisa ser igual ou maior que zero.";
    if (end <= start) return "O fim precisa ser posterior ao início.";
    if (end > state.duration + 1e-6) return "O fim do trecho ultrapassa a duração da gravação.";
    if (end - start < 0.1 - 1e-9) return "Selecione um trecho de pelo menos 0,1 segundo.";
    if (end - start > state.maxSegmentDuration + 1e-9) return `O ciclo selecionado pode ter no máximo ${decimal(state.maxSegmentDuration, 0)} segundos.`;
    return "";
  }
  function updateControls() {
    const error = state.file ? segmentError() : "";
    elements.segmentError.textContent = error; hide(elements.segmentError, !error);
    const { start, end } = segment();
    elements.segmentDuration.textContent = !error && state.file ? `${decimal(end - start)} s` : "—";
    const ready = state.status?.model_ready === true;
    elements.analyzeButton.disabled = !ready || !state.file || !!error || !elements.cycleConfirmed.checked || state.busy || state.loading;
    elements.playSegment.disabled = !!error || state.busy || !state.file;
    elements.startTime.disabled = state.busy || state.loading;
    elements.endTime.disabled = state.busy || state.loading;
    elements.cycleConfirmed.disabled = state.busy || state.loading;
    elements.uploadButton.disabled = state.busy || state.loading;
    elements.dropzone.classList.toggle("has-audio", Boolean(state.file));
    elements.uploadButtonText.textContent = state.file ? "Trocar arquivo" : "Selecionar arquivo";
    elements.audioFile.disabled = state.busy || state.loading;
    elements.clearFile.disabled = state.busy || state.loading;
    elements.loadExample.disabled = !state.examples.length || state.busy || state.loading;
    elements.exampleSelect.disabled = !state.examples.length || state.busy || state.loading;
    elements.analyzeButtonText.textContent = state.busy ? "Analisando o áudio…" : "Analisar ciclo respiratório";
    elements.analyzeHelp.textContent = state.busy ? "Processando o trecho…" :
      !ready ? "Aguardando o modelo local." :
      !elements.cycleConfirmed.checked ? "Confirme a seleção para habilitar a análise." : "";
    drawWaveform();
  }
  function resetResult() {
    state.result = null;
    hide(elements.resultContent); hide(elements.analysisProgress); hide(elements.resultEmpty, false);
    elements.resultDetails.open = false;
    elements.resultTag.textContent = state.file ? "Áudio carregado" : "Aguardando áudio";
  }
  function clearFile() {
    state.revision += 1; state.controller?.abort();
    state.file = null; state.decoded = null; state.duration = 0; state.playEnd = null; state.loading = false;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = null; elements.audioPlayer.pause(); elements.audioPlayer.removeAttribute("src"); elements.audioPlayer.load();
    elements.audioFile.value = ""; elements.startTime.value = "0"; elements.endTime.value = "";
    elements.cycleConfirmed.checked = false; hide(elements.audioWorkspace); hide(elements.referenceNote);
    resetResult(); updateControls();
  }
  async function loadFile(file, example = null) {
    if (state.busy) return;
    clearFile(); showMessage("");
    if (!file || !/\.wav$/i.test(file.name)) {
      showMessage(/\.(ogg|oga|opus)$/i.test(file?.name || "") ? voiceFormatMessage : "Escolha uma gravação de ausculta em formato WAV.");
      return;
    }
    if (!file.size) { showMessage("O arquivo está vazio. Selecione outra gravação."); return; }
    if (file.size > state.maxUploadMb * 1024 * 1024) { showMessage(`O arquivo ultrapassa o limite de ${decimal(state.maxUploadMb, 0)} MB.`); return; }
    const revision = state.revision;
    state.loading = true; updateControls();
    let audioContext;
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) throw new Error("Este navegador não permite preparar o áudio. Utilize um navegador atualizado.");
      const sourceBuffer = await file.arrayBuffer(), original = pcmWavInfo(sourceBuffer);
      const header = String.fromCharCode(...new Uint8Array(sourceBuffer, 0, Math.min(sourceBuffer.byteLength, 12)));
      if (header.startsWith("OggS")) throw new Error(voiceFormatMessage);
      if (!/^(RIFF|RIFX|RF64)/.test(header) || header.slice(8, 12) !== "WAVE") {
        throw new Error("O conteúdo do arquivo não é WAV. Escolha uma gravação de ausculta em WAV.");
      }
      audioContext = new AudioContext();
      const decoded = await audioContext.decodeAudioData(sourceBuffer);
      if (revision !== state.revision) return;
      if (!Number.isFinite(decoded.duration) || decoded.duration <= 0) throw new Error("A gravação não tem duração válida.");
      if (decoded.numberOfChannels < 1 || decoded.numberOfChannels > 2) throw new Error("A demonstração aceita áudios mono ou estéreo.");
      const sourceDuration = original && number(original.duration) ? original.duration : decoded.duration;
      if (sourceDuration > state.maxDuration) throw new Error(`A gravação ultrapassa o limite de ${decimal(state.maxDuration, 0)} segundos.`);
      state.file = file; state.decoded = decoded; state.duration = sourceDuration;
      state.audioUrl = URL.createObjectURL(file); elements.audioPlayer.src = state.audioUrl;
      elements.fileName.textContent = example?.name || file.name;
      elements.fileMeta.textContent = `${decimal(state.duration)} s · WAV`;
      elements.waveformEnd.textContent = `${decimal(state.duration)} s`;
      elements.startTime.max = String(state.duration); elements.endTime.max = state.duration.toFixed(3);
      const proposedStart = example && number(example.start) ? example.start : 0;
      const proposedEnd = example && number(example.end) ? example.end : state.duration;
      elements.startTime.value = String(Math.max(0, proposedStart));
      elements.endTime.value = String(Number(Math.min(state.duration, proposedEnd).toFixed(3)));
      elements.cycleConfirmed.checked = !!example;
      if (example) {
        const reference = typeof example.label === "number" ? labels[example.label] : example.label;
        elements.referenceNote.textContent = `Referência do exemplo: ${reference || "não informada"}.`;
        hide(elements.referenceNote, false);
      }
      hide(elements.audioWorkspace, false);
      resetResult();
      elements.audioWorkspace.scrollIntoView({ block: "nearest" });
    } catch (error) {
      if (revision === state.revision) {
        clearFile();
        showMessage(error instanceof DOMException ? "Não foi possível abrir esse WAV. Confira se o arquivo está íntegro." : error.message);
      }
    } finally {
      if (audioContext && audioContext.state !== "closed") await audioContext.close().catch(() => {});
      if (revision === state.revision) { state.loading = false; updateControls(); }
    }
  }
  function drawWaveform() {
    const canvas = elements.waveform;
    if (!state.decoded || canvas.clientWidth < 1) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2), width = Math.round(canvas.clientWidth), height = 125;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
    const samples = state.decoded.getChannelData(0), { start, end } = segment();
    const x1 = Math.max(0, Math.min(width, start / state.duration * width));
    const x2 = Math.max(0, Math.min(width, end / state.duration * width));
    if (Number.isFinite(x1) && Number.isFinite(x2) && x2 > x1) { ctx.fillStyle = "rgba(131,230,204,.12)"; ctx.fillRect(x1, 0, x2 - x1, height); }
    ctx.strokeStyle = "#83e6cc"; ctx.lineWidth = 1;
    const stride = Math.max(1, Math.floor(samples.length / width));
    for (let x = 0; x < width; x += 2) {
      const offset = Math.floor(x / width * samples.length); let low = 1, high = -1;
      for (let i = offset; i < Math.min(samples.length, offset + stride * 2); i++) { low = Math.min(low, samples[i]); high = Math.max(high, samples[i]); }
      ctx.beginPath(); ctx.moveTo(x + .5, height / 2 + low * height * .43); ctx.lineTo(x + .5, height / 2 + high * height * .43); ctx.stroke();
    }
    if (Number.isFinite(x1) && Number.isFinite(x2) && x2 > x1) {
      ctx.strokeStyle = "#cbfffc"; ctx.lineWidth = 1.5;
      [x1, x2].forEach((x) => { const safeX = Math.max(1, Math.min(width - 1, x)); ctx.beginPath(); ctx.moveTo(safeX, 0); ctx.lineTo(safeX, height); ctx.stroke(); });
    }
  }
  async function loadExample() {
    const example = state.examples[Number(elements.exampleSelect.value)];
    if (!example || state.busy || state.loading) return;
    state.loading = true; updateControls(); elements.loadExample.textContent = "Abrindo…";
    try {
      const response = await fetch(`/api/examples/${encodeURIComponent(String(example.id))}/audio`);
      if (!response.ok) throw new Error("Não foi possível carregar o exemplo de teste.");
      const blob = await response.blob();
      const name = String(example.name || `exemplo-${example.id}.wav`).replace(/[\\/]/g, "_");
      await loadFile(new File([blob], /\.wav$/i.test(name) ? name : `${name}.wav`, { type: "audio/wav" }), example);
    } catch (error) { showMessage(error.message); }
    finally { state.loading = false; elements.loadExample.textContent = "Carregar"; updateControls(); }
  }
  function renderResult(result) {
    const prediction = result.prediction;
    if (!prediction || typeof prediction.label !== "string" || !Array.isArray(prediction.scores) || !prediction.scores.length) throw new Error("O servidor retornou uma classificação incompleta.");
    const scores = prediction.scores.filter((item) => typeof item.label === "string" && number(item.score) && item.score >= 0 && item.score <= 1);
    if (scores.length !== prediction.scores.length) throw new Error("O servidor retornou pontuações inválidas.");
    elements.predictionLabel.textContent = prediction.label;
    const duration = result.segment?.duration_s;
    elements.predictionMeta.textContent = number(duration) ? `Trecho de ${decimal(duration)} s` : "";
    elements.scoreList.replaceChildren();
    scores.forEach((item) => {
      const row = document.createElement("div"); row.className = `score-row${item.label === prediction.label ? " winner" : ""}`;
      const text = document.createElement("div"); text.className = "score-label";
      const label = document.createElement("span"), value = document.createElement("span"); label.textContent = item.label; value.textContent = percent(item.score);
      text.append(label, value);
      const track = document.createElement("div"), fill = document.createElement("div"); track.className = "score-track"; fill.className = "score-fill"; fill.style.width = `${item.score * 100}%`;
      track.append(fill); row.append(text, track); elements.scoreList.append(row);
    });
    const image = result.spectrogram_png;
    if (typeof image === "string" && image.startsWith("data:image/png;base64,")) { elements.spectrogramImage.src = image; hide(elements.spectrogramImage, false); hide(elements.spectrogramUnavailable); }
    else { elements.spectrogramImage.removeAttribute("src"); hide(elements.spectrogramImage); hide(elements.spectrogramUnavailable, false); }
    const warnings = Array.isArray(result.quality?.warnings) ? result.quality.warnings.filter((x) => typeof x === "string") : [];
    elements.qualityMessage.replaceChildren(); elements.qualityMessage.classList.toggle("good", !warnings.length);
    if (warnings.length) {
      const ul = document.createElement("ul"); warnings.forEach((warning) => { const li = document.createElement("li"); li.textContent = warning; ul.append(li); }); elements.qualityMessage.append(ul);
    } else { elements.qualityMessage.textContent = "Nenhum aviso técnico foi informado para este processamento. Isso não substitui uma avaliação clínica."; }
    hide(elements.qualityMessage, false);
    state.result = result; hide(elements.resultEmpty); hide(elements.analysisProgress); hide(elements.resultContent, false);
    elements.resultTag.textContent = "Concluída";
  }
  async function analyze() {
    if (elements.analyzeButton.disabled) return;
    const error = segmentError(); if (error) { showMessage(error); return; }
    const { start, end } = segment(), revision = state.revision;
    const form = new FormData(); form.append("file", state.file); form.append("start", String(start)); form.append("end", String(end));
    state.busy = true; state.controller = new AbortController(); showMessage(""); state.result = null;
    hide(elements.resultEmpty); hide(elements.resultContent); hide(elements.analysisProgress, false); elements.resultTag.textContent = "Processando"; updateControls();
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form, signal: state.controller.signal });
      let payload;
      try { payload = await response.json(); } catch { throw new Error("O servidor não retornou uma resposta válida."); }
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "A gravação não pôde ser analisada. Confira o arquivo e os limites do ciclo.");
      if (revision === state.revision) {
        renderResult(payload);
        if (window.matchMedia("(max-width: 800px)").matches) {
          elements.resultTag.scrollIntoView({ block: "nearest" });
        }
      }
    } catch (error) {
      if (error.name !== "AbortError" && revision === state.revision) { resetResult(); showMessage(error.message || "Falha na análise. Tente novamente."); }
    } finally { state.busy = false; state.controller = null; updateControls(); }
  }

  elements.uploadButton.addEventListener("click", () => elements.audioFile.click());
  elements.audioFile.addEventListener("change", () => { const file = elements.audioFile.files?.[0]; if (file) loadFile(file); });
  ["dragenter", "dragover"].forEach((event) => elements.dropzone.addEventListener(event, (e) => { e.preventDefault(); if (!state.busy) elements.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((event) => elements.dropzone.addEventListener(event, (e) => { e.preventDefault(); elements.dropzone.classList.remove("dragover"); }));
  elements.dropzone.addEventListener("drop", (event) => {
    if (state.busy || state.loading) return;
    if (event.dataTransfer.files.length !== 1) { showMessage("Selecione uma única gravação por análise."); return; }
    loadFile(event.dataTransfer.files[0]);
  });
  elements.clearFile.addEventListener("click", clearFile);
  [elements.startTime, elements.endTime].forEach((input) => input.addEventListener("input", () => {
    elements.cycleConfirmed.checked = false; hide(elements.referenceNote); state.playEnd = null; elements.audioPlayer.pause(); resetResult(); updateControls();
  }));
  elements.cycleConfirmed.addEventListener("change", updateControls);
  elements.playSegment.addEventListener("click", async () => {
    if (segmentError()) return;
    const { start, end } = segment(); elements.audioPlayer.currentTime = start; state.playEnd = end;
    try { await elements.audioPlayer.play(); } catch { showMessage("O navegador não conseguiu iniciar a reprodução. Use os controles do áudio."); }
  });
  elements.audioPlayer.addEventListener("timeupdate", () => { if (state.playEnd !== null && elements.audioPlayer.currentTime >= state.playEnd) { elements.audioPlayer.pause(); state.playEnd = null; } });
  elements.audioPlayer.addEventListener("ended", () => { state.playEnd = null; });
  elements.loadExample.addEventListener("click", loadExample);
  elements.analyzeButton.addEventListener("click", analyze);
  elements.downloadResult.addEventListener("click", () => {
    if (!state.result) return;
    const report = { exported_at: new Date().toISOString(), filename: state.file?.name, result: state.result };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "ausculta-resultado.json"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("resize", drawWaveform);
  window.addEventListener("beforeunload", () => { state.controller?.abort(); if (state.audioUrl) URL.revokeObjectURL(state.audioUrl); });
  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => { document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active")); link.classList.add("active"); }));
  fetchStatus(); fetchExamples();
})();
