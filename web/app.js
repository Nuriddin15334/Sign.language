import { CLASSES, NO_SIGN } from './classes.js';

// Pre-trained Slovo models (github.com/ai-forever/slovo). "frames" is the window
// length the model expects; "interval" is how many 30 fps camera frames apart
// the window's frames are sampled, matching how each model was trained.
const MODELS = {
  'mvit16-4': { frames: 16, interval: 4, url: 'models/mvit16-4.onnx', size: 140510654 },
  'mvit32-2': { frames: 32, interval: 2, url: 'models/mvit32-2.onnx', size: 140785735 },
};
const SOURCE_FPS = 30;
const SIZE = 224;
const HW = SIZE * SIZE;
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];
const PAD = 114;
const CACHE_NAME = 'slovo-models-v1';

const $ = (id) => document.getElementById(id);
const els = {
  status: $('status'), video: $('video'), videoWrap: $('videoWrap'), placeholder: $('placeholder'),
  overlay: $('overlay'), overlayWord: $('overlayWord'), overlayProb: $('overlayProb'), bufferFill: $('bufferFill'),
  btnCamera: $('btnCamera'), btnSample: $('btnSample'), lblFile: $('lblFile'), fileVideo: $('fileVideo'), btnStop: $('btnStop'),
  selModel: $('selModel'), rngThreshold: $('rngThreshold'), outThreshold: $('outThreshold'),
  rngStride: $('rngStride'), outStride: $('outStride'), chkMirror: $('chkMirror'), chkSpeak: $('chkSpeak'),
  btnLoad: $('btnLoad'), fileModel: $('fileModel'), progress: $('progress'), progressFill: $('progressFill'), progressText: $('progressText'),
  transcript: $('transcript'), btnSpeakAll: $('btnSpeakAll'), btnCopy: $('btnCopy'), btnClear: $('btnClear'),
  topk: $('topk'), statBackend: $('statBackend'), statInfer: $('statInfer'), statWindow: $('statWindow'), statCount: $('statCount'),
};

const state = {
  model: null,          // { frames, interval, name }
  loadedKey: null,
  source: null,         // 'camera' | 'file' | null
  stream: null,
  frames: [],           // ring of preprocessed CHW Float32Arrays
  newFrames: 0,
  lastSampleTime: null,
  busy: false,
  pausedForInference: false,
  lastGloss: NO_SIGN,
  words: [],
  predictions: 0,
  runId: 0,
};

// ---------- settings persistence (per-viewer convenience only) ----------
const SETTINGS_KEY = 'rsl-settings';
function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    if (s.model && MODELS[s.model]) els.selModel.value = s.model;
    if (s.threshold) els.rngThreshold.value = s.threshold;
    if (s.stride) els.rngStride.value = s.stride;
    if (typeof s.mirror === 'boolean') els.chkMirror.checked = s.mirror;
    if (typeof s.speak === 'boolean') els.chkSpeak.checked = s.speak;
  } catch { /* storage unavailable */ }
}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      model: els.selModel.value, threshold: els.rngThreshold.value,
      stride: els.rngStride.value, mirror: els.chkMirror.checked, speak: els.chkSpeak.checked,
    }));
  } catch { /* storage unavailable */ }
}

// ---------- cross-origin isolation (enables multi-threaded WASM) ----------
// serve.py sends COOP/COEP headers itself. Static hosts like GitHub Pages can't,
// so a service worker adds them and the page reloads once.
async function ensureCrossOriginIsolation() {
  if (self.crossOriginIsolated || !window.isSecureContext || !('serviceWorker' in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register('sw.js');
    let reloaded = false;
    try { reloaded = sessionStorage.getItem('coi-reloaded') === '1'; } catch { /* ignore */ }
    if (reloaded) return;
    const reload = () => {
      try { sessionStorage.setItem('coi-reloaded', '1'); } catch { /* ignore */ }
      location.reload();
    };
    if (reg.active && navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });
  } catch { /* fall back to single-threaded WASM */ }
}

// ---------- worker ----------
const worker = new Worker('worker.js');
const pending = new Map();
let msgId = 0;
worker.onmessage = (e) => {
  const msg = e.data;
  const key = msg.type === 'loaded' || (msg.type === 'error' && msg.stage === 'load') ? 'load' : msg.id;
  const p = pending.get(key);
  if (!p) return;
  pending.delete(key);
  if (msg.type === 'error') p.reject(new Error(msg.message));
  else p.resolve(msg);
};
worker.onerror = (e) => setStatus('error', `Worker failed: ${e.message || 'could not start ONNX Runtime'}`);
function callWorker(msg, transfer = []) {
  const key = msg.type === 'load' ? 'load' : (msg.id = ++msgId);
  return new Promise((resolve, reject) => {
    pending.set(key, { resolve, reject });
    worker.postMessage(msg, transfer);
  });
}

// ---------- UI helpers ----------
function setStatus(kind, text) {
  els.status.dataset.state = kind;
  els.status.textContent = text;
}
function setProgress(fraction, text) {
  els.progress.hidden = fraction == null;
  if (fraction == null) return;
  els.progressFill.style.width = `${Math.round(fraction * 100)}%`;
  els.progressText.textContent = text;
}
function setSourceButtons(enabled) {
  els.btnCamera.disabled = !enabled;
  els.btnSample.disabled = !enabled;
  els.fileVideo.disabled = !enabled;
  els.lblFile.setAttribute('aria-disabled', String(!enabled));
}
const mb = (n) => `${(n / 1048576).toFixed(0)} MB`;

// ---------- model loading ----------
async function fetchWithProgress(url, expectedSize, label) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      setProgress(1, `${label}: loaded from browser cache`);
      return await hit.arrayBuffer();
    }
  } catch { cache = null; }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(res.status === 404
      ? `Model file ${url} not found on this server. Run "python serve.py" (it downloads the model), or use "Load .onnx from disk".`
      : `Could not download ${url} (HTTP ${res.status})`);
  }
  const total = Number(res.headers.get('content-length')) || expectedSize;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setProgress(Math.min(received / total, 1), `${label}: downloading ${mb(received)} / ${mb(total)}`);
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { buffer.set(c, offset); offset += c.length; }

  if (cache) {
    try { await cache.put(url, new Response(buffer.slice().buffer)); } catch { /* quota exceeded: skip caching */ }
  }
  return buffer.buffer;
}

async function loadModel(buffer, name, fallback) {
  stopSource();
  setSourceButtons(false);
  els.btnLoad.disabled = true;
  setStatus('loading', `Starting ${name}…`);
  setProgress(1, 'Initializing ONNX Runtime (first run downloads ~14 MB of WASM)…');
  const threads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  try {
    const res = await callWorker({ type: 'load', buffer, threads }, [buffer]);
    const frames = res.frames || fallback.frames;
    state.model = { name, frames, interval: fallback.interval };
    els.statBackend.textContent = `WASM (CPU, ${res.threads} thread${res.threads === 1 ? '' : 's'})`;
    els.statWindow.textContent = `${frames} frames, every ${fallback.interval}`;
    setStatus('ready', `${name} ready`);
    setProgress(null);
    setSourceButtons(true);
    els.placeholder.querySelector('p').textContent = 'Model ready. Start the camera or play the sample video.';
    if (res.threads === 1) {
      els.statBackend.textContent += ' · slow: no cross-origin isolation';
    }
  } catch (err) {
    state.model = null;
    state.loadedKey = null;
    setStatus('error', 'Model failed to load');
    setProgress(1, String(err.message || err));
  } finally {
    els.btnLoad.disabled = false;
  }
}

async function loadSelectedModel() {
  const key = els.selModel.value;
  const cfg = MODELS[key];
  try {
    setStatus('loading', `Downloading ${key}…`);
    const buffer = await fetchWithProgress(cfg.url, cfg.size, key);
    state.loadedKey = key;
    await loadModel(buffer, key, cfg);
  } catch (err) {
    setStatus('error', 'Model download failed');
    setProgress(1, String(err.message || err));
    els.btnLoad.disabled = false;
  }
}

async function loadModelFromDisk(file) {
  const m = /(\d+)-(\d+)/.exec(file.name);
  const fallback = m ? { frames: Number(m[1]), interval: Number(m[2]) } : { frames: 32, interval: 2 };
  setStatus('loading', `Reading ${file.name}…`);
  const buffer = await file.arrayBuffer();
  state.loadedKey = null;
  await loadModel(buffer, file.name.replace(/\.onnx$/i, ''), fallback);
}

// ---------- video sources ----------
function resetWindow() {
  state.frames = [];
  state.newFrames = 0;
  state.lastSampleTime = null;
  state.lastGloss = NO_SIGN;
  state.pausedForInference = false;
  els.bufferFill.style.width = '0';
}

function stopSource() {
  state.runId++;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  const v = els.video;
  v.pause();
  v.srcObject = null;
  if (v.src) { URL.revokeObjectURL(v.src); v.removeAttribute('src'); v.load(); }
  state.source = null;
  els.btnStop.disabled = true;
  els.placeholder.hidden = false;
  els.overlay.hidden = true;
  resetWindow();
}

async function startCamera() {
  stopSource();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('error', 'Camera needs https:// or http://localhost');
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' },
      audio: false,
    });
  } catch (err) {
    setStatus('error', `Camera unavailable: ${err.name === 'NotAllowedError' ? 'permission denied' : err.message}`);
    return;
  }
  els.video.srcObject = state.stream;
  state.source = 'camera';
  els.videoWrap.classList.toggle('mirror', els.chkMirror.checked);
  await beginPlayback();
}

async function startFile(url) {
  stopSource();
  els.video.src = url;
  state.source = 'file';
  // Uploaded or sample videos are shown as recorded, not mirrored.
  els.videoWrap.classList.remove('mirror');
  await beginPlayback();
}

async function beginPlayback() {
  const runId = state.runId;
  els.placeholder.hidden = true;
  els.overlay.hidden = false;
  els.overlayWord.textContent = '…';
  els.overlayProb.textContent = '';
  els.btnStop.disabled = false;
  try { await els.video.play(); } catch (err) {
    setStatus('error', `Could not play video: ${err.message}`);
    return;
  }
  scheduleFrame(runId);
}

// ---------- frame capture & preprocessing ----------
const canvas = document.createElement('canvas');
canvas.width = SIZE;
canvas.height = SIZE;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// Letterbox to 224x224 with gray (114) padding, RGB, normalized, CHW: the same
// preprocessing as Slovo's demo.py.
function preprocess(video) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r);
  const nh = Math.round(h * r);
  const dx = Math.round((SIZE - nw) / 2 - 0.1);
  const dy = Math.round((SIZE - nh) / 2 - 0.1);
  ctx.fillStyle = `rgb(${PAD},${PAD},${PAD})`;
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(video, dx, dy, nw, nh);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const out = new Float32Array(3 * HW);
  for (let i = 0, j = 0; i < HW; i++, j += 4) {
    out[i] = (px[j] - MEAN[0]) / STD[0];
    out[HW + i] = (px[j + 1] - MEAN[1]) / STD[1];
    out[2 * HW + i] = (px[j + 2] - MEAN[2]) / STD[2];
  }
  return out;
}

function scheduleFrame(runId) {
  const v = els.video;
  if (runId !== state.runId) return;
  if ('requestVideoFrameCallback' in v) {
    v.requestVideoFrameCallback((now, meta) => onFrame(runId, meta.mediaTime));
  } else {
    requestAnimationFrame(() => onFrame(runId, v.currentTime));
  }
}

function onFrame(runId, mediaTime) {
  if (runId !== state.runId || !state.model) return;
  const v = els.video;
  const step = state.model.interval / SOURCE_FPS;
  if (v.videoWidth && !v.paused) {
    // Resync the sampling clock on the first frame or if the video jumped back.
    if (state.lastSampleTime == null || mediaTime < state.lastSampleTime) state.lastSampleTime = mediaTime - step;
    if (mediaTime - state.lastSampleTime >= step - 0.004) {
      state.lastSampleTime += step;
      // After a stall (e.g. a slow inference), don't try to catch up.
      if (mediaTime - state.lastSampleTime >= step) state.lastSampleTime = mediaTime;
      addFrame(preprocess(v));
    }
  }
  scheduleFrame(runId);
}

function strideFrames() {
  return Math.max(1, Math.round(state.model.frames * Number(els.rngStride.value)));
}

function addFrame(frame) {
  const T = state.model.frames;
  state.frames.push(frame);
  if (state.frames.length > T) state.frames.shift();
  state.newFrames++;
  const stride = strideFrames();
  const ready = state.frames.length >= T;
  els.bufferFill.style.width = `${Math.min(100, (ready ? state.newFrames / stride : state.frames.length / T) * 100)}%`;
  if (!ready || state.newFrames < stride) return;
  if (!state.busy) {
    runInference();
  } else if (state.source === 'file' && !els.video.paused) {
    // For video files, wait for the model instead of skipping windows.
    els.video.pause();
    state.pausedForInference = true;
  }
}

async function runInference() {
  const T = state.model.frames;
  const runId = state.runId;
  const frames = state.frames.slice(-T);
  state.newFrames = 0;
  state.busy = true;
  // Model input is [1, 1, 3, T, 224, 224]: channel-major, then time.
  const data = new Float32Array(3 * T * HW);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < 3; c++) {
      data.set(frames[t].subarray(c * HW, (c + 1) * HW), (c * T + t) * HW);
    }
  }
  try {
    const res = await callWorker({ type: 'run', data, frames: T }, [data.buffer]);
    if (runId === state.runId) handleResult(res.probs, res.ms);
  } catch (err) {
    setStatus('error', `Inference failed: ${err.message}`);
  } finally {
    state.busy = false;
    if (runId === state.runId && state.pausedForInference) {
      state.pausedForInference = false;
      els.video.play().catch(() => {});
    }
  }
}

// ---------- results ----------
function toProbabilities(scores) {
  let sum = 0;
  let negative = false;
  for (const s of scores) { sum += s; if (s < 0) negative = true; }
  if (!negative && Math.abs(sum - 1) < 0.01) return scores;
  let max = -Infinity;
  for (const s of scores) max = Math.max(max, s);
  const out = new Float32Array(scores.length);
  let total = 0;
  for (let i = 0; i < scores.length; i++) { out[i] = Math.exp(scores[i] - max); total += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

const label = (gloss) => (gloss === NO_SIGN ? '— (no sign)' : gloss);

function handleResult(scores, ms) {
  const probs = toProbabilities(scores);
  const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
  state.predictions++;
  els.statCount.textContent = String(state.predictions);
  els.statInfer.textContent = `${Math.round(ms)} ms`;

  els.topk.replaceChildren(...idx.map((i) => {
    const li = document.createElement('li');
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = `${(probs[i] * 100).toFixed(1)}%`;
    const name = document.createElement('span');
    name.textContent = label(CLASSES[i]);
    const pct = document.createElement('span');
    pct.textContent = `${(probs[i] * 100).toFixed(1)}%`;
    li.append(fill, name, pct);
    return li;
  }));

  const best = idx[0];
  const gloss = CLASSES[best];
  const p = probs[best];
  const threshold = Number(els.rngThreshold.value);
  els.overlayWord.textContent = gloss === NO_SIGN ? '—' : gloss;
  els.overlayProb.textContent = `${Math.round(p * 100)}%`;
  els.overlay.style.opacity = gloss === NO_SIGN || p < threshold ? '0.6' : '1';

  if (gloss === NO_SIGN) {
    state.lastGloss = NO_SIGN;
  } else if (p >= threshold && gloss !== state.lastGloss) {
    state.lastGloss = gloss;
    addWord(gloss);
  }
}

function addWord(word) {
  if (!state.words.length) els.transcript.replaceChildren();
  state.words.push(word);
  els.transcript.querySelectorAll('.word.new').forEach((w) => w.classList.remove('new'));
  const span = document.createElement('span');
  span.className = 'word new';
  span.textContent = word;
  els.transcript.append(span);
  if (els.chkSpeak.checked) speak(word);
}

function clearTranscript() {
  state.words = [];
  state.lastGloss = NO_SIGN;
  const hint = document.createElement('span');
  hint.className = 'muted';
  hint.textContent = 'Recognized signs appear here.';
  els.transcript.replaceChildren(hint);
}

function speak(text) {
  if (!('speechSynthesis' in window) || !text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ru-RU';
  const voice = speechSynthesis.getVoices().find((v) => v.lang && v.lang.toLowerCase().startsWith('ru'));
  if (voice) u.voice = voice;
  speechSynthesis.speak(u);
}

// ---------- wiring ----------
function updateOutputs() {
  els.outThreshold.textContent = Number(els.rngThreshold.value).toFixed(2);
  els.outStride.textContent = `${Math.round(Number(els.rngStride.value) * 100)}%`;
}

loadSettings();
updateOutputs();
ensureCrossOriginIsolation();

els.btnLoad.addEventListener('click', loadSelectedModel);
els.fileModel.addEventListener('change', () => {
  const f = els.fileModel.files[0];
  if (f) loadModelFromDisk(f);
  els.fileModel.value = '';
});
els.selModel.addEventListener('change', () => {
  saveSettings();
  if (state.loadedKey && state.loadedKey !== els.selModel.value) {
    setStatus('idle', `Press "Load model" to switch to ${els.selModel.value}`);
  }
});
els.rngThreshold.addEventListener('input', () => { updateOutputs(); saveSettings(); });
els.rngStride.addEventListener('input', () => { updateOutputs(); saveSettings(); });
els.chkSpeak.addEventListener('change', saveSettings);
els.chkMirror.addEventListener('change', () => {
  saveSettings();
  if (state.source === 'camera') els.videoWrap.classList.toggle('mirror', els.chkMirror.checked);
});

els.btnCamera.addEventListener('click', startCamera);
els.btnSample.addEventListener('click', () => startFile('sample.mp4'));
els.fileVideo.addEventListener('change', () => {
  const f = els.fileVideo.files[0];
  if (f) startFile(URL.createObjectURL(f));
  els.fileVideo.value = '';
});
els.btnStop.addEventListener('click', stopSource);

els.btnClear.addEventListener('click', clearTranscript);
els.btnSpeakAll.addEventListener('click', () => speak(state.words.join(' ')));
els.btnCopy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(state.words.join(' '));
    els.btnCopy.textContent = 'Copied';
    setTimeout(() => { els.btnCopy.textContent = 'Copy'; }, 1200);
  } catch { /* clipboard unavailable */ }
});
