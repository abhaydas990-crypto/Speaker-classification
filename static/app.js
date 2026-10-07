"use strict";

const $ = (id) => document.getElementById(id);
const TARGET_SR = 16000;
const MAX_REC_SEC = 30;
const MAX_SEND_SEC = 60;
const EMOJI = {
  dog: "🐕", cat: "🐈", cow: "🐄", pig: "🐖", rooster: "🐓",
  hen: "🐔", sheep: "🐑", frog: "🐸", crow: "🐦", insects: "🦗",
};
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (x) => (x * 100).toFixed(1) + "%";
let busy = false;

/* ---------------------------------------------------------- small UI helpers */
function setStatus(msg, kind) {
  const el = $("status");
  el.className = "status" + (kind === "err" ? " err" : "");
  el.innerHTML = "";
  if (kind === "busy") {
    const spinner = document.createElement("div");
    spinner.className = "spin";
    el.appendChild(spinner);
  }
  if (msg) el.appendChild(document.createTextNode(msg));
}

function showBanner(msg) {
  const b = $("banner");
  b.textContent = msg;
  b.classList.toggle("show", Boolean(msg));
}

for (const t of ["mic", "file"]) {
  $(`tab-${t}`).addEventListener("click", () => {
    if (recording) return;
    for (const u of ["mic", "file"]) {
      $(`tab-${u}`).setAttribute("aria-selected", String(u === t));
      $(`panel-${u}`).hidden = u !== t;
    }
  });
}

/* ---------------------------------------------------------- WAV encoding (16 kHz mono) */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function toWav16k(blob) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const raw = await blob.arrayBuffer();
    const decoded = await new Promise((resolve, reject) => ctx.decodeAudioData(raw, resolve, reject));
    const seconds = Math.min(decoded.duration, MAX_SEND_SEC);
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(seconds * TARGET_SR)), TARGET_SR);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return encodeWav(rendered.getChannelData(0), TARGET_SR);
  } finally {
    ctx.close();
  }
}

/* ---------------------------------------------------------- send to server + render */
async function analyze(blob, displayName) {
  if (busy) return;
  busy = true;
  $("recbtn").disabled = true;
  try {
    setStatus("Analysing…", "busy");
    let payload = blob;
    let sendName = displayName;
    try {
      payload = await toWav16k(blob);
      sendName = displayName.replace(/\.[^.]+$/, "") + ".wav";
    } catch (err) {
      console.warn("Browser could not decode audio client-side; sending original file", err);
    }

    const form = new FormData();
    form.append("audio", payload, sendName);
    const resp = await fetch("/api/predict", { method: "POST", body: form });
    const data = await resp.json().catch(() => ({ error: "Unexpected server response." }));
    if (!resp.ok) throw new Error(data.error || "Request failed.");

    setStatus("");
    render(data);
    addHistory(displayName, data);
  } catch (err) {
    setStatus(err.message || String(err), "err");
  } finally {
    busy = false;
    $("recbtn").disabled = false;
  }
}

function verdictText(r) {
  const name = cap(r.label);
  if (r.uncertain) return { kind: "NOT SURE", cls: "unsure", name: `Maybe ${name}`, emoji: "🤔" };
  if (r.kind === "human") return { kind: "HUMAN VOICE", cls: "human", name, emoji: "🗣️" };
  return { kind: "ANIMAL SOUND", cls: "animal", name, emoji: EMOJI[r.label] || "🐾" };
}

function render(r) {
  const v = verdictText(r);
  $("result").hidden = false;
  $("r-emoji").textContent = v.emoji;
  const kindEl = $("r-kind");
  kindEl.textContent = v.kind;
  kindEl.className = "kind " + v.cls;
  $("r-name").textContent = v.name;
  $("r-conf").textContent = `${pct(r.confidence)} confident · ${r.duration.toFixed(1)}s analysed`;

  const note = $("r-note");
  note.hidden = !r.uncertain;
  note.textContent =
    "The model isn't confident about this clip. It may be a person or sound it wasn't trained on, " +
    "or the recording may be noisy or very short.";

  $("s-h").style.width = pct(r.p_human);
  $("s-a").style.width = pct(r.p_animal);
  $("l-h").textContent = `🗣️ Human ${pct(r.p_human)}`;
  $("l-a").textContent = `🐾 Animal ${pct(r.p_animal)}`;

  const top = $("r-top");
  top.innerHTML = "";
  for (const t of r.top) {
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("div");
    label.className = "lbl";
    label.textContent = (t.kind === "human" ? "🗣️ " : `${EMOJI[t.label] || "🐾"} `) + cap(t.label);
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.className = t.kind;
    bar.appendChild(fill);
    const value = document.createElement("div");
    value.className = "pct";
    value.textContent = pct(t.prob);
    row.append(label, bar, value);
    top.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${t.prob * 100}%`; }));
  }

  const winBox = $("r-win");
  winBox.innerHTML = "";
  for (const w of r.windows) {
    const chip = document.createElement("span");
    chip.className = "chip " + w.kind;
    chip.append(document.createTextNode(cap(w.label) + " "));
    const small = document.createElement("small");
    small.textContent = `${w.start.toFixed(1)}s · ${Math.round(w.confidence * 100)}%`;
    chip.appendChild(small);
    winBox.appendChild(chip);
  }
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function addHistory(name, r) {
  $("histcard").hidden = false;
  const li = document.createElement("li");
  const src = document.createElement("span");
  src.className = "src";
  src.textContent = name;
  const res = document.createElement("span");
  res.className = "res";
  const tag = r.uncertain
    ? `🤔 Not sure (${cap(r.label)}?)`
    : (r.kind === "human" ? "🗣️ " : `${EMOJI[r.label] || "🐾"} `) + cap(r.label);
  res.textContent = `${tag} · ${Math.round(r.confidence * 100)}%`;
  li.append(src, res);
  $("hist").prepend(li);
}

/* ---------------------------------------------------------- microphone recording */
let recording = false;
let mediaRecorder = null;
let micStream = null;
let recordedChunks = [];
let timerHandle = null;
let recordStartedAt = 0;
let audioCtx = null;
let analyserNode = null;
let vizFrame = null;

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
}

function drawViz() {
  const canvas = $("viz");
  const ctx2d = canvas.getContext("2d");
  const data = new Uint8Array(analyserNode.frequencyBinCount);
  const accentColor = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const loop = () => {
    analyserNode.getByteFrequencyData(data);
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    const bars = 64;
    const barWidth = canvas.width / bars;
    ctx2d.fillStyle = accentColor;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i * data.length) / bars / 2)] / 255;
      const h = Math.max(3, v * canvas.height * 0.95);
      ctx2d.fillRect(i * barWidth + 2, (canvas.height - h) / 2, barWidth - 4, h);
    }
    vizFrame = requestAnimationFrame(loop);
  };
  loop();
}

function clearViz() {
  cancelAnimationFrame(vizFrame);
  const canvas = $("viz");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

async function startRecording() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus("Your browser doesn't support microphone recording. Try Chrome, Edge, Firefox or Safari, or upload a file instead.", "err");
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    setStatus(
      denied
        ? "Microphone access was blocked. Allow it in the browser's address bar and try again."
        : `Could not access a microphone: ${err.message || err.name}`,
      "err"
    );
    return;
  }

  const mimeType = pickMimeType();
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStopped;
  mediaRecorder.start(250);

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 256;
  audioCtx.createMediaStreamSource(micStream).connect(analyserNode);
  drawViz();

  recording = true;
  recordStartedAt = Date.now();
  $("recbtn").classList.add("on");
  $("recbtn").setAttribute("aria-label", "Stop recording");
  $("player").hidden = true;
  setStatus("Recording… tap the button again to stop.");

  timerHandle = setInterval(() => {
    const elapsed = (Date.now() - recordStartedAt) / 1000;
    $("timer").textContent = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`;
    if (elapsed >= MAX_REC_SEC) stopRecording();
  }, 200);
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(timerHandle);
  $("recbtn").classList.remove("on");
  $("recbtn").setAttribute("aria-label", "Start recording");
  mediaRecorder.stop();
  micStream.getTracks().forEach((t) => t.stop());
  clearViz();
  if (audioCtx) audioCtx.close();
}

function onRecordingStopped() {
  const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
  const elapsedMs = Date.now() - recordStartedAt;
  if (blob.size < 1000 || elapsedMs < 700) {
    setStatus("That was too short - hold on for at least a second.", "err");
    return;
  }
  const url = URL.createObjectURL(blob);
  const player = $("player");
  player.src = url;
  player.hidden = false;
  const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
  analyze(blob, `Recording ${new Date().toLocaleTimeString()}.${ext}`);
}

$("recbtn").addEventListener("click", () => (recording ? stopRecording() : startRecording()));

/* ---------------------------------------------------------- file upload */
async function handleFiles(fileList) {
  for (const file of Array.from(fileList)) {
    const player = $("player");
    player.src = URL.createObjectURL(file);
    player.hidden = false;
    await analyze(file, file.name);
  }
}

$("file").addEventListener("change", (e) => {
  handleFiles(e.target.files);
  e.target.value = "";
});

const dropZone = $("drop");
["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("over"); })
);
dropZone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

/* ---------------------------------------------------------- startup */
(async () => {
  if (!window.isSecureContext) {
    showBanner("Microphone recording needs a secure page - open this app at http://localhost:5000 (not by IP address).");
  }
  try {
    const info = await (await fetch("/api/info")).json();
    if (!info.ready) showBanner("The model is still loading - wait a few seconds and refresh.");
    const knownBox = $("known");
    knownBox.hidden = false;
    knownBox.innerHTML = "";
    const peopleLine = document.createElement("div");
    peopleLine.innerHTML = `<b>🗣️ People it knows (${info.people.length}):</b> `;
    peopleLine.append(document.createTextNode(info.people.map(cap).join(", ")));
    const animalsLine = document.createElement("div");
    animalsLine.innerHTML = `<b>🐾 Animals it knows (${info.animals.length}):</b> `;
    animalsLine.append(document.createTextNode(info.animals.map(cap).join(", ")));
    knownBox.append(peopleLine, animalsLine);
  } catch (err) {
    showBanner("Can't reach the server. Make sure `python app.py` is running.");
  }
})();
