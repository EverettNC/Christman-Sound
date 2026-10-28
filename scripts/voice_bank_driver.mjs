// =============================================================================
// © 2025 Everett Nathaniel Christman & Misty Gail Christman
// The Christman AI Project — Luma Cognify AI
// =============================================================================
// voice_bank_driver.mjs — runs Codec War's own frame analysis and bank builder
// (CODECWAR5TH src/lib/audio/frames.ts, align.ts, bank.ts, stitch.ts) under
// node and writes a frame voice's bank. Called by scripts/build_voice_bank.py;
// node strips the TypeScript types, nothing is installed or rewritten.
//
//   node voice_bank_driver.mjs JOB.json
//
// JOB.json: { "codecwar": "<checkout>", "out": "<folder>", "minConf": 0.9,
//             "recordings": [{ "path": "<22050 Hz 16-bit mono WAV>", "session": "<name>" }],
//             "tokens": [{ "rec", "word", "start", "end", "conf", "phones" }] }
// Writes <out>/frames.bin and <out>/bank.json. frames.bin holds, per frame:
// pitch float32, envelope int16 x 96 (0.01 dB), aperiodicity uint16 x 8
// (1/65535), level int16 (0.01 dB), little-endian.

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const job = JSON.parse(readFileSync(process.argv[2], "utf8"));
const lib = (f) => import(pathToFileURL(join(job.codecwar, "src", "lib", "audio", f)).href);
const { analyze, envGridHz, AP_EDGES, ENV_POINTS, AP_BANDS, FRAME_SR, HOP } = await lib("frames.ts");
const { buildBank } = await lib("bank.ts");
const { recordingStats, pauseFrames } = await lib("stitch.ts");

function readWav(path) {
  const b = readFileSync(path);
  if (b.toString("ascii", 0, 4) !== "RIFF" || b.toString("ascii", 8, 12) !== "WAVE") throw new Error(`${path}: not a WAV file`);
  let pos = 12;
  let fmt = null;
  let data = null;
  while (pos + 8 <= b.length) {
    const id = b.toString("ascii", pos, pos + 4);
    const size = b.readUInt32LE(pos + 4);
    if (id === "fmt ") fmt = { tag: b.readUInt16LE(pos + 8), ch: b.readUInt16LE(pos + 10), sr: b.readUInt32LE(pos + 12), bits: b.readUInt16LE(pos + 22) };
    if (id === "data") data = b.subarray(pos + 8, pos + 8 + size);
    pos += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error(`${path}: no fmt or data chunk`);
  if (fmt.tag !== 1 || fmt.bits !== 16 || fmt.ch !== 1 || fmt.sr !== FRAME_SR)
    throw new Error(`${path}: must be ${FRAME_SR} Hz 16-bit mono PCM (is ${fmt.sr} Hz, ${fmt.bits}-bit, ${fmt.ch} channel(s), format ${fmt.tag})`);
  const n = data.length / 2;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2) / 32768;
  return x;
}

const tracks = job.recordings.map((r) => analyze(readWav(r.path), FRAME_SR));
const bank = buildBank(tracks, job.tokens, job.minConf);
const stats = recordingStats(bank, job.recordings.map((r) => r.session));
const per = 4 + 2 * ENV_POINTS + 2 * AP_BANDS + 2;
const total = tracks.reduce((a, t) => a + t.n, 0);
const buf = Buffer.alloc(total * per);
let off = 0;
const q16 = (v) => Math.max(-32768, Math.min(32767, Math.round(v * 100)));
const recordings = [];
let frame0 = 0;
tracks.forEach((t, ri) => {
  recordings.push({
    path: job.recordings[ri].path,
    session: job.recordings[ri].session,
    n: t.n,
    frame0,
    stats: { ltas: Array.from(stats[ri].ltas), level: stats[ri].level },
    pause: pauseFrames(bank, ri),
  });
  for (let i = 0; i < t.n; i++) {
    buf.writeFloatLE(t.f0[i], off);
    off += 4;
    for (let j = 0; j < ENV_POINTS; j++) {
      buf.writeInt16LE(q16(t.envDb[i * ENV_POINTS + j]), off);
      off += 2;
    }
    for (let b = 0; b < AP_BANDS; b++) {
      buf.writeUInt16LE(Math.round(t.ap[i * AP_BANDS + b] * 65535), off);
      off += 2;
    }
    buf.writeInt16LE(q16(t.rmsDb[i]), off);
    off += 2;
  }
  frame0 += t.n;
});
writeFileSync(join(job.out, "frames.bin"), buf);
writeFileSync(
  join(job.out, "bank.json"),
  JSON.stringify({ sr: FRAME_SR, hop: HOP, envHz: Array.from(envGridHz(FRAME_SR)), bandEdges: [...AP_EDGES], recordings, tokens: job.tokens, units: bank.units, skipped: bank.skipped }),
);
console.log(JSON.stringify({ frames: total, units: bank.units.length, skipped: bank.skipped.length, bytes: buf.length }));
