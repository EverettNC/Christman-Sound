# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The frame voice (2026-10-28): christman_vocoder.c and christman_sound/stitch.py
are ports of Codec War's vocoder.ts and stitch.ts, and are held to them here.

  * The renderer is compared with Codec War's own vocoder.ts, run under node,
    sample for sample (allowed difference 1e-5, as for christman_synth; on
    Everett's Mac every sample has been identical).
  * The stitcher is compared with Codec War's own stitch.ts on a bank Codec
    War builds from its own synthetic test recordings: the same units, the
    same joins, the same applied values and the same frames.
  * Refusals: every control is applied or refused with the reason.

Codec War is found at $CODECWAR_SRC, else at ../CODECWAR5TH beside this
repository. Without node or a checkout the comparisons report NOT RUN.
"""

from __future__ import annotations

import array
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_sound.dsp import ChristmanDSP, SynthRefused  # noqa: E402
from christman_sound import stitch as cs_stitch  # noqa: E402


def codecwar_audio() -> Path:
    src = os.environ.get("CODECWAR_SRC") or str(ROOT.parent / "CODECWAR5TH")
    audio = Path(src).expanduser() / "src" / "lib" / "audio"
    node = shutil.which("node")
    if not node or not (audio / "vocoder.ts").is_file() or not (audio / "stitch.ts").is_file():
        pytest.skip("not run: needs node on PATH and a CODECWAR5TH checkout with the frame mouth (b148879 or later) "
                    f"at $CODECWAR_SRC or {ROOT.parent / 'CODECWAR5TH'} (node={'found' if node else 'missing'}, "
                    f"vocoder.ts={'found' if (audio / 'vocoder.ts').is_file() else 'missing'})")
    return audio


@pytest.fixture(scope="module")
def dsp(tmp_path_factory) -> ChristmanDSP:
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"build_dsp.py failed:\n{proc.stdout}\n{proc.stderr}"
    d = ChristmanDSP(out)
    assert d.available and d.vocoder_available, (d.error, d.vocoder_error)
    return d


# ------------------------------------------------------------ the renderer

_VOCODER_DRIVER = r"""
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [dir, inPath, out] = process.argv.slice(2);
const { renderFrames } = await import(pathToFileURL(dir + "/vocoder.ts").href);
const y = renderFrames(JSON.parse(readFileSync(inPath, "utf8")));
writeFileSync(out, Buffer.from(y.buffer));
"""


def _frames_case(n: int, kind: str):
    """Frames with known shapes: a vowel-like envelope, pitch moving, noise where unvoiced."""
    hop, sr = 110, 22050
    top = 1127 * math.log(1 + (sr / 2) / 700)
    env_hz = [700 * (math.exp((top * j) / 95 / 1127) - 1) for j in range(96)]
    env_hz[0], env_hz[-1] = 0.0, sr / 2
    edges = [0, 600, 1200, 2000, 3000, 4000, 5500, 7500, 11025]
    f0, env, ap = [], [], []
    for i in range(n):
        voiced = kind == "voiced" or (kind == "mixed" and (i // 30) % 2 == 0)
        f0.append(120 + 40 * math.sin(i / 25) if voiced else 0.0)
        for f in env_hz:
            env.append(-45 + 25 * math.exp(-((f - 700 - 300 * math.sin(i / 17)) / 250) ** 2)
                       + 18 * math.exp(-((f - 1800) / 400) ** 2) - (0 if voiced else 10))
        ap.extend(([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if voiced else [1.0] * 8))
    return {"sr": sr, "hop": hop, "f0": f0, "envDb": env, "envHz": env_hz, "ap": ap, "bandEdges": edges}


@pytest.mark.parametrize("kind,jitter,shimmer,seed", [
    ("voiced", 0.0, 0.0, 1), ("voiced", 0.02, 0.06, 7), ("unvoiced", 0.0, 0.0, 3), ("mixed", 0.015, 0.05, 11),
])
def test_vocoder_matches_codec_war_sample_for_sample(dsp, tmp_path, kind, jitter, shimmer, seed):
    audio = codecwar_audio()
    case = {**_frames_case(240, kind), "jitter": jitter, "shimmer": shimmer, "seed": seed}
    (tmp_path / "in.json").write_text(json.dumps(case))
    (tmp_path / "d.mjs").write_text(_VOCODER_DRIVER)
    subprocess.run([shutil.which("node"), str(tmp_path / "d.mjs"), str(audio), str(tmp_path / "in.json"),
                    str(tmp_path / "ref.f32")], check=True, capture_output=True, text=True)
    ref = array.array("f")
    ref.frombytes((tmp_path / "ref.f32").read_bytes())
    ours = dsp.vocoder_render(case["f0"], case["envDb"], case["envHz"], case["ap"], case["bandEdges"],
                              case["hop"], case["sr"], jitter, shimmer, seed)
    assert len(ours) == len(ref) == 240 * 110
    diffs = [abs(a - b) for a, b in zip(ours, ref)]
    print(f"{kind}: {len(ref)} samples, {sum(1 for d in diffs if d == 0.0)} identical, max difference {max(diffs):.3e}")
    assert max(diffs) <= 1e-5


def test_vocoder_refuses_with_the_engines_reason(dsp):
    case = _frames_case(20, "voiced")
    bad_ap = list(case["ap"])
    bad_ap[3] = 1.5
    with pytest.raises(SynthRefused) as e:
        dsp.vocoder_render(case["f0"], case["envDb"], case["envHz"], bad_ap, case["bandEdges"], 110, 22050, 0, 0, 1)
    assert e.value.code == 3
    with pytest.raises(SynthRefused) as e:
        dsp.vocoder_render(case["f0"], case["envDb"], case["envHz"], case["ap"], case["bandEdges"], 110, 44100, 0, 0, 1)
    assert e.value.code == 2
    with pytest.raises(SynthRefused) as e:
        dsp.vocoder_render(case["f0"], case["envDb"], case["envHz"], case["ap"], case["bandEdges"], 110, 22050, 0.5, 0, 1)
    assert e.value.code == 3
    with pytest.raises(SynthRefused) as e:
        dsp.vocoder_render(case["f0"], case["envDb"][:-1], case["envHz"], case["ap"], case["bandEdges"], 110, 22050, 0, 0, 1)
    assert e.value.code == 1


# ------------------------------------------------------------ the stitcher

_STITCH_DRIVER = r"""
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [dir, jobPath, out] = process.argv.slice(2);
const job = JSON.parse(readFileSync(jobPath, "utf8"));
const imp = (f) => import(pathToFileURL(join(dir, f)).href);
const F = await imp("frames.ts");
const { record } = await imp("synthetic.test-helpers.ts");
const { buildBank } = await imp("bank.ts");
const S = await imp("stitch.ts");
const { ENV_POINTS: E, AP_BANDS: B } = F;
const { track, tokens } = record(job.words, 3);
const built = buildBank([track], tokens, 0.9);
// Quantise exactly as a bank on disk holds the frames, and read them back.
const t = track;
const q = { ...t, f0: Float32Array.from(t.f0), envDb: t.envDb.map((v) => Math.max(-32768, Math.min(32767, Math.round(v * 100))) / 100),
  ap: t.ap.map((v) => Math.round(v * 65535) / 65535), rmsDb: t.rmsDb.map((v) => Math.max(-32768, Math.min(32767, Math.round(v * 100))) / 100) };
q.f0 = Float64Array.from(q.f0);
const bank = { ...built, tracks: [q] };
const stats = S.recordingStats(bank, ["one"]);
const per = 4 + 2 * E + 2 * B + 2;
const buf = Buffer.alloc(q.n * per);
let o = 0;
for (let i = 0; i < q.n; i++) {
  buf.writeFloatLE(q.f0[i], o); o += 4;
  for (let j = 0; j < E; j++) { buf.writeInt16LE(Math.round(q.envDb[i * E + j] * 100), o); o += 2; }
  for (let b = 0; b < B; b++) { buf.writeUInt16LE(Math.round(q.ap[i * B + b] * 65535), o); o += 2; }
  buf.writeInt16LE(Math.round(q.rmsDb[i] * 100), o); o += 2;
}
writeFileSync(join(job.bankDir, "frames.bin"), buf);
writeFileSync(join(job.bankDir, "bank.json"), JSON.stringify({ sr: F.FRAME_SR, hop: F.HOP, envHz: Array.from(F.envGridHz()), bandEdges: [...F.AP_EDGES],
  recordings: [{ path: "synthetic", session: "one", n: q.n, frame0: 0, stats: { ltas: Array.from(stats[0].ltas), level: stats[0].level }, pause: S.pauseFrames(bank, 0) }],
  tokens: built.tokens, units: built.units, skipped: built.skipped }));
const results = [];
for (const c of job.cases) {
  const words = c.words.map(([word, phones]) => ({ word, phones }));
  const sel = S.selectUnits(bank, words);
  const st = S.joinUnits(bank, sel);
  let v = null, refused = null;
  try { v = S.applyVoice(st, job.measures, { ...S.NEUTRAL_CONTROLS, ...c.controls }, stats); } catch (e) { refused = e.message; }
  results.push({ units: sel.units, joins: sel.joins, wholeWords: sel.wholeWords, joinsAt: st.joinsAt,
    stitched: { f0: Array.from(st.track.f0), env: Array.from(st.track.envDb), ap: Array.from(st.track.ap) },
    refused, voiced: v && { f0: Array.from(v.track.f0), env: Array.from(v.track.envDb), ap: Array.from(v.track.ap), phones: v.phones,
      jitter: v.jitter, shimmer: v.shimmer, applied: v.applied.map((a) => [a.name, a.value]) } });
}
writeFileSync(out, JSON.stringify(results));
"""

WORDS = [
    {"word": "mass", "phones": ["M", "AA", "S"]}, {"word": "tea", "phones": ["T", "IY"]},
    {"word": "loom", "phones": ["L", "UW", "M"], "pauseAfter": True}, {"word": "seat", "phones": ["S", "IY", "T"]},
    {"word": "mass", "phones": ["M", "AA", "S"], "pauseAfter": True}, {"word": "loom", "phones": ["L", "UW", "M"]},
    {"word": "tea", "phones": ["T", "IY"]}, {"word": "seat", "phones": ["S", "IY", "T"]},
]
CASES = [
    {"words": [["mass", ["M", "AA", "S"]], ["tea", ["T", "IY"]]], "controls": {}},
    {"words": [["tea", ["T", "IY"]], ["mass", ["M", "AA", "S"]], ["loom", ["L", "UW", "M"]]], "controls": {}},
    {"words": [["seam", ["S", "IY", "M"]], ["tool", ["T", "UW", "L"]]],
     "controls": {"breath": 0.8, "intonation": 0.3, "timbre": 0.7, "articulation": 0.6, "prosody": 0.9,
                  "resonance": "head", "rate": 1.2, "f0": 180, "jitter": 0.7, "shimmer": 0.6}},
    {"words": [["mass", ["M", "AA", "S"]]], "controls": {"affect": "calm"}},
]


def _measures(ltas):
    return {"f0_median_hz": 140.0, "phrase_range_st": 1.0, "syllable_rate": 4.0, "voiced_aperiodicity": 0.2,
            "ltas_db": ltas, "level_db": -20.0, "jitter": 0.02, "shimmer": 0.06, "jitter_carried": 0.005,
            "shimmer_carried": 0.02, "hnr_db": 10.0, "cpp_db": 15.0}


def _close(a, b, tol=1e-9):
    assert len(a) == len(b)
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    assert worst <= tol, worst
    return worst


def test_stitch_matches_codec_war(dsp, tmp_path):
    audio = codecwar_audio()
    if not (audio / "synthetic.test-helpers.ts").is_file():
        pytest.skip("not run: the Codec War checkout has no synthetic.test-helpers.ts (b148879 or later)")
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    ltas_probe = [0.0] * 96
    job = {"words": WORDS, "bankDir": str(bank_dir), "cases": CASES, "measures": _measures(ltas_probe)}
    (tmp_path / "job.json").write_text(json.dumps(job))
    (tmp_path / "d.mjs").write_text(_STITCH_DRIVER)
    subprocess.run([shutil.which("node"), str(tmp_path / "d.mjs"), str(audio), str(tmp_path / "job.json"),
                    str(tmp_path / "out.json")], check=True, capture_output=True, text=True)
    ref = json.loads((tmp_path / "out.json").read_text())
    bank = cs_stitch.FrameBank(bank_dir)
    vowels = [k for k, (_, v) in dsp.synth_phone_formants().items() if v]
    worst = 0.0
    for case, r in zip(CASES, ref):
        words = [(w, p) for w, p in case["words"]]
        sel = cs_stitch.select_units(bank, words)
        assert sel.units == r["units"] and sel.joins == r["joins"] and sel.whole_words == r["wholeWords"]
        st = cs_stitch.join_units(bank, sel)
        assert st.joins_at == r["joinsAt"]
        worst = max(worst, _close(st.track.f0, r["stitched"]["f0"]), _close(st.track.env, r["stitched"]["env"]),
                    _close(st.track.ap, r["stitched"]["ap"]))
        ctl = {**cs_stitch.NEUTRAL_CONTROLS, **case["controls"]}
        if r["refused"]:
            with pytest.raises(cs_stitch.StitchRefused) as e:
                cs_stitch.apply_voice(st, _measures(ltas_probe), ctl, bank.session_stats(), bank.env_hz, vowels,
                                      bank.hop, bank.sr)
            assert str(e.value) == r["refused"]
            continue
        v = cs_stitch.apply_voice(st, _measures(ltas_probe), ctl, bank.session_stats(), bank.env_hz, vowels,
                                  bank.hop, bank.sr)
        assert v.phones == r["voiced"]["phones"]
        assert [a["name"] for a in v.applied] == [n for n, _ in r["voiced"]["applied"]]
        worst = max(worst, _close([a["value"] for a in v.applied], [x for _, x in r["voiced"]["applied"]]),
                    _close(v.track.f0, r["voiced"]["f0"]), _close(v.track.env, r["voiced"]["env"]),
                    _close(v.track.ap, r["voiced"]["ap"]), _close([v.jitter, v.shimmer],
                                                                   [r["voiced"]["jitter"], r["voiced"]["shimmer"]]))
    print(f"stitch.py against stitch.ts: same units, joins and applied parameters; largest difference {worst:.2e}")


# ------------------------------------------------------------ refusals

def test_a_missing_voice_folder_names_what_is_missing(tmp_path):
    with pytest.raises(cs_stitch.FrameVoiceUnavailable, match="voice.json is missing"):
        cs_stitch.load_frame_voice(tmp_path)
    (tmp_path / "voice.json").write_text(json.dumps({"name": "Test"}))
    with pytest.raises(cs_stitch.FrameVoiceUnavailable, match="bank.json is missing"):
        cs_stitch.load_frame_voice(tmp_path)


def test_controls_are_checked_by_name_and_range():
    good = dict(cs_stitch.NEUTRAL_CONTROLS)
    assert cs_stitch.check_controls(good) == good
    for bad, pattern in [({"breath": 1.5}, "breath"), ({"rate": 2.0}, "rate"), ({"resonance": "nasal"}, "resonance"),
                         ({"f0": 20}, "f0"), ({"affect": "calm"}, 'affect "calm" is not applied'),
                         ({"warmth": 0.4}, "not controls of this mouth")]:
        with pytest.raises(cs_stitch.StitchRefused, match=pattern):
            cs_stitch.check_controls({**good, **bad})


def test_an_unmeasured_profile_is_refused_by_name():
    m = _measures([0.0] * 96)
    del m["jitter"]
    m["cpp_db"] = float("nan")
    with pytest.raises(cs_stitch.StitchRefused, match="does not measure: jitter, cpp_db"):
        cs_stitch.check_measures(m, 96)
