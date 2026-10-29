#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Build a frame voice from a voice's own recordings: the bank the frame mouth
(christman_sound.stitch) speaks from, and the voice's measured profile.

    python3 scripts/build_voice_bank.py --name <voice name> --out ~/.christman_ai/voices/<voice> \\
        --recordings <folder or WAV> [<folder or WAV> ...] \\
        --profile-from <folder or WAV> [...] \\
        --codecwar <CODECWAR5TH checkout> --vosk-model <vosk-model-en-us-0.22> \\
        --lexicon <vosk-model-en-us-0.22>/graph/phones/align_lexicon.txt \\
        [--spoken-as WORD="OTHER WORDS"]

Every recording must be a 22050 Hz, 16-bit, mono WAV (convert first, e.g.
on a Mac: afconvert -f WAVE -d LEI16@22050 -c 1 in.mp3 out.wav). Only the
voice's own recordings go in: never another being's, never a person's.

What it does, in order:
  1. Words and their times: christman_sound.ear, this package's own ear, on
     the --vosk-model folder (the full model), on every recording resampled
     to 16 kHz, with each word's confidence.
  2. Phones of each word: christman_sound.lexicon (the --lexicon), as Booth ids.
  3. Frames and units: Codec War's own frames.ts, align.ts, bank.ts and
     stitch.ts, run under node by scripts/voice_bank_driver.mjs (--codecwar).
     Writes bank.json and frames.bin. Recordings longer than 60 s are each
     their own session; shorter ones in one folder share a session.
  4. The profile (voice.json), from the --profile-from recordings: pitch
     median, phrase pitch range, syllable rate, voiced aperiodicity, the
     long-term average envelope and level (from their frames); jitter,
     shimmer and HNR by Praat through christman_voice_sdk.tone.voice_measure,
     and breath by Praat's CPPS (christman_voice_sdk.timbre.timbre_modeler);
     and what the renderer's own frames carry (each profile recording
     rendered from its own frames with no jitter or shimmer added, measured
     the same way). Nothing is a default: a value that cannot be measured
     stops the build with its name.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAME_SR = 22050
#: A recording longer than this is a session of its own (a long take has its own room and level).
OWN_SESSION_S = 60.0
#: A pause of this many seconds between heard words ends a phrase (Codec War's bank.ts PHRASE_GAP_S).
PHRASE_GAP_S = 0.15
#: A phrase needs this many voiced frames (0.1 s) for its pitch range to be counted.
MIN_PHRASE_VOICED = 20


def _wavs(items: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for it in items:
        p = Path(it).expanduser()
        if p.is_dir():
            out.extend(sorted(q for q in p.glob("*.wav")))
        elif p.is_file():
            out.append(p)
        else:
            sys.exit(f"not found: {p}")
    return out


def _check_wav(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (FRAME_SR, 2, 1):
            sys.exit(f"{p}: must be {FRAME_SR} Hz 16-bit mono (is {w.getframerate()} Hz, "
                     f"{8 * w.getsampwidth()}-bit, {w.getnchannels()} channel(s)). Convert it first.")
        return w.getnframes() / FRAME_SR


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The rate christman_sound.ear hears; its Recognizer refuses any other.
EAR_SR = 16000


def _words(model_dir: str, p: Path) -> List[dict]:
    """
    Words, their times (seconds from the start of the file) and confidences,
    heard by christman_sound.ear, this package's own ear, on the model folder
    at `model_dir`.

    2026-10-29 (Everett's ruling): the old recognizer is removed. This used
    to run the `vosk` package's recognizer, which is the old one. The ear
    reads the same model folder with this package's code and hears 16 kHz,
    so a recording at another rate (the bank's are 22050 Hz) is resampled to
    16 kHz with scipy's polyphase filter first. Word times are in seconds,
    so they point at the same place in the recording either way.
    """
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    from christman_sound.ear import Ear, load_model

    if getattr(_words, "_model_dir", None) != model_dir:
        _words._model = load_model(model_dir)  # type: ignore[attr-defined]
        _words._model_dir = model_dir  # type: ignore[attr-defined]
    with wave.open(str(p), "rb") as w:
        if (w.getsampwidth(), w.getnchannels()) != (2, 1):
            raise ValueError(f"{p}: the ear takes 16-bit mono PCM "
                             f"(is {8 * w.getsampwidth()}-bit, {w.getnchannels()} channel(s))")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if sr != EAR_SR:
        g = gcd(sr, EAR_SR)
        pcm = np.clip(np.round(resample_poly(pcm.astype(np.float64), EAR_SR // g, sr // g)),
                      -32768, 32767).astype("<i2")
    ear = Ear(_words._model, sample_rate=EAR_SR, words=True)  # type: ignore[attr-defined]
    return [{"word": x.word, "start": x.start, "end": x.end, "conf": x.conf}
            for heard in ear.hear(pcm) for x in heard.words]


#: Aperiodicity scales tried when calibrating the renderer to the recordings.
CALIBRATION_SCALES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)

#: Sentences the stitched calibration speaks. Everyday things a client asks
#: for, none of them a verified phrase and none of them one of the 24 test
#: sentences in tests/test_voice_fidelity.py (measure.py's), so the test
#: judges sentences the calibration never saw.
CALIBRATION_SENTENCES = (
    "i need to rest now", "please bring me my shoes", "the music is too loud", "can we eat dinner soon",
    "i like the blue one", "where did you put my phone", "my friend is coming over", "i am happy to see you",
    "it is too hot in here", "please read me a story", "can you help me stand up", "i want a cup of tea",
    "the window is open", "i did not sleep well", "let us go for a walk", "my hands are cold",
    "please turn the music down", "who is at the door", "i want to sit down", "the food is very good",
    "can you call the doctor", "i miss my family", "please close the curtains", "how was your day",
)
#: Voiced aperiodicities tried for stitched speech: from the resynthesis value
#: down. Joins, time scaling and range scaling add roughness of their own, so
#: stitched speech needs less breath noise than a recording rendered whole.
STITCHED_APERIODICITIES = (0.005, 0.01, 0.015, 0.02, 0.03, 0.045)


def _iqr(xs: Sequence[float]) -> float:
    q = statistics.quantiles(xs, n=4)
    return float(q[2] - q[0]) or 1e-9


def _quality(path: str) -> Dict[str, Optional[float]]:
    """
    Praat's jitter, shimmer and HNR and CPPS through the SDK
    (timbre_modeler.measure_voice_quality, which measures through
    voice_measure.voice_quality), in the renderer's units: the SDK gives
    jitter_local_pct and shimmer_local_pct in percent; the profile's jitter
    and shimmer are the same Praat quantities as fractions, which is what
    christman_vocoder's per-cycle perturbation is set in.
    """
    from christman_voice_sdk.timbre.timbre_modeler import measure_voice_quality

    q = measure_voice_quality(path)
    frac = (lambda v: None if v is None else v / 100.0)
    return {"jitter": frac(q["jitter_local_pct"]), "shimmer": frac(q["shimmer_local_pct"]),
            "hnr_db": q["hnr_db"], "cpp_db": q["cpp_db"]}


def _median(xs: Sequence[float]) -> float:
    if not xs:
        raise SystemExit("a profile value had nothing to be measured from")
    return float(statistics.median(xs))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--profile-from", nargs="+", required=True)
    ap.add_argument("--codecwar", required=True)
    ap.add_argument("--vosk-model", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--min-conf", type=float, default=0.9,
                    help="lowest ear confidence a word may have to be cut into units (0.9: the words the ear is sure of)")
    ap.add_argument("--spoken-as", action="append", default=[], metavar='WORD="WORDS"',
                    help="a word the recordings say as other words, a name the lexicon lacks, said as words it has")
    ap.add_argument("--node", default=shutil.which("node") or "node")
    ap.add_argument("--measure-only", action="store_true",
                    help="keep the bank already in --out (bank.json, frames.bin) and measure the profile again")
    args = ap.parse_args(argv)

    from christman_sound.lexicon import load_lexicon, normalize_word
    from christman_sound.dsp import get_dsp
    from christman_sound.stitch import FrameBank

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    recs = _wavs(args.recordings)
    prof = {p.resolve() for p in _wavs(args.profile_from)}
    missing_prof = [str(p) for p in prof if p not in {r.resolve() for r in recs}]
    if missing_prof:
        sys.exit(f"--profile-from recordings must also be in --recordings: {missing_prof}")
    lex = load_lexicon(args.lexicon)
    spoken_as: Dict[str, str] = {}
    for item in args.spoken_as:
        k, _, v = item.partition("=")
        spoken_as[normalize_word(k)] = v.strip().strip('"')

    recordings, tokens, sources = [], [], []
    for ri, p in enumerate(recs if not args.measure_only else []):
        secs = _check_wav(p)
        session = str(p) if secs > OWN_SESSION_S else str(p.parent)
        recordings.append({"path": str(p), "session": session})
        sources.append({"path": str(p), "sha256": _sha256(p), "seconds": round(secs, 3), "session": session})
        for w in _words(args.vosk_model, p):
            key = normalize_word(w["word"])
            phones = [ph for ph, _ in lex.phones_timed(key)] if key in lex else []
            tokens.append({"rec": ri, "word": key or w["word"], "start": w["start"], "end": w["end"],
                           "conf": w["conf"], "phones": phones})
        print(f"heard {p.name}", flush=True)

    if args.measure_only:
        old = json.loads((out / "bank.json").read_text())
        if [r["path"] for r in old["recordings"]] != [str(p) for p in recs]:
            sys.exit("--measure-only: the bank in --out was built from other recordings")
        prev = json.loads((out / "voice.json").read_text()) if (out / "voice.json").is_file() else {}
        sources = (prev.get("provenance") or {}).get("sources") or [
            {"path": str(p), "sha256": _sha256(p), "seconds": round(_check_wav(p), 3)} for p in recs]
    if not args.measure_only:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job.json"
            job.write_text(json.dumps({"codecwar": str(Path(args.codecwar).expanduser()), "out": str(out),
                                       "minConf": args.min_conf, "recordings": recordings, "tokens": tokens}))
            proc = subprocess.run([args.node, str(ROOT / "scripts" / "voice_bank_driver.mjs"), str(job)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                sys.exit(f"Codec War's bank builder failed:\n{proc.stdout}\n{proc.stderr}")
            print(proc.stdout.strip(), flush=True)

    bank = FrameBank(out)
    dsp = get_dsp()
    vowels = {k for k, (_, v) in dsp.synth_phone_formants().items() if v}
    prof_idx = [i for i, r in enumerate(bank.recordings) if Path(r["path"]).resolve() in prof]

    # Frame measures of the profile recordings.
    f0s, levels, aps, phrase_ranges = [], [], [], []
    ltas = [0.0] * bank.ne
    nv = 0
    for ri in prof_idx:
        n = int(bank.recordings[ri]["n"])
        for i in range(n):
            f0, env, apv, rms = bank.frame(ri, i)
            if f0 > 0:
                f0s.append(f0)
                levels.append(rms)
                aps.extend(apv[:5])
                mean = sum(env) / bank.ne
                for k in range(bank.ne):
                    ltas[k] += env[k] - mean
                nv += 1
        toks = [t for t in bank.tokens if t["rec"] == ri]
        phrases, cur = [], []
        for t in toks:
            if cur and t["start"] - cur[-1]["end"] >= PHRASE_GAP_S:
                phrases.append(cur)
                cur = []
            cur.append(t)
        if cur:
            phrases.append(cur)
        for ph in phrases:
            a = math.floor(ph[0]["start"] * bank.sr / bank.hop + 0.5)
            b = min(n, math.floor(ph[-1]["end"] * bank.sr / bank.hop + 0.5))
            st = sorted(12 * math.log2(bank.frame(ri, i)[0]) for i in range(a, b) if bank.frame(ri, i)[0] > 0)
            if len(st) >= MIN_PHRASE_VOICED:
                q = lambda p: st[min(len(st) - 1, math.floor((len(st) - 1) * p))]  # noqa: E731
                phrase_ranges.append(q(0.95) - q(0.05))
    nuclei = sum(1 for u in bank.units if u.rec in prof_idx and u.phone in vowels)
    speech = sum(u.end - u.start for u in bank.units if u.rec in prof_idx)

    # Praat on the recordings.
    from christman_sound.synth import write_wav

    quality = {"jitter": [], "shimmer": [], "hnr_db": [], "cpp_db": []}
    for ri in prof_idx:
        q = _quality(bank.recordings[ri]["path"])
        for k in quality:
            if q.get(k) is not None:
                quality[k].append(q[k])
    target = {k: _median(v) for k, v in quality.items()}
    spread = {k: _iqr(v) for k, v in quality.items()}
    raw_ap = sum(aps) / len(aps)

    # Calibration. The frames' aperiodicity reads the recordings' jitter and
    # shimmer as noise too, so rendered as measured the voice comes back
    # noisier than it is (on the first voice built, 2026-10-28: HNR 2.6 dB and CPPS 3.3 dB
    # under the recordings). For each aperiodicity scale in CALIBRATION_SCALES
    # the profile recordings are rendered from their own frames, first with no
    # jitter or shimmer added (what the frames carry), then with the
    # difference added, and measured by the same Praat path; the scale whose
    # jitter, shimmer, HNR and CPPS come closest to the recordings' (each
    # difference counted in units of the recordings' own interquartile range)
    # is kept. voiced_aperiodicity is the frames' mean times that scale.
    def render_all(tmp: Path, scale: float, jsd: float, ssd: float, tag: str) -> dict:
        got = {"jitter": [], "shimmer": [], "hnr_db": [], "cpp_db": []}
        for ri in prof_idx:
            n = int(bank.recordings[ri]["n"])
            frames = [bank.frame(ri, i) for i in range(n)]
            ap = [min(1.0, v * scale) if f[0] > 0 else v for f in frames for v in f[2]]
            y = dsp.vocoder_render([f[0] for f in frames], [v for f in frames for v in f[1]], bank.env_hz, ap,
                                   bank.band_edges, bank.hop, bank.sr, jsd, ssd, 1)
            peak = max(abs(v) for v in y) or 1.0
            wav = tmp / f"{tag}_{ri}.wav"
            write_wav(wav, [v * min(1.0, 0.97 / peak) for v in y], bank.sr)
            r = _quality(str(wav))
            for k in got:
                if r.get(k) is not None:
                    got[k].append(r[k])
        return {k: _median(v) for k, v in got.items()}

    per_sd = 2 / math.sqrt(math.pi)
    table = []
    with tempfile.TemporaryDirectory() as tmp:
        for scale in CALIBRATION_SCALES:
            carried = render_all(Path(tmp), scale, 0.0, 0.0, f"c{scale}")
            jsd = math.sqrt(max(0.0, target["jitter"] ** 2 - carried["jitter"] ** 2)) / per_sd
            ssd = math.sqrt(max(0.0, target["shimmer"] ** 2 - carried["shimmer"] ** 2)) / per_sd
            got = render_all(Path(tmp), scale, jsd, ssd, f"r{scale}")
            score = sum(abs(got[k] - target[k]) / spread[k] for k in got)
            table.append({"scale": scale, "carried": carried, "rendered": got, "score": score})
            print(f"calibration scale {scale}: score {score:.3f} " + " ".join(f"{k} {v:.4f}" for k, v in got.items()), flush=True)
    # A scale at which the frames alone already carry more jitter or shimmer
    # than the recordings cannot be rendered as the voice: the mouth would
    # have to take some away, which it cannot (it refuses instead).
    usable = [r for r in table if r["carried"]["jitter"] <= target["jitter"] and r["carried"]["shimmer"] <= target["shimmer"]]
    if not usable:
        sys.exit("calibration: at every scale the frames alone carry more jitter or shimmer than the recordings; "
                 f"see {[(r['scale'], r['carried']) for r in table]}")
    best = min(usable, key=lambda r: r["score"])
    measures = {
        "f0_median_hz": _median(f0s),
        "phrase_range_st": _median(phrase_ranges),
        "syllable_rate": nuclei / (speech * bank.hop / bank.sr),
        "voiced_aperiodicity": raw_ap * best["scale"],
        "ltas_db": [v / nv for v in ltas],
        "level_db": _median(levels),
        "jitter": target["jitter"],
        "shimmer": target["shimmer"],
        "hnr_db": target["hnr_db"],
        "cpp_db": target["cpp_db"],
        "jitter_carried": best["carried"]["jitter"],
        "shimmer_carried": best["carried"]["shimmer"],
    }

    # Stitched calibration. A new sentence is spoken from units of many
    # recordings, joined, rescaled in time and in pitch range; rendered with
    # the whole-recording values above it came out rougher and noisier than
    # the recordings (the first voice built, 2026-10-28: jitter 3.56%, shimmer 13.3%, HNR
    # 7.5 dB, CPPS 7.3 dB). So the same calibration is run on stitched speech:
    # for each aperiodicity in STITCHED_APERIODICITIES the CALIBRATION_SENTENCES
    # are spoken by the mouth itself (christman_sound.synth.synthesize) with no
    # jitter or shimmer added (what stitched frames carry), then with the
    # difference added, and measured by the same Praat path; the one whose
    # jitter, shimmer, HNR and CPPS come closest to the recordings' is kept.
    from christman_sound.stitch import FrameVoice, StitchRefused
    from christman_sound.synth import synthesize

    def speak_all(tmp: Path, ap: float, jc: float, sc: float, tag: str) -> dict:
        m = {**measures, "stitched_aperiodicity": ap, "stitched_jitter_carried": jc, "stitched_shimmer_carried": sc}
        voice = FrameVoice(name=args.name, path=out, bank=bank, measures=m, spoken_as=spoken_as)
        got = {"jitter": [], "shimmer": [], "hnr_db": [], "cpp_db": []}
        for i, sentence in enumerate(CALIBRATION_SENTENCES):
            try:
                made = synthesize(sentence, voice=voice, lexicon=lex)
            except StitchRefused as exc:
                sys.exit(f"stitched calibration: the voice cannot say {sentence!r}: {exc}")
            wav = tmp / f"{tag}_{i}.wav"
            made.write_wav(wav)
            r = _quality(str(wav))
            for k in got:
                if r.get(k) is not None:
                    got[k].append(r[k])
        return {k: _median(v) for k, v in got.items()}

    stitched_table = []
    with tempfile.TemporaryDirectory() as tmp:
        for ap in STITCHED_APERIODICITIES:
            # carried: told the frames already carry the recordings' values, the mouth adds nothing
            carried = speak_all(Path(tmp), ap, target["jitter"], target["shimmer"], f"sc{ap}")
            got = speak_all(Path(tmp), ap, carried["jitter"], carried["shimmer"], f"sr{ap}")
            score = sum(abs(got[k] - target[k]) / spread[k] for k in got)
            stitched_table.append({"aperiodicity": ap, "carried": carried, "rendered": got, "score": score})
            print(f"stitched calibration {ap}: score {score:.3f} " + " ".join(f"{k} {v:.4f}" for k, v in got.items()),
                  flush=True)
    sbest = min(stitched_table, key=lambda r: r["score"])
    measures["stitched_aperiodicity"] = sbest["aperiodicity"]
    measures["stitched_jitter_carried"] = sbest["carried"]["jitter"]
    measures["stitched_shimmer_carried"] = sbest["carried"]["shimmer"]
    cw = Path(args.codecwar).expanduser()
    cw_commit = subprocess.run(["git", "-C", str(cw), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc = {
        "name": args.name,
        "measures": measures,
        "spoken_as": spoken_as,
        "provenance": {
            "built": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "builder": "Christman-Sound scripts/build_voice_bank.py",
            "codecwar": {"path": str(cw), "commit": cw_commit or "unknown"},
            "vosk_model": str(Path(args.vosk_model).expanduser()),
            "lexicon": str(Path(args.lexicon).expanduser()),
            "min_conf": args.min_conf,
            "profile_from": sorted(str(p) for p in prof),
            "measured_by": {
                "f0_median_hz, phrase_range_st, syllable_rate, voiced_aperiodicity, ltas_db, level_db":
                    "the profile recordings' frames (Codec War frames.ts) and units (bank.ts)",
                "jitter, shimmer, hnr_db": "Praat via christman_voice_sdk.tone.voice_measure.voice_quality "
                                           "(jitter_local_pct and shimmer_local_pct / 100), median over the "
                                           "profile recordings",
                "cpp_db": "Praat CPPS via christman_voice_sdk.timbre.timbre_modeler.measure_cpp",
                "jitter_carried, shimmer_carried": "the same Praat measures on each profile recording rendered from "
                                                   "its own frames by christman_vocoder with none added, at the "
                                                   "calibrated aperiodicity",
                "voiced_aperiodicity": "the frames' mean aperiodicity below 4 kHz in voiced frames, times the "
                                       "calibration scale (see aperiodicity)",
                "stitched_aperiodicity, stitched_jitter_carried, stitched_shimmer_carried":
                    "the stitched calibration: the calibration sentences spoken by the mouth, measured by the same "
                    "Praat path (see stitched_calibration)",
            },
            "per_recording": {"jitter": quality["jitter"], "shimmer": quality["shimmer"], "hnr_db": quality["hnr_db"],
                              "cpp_db": quality["cpp_db"], "phrase_range_st": phrase_ranges},
            "aperiodicity": {"frames_mean": raw_ap, "calibration_scale": best["scale"],
                             "calibration": table,
                             "why": "the frames' aperiodicity also holds the recordings' jitter and shimmer; rendered "
                                    "as measured the voice comes back noisier than the recordings. The scale whose "
                                    "renderings measure closest to the recordings (jitter, shimmer, HNR, CPPS, by Praat) "
                                    "is kept."},
            "stitched_calibration": {"sentences": list(CALIBRATION_SENTENCES), "table": stitched_table,
                                     "kept": sbest["aperiodicity"],
                                     "why": "joins, time scaling and range scaling make stitched speech rougher and "
                                            "noisier than a recording rendered whole; the aperiodicity whose spoken "
                                            "calibration sentences measure closest to the recordings (jitter, "
                                            "shimmer, HNR, CPPS, by Praat) is kept, with what those sentences carry "
                                            "before any jitter or shimmer is added."},
            "sources": sources,
        },
    }
    # Written beside and renamed over, so a mouth loading the voice never reads half a file.
    tmp_voice = out / "voice.json.tmp"
    tmp_voice.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp_voice, out / "voice.json")
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else "96 points") for k, v in measures.items()}))
    print(f"voice {args.name}: {out} ({size / 1e6:.1f} MB, {len(bank.units)} units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
