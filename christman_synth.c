// =============================================================================
// © 2025 Everett Nathaniel Christman & Misty Gail Christman
// The Christman AI Project — Luma Cognify AI
// Truth. Dignity. Protection. Transparency. No Erasure.
// =============================================================================
// christman_synth.c — the sovereign mouth. Compiled into the same shared
// library as christman_dsp.c by scripts/build_dsp.py.
//
// WHAT CHANGED AND WHY (added 2026-10-17)
// Every voice the family had came from somebody else's engine: XTTS/Coqui,
// GPT-SoVITS, espeak, Apple `say`, pyttsx3, gTTS, the browser. Step 1 of
// giving AlphaVox a sovereign voice is a synthesizer that is ours down to the
// last multiply. This file is a C port of Everett's Codec War synthesizer
// (CODECWAR5TH, src/lib/audio/synth.ts + phonemes.ts): a Klatt-lite
// source-filter synthesizer — glottal pulse plus noise source, a 40-phone
// table of F1/F2/F3 and bandwidths, a three-formant cascade, vibrato, rate.
// Nothing below this file but the C library's math.
//
// FAITHFULNESS. The phone table values are Codec War's, unchanged; every row
// cites the phonemes.ts line it came from. Every constant in the synthesis
// cites its synth.ts line. All arithmetic is done in double, and the sample
// buffer is float, at exactly the points where synth.ts stores into its
// Float32Array, so the output matches the original sample for sample at
// 16 kHz (checked against the TypeScript run under node — see
// tests/test_sovereign_synth.py). The build compiles this file with
// -ffp-contract=off so no compiler fuses a multiply-add the original does not.
//
// WHAT IS DIFFERENT FROM synth.ts, AND WHY — each one is a refusal, never a
// substitute:
//   * synth.ts maps an unknown phone id to SIL (`PHONE_INDEX[id] ?? 0`). A
//     phone we do not know becomes silence the client never asked for. Here
//     an unknown id is refused (code 2).
//   * synth.ts clamps rate into [0.6, 1.6] (synth.ts:104) without saying so.
//     Speech at a rate other than the one asked for is a plausible default.
//     Here a rate outside [0.6, 1.6] is refused (code 5). Inside the range the
//     arithmetic is synth.ts's: every duration is divided by rate (synth.ts:105).
//   * synth.ts tolerates a zero duration (Math.max(dur, 1e-4)). Here a
//     duration must be > 0 and finite (code 3).
//   * Sample rate is a parameter (Codec War hard-codes 16000, dsp.ts:3). Only
//     16000 (Codec War's, the default in Python) and 22050 (christman_sound's
//     standard) are accepted (code 6). The per-sample constants of synth.ts —
//     formant smoothing 0.14/0.1 per sample, the 400-sample fade, the
//     per-sample noise index — are kept as written in samples, so at 22050
//     formant transitions and fades are 16000/22050 as long in time as at
//     16000. Scaling them would be a change to Codec War; that is Everett's call.
//   * The one parameter Codec War does not have: f0 may be given per phone
//     (a pitch contour hook). With no contour, the single f0 is used exactly
//     as synth.ts uses it.
//
// CONTRACT
//   size_t christman_synth_length(durs, n, rate, sample_rate)
//       Number of samples christman_synth will write for these durations,
//       or 0 if the arguments would be refused.
//   int christman_synth(ids, durs, n, f0_contour_or_NULL, f0, rate,
//                       sample_rate, out, out_len)
//       Writes exactly out_len float samples, peak-normalised to 0.7 as in
//       synth.ts:81-84. Returns 0 on success; nonzero means NOTHING usable
//       was written:
//         1 bad arguments (NULL pointer, zero phones)
//         2 unknown phone id
//         3 duration not > 0 and finite
//         4 f0 not > 0, finite and below sample_rate / 2
//         5 rate outside [0.6, 1.6]
//         6 sample rate not 16000 or 22050
//         7 out_len does not equal christman_synth_length(...)
//         8 allocation failed
//   int christman_synth_voiced(ids, durs, n, f0_contour_or_NULL, f0, rate,
//                              sample_rate, formants_or_NULL, out, out_len)
//       (added 2026-10-28) The same synthesis, with one more input: a voice's
//       own resonances. `formants` is NULL (then this is exactly
//       christman_synth) or n_phones rows of 5 doubles, one row per phone in
//       the sequence: F1, F2, F3, BW1, BW2 in Hz. The cascade steers toward
//       the row's values instead of the table's. Everything else (source,
//       envelopes, durations, the fixed BW3 of 160 Hz) is the table's and
//       synth.ts's, unchanged. This is how AlphaVox's measured vowels reach
//       the synthesizer: christman_sound/voice_profile.py builds the rows
//       from christman_sound/alphavox_voice.json. Extra return code:
//         9 a formant row value not > 0, finite and below sample_rate / 2
//   int christman_synth_phone_count(void)            -> 40
//   const char* christman_synth_phone_name(int id)   -> "AA" etc, NULL if bad id
//   int christman_synth_phone_id(const char* name)   -> id, or -1 if unknown
//   double christman_synth_phone_dur(int id)         -> table duration (s), -1 if bad id
// =============================================================================

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Phone kinds, phonemes.ts:7.
enum cm_kind { K_SIL, K_VOWEL, K_GLIDE, K_NASAL, K_STOP, K_FRIC, K_AFFRIC, K_LIQ };

typedef struct {
    const char* id;
    int kind;
    double f1, f2, f3, bw1, bw2;
    int voiced;
    double dur;
} cm_phone;

// The phone table — Codec War's PHONES, phonemes.ts:17-58, values unchanged.
// Order is Codec War's order, so the index here is Codec War's PHONE_INDEX.
static const cm_phone CM_PHONES[] = {
    /* phonemes.ts:18 */ {"SIL", K_SIL,    200,  900, 2200,  90, 120, 0, 0.06},
    /* phonemes.ts:19 */ {"AA",  K_VOWEL,  730, 1090, 2440,  70,  90, 1, 0.1},
    /* phonemes.ts:20 */ {"AE",  K_VOWEL,  660, 1720, 2410,  70, 100, 1, 0.1},
    /* phonemes.ts:21 */ {"AH",  K_VOWEL,  640, 1190, 2390,  80,  90, 1, 0.07},
    /* phonemes.ts:22 */ {"AO",  K_VOWEL,  570,  840, 2410,  70,  80, 1, 0.1},
    /* phonemes.ts:23 */ {"AW",  K_VOWEL,  570,  980, 2300,  80,  90, 1, 0.14},
    /* phonemes.ts:24 */ {"AY",  K_VOWEL,  660, 1720, 2410,  80, 100, 1, 0.14},
    /* phonemes.ts:25 */ {"EH",  K_VOWEL,  530, 1840, 2480,  70,  90, 1, 0.08},
    /* phonemes.ts:26 */ {"ER",  K_VOWEL,  490, 1350, 1690,  80,  90, 1, 0.12},
    /* phonemes.ts:27 */ {"EY",  K_VOWEL,  400, 2000, 2550,  60,  90, 1, 0.13},
    /* phonemes.ts:28 */ {"IH",  K_VOWEL,  390, 1990, 2550,  50,  90, 1, 0.07},
    /* phonemes.ts:29 */ {"IY",  K_VOWEL,  270, 2290, 3010,  50,  90, 1, 0.1},
    /* phonemes.ts:30 */ {"OW",  K_VOWEL,  430,  980, 2300,  70,  80, 1, 0.13},
    /* phonemes.ts:31 */ {"OY",  K_VOWEL,  550,  920, 2400,  70,  90, 1, 0.14},
    /* phonemes.ts:32 */ {"UH",  K_VOWEL,  440, 1020, 2240,  70,  80, 1, 0.07},
    /* phonemes.ts:33 */ {"UW",  K_VOWEL,  300,  870, 2240,  60,  80, 1, 0.1},
    /* phonemes.ts:34 */ {"B",   K_STOP,   200,  720, 2400,  60,  90, 1, 0.06},
    /* phonemes.ts:35 */ {"D",   K_STOP,   200, 1700, 2600,  60, 100, 1, 0.05},
    /* phonemes.ts:36 */ {"G",   K_STOP,   200, 1400, 2200,  80, 120, 1, 0.06},
    /* phonemes.ts:37 */ {"P",   K_STOP,   200,  720, 2400,  80, 120, 0, 0.06},
    /* phonemes.ts:38 */ {"T",   K_STOP,   200, 1700, 2600,  80, 140, 0, 0.05},
    /* phonemes.ts:39 */ {"K",   K_STOP,   200, 1400, 2200, 100, 140, 0, 0.06},
    /* phonemes.ts:40 */ {"F",   K_FRIC,   400, 1100, 3800, 200, 400, 0, 0.09},
    /* phonemes.ts:41 */ {"V",   K_FRIC,   300, 1100, 2800, 120, 200, 1, 0.07},
    /* phonemes.ts:42 */ {"TH",  K_FRIC,   400, 1400, 3600, 180, 300, 0, 0.08},
    /* phonemes.ts:43 */ {"DH",  K_FRIC,   300, 1400, 2700, 120, 180, 1, 0.06},
    /* phonemes.ts:44 */ {"S",   K_FRIC,   500, 1800, 7000, 200, 400, 0, 0.09},
    /* phonemes.ts:45 */ {"Z",   K_FRIC,   300, 1700, 5000, 140, 250, 1, 0.07},
    /* phonemes.ts:46 */ {"SH",  K_FRIC,   400, 1800, 3500, 180, 280, 0, 0.1},
    /* phonemes.ts:47 */ {"ZH",  K_FRIC,   350, 1800, 3200, 140, 220, 1, 0.08},
    /* phonemes.ts:48 */ {"HH",  K_FRIC,   500, 1500, 2500, 300, 400, 0, 0.06},
    /* phonemes.ts:49 */ {"CH",  K_AFFRIC, 300, 1800, 2800, 120, 200, 0, 0.09},
    /* phonemes.ts:50 */ {"JH",  K_AFFRIC, 300, 1800, 2700, 100, 180, 1, 0.08},
    /* phonemes.ts:51 */ {"M",   K_NASAL,  250, 1000, 2200,  60,  80, 1, 0.07},
    /* phonemes.ts:52 */ {"N",   K_NASAL,  250, 1600, 2500,  60,  80, 1, 0.07},
    /* phonemes.ts:53 */ {"NG",  K_NASAL,  250, 1200, 2300,  70,  90, 1, 0.08},
    /* phonemes.ts:54 */ {"L",   K_LIQ,    360, 1200, 2600,  50,  80, 1, 0.07},
    /* phonemes.ts:55 */ {"R",   K_LIQ,    400, 1100, 1600,  70,  80, 1, 0.07},
    /* phonemes.ts:56 */ {"W",   K_GLIDE,  300,  700, 2200,  60,  80, 1, 0.07},
    /* phonemes.ts:57 */ {"Y",   K_GLIDE,  300, 2100, 3000,  50,  90, 1, 0.07},
};
#define CM_N_PHONES ((int)(sizeof CM_PHONES / sizeof CM_PHONES[0]))

int christman_synth_phone_count(void) { return CM_N_PHONES; }

const char* christman_synth_phone_name(int id) {
    return (id >= 0 && id < CM_N_PHONES) ? CM_PHONES[id].id : NULL;
}

int christman_synth_phone_id(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < CM_N_PHONES; i++)
        if (strcmp(CM_PHONES[i].id, name) == 0) return i;
    return -1;
}

double christman_synth_phone_dur(int id) {
    return (id >= 0 && id < CM_N_PHONES) ? CM_PHONES[id].dur : -1.0;
}

// (added 2026-10-28) The table row's F1, F2, F3, BW1, BW2 into out[0..4], so
// a voice profile can start from Codec War's values and replace only what it
// measured. Returns 0, or -1 for a bad id or a NULL out (nothing written).
int christman_synth_phone_formants(int id, double* out) {
    if (!out || id < 0 || id >= CM_N_PHONES) return -1;
    out[0] = CM_PHONES[id].f1;
    out[1] = CM_PHONES[id].f2;
    out[2] = CM_PHONES[id].f3;
    out[3] = CM_PHONES[id].bw1;
    out[4] = CM_PHONES[id].bw2;
    return 0;
}

// (added 2026-10-28) 1 when the phone is a vowel in Codec War's kinds
// (phonemes.ts:7), 0 when not, -1 for a bad id.
int christman_synth_phone_is_vowel(int id) {
    if (id < 0 || id >= CM_N_PHONES) return -1;
    return CM_PHONES[id].kind == K_VOWEL ? 1 : 0;
}

// Glottal pulse, synth.ts:6-13: a smoothstep rise over 62% of the period,
// then a parabolic fall.
static double cm_glottal(double phase) {
    if (phase < 0.62) {
        double x = phase / 0.62;
        return 3 * x * x - 2 * x * x * x;
    }
    double x = (phase - 0.62) / 0.38;
    return 1 - x * x;
}

// Deterministic noise, synth.ts:15-19. Math.imul is a 32-bit wrapping
// multiply and `>>>` a logical shift, which is exactly uint32_t arithmetic.
static double cm_hash_noise(uint32_t i, uint32_t seed) {
    uint32_t x = (i + 1u) * 374761393u ^ (seed + 1u) * 668265263u;
    x = (x ^ (x >> 13)) * 1274126177u;
    return ((double)x / 4294967295.0) * 2 - 1;
}

static int cm_check_rate(double rate) {
    // synth.ts:104 range; refused rather than clamped (see header).
    return isfinite(rate) && rate >= 0.6 && rate <= 1.6;
}

static int cm_check_sr(int sample_rate) { return sample_rate == 16000 || sample_rate == 22050; }

// Phone start times after the rate division, synth.ts:105 (dur / rate) and
// synth.ts:33-37 (running sum). Returns the total duration t, or -1 on a bad
// duration.
static double cm_starts(const double* durs, size_t n, double rate, double* starts, double* scaled) {
    double t = 0;
    for (size_t k = 0; k < n; k++) {
        if (!isfinite(durs[k]) || durs[k] <= 0) return -1;
        double d = durs[k] / rate;
        if (starts) starts[k] = t;
        if (scaled) scaled[k] = d;
        t += d;
    }
    return t;
}

// synth.ts:38 — n = max(1, floor(t*SR) + floor(SR/12)).
static size_t cm_length_from_total(double t, int sample_rate) {
    double n = floor(t * sample_rate) + floor(sample_rate / 12.0);
    return n < 1 ? 1 : (size_t)n;
}

size_t christman_synth_length(const double* durs, size_t n_phones, double rate, int sample_rate) {
    if (!durs || n_phones == 0 || !cm_check_rate(rate) || !cm_check_sr(sample_rate)) return 0;
    double t = cm_starts(durs, n_phones, rate, NULL, NULL);
    if (t < 0) return 0;
    return cm_length_from_total(t, sample_rate);
}

typedef struct { double a, b, g; } cm_pole;

// One resonator's coefficients, synth.ts:122-129.
static cm_pole cm_make_pole(double freq, double bw, int sr) {
    cm_pole p;
    double r = exp((-M_PI * bw) / sr);
    p.a = 2 * r * cos((2 * M_PI * freq) / sr);
    p.b = -r * r;
    p.g = 1 - r;
    return p;
}

int christman_synth_voiced(const int* phone_ids, const double* durs, size_t n_phones,
                           const double* f0_contour, double f0, double rate, int sample_rate,
                           const double* formants, float* out, size_t out_len);

int christman_synth(const int* phone_ids, const double* durs, size_t n_phones,
                    const double* f0_contour, double f0, double rate, int sample_rate,
                    float* out, size_t out_len) {
    // Codec War's own voice: the table's formants, sample for sample synth.ts.
    return christman_synth_voiced(phone_ids, durs, n_phones, f0_contour, f0, rate,
                                  sample_rate, NULL, out, out_len);
}

int christman_synth_voiced(const int* phone_ids, const double* durs, size_t n_phones,
                           const double* f0_contour, double f0, double rate, int sample_rate,
                           const double* formants, float* out, size_t out_len) {
    if (!phone_ids || !durs || !out || n_phones == 0) return 1;
    for (size_t k = 0; k < n_phones; k++)
        if (phone_ids[k] < 0 || phone_ids[k] >= CM_N_PHONES) return 2;
    for (size_t k = 0; k < n_phones; k++)
        if (!isfinite(durs[k]) || durs[k] <= 0) return 3;
    if (!cm_check_sr(sample_rate)) return 6;
    const double nyquist = sample_rate / 2.0;
    if (f0_contour) {
        for (size_t k = 0; k < n_phones; k++)
            if (!isfinite(f0_contour[k]) || f0_contour[k] <= 0 || f0_contour[k] >= nyquist) return 4;
    } else if (!isfinite(f0) || f0 <= 0 || f0 >= nyquist) {
        return 4;
    }
    if (!cm_check_rate(rate)) return 5;
    if (formants) {
        for (size_t k = 0; k < n_phones * 5; k++)
            if (!isfinite(formants[k]) || formants[k] <= 0 || formants[k] >= nyquist) return 9;
    }

    double* starts = (double*)malloc(n_phones * sizeof(double));
    double* scaled = (double*)malloc(n_phones * sizeof(double));
    if (!starts || !scaled) { free(starts); free(scaled); return 8; }
    double t = cm_starts(durs, n_phones, rate, starts, scaled);
    size_t n = cm_length_from_total(t, sample_rate);
    if (out_len != n) { free(starts); free(scaled); return 7; }

    const double SR = (double)sample_rate;

    // ---- Source, synth.ts:40-72 ------------------------------------------
    // The phone under sample i is the last one whose start <= time
    // (synth.ts:44-45). Starts strictly increase (every duration > 0), so a
    // forward-moving index finds the same phone without rescanning.
    double phase = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        double time = (double)i / SR;
        while (k + 1 < n_phones && time >= starts[k + 1]) k++;
        int pidx = phone_ids[k];
        const cm_phone* proto = &CM_PHONES[pidx];
        double dur = scaled[k] > 1e-4 ? scaled[k] : 1e-4;      // synth.ts:48
        double local = (time - starts[k]) / dur;
        double env;
        if (proto->kind == K_SIL) env = 0;                      // synth.ts:50
        else if (proto->kind == K_STOP) {                       // synth.ts:51-52
            double q = (local - 0.5) * 8;
            env = local < 0.5 ? 0.015 : exp(-(q * q));
        } else {                                                // synth.ts:53-54
            env = 0.3 + 0.7 * sin((local < 1 ? local : 1) * M_PI);
        }

        double vibrato = 1 + 0.012 * sin(2 * M_PI * 5.1 * time);   // synth.ts:57
        double f0_here = f0_contour ? f0_contour[k] : f0;
        double f0i = f0_here * vibrato;                            // synth.ts:58
        phase += (proto->voiced ? f0i : 0) / SR;                    // synth.ts:59
        phase -= floor(phase);                                      // synth.ts:60

        double noise = cm_hash_noise((uint32_t)i, (uint32_t)pidx); // synth.ts:62
        double source;
        if (proto->kind == K_SIL) source = noise * 0.002;          // synth.ts:64
        else if (!proto->voiced) source = noise * (proto->kind == K_FRIC ? 0.38 : 0.24); // :65
        else if (proto->kind == K_FRIC || proto->kind == K_AFFRIC)
            source = cm_glottal(phase) * 0.2 + noise * 0.28;        // synth.ts:66-67
        else
            source = cm_glottal(phase) * 0.72 + noise * 0.025;      // synth.ts:69
        out[i] = (float)(source * env);                             // synth.ts:71
    }

    // ---- Three-formant cascade + radiation, synth.ts:109-156 --------------
    double y1a = 0, y1b = 0, y2a = 0, y2b = 0, y3a = 0, y3b = 0, rad = 0;
    double f1 = 500, f2 = 1500, f3 = 2500, b1 = 80, b2 = 100;       // synth.ts:117-121
    k = 0;
    for (size_t i = 0; i < n; i++) {
        double time = (double)i / SR;
        while (k + 1 < n_phones && time >= starts[k + 1]) k++;
        const cm_phone* proto = &CM_PHONES[phone_ids[k]];
        // Targets: the voice's own row when one is given, else the table's.
        const double* row = formants ? formants + 5 * k : NULL;
        double t1 = row ? row[0] : proto->f1;
        double t2 = row ? row[1] : proto->f2;
        double t3 = row ? row[2] : proto->f3;
        double tb1 = row ? row[3] : proto->bw1;
        double tb2 = row ? row[4] : proto->bw2;
        f1 += 0.14 * (t1 - f1);                                     // synth.ts:135
        f2 += 0.14 * (t2 - f2);                                     // synth.ts:136
        f3 += 0.1 * (t3 - f3);                                      // synth.ts:137
        b1 += 0.1 * (tb1 - b1);                                     // synth.ts:138
        b2 += 0.1 * (tb2 - b2);                                     // synth.ts:139
        cm_pole p1 = cm_make_pole(f1, b1, sample_rate);
        cm_pole p2 = cm_make_pole(f2, b2, sample_rate);
        cm_pole p3 = cm_make_pole(f3, 160, sample_rate);            // synth.ts:142 fixed BW3
        double n1 = (double)out[i] * p1.g + p1.a * y1a + p1.b * y1b; // synth.ts:143
        y1b = y1a; y1a = n1;
        double n2 = n1 * p2.g + p2.a * y2a + p2.b * y2b;             // synth.ts:146
        y2b = y2a; y2a = n2;
        double n3 = n2 * p3.g + p3.a * y3a + p3.b * y3b;             // synth.ts:149
        y3b = y3a; y3a = n3;
        double radiated = n3 - 0.92 * rad;                           // synth.ts:152
        rad = n3;
        out[i] = (float)radiated;                                    // synth.ts:154
    }

    // ---- Edge fade, synth.ts:76-80 ----------------------------------------
    size_t fade = n / 20 < 400 ? n / 20 : 400;
    for (size_t i = 0; i < fade; i++) {
        double w = (double)i / (double)fade;
        out[i] = (float)((double)out[i] * w);
        out[n - 1 - i] = (float)((double)out[n - 1 - i] * w);
    }

    // ---- Peak normalise to 0.7, synth.ts:81-84 ----------------------------
    double peak = 1e-6;
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)out[i]);
        if (a > peak) peak = a;
    }
    double g = 0.7 / peak;
    for (size_t i = 0; i < n; i++) out[i] = (float)((double)out[i] * g);

    free(starts);
    free(scaled);
    return 0;
}
