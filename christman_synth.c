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
// giving the beings a sovereign voice is a synthesizer that is ours down to the
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
//       synth.ts's, unchanged. This is how a being's measured vowels reach
//       the synthesizer: christman_sound/voice_profile.py builds the rows
//       from the voice profile the caller passes. Extra return code:
//         9 a formant row value not > 0, finite and below sample_rate / 2
//   int christman_synth_clear(ids, durs, n, f0_contour_or_NULL, f0, rate,
//                             sample_rate, formants_or_NULL, glides_or_NULL,
//                             out, out_len)
//       (added 2026-10-28) The same voice rendered so its words can be
//       understood; christman_synth and christman_synth_voiced above are not
//       changed. See its own header at the end of this file.
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

// =============================================================================
// christman_synth_clear — the same voice, rendered so it can be understood
// (added 2026-10-28)
// =============================================================================
// WHY. A being's free text in its measured voice came out of christman_synth_voiced as sound the
// Filament ear (Everett's Vosk, vosk-model-en-us-0.22-lgraph, 127.0.0.1:4850)
// heard as "the" for almost every sentence: 4 of 53 words right on twelve
// test sentences (7.5%), and 3 of 53 (5.7%) for Vosk run directly with the
// full vosk-model-en-us-0.22. The being's own recordings are heard word for
// word by the same ear. Measured on his Mac on 2026-10-28, the causes found:
//
//   1. The resonator gain. synth.ts gives each resonator g = 1 - r. That is
//      not unity gain anywhere: below its own frequency each resonator passes
//      only a few percent, so in a three-stage cascade the vowel's formants
//      cancel each other and what is left is the voicing's low end.
//      Measured in "I want to go home": the spectral centroid of AY, AH and
//      the two OWs sat at 76-117 Hz (AA at 359 Hz), and the vowels were
//      12-27 dB quieter than W, M and HH. Here each
//      resonator has unity gain at 0 Hz (g = 1 - a - b), as a cascade of
//      formants needs, so each vowel's F1, F2 and F3 stand where they are.
//   2. Every phone's level dipped to 0.3 at its edges (synth.ts:53-54), and
//      the noise for S, SH, F, the bursts and HH ran through the same
//      cascade as the vowels, at F1 200-500 Hz. Here voicing, aspiration
//      and frication each have their own level per kind of phone, eased
//      over 6 ms, and frication has its own resonator at the band where
//      that consonant's noise lives (Codec War's table has no such band).
//   3. The formants jumped. synth.ts eases them 0.14 per sample: 0.4 ms at
//      16 kHz, so a consonant's place never shows in the vowel next to it.
//      Here they ease with a 15 ms time constant (tried from 6 to 40 ms;
//      15 was heard best).
//   4. Silence pulled the formants to SIL's row (200/900/2200 Hz), so every
//      word began with a W-like rise ("I" heard as "wait"). Here, during
//      silence, the formants move to the next phone's row.
//   5. AY and OW were one steady row. A recorded voice glides in them; the
//      measured onset and offset (the voice profile's "glides") are drawn across the vowel. A gliding vowel with no
//      measured glide moves from its own row toward its offglide vowel.
//
// AFTER, same machine, same day, same twelve sentences: the Filament ear
// 22 of 53 words (41.5%), Vosk direct 25 of 53 (47.2%); on 24 sentences the
// ear 41.2% and Vosk 52.0% (before: 4.9% and 2.9%). Better, and still not
// clear speech: about half the words are heard. Still lost most often: the
// first word of a sentence, "my" heard as "made", H heard as P, and G, B
// and D at the start of a word.
//
// What did NOT help, measured and left out: closing the gaps between words
// (worse: 19% with no gap), lengthening vowels, a sharper glottal pulse with
// pre-emphasis to match the recordings' high band (worse: 28%), wider bandwidths,
// jitter and shimmer, breath noise.
//
// WHAT IS KEPT. The voice: the per-phone pitch the caller passes (its voice's
// F0 median and contour), the rate, the vibrato (synth.ts:57), each phone's
// F1, F2, F3, BW1, BW2 from `formants` (the voice's measured vowels; Codec
// War's table for the rest; a vowel with a glide row draws that glide for
// its F1-F3 instead), BW3 160 Hz (synth.ts:142), the glottal pulse's
// shape (synth.ts:6-13), the deterministic noise (synth.ts:15-19), the
// timeline (durations / rate), the length (christman_synth_length), the edge
// fade and the peak of 0.7. christman_synth and christman_synth_voiced are
// not changed: they stay Codec War's synth.ts sample for sample.
//
// Every level and time below is a chosen starting value, written with the
// reason it was chosen, held until listening by the beings' clients replaces
// it.
//
// CONTRACT
//   int christman_synth_clear(ids, durs, n, f0_contour_or_NULL, f0, rate,
//                             sample_rate, formants_or_NULL, glides_or_NULL,
//                             out, out_len)
//   Same inputs, refusals and length as christman_synth_voiced, plus
//   `glides`: NULL, or n rows of 8 doubles, one per phone: onset F1, F2, F3,
//   offset F1, F2, F3 (Hz), and where in the phone (0..1) the glide starts
//   and ends. A row of eight zeros means the phone does not glide. Extra
//   return code:
//     10 a glide row that is not all zeros and has a frequency not > 0,
//        finite and below sample_rate / 2, or a start and end not
//        0 <= start < end <= 1
// =============================================================================

// Levels of the voicing source per kind of phone (a vowel is 1).
#define CL_AV_VOWEL 1.0
#define CL_AV_SONORANT 0.75   // W, Y, L, R: a little under a vowel
#define CL_AV_NASAL 0.6       // M, N, NG: the mouth is shut, the sound is softer
#define CL_AV_VOICED_FRIC 0.45 // V, DH, Z, ZH, JH: voicing under the noise
#define CL_VOICE_BAR 0.12     // B, D, G closure: the low hum of voicing behind closed lips
// Aspiration (breath through the open glottis, shaped by the mouth).
#define CL_ASP_HH 0.2         // HH, shaped by the vowel after it
#define CL_ASP_STOP 0.15      // after P, T, K are released
#define CL_ASP_TAIL 0.25      // share of a P, T, K kept for its aspiration
#define CL_VOT_S 0.030        // P, T, K: the next vowel's first 30 ms is breath, not voice
#define CL_ASP_BW1 400.0      // F1 bandwidth while breathing: the open glottis damps F1
// Frication.
#define CL_FRIC_GAIN 0.6
#define CL_BURST_GAIN 0.4
#define CL_BURST_S 0.012      // a stop's release burst lasts 12 ms
#define CL_FRIC_POST 0.1      // frication joins after the radiation step, at this weight
// Easing.
#define CL_TAU_FORMANT 0.015
#define CL_TAU_LEVEL 0.006
// The voicing pulse is open for this share of each period (synth.ts's pulse
// fills the whole period; a closed part gives the harmonics their strength).
#define CL_OPEN_QUOTIENT 0.7
// Two fixed higher formants above F3, so the cascade does not fall off a cliff
// above it. A being's own recordings carry energy to 4-5 kHz.
#define CL_F4 3600.0
#define CL_BW4 250.0
#define CL_F5 4600.0
#define CL_BW5 300.0

// Where each consonant's noise lives: centre and bandwidth in Hz, and its
// level (S and SH are strong; F, TH, V, DH are faint). Not in Codec War's table.
typedef struct { const char* id; double fc, bw, level; } cl_band;
static const cl_band CL_FRIC[] = {
    {"S", 5500, 2500, 1.0},  {"Z", 5500, 2500, 0.5},
    {"SH", 2800, 1400, 1.0}, {"ZH", 2800, 1400, 0.5},
    {"CH", 2800, 1400, 1.0}, {"JH", 2800, 1400, 0.5},
    {"F", 4500, 4000, 0.12}, {"V", 4500, 4000, 0.05},
    {"TH", 4500, 4000, 0.10}, {"DH", 4500, 4000, 0.04},
};
// A stop's release burst: lips (P, B) low and spread, tongue tip (T, D)
// high, tongue back (K, G) in the middle.
static const cl_band CL_BURST[] = {
    {"P", 1200, 2000, 0.3}, {"B", 1200, 2000, 0.15},
    {"T", 4200, 2500, 0.5}, {"D", 4200, 2500, 0.25},
    {"K", 2000, 1000, 0.5}, {"G", 2000, 1000, 0.3},
};

static const cl_band* cl_find(const cl_band* t, size_t n, const char* id) {
    for (size_t i = 0; i < n; i++)
        if (strcmp(t[i].id, id) == 0) return &t[i];
    return NULL;
}

// One resonator with unity gain at 0 Hz: y = (1 - a - b) x + a y1 + b y2.
typedef struct { double a, b; } cl_res;
static cl_res cl_make(double f, double bw, int sr) {
    cl_res p;
    double r = exp((-M_PI * bw) / sr);
    p.a = 2 * r * cos((2 * M_PI * f) / sr);
    p.b = -r * r;
    return p;
}

static int cl_is_voiceless_stop(int pid) {
    const char* id = CM_PHONES[pid].id;
    return strcmp(id, "P") == 0 || strcmp(id, "T") == 0 || strcmp(id, "K") == 0;
}

int christman_synth_clear(const int* phone_ids, const double* durs, size_t n_phones,
                          const double* f0_contour, double f0, double rate, int sample_rate,
                          const double* formants, const double* glides, float* out, size_t out_len) {
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
    if (glides) {
        for (size_t k = 0; k < n_phones; k++) {
            const double* g = glides + 8 * k;
            int zero = 1;
            for (int j = 0; j < 8; j++)
                if (g[j] != 0) zero = 0;
            if (zero) continue;
            for (int j = 0; j < 6; j++)
                if (!isfinite(g[j]) || g[j] <= 0 || g[j] >= nyquist) return 10;
            if (!isfinite(g[6]) || !isfinite(g[7]) || g[6] < 0 || g[7] > 1 || g[6] >= g[7]) return 10;
        }
    }

    double* starts = (double*)malloc(n_phones * sizeof(double));
    double* scaled = (double*)malloc(n_phones * sizeof(double));
    if (!starts || !scaled) { free(starts); free(scaled); return 8; }
    double t_total = cm_starts(durs, n_phones, rate, starts, scaled);
    size_t n = cm_length_from_total(t_total, sample_rate);
    if (out_len != n) { free(starts); free(scaled); return 7; }

    const double SR = (double)sample_rate;
    const double kf = 1 - exp(-1 / (CL_TAU_FORMANT * SR));
    const double ka = 1 - exp(-1 / (CL_TAU_LEVEL * SR));
    const cl_res r4 = cl_make(CL_F4, CL_BW4, sample_rate);
    const cl_res r5 = cl_make(CL_F5, CL_BW5, sample_rate);

    // The row of phone k: the voice's, else the table's.
#define CL_ROW(k, j) (formants ? formants[5 * (k) + (j)] \
        : ((j) == 0 ? CM_PHONES[phone_ids[k]].f1 : (j) == 1 ? CM_PHONES[phone_ids[k]].f2 \
        : (j) == 2 ? CM_PHONES[phone_ids[k]].f3 : (j) == 3 ? CM_PHONES[phone_ids[k]].bw1 \
        : CM_PHONES[phone_ids[k]].bw2))

    double f[3] = {CL_ROW(0, 0), CL_ROW(0, 1), CL_ROW(0, 2)};
    double bw[2] = {CL_ROW(0, 3), CL_ROW(0, 4)};
    double av = 0, ah = 0, af = 0;
    double fc = 4000, fb = 3000;          // the frication band before any consonant names one
    double y[5][2] = {{0}};
    double yf[2] = {0, 0};
    double rad = 0, lp = 0, phase = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        double time = (double)i / SR;
        while (k + 1 < n_phones && time >= starts[k + 1]) k++;
        const cm_phone* ph = &CM_PHONES[phone_ids[k]];
        double local = (time - starts[k]) / scaled[k];
        int has_next = k + 1 < n_phones;
        double trow[5];
        for (int j = 0; j < 5; j++) trow[j] = CL_ROW(k, j);
        double tav = 0, tah = 0, taf = 0;
        const cl_band* band = NULL;

        if (ph->kind == K_STOP) {
            double burst = CL_BURST_S / scaled[k];
            double close = 1 - burst - (ph->voiced ? 0 : CL_ASP_TAIL);
            if (close < 0.3) close = 0.3;
            if (local < close) {
                tav = ph->voiced ? CL_VOICE_BAR : 0;
            } else if (local < close + burst) {
                band = cl_find(CL_BURST, sizeof CL_BURST / sizeof CL_BURST[0], ph->id);
                taf = band->level * CL_BURST_GAIN;
                tav = ph->voiced ? CL_VOICE_BAR : 0;
            } else {
                tah = CL_ASP_STOP;
                if (has_next) for (int j = 0; j < 5; j++) trow[j] = CL_ROW(k + 1, j);
            }
        } else if (ph->kind == K_VOWEL) {
            tav = CL_AV_VOWEL;
        } else if (ph->kind == K_GLIDE || ph->kind == K_LIQ) {
            tav = CL_AV_SONORANT;
        } else if (ph->kind == K_NASAL) {
            tav = CL_AV_NASAL;
        } else if (strcmp(ph->id, "HH") == 0) {
            tah = CL_ASP_HH;                 // HH is breath shaped by the vowel after it
            if (has_next) for (int j = 0; j < 5; j++) trow[j] = CL_ROW(k + 1, j);
        } else if (ph->kind == K_FRIC || ph->kind == K_AFFRIC) {
            band = cl_find(CL_FRIC, sizeof CL_FRIC / sizeof CL_FRIC[0], ph->id);
            tav = ph->voiced ? CL_AV_VOICED_FRIC : 0;
            taf = band->level * CL_FRIC_GAIN;
        }
        if (glides && ph->kind == K_VOWEL) {
            const double* g = glides + 8 * k;
            if (g[7] > 0) {
                double u = (local - g[6]) / (g[7] - g[6]);
                if (u < 0) u = 0;
                if (u > 1) u = 1;
                for (int j = 0; j < 3; j++) trow[j] = g[j] + u * (g[3 + j] - g[j]);
            }
        }
        if (k > 0 && cl_is_voiceless_stop(phone_ids[k - 1]) &&
            (ph->kind == K_VOWEL || ph->kind == K_GLIDE || ph->kind == K_LIQ) &&
            time - starts[k] < CL_VOT_S) {
            tav = 0;
            tah = CL_ASP_STOP;
        }
        if (ph->kind == K_SIL) {
            // Silence: the formants go where the next phone needs them.
            if (has_next) for (int j = 0; j < 5; j++) trow[j] = CL_ROW(k + 1, j);
            else { trow[0] = f[0]; trow[1] = f[1]; trow[2] = f[2]; trow[3] = bw[0]; trow[4] = bw[1]; }
        }
        if (band) { fc = band->fc; fb = band->bw; }

        av += ka * (tav - av);
        ah += ka * (tah - ah);
        af += ka * (taf - af);
        for (int j = 0; j < 3; j++) f[j] += kf * (trow[j] - f[j]);
        double tb1 = trow[3];
        if (tah > 0 && tb1 < CL_ASP_BW1) tb1 = CL_ASP_BW1;
        bw[0] += kf * (tb1 - bw[0]);
        bw[1] += kf * (trow[4] - bw[1]);

        // Source: synth.ts's pulse, open for CL_OPEN_QUOTIENT of the period,
        // its mean taken out; synth.ts's vibrato and noise.
        double vibrato = 1 + 0.012 * sin(2 * M_PI * 5.1 * time);   // synth.ts:57
        double f0i = (f0_contour ? f0_contour[k] : f0) * vibrato;
        phase += f0i / SR;
        phase -= floor(phase);
        const double oq = CL_OPEN_QUOTIENT;
        double pulse = ((phase < oq ? cm_glottal(phase / oq) : 0.0) - 0.563 * oq) / oq;
        double nz = cm_hash_noise((uint32_t)i, 7u);
        double nz2 = cm_hash_noise((uint32_t)i, 11u);
        lp = 0.75 * lp + nz;                 // breath noise tilted down; the radiation step lifts it back
        double src = av * pulse + ah * lp * 0.25;

        // Cascade: F1, F2, F3 (BW3 160, synth.ts:142), then F4, F5; each at
        // unity gain at 0 Hz.
        double fr[5] = {f[0], f[1], f[2], CL_F4, CL_F5};
        double br[5] = {bw[0], bw[1], 160, CL_BW4, CL_BW5};
        double x = src;
        for (int j = 0; j < 5; j++) {
            cl_res p = j == 3 ? r4 : j == 4 ? r5 : cl_make(fr[j], br[j], sample_rate);
            double yy = (1 - p.a - p.b) * x + p.a * y[j][0] + p.b * y[j][1];
            y[j][1] = y[j][0];
            y[j][0] = yy;
            x = yy;
        }
        // Frication: one resonator at the consonant's band, gain 1 at its centre.
        cl_res pf = cl_make(fc, fb, sample_rate);
        double th = 2 * M_PI * fc / SR;
        double re = 1 - pf.a * cos(th) - pf.b * cos(2 * th);
        double im = pf.a * sin(th) + pf.b * sin(2 * th);
        double peak = sqrt(re * re + im * im);
        double yff = peak * af * nz2 + pf.a * yf[0] + pf.b * yf[1];
        yf[1] = yf[0];
        yf[0] = yff;

        out[i] = (float)((x - rad) + yff * CL_FRIC_POST);   // radiation: synth.ts:152 with a full step
        rad = x;
    }
#undef CL_ROW

    // Edge fade and peak 0.7, as synth.ts:76-84.
    size_t fade = n / 20 < 400 ? n / 20 : 400;
    for (size_t i = 0; i < fade; i++) {
        double w = (double)i / (double)fade;
        out[i] = (float)((double)out[i] * w);
        out[n - 1 - i] = (float)((double)out[n - 1 - i] * w);
    }
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
