// =============================================================================
// © 2025 Everett Nathaniel Christman & Misty Gail Christman
// The Christman AI Project — Luma Cognify AI
// Truth. Dignity. Protection. Transparency. No Erasure.
// =============================================================================
// christman_vocoder.c — a voice rendered from its own measured frames.
// The C port of Codec War's frame mouth (CODECWAR5TH src/lib/audio/vocoder.ts
// and the fft of frames.ts, main b148879), step for step, and tested against
// it sample for sample (tests/test_frame_voice.py). Codec War is the canonical
// source; a change is made there first. Compiled into the same shared library
// as christman_dsp.c and christman_synth.c by scripts/build_dsp.py, with
// -ffp-contract=off so no multiply and add are fused where JavaScript does not.
// Nothing below this file but the C library's math.
//
// WHAT THIS IS AND WHY (added 2026-10-28)
// Everett listened to christman_synth_clear and said it was "super robotic and
// nothing like the files I gave", and ruled: "if you add the mathematical
// numbers precisely to the matching voice pattern, there shouldn't be any
// difference." christman_synth_clear is driven by about a dozen numbers per
// vowel. This renderer is driven by the voice's own frames, one every hop
// samples (110 samples at 22050 Hz, about 5 ms), measured from the voice's
// recordings by christman_sound/vocoder.py:
//   * f0       the pitch of the frame in Hz, 0 for an unvoiced frame;
//   * env_db   the full spectral envelope, ne values in dB on a frequency grid
//              (96 points in the first bank built), not a few formants;
//   * ap       the aperiodicity (noise fraction of the power) in nb bands, so
//              the noise in S, SH, F, H, the stop bursts and the breath in the
//              voiced sounds comes from the recordings themselves.
// and by four numbers measured from the same recordings for the whole voice:
//   * jitter   cycle-to-cycle period perturbation (standard deviation as a
//              fraction of the period) added to each glottal cycle;
//   * shimmer  cycle-to-cycle amplitude perturbation, the same way;
// (the voice's breathiness and harmonics-to-noise ratio are in ap; its
// intonation, cadence and resonance are in f0, the timing of the frames and
// env_db. christman_sound/vocoder.py says how each is measured and checks the
// rendered output against the recordings.)
//
// HOW (a source-filter vocoder written here from first principles)
//   Periodic part: one pulse per glottal cycle, placed by accumulating phase
//   from the frame pitch, at fractional sample positions. Each pulse is the
//   minimum-phase impulse response whose power spectrum is
//   envelope x (1 - aperiodicity) at that instant, scaled by sqrt(period in
//   samples), so a pulse train has unit power before the envelope is applied.
//   Aperiodic part: white Gaussian noise in frames of 2 x hop samples under a
//   square-root Hann window (whose squares sum to one at 50% overlap, so the
//   noise power is constant), each filtered by the minimum-phase response
//   whose power spectrum is envelope x aperiodicity at that frame.
//   Envelope units: power per sample, so the two parts add to the envelope.
//   Minimum phase comes from the folded real cepstrum (a log spectrum with
//   all its energy at the start of the response, as a vocal tract has).
//
// CONTRACT
//   size_t christman_vocoder_length(n_frames, hop)  -> n_frames * hop
//   int christman_vocoder_render(n_frames, hop, sample_rate,
//         f0[n_frames], env_db[n_frames * ne], ne, env_hz[ne],
//         ap[n_frames * nb], nb, band_edges_hz[nb + 1],
//         jitter, shimmer, seed, out, out_len)
//     Writes exactly out_len floats (out_len must equal n_frames * hop).
//     Returns 0 on success; nonzero means NOTHING usable was written:
//       1 bad arguments (NULL pointer, zero frames, ne < 2, nb < 1, hop < 1,
//         out_len wrong)
//       2 sample rate outside 8000..24000 (the 1024-point FFT must hold three
//         periods at 70 Hz: 3 x 24000 / 70 = 1029 is the edge, so 24000 is
//         the highest rate this size serves)
//       3 a value that is not finite, a negative pitch, an aperiodicity
//         outside [0, 1], jitter or shimmer outside [0, 0.2], or an env_hz
//         grid that does not rise from 0 to sample_rate / 2
//   The output is NOT normalised: its level is the recordings' level, which
//   is part of the voice. The caller checks it fits in [-1, 1].
//   The same seed gives the same samples, every time.
// =============================================================================

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CV_NFFT 1024
// One cycle's perturbation stays within 20% of the period and 50% of the
// amplitude (vocoder.ts PERIOD_CLAMP, AMP_CLAMP).
#define CV_PERIOD_CLAMP 0.2
#define CV_AMP_CLAMP 0.5
#define CV_HALF (CV_NFFT / 2 + 1)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void cv_fft(double *re, double *im, int inverse) {
    int n = CV_NFFT;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                double xr = re[b] * cr - im[b] * ci;
                double xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
                double nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

// xorshift32, exact in 32-bit integer arithmetic, as vocoder.ts's Rng.
typedef struct { uint32_t s; int has; double spare; } cv_rng;

static void cv_seed(cv_rng *r, unsigned int seed) {
    // (Math.imul(seed >>> 0, 2654435761) ^ 0x9e3779b9) >>> 0 || 0x6d2b79f5
    uint32_t s = ((uint32_t)seed * 2654435761u) ^ 0x9e3779b9u;
    r->s = s ? s : 0x6d2b79f5u;
    r->has = 0;
    r->spare = 0.0;
}

static double cv_uniform(cv_rng *r) {
    uint32_t s = r->s;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    r->s = s;
    return ((double)s + 0.5) / 4294967296.0;
}

static double cv_gauss(cv_rng *r) {
    if (r->has) { r->has = 0; return r->spare; }
    double u = cv_uniform(r), v = cv_uniform(r);
    double m = sqrt(-2.0 * log(u));
    r->spare = m * sin(2.0 * M_PI * v);
    r->has = 1;
    return m * cos(2.0 * M_PI * v);
}

typedef struct {
    int n, hop, sr, ne, nb;
    const double *f0, *env_db, *env_hz, *ap, *edges;
    int grid_j[CV_HALF];      // grid interval each bin falls in
    double grid_w[CV_HALF];   // weight of grid point j + 1
    int band_j[CV_HALF];
    double band_w[CV_HALF];
} cv_ctx;

static void cv_prepare(cv_ctx *c) {
    for (int k = 0; k < CV_HALF; k++) {
        double f = (double)k * c->sr / CV_NFFT;
        int j = 0;
        while (j < c->ne - 2 && c->env_hz[j + 1] <= f) j++;
        double span = c->env_hz[j + 1] - c->env_hz[j];
        double w = span > 0 ? (f - c->env_hz[j]) / span : 0.0;
        c->grid_j[k] = j; c->grid_w[k] = w < 0 ? 0 : (w > 1 ? 1 : w);
        // aperiodicity: linear between band centres, flat beyond the outer ones
        if (c->nb == 1) { c->band_j[k] = 0; c->band_w[k] = 0; continue; }
        double first = 0.5 * (c->edges[0] + c->edges[1]);
        double last = 0.5 * (c->edges[c->nb - 1] + c->edges[c->nb]);
        if (f <= first) { c->band_j[k] = 0; c->band_w[k] = 0; continue; }
        if (f >= last) { c->band_j[k] = c->nb - 2; c->band_w[k] = 1; continue; }
        int b = 0;
        while (b < c->nb - 2 && 0.5 * (c->edges[b + 1] + c->edges[b + 2]) <= f) b++;
        double cb = 0.5 * (c->edges[b] + c->edges[b + 1]);
        double cn = 0.5 * (c->edges[b + 1] + c->edges[b + 2]);
        c->band_j[k] = b; c->band_w[k] = (f - cb) / (cn - cb);
    }
}

// Natural-log power envelope and aperiodicity per bin at fractional frame fi.
static void cv_spectrum_at(const cv_ctx *c, double fi, double *lnp, double *apb) {
    if (fi < 0) fi = 0;
    if (fi > c->n - 1) fi = c->n - 1;
    int i0 = (int)floor(fi);
    int i1 = i0 + 1 < c->n ? i0 + 1 : i0;
    double t = fi - i0;
    const double *e0 = c->env_db + (size_t)i0 * c->ne, *e1 = c->env_db + (size_t)i1 * c->ne;
    const double *a0 = c->ap + (size_t)i0 * c->nb, *a1 = c->ap + (size_t)i1 * c->nb;
    const double db2ln = log(10.0) / 10.0;
    for (int k = 0; k < CV_HALF; k++) {
        int j = c->grid_j[k];
        double w = c->grid_w[k];
        double d0 = e0[j] * (1 - w) + e0[j + 1] * w;
        double d1 = e1[j] * (1 - w) + e1[j + 1] * w;
        lnp[k] = (d0 * (1 - t) + d1 * t) * db2ln;
        int b = c->band_j[k];
        double bw = c->band_w[k];
        double x0, x1;
        if (c->nb == 1) { x0 = a0[0]; x1 = a1[0]; }
        else {
            x0 = a0[b] * (1 - bw) + a0[b + 1] * bw;
            x1 = a1[b] * (1 - bw) + a1[b + 1] * bw;
        }
        apb[k] = x0 * (1 - t) + x1 * t;
    }
}

// Minimum-phase spectrum (re, im over the full FFT) from a natural-log
// magnitude over the half spectrum. delay: fractional samples to shift by.
static void cv_minphase(const double *lnmag, double delay, double *re, double *im) {
    for (int k = 0; k < CV_HALF; k++) { re[k] = lnmag[k]; im[k] = 0; }
    for (int k = CV_HALF; k < CV_NFFT; k++) { re[k] = lnmag[CV_NFFT - k]; im[k] = 0; }
    cv_fft(re, im, 1);  // real cepstrum
    for (int q = 1; q < CV_NFFT / 2; q++) re[q] *= 2.0;
    for (int q = CV_NFFT / 2 + 1; q < CV_NFFT; q++) re[q] = 0.0;
    for (int q = 0; q < CV_NFFT; q++) im[q] = 0.0;
    cv_fft(re, im, 0);  // log spectrum, minimum phase
    for (int k = 0; k < CV_NFFT; k++) {
        double kk = k <= CV_NFFT / 2 ? k : k - CV_NFFT;
        double mag = exp(re[k]);
        double ph = im[k] - 2.0 * M_PI * kk * delay / CV_NFFT;
        re[k] = mag * cos(ph);
        im[k] = mag * sin(ph);
    }
}

static double cv_f0_at(const cv_ctx *c, double fi) {
    if (fi < 0) fi = 0;
    if (fi > c->n - 1) fi = c->n - 1;
    int i0 = (int)floor(fi);
    int i1 = i0 + 1 < c->n ? i0 + 1 : i0;
    double t = fi - i0;
    double a = c->f0[i0], b = c->f0[i1];
    if (a > 0 && b > 0) return a * (1 - t) + b * t;
    // At a voicing edge the nearer frame decides.
    return t < 0.5 ? a : b;
}

size_t christman_vocoder_length(int n_frames, int hop) {
    if (n_frames <= 0 || hop <= 0) return 0;
    return (size_t)n_frames * (size_t)hop;
}

int christman_vocoder_render(int n_frames, int hop, int sample_rate,
                             const double *f0, const double *env_db, int ne, const double *env_hz,
                             const double *ap, int nb, const double *band_edges_hz,
                             double jitter, double shimmer, unsigned int seed,
                             float *out, size_t out_len) {
    if (!f0 || !env_db || !env_hz || !ap || !band_edges_hz || !out) return 1;
    if (n_frames <= 0 || hop <= 0 || ne < 2 || nb < 1) return 1;
    if (out_len != (size_t)n_frames * (size_t)hop) return 1;
    if (sample_rate < 8000 || sample_rate > 24000) return 2;
    if (!isfinite(jitter) || !isfinite(shimmer) || jitter < 0 || jitter > 0.2 || shimmer < 0 || shimmer > 0.2) return 3;
    if (env_hz[0] != 0.0 || fabs(env_hz[ne - 1] - sample_rate / 2.0) > 1e-6) return 3;
    for (int j = 1; j < ne; j++) if (!(env_hz[j] > env_hz[j - 1])) return 3;
    if (band_edges_hz[0] != 0.0) return 3;
    for (int b = 1; b <= nb; b++) if (!(band_edges_hz[b] > band_edges_hz[b - 1])) return 3;
    for (int i = 0; i < n_frames; i++) {
        if (!isfinite(f0[i]) || f0[i] < 0) return 3;
        for (int j = 0; j < ne; j++) if (!isfinite(env_db[(size_t)i * ne + j])) return 3;
        for (int b = 0; b < nb; b++) {
            double a = ap[(size_t)i * nb + b];
            if (!isfinite(a) || a < 0 || a > 1) return 3;
        }
    }

    cv_ctx *c = (cv_ctx *)calloc(1, sizeof(cv_ctx));
    double *re = (double *)malloc(sizeof(double) * CV_NFFT);
    double *im = (double *)malloc(sizeof(double) * CV_NFFT);
    double *nre = (double *)malloc(sizeof(double) * CV_NFFT);
    double *nim = (double *)malloc(sizeof(double) * CV_NFFT);
    double *lnp = (double *)malloc(sizeof(double) * CV_HALF);
    double *apb = (double *)malloc(sizeof(double) * CV_HALF);
    double *lnm = (double *)malloc(sizeof(double) * CV_HALF);
    double *acc = (double *)calloc(out_len + CV_NFFT, sizeof(double));
    if (!c || !re || !im || !nre || !nim || !lnp || !apb || !lnm || !acc) {
        free(c); free(re); free(im); free(nre); free(nim); free(lnp); free(apb); free(lnm); free(acc);
        return 1;
    }
    c->n = n_frames; c->hop = hop; c->sr = sample_rate; c->ne = ne; c->nb = nb;
    c->f0 = f0; c->env_db = env_db; c->env_hz = env_hz; c->ap = ap; c->edges = band_edges_hz;
    cv_prepare(c);
    cv_rng rng;
    cv_seed(&rng, seed);
    const double floor_ln = log(1e-12);

    // Periodic part: one minimum-phase pulse per glottal cycle.
    double phase = 1.0, pscale = 1.0, ascale = 1.0;
    for (size_t s = 0; s < out_len; s++) {
        double fi = (double)s / hop;
        double f = cv_f0_at(c, fi);
        if (f <= 0) { phase = 1.0; continue; }  // the next voiced sample starts a cycle
        double inc = f / sample_rate / pscale;
        double before = phase;
        phase += inc;
        if (before >= 1.0 || phase >= 1.0) {
            double over = before >= 1.0 ? 0.0 : (phase - 1.0) / inc;  // samples since the crossing
            double pos = (double)s - over;
            if (pos < 0) pos = 0;
            phase = before >= 1.0 ? 0.0 : phase - 1.0;
            double period = sample_rate / f * pscale;
            cv_spectrum_at(c, pos / hop, lnp, apb);
            for (int k = 0; k < CV_HALF; k++) {
                double per = 1.0 - apb[k];
                double l = per > 1e-12 ? lnp[k] + log(per) : floor_ln + lnp[k];
                lnm[k] = 0.5 * l;
            }
            size_t p0 = (size_t)floor(pos);
            cv_minphase(lnm, pos - (double)p0, re, im);
            cv_fft(re, im, 1);
            double amp = sqrt(period) * ascale;
            for (int m = 0; m < CV_NFFT; m++) acc[p0 + m] += amp * re[m];
            // the next cycle's perturbations (the voice's measured jitter and shimmer)
            double gj = cv_gauss(&rng), gs = cv_gauss(&rng);
            pscale = 1.0 + jitter * gj;
            if (pscale < 1.0 - CV_PERIOD_CLAMP) pscale = 1.0 - CV_PERIOD_CLAMP;
            if (pscale > 1.0 + CV_PERIOD_CLAMP) pscale = 1.0 + CV_PERIOD_CLAMP;
            ascale = 1.0 + shimmer * gs;
            if (ascale < 1.0 - CV_AMP_CLAMP) ascale = 1.0 - CV_AMP_CLAMP;
            if (ascale > 1.0 + CV_AMP_CLAMP) ascale = 1.0 + CV_AMP_CLAMP;
        }
    }

    // Aperiodic part: windowed white noise, filtered frame by frame.
    int seg = 2 * hop;
    if (seg > CV_NFFT) seg = CV_NFFT;
    for (int i = 0; i < n_frames; i++) {
        cv_spectrum_at(c, (double)i, lnp, apb);
        int any = 0;
        for (int k = 0; k < CV_HALF; k++) {
            double l = apb[k] > 1e-12 ? lnp[k] + log(apb[k]) : floor_ln + lnp[k];
            lnm[k] = 0.5 * l;
            if (apb[k] > 1e-6) any = 1;
        }
        if (!any) continue;
        cv_minphase(lnm, 0.0, re, im);
        for (int m = 0; m < CV_NFFT; m++) { nre[m] = 0; nim[m] = 0; }
        for (int m = 0; m < seg; m++) {
            double w = sqrt(0.5 - 0.5 * cos(2.0 * M_PI * m / seg));
            nre[m] = cv_gauss(&rng) * w;
        }
        cv_fft(nre, nim, 0);
        for (int k = 0; k < CV_NFFT; k++) {
            double a = nre[k] * re[k] - nim[k] * im[k];
            double b = nre[k] * im[k] + nim[k] * re[k];
            nre[k] = a; nim[k] = b;
        }
        cv_fft(nre, nim, 1);
        long start = (long)i * hop - hop;
        for (int m = 0; m < CV_NFFT; m++) {
            long at = start + m;
            if (at >= 0 && (size_t)at < out_len + CV_NFFT) acc[at] += nre[m];
        }
    }

    for (size_t s = 0; s < out_len; s++) out[s] = (float)acc[s];
    free(c); free(re); free(im); free(nre); free(nim); free(lnp); free(apb); free(lnm); free(acc);
    return 0;
}
