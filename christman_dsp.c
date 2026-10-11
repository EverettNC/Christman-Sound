// christman_dsp.c
// The Christman Bare-Metal Audio Engine
#include <math.h>
#include <stddef.h>

// Calculate RMS (Root Mean Square) Energy for Arousal/Dominance
void christman_rms(const float* audio, size_t length, float* out_rms) {
    if (length == 0) {
        *out_rms = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        sum += audio[i] * audio[i];
    }
    *out_rms = sqrtf(sum / (float)length);
}

// Calculate Zero Crossing Rate for Valence Smoothness
void christman_zcr(const float* audio, size_t length, float* out_zcr) {
    if (length < 2) {
        *out_zcr = 0.0f;
        return;
    }

    size_t crossings = 0;
    for (size_t i = 1; i < length; i++) {
        // Detect if the wave crossed the zero line
        if ((audio[i] > 0.0f && audio[i - 1] <= 0.0f) || 
            (audio[i] <= 0.0f && audio[i - 1] > 0.0f)) {
            crossings++;
        }
    }
    *out_zcr = (float)crossings / (float)length;
}

#include <stdlib.h> // Required for malloc/free

// Bare-metal YIN Pitch Detection Algorithm (de Cheveigné & Kawahara, 2002)
//
// Operates on ONE analysis frame (e.g. 2048 samples). The difference
// function is O(half_len^2); call it per frame from the contour wrapper,
// never on a whole recording. Writes 0.0 for an unvoiced frame.
void christman_yin(const float* audio, size_t length, int sample_rate, float threshold, float* out_pitch) {
    *out_pitch = 0.0f; // defined on every path, including early returns
    if (length < 4 || sample_rate <= 0) {
        return;
    }
    int half_len = (int)(length / 2);
    if (half_len <= 0) {
        return;
    }

    float* yin_buffer = (float*)malloc(half_len * sizeof(float));
    if (!yin_buffer) {
        *out_pitch = 0.0f;
        return;
    }

    // 1. Difference function
    for (int tau = 0; tau < half_len; tau++) {
        yin_buffer[tau] = 0.0f;
        for (int i = 0; i < half_len; i++) {
            float delta = audio[i] - audio[i + tau];
            yin_buffer[tau] += delta * delta;
        }
    }

    // 2. Cumulative mean normalized difference
    yin_buffer[0] = 1.0f;
    float running_sum = 0.0f;
    for (int tau = 1; tau < half_len; tau++) {
        running_sum += yin_buffer[tau];
        if (running_sum == 0.0f) {
            yin_buffer[tau] = 1.0f;
        } else {
            yin_buffer[tau] = yin_buffer[tau] * tau / running_sum;
        }
    }

    // 3. Absolute thresholding (find the period)
    int tau_estimate = -1;
    for (int tau = 2; tau < half_len; tau++) {
        if (yin_buffer[tau] < threshold) {
            // Found a dip, now find the local minimum
            while (tau + 1 < half_len && yin_buffer[tau + 1] < yin_buffer[tau]) {
                tau++;
            }
            tau_estimate = tau;
            break;
        }
    }

    // 4. Convert period to frequency
    if (tau_estimate != -1 && tau_estimate != 0) {
        *out_pitch = (float)sample_rate / (float)tau_estimate;
    } else {
        *out_pitch = 0.0f; // Unvoiced / No pitch detected
    }

    free(yin_buffer);
}

// Add this to christman_dsp.c

// Bare-metal Linear Predictive Coding (LPC) using Levinson-Durbin recursion
//
// Contract: out_a has order+1 slots. On ANY early return the caller gets
// a[0] = 1 and a[1..order] = 0 — a flat (all-pole-free) filter — never
// uninitialised memory. Returns 0 on success, nonzero when no coefficients
// could be computed (too-short frame, silent frame, allocation failure, or
// a numerically unstable recursion), so the caller can report "unmeasured"
// instead of reading zeros as a measurement.
int christman_lpc(const float* audio, size_t length, int order, float* out_a) {
    if (order <= 0) {
        return 1;
    }
    // Initialise FIRST (medical-grade review: out_a was left uninitialised
    // on the malloc-failure path below).
    out_a[0] = 1.0f;
    for (int i = 1; i <= order; i++) out_a[i] = 0.0f;

    // Guard the size_t underflow: with length < order+1, `length - lag`
    // wrapped to ~2^64 and the autocorrelation loop read off the end of
    // the buffer (segfault on any frame shorter than the LPC order).
    if (length == 0 || (size_t)order >= length) {
        return 2;
    }

    // 1. Calculate Autocorrelation
    float* r = (float*)malloc((order + 1) * sizeof(float));
    if (!r) return 3;

    for (int lag = 0; lag <= order; lag++) {
        r[lag] = 0.0f;
        for (size_t i = 0; i < length - lag; i++) {
            r[lag] += audio[i] * audio[i + lag];
        }
    }

    // 2. Levinson-Durbin Recursion
    if (r[0] <= 0.0f) {
        // Silent frame: no energy, no spectrum to model.
        free(r);
        return 4;
    }

    float error = r[0];
    float* k = (float*)malloc((order + 1) * sizeof(float));
    if (!k) { free(r); return 3; }

    int status = 0;
    for (int i = 1; i <= order; i++) {
        // Guard: prediction error must stay strictly positive. It reaches
        // zero (or goes negative from float round-off) on a perfectly
        // predictable or degenerate frame; dividing by it produced inf/NaN
        // coefficients that were handed back as if measured.
        if (error <= 0.0f) {
            for (int j = i; j <= order; j++) out_a[j] = 0.0f;
            status = 5;
            break;
        }
        float sum = 0.0f;
        for (int j = 1; j < i; j++) {
            sum += out_a[j] * r[i - j];
        }
        k[i] = (r[i] - sum) / error;
        out_a[i] = k[i];

        // Update previous coefficients
        for (int j = 1; j <= i / 2; j++) {
            float temp = out_a[j] - k[i] * out_a[i - j];
            out_a[i - j] -= k[i] * out_a[j];
            out_a[j] = temp;
        }
        error *= (1.0f - k[i] * k[i]);
    }

    free(k);
    free(r);
    return status;
}