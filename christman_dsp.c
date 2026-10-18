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
// =============================================================================
// OUTPUT SIDE — christman_play (added 2026-10-13)
//
// WHAT CHANGED AND WHY
// Every speaker path the family ever had went through somebody else's
// library: pygame (SDL) in the SDK, sounddevice (PortAudio) in the Voice
// Creation Center. Everett's ruling: the engine measures AND plays, and
// nothing in the chain belongs to anyone else. The only thing below this
// function is the operating system's own audio layer — AudioToolbox on
// Darwin, ALSA on Linux — which is where every player on earth ends,
// including the ones we just removed. No pip package anywhere.
//
// On Linux, libasound is found at run time by dlopen with the handful of
// symbols declared here, so the .so links against nothing and a machine
// with no sound system still loads the measurement side. On Darwin the
// AudioToolbox framework is linked by scripts/build_dsp.py.
//
// Contract: blocking. Plays `length` interleaved float32 samples at
// `sample_rate` with `channels` channels and returns when the device has
// played out the last sample. Returns 0 only then. Nonzero means the clip
// was NOT played in full, with the reason in the code. Codes 1-5 and 7
// come before any sample reaches the device; 6 and 8 come after part of
// the clip may already have been heard:
//   1  bad arguments            2  no audio system on this platform
//   3  OS audio library absent  4  device could not be opened
//   5  device rejected format   6  write failed mid-stream
//   7  allocation failed        8  device failed while playing out the end
//
// (2026-10-18) THE LAST BUFFER HAS TO DRAIN, AND THE DEVICE HAS TO SAY SO.
// On Linux the result of snd_pcm_drain was thrown away, so a device that
// failed while playing out the last buffered audio (unplugged, suspended)
// still returned 0. It now returns 8. On Darwin a queue that stopped on
// its own, before this code asked it to, still set `finished` and returned
// 0, and a failed AudioQueueEnqueueBuffer was counted as frames handed
// over. Both now return 6.
// `christman_play_backend` names the backend ("alsa", "coreaudio", or
// "none") so a caller can report what it is about to use, not guess.
// =============================================================================
#include <string.h>

#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <pthread.h>

typedef struct {
    const float* data;
    size_t frames_total;
    size_t frames_done;
    int channels;
    int stopping;   /* async stop requested once all audio was handed over */
    int finished;   /* CoreAudio reported the queue is no longer running */
    int failed;     /* a buffer could not be enqueued; the clip is not whole */
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} cm_play_state;

/* Fill one buffer. Returns the number of frames written (0 = nothing left). */
static size_t cm_fill(cm_play_state* st, AudioQueueRef aq, AudioQueueBufferRef buf) {
    if (st->failed) return 0;   /* the clip is already broken; hand over nothing more */
    size_t frames_left = st->frames_total - st->frames_done;
    size_t frames_fit  = buf->mAudioDataBytesCapacity / (sizeof(float) * st->channels);
    size_t n = frames_left < frames_fit ? frames_left : frames_fit;
    if (n == 0) return 0;
    memcpy(buf->mAudioData, st->data + st->frames_done * st->channels, n * st->channels * sizeof(float));
    buf->mAudioDataByteSize = (UInt32)(n * st->channels * sizeof(float));
    if (AudioQueueEnqueueBuffer(aq, buf, 0, NULL) != noErr) { st->failed = 1; return 0; }
    st->frames_done += n;
    return n;
}

/* A buffer came back from the hardware. Refill it, or — once everything has
 * been handed over — ask the queue to stop AFTER the buffers still in flight
 * have played. Finishing is signalled by the IsRunning listener, not here;
 * declaring done on the first empty callback cut the last ~200 ms of every
 * utterance (found on Everett's Mac, 2026-10-13: 1.5 s of audio returned in 1.33 s). */
static void cm_aq_callback(void* user, AudioQueueRef aq, AudioQueueBufferRef buf) {
    cm_play_state* st = (cm_play_state*)user;
    if (cm_fill(st, aq, buf) == 0 && !st->stopping) {
        pthread_mutex_lock(&st->lock);
        st->stopping = 1;
        pthread_mutex_unlock(&st->lock);
        AudioQueueStop(aq, false);
    }
}

static void cm_aq_running(void* user, AudioQueueRef aq, AudioQueuePropertyID id) {
    cm_play_state* st = (cm_play_state*)user;
    UInt32 running = 0, size = sizeof running;
    if (AudioQueueGetProperty(aq, kAudioQueueProperty_IsRunning, &running, &size) != noErr) running = 0;
    if (!running) {
        pthread_mutex_lock(&st->lock);
        st->finished = 1;
        pthread_cond_signal(&st->cond);
        pthread_mutex_unlock(&st->lock);
    }
}

const char* christman_play_backend(void) { return "coreaudio"; }

int christman_play(const float* audio, size_t length, int sample_rate, int channels) {
    if (!audio || length == 0 || sample_rate <= 0 || channels < 1 || channels > 2 || length % channels) return 1;
    cm_play_state st;
    st.data = audio; st.frames_total = length / channels; st.frames_done = 0;
    st.channels = channels; st.stopping = 0; st.finished = 0; st.failed = 0;
    pthread_mutex_init(&st.lock, NULL); pthread_cond_init(&st.cond, NULL);

    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof fmt);
    fmt.mSampleRate = (Float64)sample_rate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel = 32;
    fmt.mChannelsPerFrame = (UInt32)channels;
    fmt.mBytesPerFrame = 4 * (UInt32)channels;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = fmt.mBytesPerFrame;

    AudioQueueRef aq = NULL;
    if (AudioQueueNewOutput(&fmt, cm_aq_callback, &st, NULL, kCFRunLoopCommonModes, 0, &aq) != noErr) return 4;

    if (AudioQueueAddPropertyListener(aq, kAudioQueueProperty_IsRunning, cm_aq_running, &st) != noErr) {
        AudioQueueDispose(aq, true); return 4;
    }

    const int NBUF = 3;
    const UInt32 bytes_per_buf = (UInt32)(sample_rate / 10) * fmt.mBytesPerFrame; // 100 ms
    AudioQueueBufferRef bufs[3];
    /* Prime: fill only as many buffers as the audio needs. A clip shorter than
     * 300 ms must not trip the stop before the queue has started. */
    for (int i = 0; i < NBUF; i++) {
        if (AudioQueueAllocateBuffer(aq, bytes_per_buf, &bufs[i]) != noErr) { AudioQueueDispose(aq, true); return 7; }
        if (cm_fill(&st, aq, bufs[i]) == 0) break;
    }
    if (st.failed) {                              /* a primed buffer was refused */
        AudioQueueDispose(aq, true);
        pthread_mutex_destroy(&st.lock); pthread_cond_destroy(&st.cond);
        return 6;
    }
    if (AudioQueueStart(aq, NULL) != noErr) { AudioQueueDispose(aq, true); return 4; }
    if (st.frames_done == st.frames_total) {      /* whole clip fit in the primed buffers */
        pthread_mutex_lock(&st.lock);
        st.stopping = 1;
        pthread_mutex_unlock(&st.lock);
        AudioQueueStop(aq, false);                /* stops after they have played */
    }

    pthread_mutex_lock(&st.lock);
    while (!st.finished) pthread_cond_wait(&st.cond, &st.lock);
    /* Whole only if every frame was enqueued and the queue stopped because we
     * asked it to after the last one. Anything else stopped it early. */
    int whole = st.stopping && !st.failed && st.frames_done == st.frames_total;
    pthread_mutex_unlock(&st.lock);
    AudioQueueDispose(aq, true);
    pthread_mutex_destroy(&st.lock); pthread_cond_destroy(&st.cond);
    return whole ? 0 : 6;
}

#elif defined(__linux__)
#include <dlfcn.h>
#include <stdint.h>

// The few libasound symbols we use, declared here so no headers are needed.
typedef struct _snd_pcm snd_pcm_t;
typedef long snd_pcm_sframes_t;
typedef int (*fn_open)(snd_pcm_t**, const char*, int, int);
typedef int (*fn_set_params)(snd_pcm_t*, int, int, unsigned, unsigned, int, unsigned);
typedef snd_pcm_sframes_t (*fn_writei)(snd_pcm_t*, const void*, unsigned long);
typedef int (*fn_drain)(snd_pcm_t*);
typedef int (*fn_close)(snd_pcm_t*);
typedef int (*fn_recover)(snd_pcm_t*, int, int);
enum { CM_SND_PCM_STREAM_PLAYBACK = 0, CM_SND_PCM_FORMAT_S16_LE = 2, CM_SND_PCM_ACCESS_RW_INTERLEAVED = 3 };

const char* christman_play_backend(void) { return "alsa"; }

int christman_play(const float* audio, size_t length, int sample_rate, int channels) {
    if (!audio || length == 0 || sample_rate <= 0 || channels < 1 || channels > 2 || length % channels) return 1;
    void* lib = dlopen("libasound.so.2", RTLD_NOW);
    if (!lib) return 3;
    fn_open  p_open  = (fn_open) dlsym(lib, "snd_pcm_open");
    fn_set_params p_set = (fn_set_params) dlsym(lib, "snd_pcm_set_params");
    fn_writei p_writei = (fn_writei) dlsym(lib, "snd_pcm_writei");
    fn_drain p_drain = (fn_drain) dlsym(lib, "snd_pcm_drain");
    fn_close p_close = (fn_close) dlsym(lib, "snd_pcm_close");
    fn_recover p_recover = (fn_recover) dlsym(lib, "snd_pcm_recover");
    if (!p_open || !p_set || !p_writei || !p_drain || !p_close || !p_recover) { dlclose(lib); return 3; }

    snd_pcm_t* pcm = NULL;
    if (p_open(&pcm, "default", CM_SND_PCM_STREAM_PLAYBACK, 0) < 0) { dlclose(lib); return 4; }
    if (p_set(pcm, CM_SND_PCM_FORMAT_S16_LE, CM_SND_PCM_ACCESS_RW_INTERLEAVED,
              (unsigned)channels, (unsigned)sample_rate, 1, 200000) < 0) { p_close(pcm); dlclose(lib); return 5; }

    size_t frames_total = length / channels;
    const size_t chunk = 1024;
    int16_t* buf = (int16_t*)malloc(chunk * channels * sizeof(int16_t));
    if (!buf) { p_close(pcm); dlclose(lib); return 7; }

    int status = 0;
    size_t done = 0;
    while (done < frames_total) {
        size_t n = frames_total - done < chunk ? frames_total - done : chunk;
        for (size_t i = 0; i < n * channels; i++) {
            float s = audio[done * channels + i];
            if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
            buf[i] = (int16_t)(s * 32767.0f);
        }
        snd_pcm_sframes_t w = p_writei(pcm, buf, n);
        if (w < 0) {
            if (p_recover(pcm, (int)w, 1) < 0) { status = 6; break; }
            continue; // underrun recovered; retry the same chunk
        }
        done += (size_t)w;
    }
    /* Blocks until the device has played what it holds. Its failure is ours. */
    int drained = p_drain(pcm);
    if (status == 0 && drained < 0) status = 8;
    p_close(pcm);
    free(buf);
    dlclose(lib);
    return status;
}

#else
const char* christman_play_backend(void) { return "none"; }
int christman_play(const float* audio, size_t length, int sample_rate, int channels) {
    (void)audio; (void)length; (void)sample_rate; (void)channels;
    return 2;
}
#endif
