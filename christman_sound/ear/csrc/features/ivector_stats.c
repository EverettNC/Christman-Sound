/* i-vector statistics and their solution, in a fixed order of arithmetic.
 *
 * Fifteen steps of conjugate gradient do not converge, so the answer depends on
 * the last bits of every sum. Library matrix routines may sum in a different
 * order from run to run (threads, alignment); these loops never do. The same
 * input always gives the same i-vector.
 */
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* y = A x, A full symmetric n x n, row major. */
static void symv(const double *A, const double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        const double *row = A + (size_t)i * n;
        double s = 0.0;
        for (int j = 0; j < n; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

/* Exact solve of A x = b by Cholesky; returns 0 on success. */
static int cholesky_solve(const double *A, const double *b, double *x, int n) {
    double *L = calloc((size_t)n * n, sizeof(double));
    if (!L) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; k++) s -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
            if (i == j) {
                if (!(s > 0.0)) { free(L); return -1; }
                L[(size_t)i * n + i] = sqrt(s);
            } else {
                L[(size_t)i * n + j] = s / L[(size_t)j * n + j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[(size_t)i * n + k] * x[k];
        x[i] = s / L[(size_t)i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = x[i];
        for (int k = i + 1; k < n; k++) s -= L[(size_t)k * n + i] * x[k];
        x[i] = s / L[(size_t)i * n + i];
    }
    free(L);
    return 0;
}

/* Linear conjugate gradient for A x = b; x is the starting guess and the answer.
 * Stops after max_iters steps (negative: no limit), after n + 5 steps, or at an
 * exact zero residual. The residual is recomputed from scratch when its squared
 * norm has fallen below 1e-4 of, or grown past 1e4 times, its value when last
 * computed. If the residual ends worse than it started, A x = b is solved exactly.
 * Returns the number of steps taken, or -1 if memory ran out. */
int earcore_linear_cg(int n, const double *A, const double *b, double *x, int max_iters) {
    double *mem = malloc(sizeof(double) * 4 * (size_t)n);
    if (!mem) return -1;
    double *r = mem, *p = mem + n, *Ap = mem + 2 * n, *x_orig = mem + 3 * n;
    symv(A, x, Ap, n);
    for (int i = 0; i < n; i++) {
        p[i] = b[i] - Ap[i];
        r[i] = -p[i];
        x_orig[i] = x[i];
    }
    double r_cur = dot(r, r, n), r_initial = r_cur, r_recompute = r_cur;
    const double max_error_sq = DBL_MIN;
    const float rf = 0.01f;
    const double factor = (double)(rf * rf), inv_factor = 1.0 / factor;
    int k = 0;
    for (; k < n + 5 && k != max_iters; k++) {
        symv(A, p, Ap, n);
        double alpha = -dot(p, r, n) / dot(p, Ap, n);
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
        for (int i = 0; i < n; i++) r[i] += alpha * Ap[i];
        double r_next = dot(r, r, n);
        if (r_next < factor * r_recompute || r_next > inv_factor * r_recompute) {
            symv(A, x, r, n);
            for (int i = 0; i < n; i++) r[i] -= b[i];
            r_next = dot(r, r, n);
            r_recompute = r_next;
        }
        if (r_next <= max_error_sq) break;
        double beta = r_next / r_cur;
        for (int i = 0; i < n; i++) p[i] = beta * p[i] - r[i];
        r_cur = r_next;
    }
    if (r_cur > r_initial && r_cur > r_initial + 1.0e-10 * dot(b, b, n)) {
        if (cholesky_solve(A, b, x, n) != 0) memcpy(x, x_orig, sizeof(double) * (size_t)n);
    }
    free(mem);
    return k;
}

/* Add one block of frames to the extractor statistics.
 *   feats  N x F  the frames without CMVN, after splice and LDA
 *   idx    N x K  Gaussian index per kept posterior (-1 for none)
 *   post   N x K  posterior, already times the posterior scale and the frame weight
 *   sim    G x F x D  Sigma_g^-1 M_g
 *   u      G x P  M_g^T Sigma_g^-1 M_g, lower triangle row by row, P = D (D + 1) / 2
 * linear (D) and quad (D x D, full) are added to; the total weight is returned
 * (NaN if memory ran out). Gaussians are visited in index order, each once. */
double earcore_ivector_acc(int N, int K, int F, int D, int G,
                           const double *feats, const int *idx, const double *post,
                           const double *sim, const double *u,
                           double *linear, double *quad) {
    const int P = D * (D + 1) / 2;
    double *wf = calloc((size_t)G * F, sizeof(double));
    double *gamma = calloc((size_t)G, sizeof(double));
    unsigned char *seen = calloc((size_t)G, 1);
    double *packed = calloc((size_t)P, sizeof(double));
    double tot = 0.0;
    if (!wf || !gamma || !seen || !packed) {
        free(wf); free(gamma); free(seen); free(packed);
        return NAN;
    }
    for (int t = 0; t < N; t++) {
        const double *f = feats + (size_t)t * F;
        for (int j = 0; j < K; j++) {
            int g = idx[(size_t)t * K + j];
            double w = post[(size_t)t * K + j];
            if (g < 0 || w == 0.0) continue;
            seen[g] = 1;
            gamma[g] += w;
            double *row = wf + (size_t)g * F;
            for (int d = 0; d < F; d++) row[d] += w * f[d];
        }
    }
    for (int g = 0; g < G; g++) {
        if (!seen[g]) continue;
        const double *row = wf + (size_t)g * F;
        const double *s = sim + (size_t)g * F * D;
        for (int d = 0; d < F; d++) {
            double v = row[d];
            const double *sd = s + (size_t)d * D;
            for (int i = 0; i < D; i++) linear[i] += v * sd[i];
        }
        const double *ug = u + (size_t)g * P;
        double gm = gamma[g];
        for (int k = 0; k < P; k++) packed[k] += gm * ug[k];
        tot += gm;
    }
    int k = 0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++, k++) {
            quad[(size_t)i * D + j] += packed[k];
            if (i != j) quad[(size_t)j * D + i] += packed[k];
        }
    }
    free(wf); free(gamma); free(seen); free(packed);
    return tot;
}
