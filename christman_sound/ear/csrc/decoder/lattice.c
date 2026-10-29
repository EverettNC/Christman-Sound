/* Reading the token lattice back: topological order, best path, the word
 * lattice (determinized on words in the (graph, acoustic) lattice semiring and
 * pruned at lattice_beam), and MBR word posteriors over it (the edit-distance
 * forward-backward of Xu et al., the way MBR confidences are computed). */
#include "decoder_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int lfail(ec_dec *d, const char *msg) {
  snprintf(d->err, sizeof(d->err), "%s", msg);
  return -1;
}

static int lgrow(void **p, int64_t *cap, int64_t need, size_t elem) {
  if (need <= *cap) return 0;
  int64_t nc = *cap ? *cap : 256;
  while (nc < need) nc *= 2;
  void *q = realloc(*p, (size_t)nc * elem);
  if (!q) return -1;
  *p = q;
  *cap = nc;
  return 0;
}

void ec_lat_free(RawLat *L) {
  free(L->vert_tok); free(L->vert_frame); free(L->arc_start);
  free(L->dst); free(L->il); free(L->ol); free(L->g); free(L->a);
  free(L->fin); free(L->beta);
  free(L->p_il); free(L->p_ol); free(L->p_g); free(L->p_a);
  memset(L, 0, sizeof(*L));
}

/* Lattice weights compare by total, then by graph part. */
static inline int w_less(double g1, double a1, double g2, double a2) {
  double t1 = g1 + a1, t2 = g2 + a2;
  if (t1 < t2) return 1;
  if (t1 > t2) return 0;
  return g1 < g2;
}

/* Build d->lat from the live tokens. */
static int build_raw(ec_dec *d, int use_final) {
  RawLat *L = &d->lat;
  if (!d->initialized) return lfail(d, "no decoding to read");
  if (d->finalized && !use_final)
    return lfail(d, "decoding is finalized; only the final lattice can be read");
  int32_t T = d->nframes;
  int64_t ntok = 0, nlink = 0;
  for (int32_t f = 0; f <= T; f++)
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) {
      ntok++;
      for (int32_t l = d->toks[t].links; l >= 0; l = d->links[l].next) nlink++;
    }
  int64_t ncap0 = L->n_cap;
  if (lgrow((void **)&L->vert_tok, &L->n_cap, ntok + 1, sizeof(int32_t))) return lfail(d, "out of memory");
  int64_t c;
  c = ncap0; if (lgrow((void **)&L->vert_frame, &c, L->n_cap, sizeof(int32_t))) return lfail(d, "out of memory");
  c = ncap0; if (lgrow((void **)&L->arc_start, &c, L->n_cap, sizeof(int32_t))) return lfail(d, "out of memory");
  c = ncap0; if (lgrow((void **)&L->fin, &c, L->n_cap, sizeof(float))) return lfail(d, "out of memory");
  c = ncap0; if (lgrow((void **)&L->beta, &c, L->n_cap, sizeof(double))) return lfail(d, "out of memory");
  int64_t acap0 = L->a_cap;
  if (lgrow((void **)&L->dst, &L->a_cap, nlink + 1, sizeof(int32_t))) return lfail(d, "out of memory");
  c = acap0; if (lgrow((void **)&L->il, &c, L->a_cap, sizeof(int32_t))) return lfail(d, "out of memory");
  c = acap0; if (lgrow((void **)&L->ol, &c, L->a_cap, sizeof(int32_t))) return lfail(d, "out of memory");
  c = acap0; if (lgrow((void **)&L->g, &c, L->a_cap, sizeof(float))) return lfail(d, "out of memory");
  c = acap0; if (lgrow((void **)&L->a, &c, L->a_cap, sizeof(float))) return lfail(d, "out of memory");
  L->bytes = L->n_cap * (3 * 4 + 4 + 8) + L->a_cap * (3 * 4 + 2 * 4);

  /* token index -> vert index; topological order inside each frame over the
   * epsilon links (Kahn), frames in order. */
  int32_t *pos = malloc((size_t)(d->tok_used > 0 ? d->tok_used : 1) * sizeof(int32_t));
  int32_t *indeg = calloc((size_t)(d->tok_used > 0 ? d->tok_used : 1), sizeof(int32_t));
  int32_t *stack = malloc((size_t)(ntok + 1) * sizeof(int32_t));
  if (!pos || !indeg || !stack) {
    free(pos); free(indeg); free(stack);
    return lfail(d, "out of memory");
  }
  int32_t n = 0;
  for (int32_t f = 0; f <= T; f++) {
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) indeg[t] = 0;
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next)
      for (int32_t l = d->toks[t].links; l >= 0; l = d->links[l].next)
        if (d->links[l].ilabel == 0) indeg[d->links[l].next_tok]++;
    int32_t sp = 0, placed = 0;
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next)
      if (indeg[t] == 0) stack[sp++] = t;
    /* the list is newest-first; walk sources oldest-first */
    for (int32_t i = 0, j = sp - 1; i < j; i++, j--) {
      int32_t x = stack[i]; stack[i] = stack[j]; stack[j] = x;
    }
    int32_t head = 0;
    while (head < sp) {
      int32_t t = stack[head++];
      pos[t] = n;
      L->vert_tok[n] = t;
      L->vert_frame[n] = f;
      n++;
      placed++;
      for (int32_t l = d->toks[t].links; l >= 0; l = d->links[l].next)
        if (d->links[l].ilabel == 0 && --indeg[d->links[l].next_tok] == 0)
          stack[sp++] = d->links[l].next_tok;
    }
    int32_t in_frame = 0;
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) in_frame++;
    if (placed != in_frame) {
      free(pos); free(indeg); free(stack);
      return lfail(d, "epsilon cycle in the token lattice");
    }
  }
  L->n = n;
  /* final costs on the newest frame */
  int any_final = 0;
  if (use_final) {
    if (d->finalized) any_final = d->final_any;
    else {
      float bw;
      ec_final_costs(d, &any_final, &bw, NULL);
    }
  }
  int32_t na = 0;
  L->start = -1;
  for (int32_t i = 0; i < n; i++) {
    int32_t t = L->vert_tok[i], f = L->vert_frame[i];
    L->arc_start[i] = na;
    if (f == T) L->fin[i] = use_final ? ec_tok_final_cost(d, t, any_final) : 0.0f;
    else L->fin[i] = INFINITY;
    if (f == 0 && d->toks[t].state == (int32_t)d->g->start && L->start < 0) L->start = i;
    for (int32_t l = d->toks[t].links; l >= 0; l = d->links[l].next) {
      const Link *k = &d->links[l];
      L->dst[na] = pos[k->next_tok];
      L->il[na] = k->ilabel;
      L->ol[na] = k->olabel;
      L->g[na] = k->graph_cost;
      L->a[na] = k->ilabel != 0 ? k->ac_cost - d->cost_offsets[f] : k->ac_cost;
      na++;
    }
  }
  L->arc_start[n] = na;
  L->narcs = na;
  free(pos); free(indeg); free(stack);
  /* best cost to the end */
  for (int32_t i = n - 1; i >= 0; i--) {
    double b = L->fin[i];
    for (int32_t j = L->arc_start[i]; j < L->arc_start[i + 1]; j++) {
      double v = (double)L->g[j] + (double)L->a[j] + L->beta[L->dst[j]];
      if (v < b) b = v;
    }
    L->beta[i] = b;
  }
  if (L->start < 0) return lfail(d, "start token was pruned away");
  return 0;
}

int64_t ec_dec_best_path(ec_dec *d, int32_t use_final, double *costs3) {
  if (build_raw(d, use_final)) return -1;
  RawLat *L = &d->lat;
  int32_t n = L->n;
  double *cg = malloc((size_t)n * sizeof(double)), *ca = malloc((size_t)n * sizeof(double));
  int32_t *pred = malloc((size_t)n * sizeof(int32_t)), *parc = malloc((size_t)n * sizeof(int32_t));
  if (!cg || !ca || !pred || !parc) {
    free(cg); free(ca); free(pred); free(parc);
    return lfail(d, "out of memory");
  }
  for (int32_t i = 0; i < n; i++) cg[i] = ca[i] = INFINITY, pred[i] = parc[i] = -1;
  cg[L->start] = ca[L->start] = 0.0;
  for (int32_t i = 0; i < n; i++) {
    if (cg[i] == INFINITY) continue;
    for (int32_t j = L->arc_start[i]; j < L->arc_start[i + 1]; j++) {
      int32_t k = L->dst[j];
      double g = cg[i] + L->g[j], a = ca[i] + L->a[j];
      if (cg[k] == INFINITY || w_less(g, a, cg[k], ca[k])) {
        cg[k] = g; ca[k] = a; pred[k] = i; parc[k] = j;
      }
    }
  }
  int32_t best = -1;
  double bg = INFINITY, ba = INFINITY;
  for (int32_t i = 0; i < n; i++) {
    if (L->fin[i] == INFINITY || cg[i] == INFINITY) continue;
    double g = cg[i] + L->fin[i];
    if (best < 0 || w_less(g, ca[i], bg, ba)) { best = i; bg = g; ba = ca[i]; }
  }
  if (best < 0) {
    free(cg); free(ca); free(pred); free(parc);
    return lfail(d, "no path reaches the end of the lattice");
  }
  int32_t len = 0;
  for (int32_t i = best; pred[i] >= 0; i = pred[i]) len++;
  free(L->p_il); free(L->p_ol); free(L->p_g); free(L->p_a);
  L->path_cap = len + 1;
  L->p_il = malloc((size_t)L->path_cap * sizeof(int32_t));
  L->p_ol = malloc((size_t)L->path_cap * sizeof(int32_t));
  L->p_g = malloc((size_t)L->path_cap * sizeof(float));
  L->p_a = malloc((size_t)L->path_cap * sizeof(float));
  if (!L->p_il || !L->p_ol || !L->p_g || !L->p_a) {
    free(cg); free(ca); free(pred); free(parc);
    return lfail(d, "out of memory");
  }
  int32_t k = len;
  for (int32_t i = best; pred[i] >= 0; i = pred[i]) {
    int32_t j = parc[i];
    k--;
    L->p_il[k] = L->il[j];
    L->p_ol[k] = L->ol[j];
    L->p_g[k] = L->g[j];
    L->p_a[k] = L->a[j];
  }
  L->path_len = len;
  costs3[0] = bg;
  costs3[1] = ba;
  costs3[2] = L->fin[best];
  free(cg); free(ca); free(pred); free(parc);
  return len;
}

void ec_dec_path_arcs(const ec_dec *d, int32_t *ilabels, int32_t *olabels, float *graph,
                      float *acoustic) {
  const RawLat *L = &d->lat;
  for (int32_t i = 0; i < L->path_len; i++) {
    ilabels[i] = L->p_il[i];
    olabels[i] = L->p_ol[i];
    graph[i] = L->p_g[i];
    acoustic[i] = L->p_a[i];
  }
}

/* ---------------------------------------------------------------------------
 * Word lattice. Subset construction over the raw lattice with words as the
 * labels and every word-0 arc as epsilon. A subset element is a raw vert and
 * its residual weight (graph, acoustic). Each new subset is normalized by its
 * smallest element (that weight goes on the word arc), closed over epsilon,
 * and then kept only where it can still emit a word or end: that minimal
 * subset is the word-lattice state. States are expanded best-first by
 * forward + backward cost, and an arc whose best path through it is more than
 * lattice_beam worse than the best path is dropped.
 * ------------------------------------------------------------------------- */

#define SUBSET_DELTA (1.0f / 1024.0f)

typedef struct { int32_t vert; float g, a; } Elem;
typedef struct { int32_t off, len; double F, B; float fg, fa; int done, final_ok; uint32_t hash; } DState;
typedef struct { int32_t src, dst, word; float g, a; } DArc;
typedef struct { int32_t word, vert; float g, a; } Cand;

typedef struct {
  Elem *elems; int64_t ne, ne_cap;
  DState *st; int64_t ns, ns_cap;
  DArc *arcs; int64_t na, na_cap;
  int32_t *htab; uint32_t hmask;            /* subset hash -> state */
  /* closure scratch, indexed by raw vert */
  uint32_t *stamp; uint32_t cur_stamp;
  float *rg, *ra;
  int32_t *heap; int64_t heap_cap;
  Elem *work; int64_t work_cap;
  Cand *cand; int64_t cand_cap;
  /* best-first queue of (key, state) */
  double *qk; int32_t *qs; int64_t nq, q_cap;
} WL;

static void wl_free(WL *w) {
  free(w->elems); free(w->st); free(w->arcs); free(w->htab); free(w->stamp);
  free(w->rg); free(w->ra); free(w->heap); free(w->work); free(w->cand);
  free(w->qk); free(w->qs);
}

static uint32_t subset_hash(const Elem *e, int32_t n) {
  uint32_t h = 2166136261u;
  for (int32_t i = 0; i < n; i++) h = (h ^ (uint32_t)e[i].vert) * 16777619u;
  return h ^ (uint32_t)n;
}

static int subset_equal(const Elem *x, const Elem *y, int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    if (x[i].vert != y[i].vert) return 0;
    if (fabsf(x[i].g - y[i].g) > SUBSET_DELTA || fabsf(x[i].a - y[i].a) > SUBSET_DELTA) return 0;
  }
  return 1;
}

static void heap_push(int32_t *h, int64_t *n, int32_t v) {
  int64_t i = (*n)++;
  h[i] = v;
  while (i > 0) {
    int64_t p = (i - 1) / 2;
    if (h[p] <= h[i]) break;
    int32_t t = h[p]; h[p] = h[i]; h[i] = t;
    i = p;
  }
}
static int32_t heap_pop(int32_t *h, int64_t *n) {
  int32_t top = h[0];
  h[0] = h[--(*n)];
  int64_t i = 0;
  for (;;) {
    int64_t l = 2 * i + 1, r = l + 1, m = i;
    if (l < *n && h[l] < h[m]) m = l;
    if (r < *n && h[r] < h[m]) m = r;
    if (m == i) break;
    int32_t t = h[m]; h[m] = h[i]; h[i] = t;
    i = m;
  }
  return top;
}

/* Close the subset in w->work[0..n) over word-0 arcs and keep the minimal part
 * in w->work (sorted by vert). Returns its size, or -1. */
static int64_t closure(const RawLat *L, WL *w, int64_t n) {
  w->cur_stamp++;
  if (lgrow((void **)&w->heap, &w->heap_cap, L->n + 1, sizeof(int32_t))) return -1;
  int64_t nh = 0;
  for (int64_t i = 0; i < n; i++) {
    int32_t v = w->work[i].vert;
    if (w->stamp[v] != w->cur_stamp) {
      w->stamp[v] = w->cur_stamp;
      w->rg[v] = w->work[i].g;
      w->ra[v] = w->work[i].a;
      heap_push(w->heap, &nh, v);
    } else if (w_less(w->work[i].g, w->work[i].a, w->rg[v], w->ra[v])) {
      w->rg[v] = w->work[i].g;
      w->ra[v] = w->work[i].a;
    }
  }
  int64_t m = 0;
  while (nh > 0) {
    int32_t v = heap_pop(w->heap, &nh);
    float g = w->rg[v], a = w->ra[v];
    int keep = (L->fin[v] != INFINITY);
    for (int32_t j = L->arc_start[v]; j < L->arc_start[v + 1]; j++) {
      if (L->ol[j] != 0) { keep = 1; continue; }
      int32_t u = L->dst[j];
      float ng = g + L->g[j], na = a + L->a[j];
      if (w->stamp[u] != w->cur_stamp) {
        w->stamp[u] = w->cur_stamp;
        w->rg[u] = ng; w->ra[u] = na;
        heap_push(w->heap, &nh, u);
      } else if (w_less(ng, na, w->rg[u], w->ra[u])) {
        w->rg[u] = ng; w->ra[u] = na;
      }
    }
    if (keep && L->beta[v] != INFINITY) {
      if (lgrow((void **)&w->work, &w->work_cap, m + 1, sizeof(Elem))) return -1;
      /* work[0..n) has been read already; v pops in increasing order */
      w->work[m].vert = v; w->work[m].g = g; w->work[m].a = a;
      m++;
    }
  }
  return m;
}

/* Find or add the state for the minimal subset in w->work[0..m). */
static int32_t find_or_add_state(WL *w, int64_t m, const RawLat *L, int *is_new) {
  uint32_t h = subset_hash(w->work, (int32_t)m);
  uint32_t i = h & w->hmask;
  while (w->htab[i] >= 0) {
    DState *s = &w->st[w->htab[i]];
    if (s->hash == h && s->len == m && subset_equal(&w->elems[s->off], w->work, (int32_t)m)) {
      *is_new = 0;
      return w->htab[i];
    }
    i = (i + 1) & w->hmask;
  }
  if (lgrow((void **)&w->st, &w->ns_cap, w->ns + 1, sizeof(DState))) return -1;
  if (lgrow((void **)&w->elems, &w->ne_cap, w->ne + m, sizeof(Elem))) return -1;
  int32_t id = (int32_t)w->ns++;
  DState *s = &w->st[id];
  s->off = (int32_t)w->ne;
  s->len = (int32_t)m;
  s->hash = h;
  s->F = INFINITY;
  s->done = 0;
  memcpy(&w->elems[w->ne], w->work, (size_t)m * sizeof(Elem));
  w->ne += m;
  double B = INFINITY;
  float fg = INFINITY, fa = INFINITY;
  int fok = 0;
  for (int64_t k = 0; k < m; k++) {
    const Elem *e = &w->work[k];
    double b = (double)e->g + (double)e->a + L->beta[e->vert];
    if (b < B) B = b;
    if (L->fin[e->vert] != INFINITY) {
      float g = e->g + L->fin[e->vert], a = e->a;
      if (!fok || w_less(g, a, fg, fa)) { fg = g; fa = a; fok = 1; }
    }
  }
  s->B = B; s->fg = fg; s->fa = fa; s->final_ok = fok;
  w->htab[i] = id;
  if ((uint64_t)w->ns * 2 > (uint64_t)w->hmask + 1) {
    uint32_t nc = (w->hmask + 1) * 2;
    int32_t *nt = malloc((size_t)nc * sizeof(int32_t));
    if (!nt) return -1;
    for (uint32_t k = 0; k < nc; k++) nt[k] = -1;
    for (int64_t k = 0; k < w->ns; k++) {
      uint32_t j = w->st[k].hash & (nc - 1);
      while (nt[j] >= 0) j = (j + 1) & (nc - 1);
      nt[j] = (int32_t)k;
    }
    free(w->htab);
    w->htab = nt;
    w->hmask = nc - 1;
  }
  *is_new = 1;
  return id;
}

static int q_push(WL *w, double key, int32_t s) {
  int64_t cap = w->q_cap;
  if (lgrow((void **)&w->qk, &cap, w->nq + 1, sizeof(double))) return -1;
  if (lgrow((void **)&w->qs, &w->q_cap, w->nq + 1, sizeof(int32_t))) return -1;
  int64_t i = w->nq++;
  w->qk[i] = key; w->qs[i] = s;
  while (i > 0) {
    int64_t p = (i - 1) / 2;
    if (w->qk[p] <= w->qk[i]) break;
    double tk = w->qk[p]; w->qk[p] = w->qk[i]; w->qk[i] = tk;
    int32_t ts = w->qs[p]; w->qs[p] = w->qs[i]; w->qs[i] = ts;
    i = p;
  }
  return 0;
}
static int32_t q_pop(WL *w) {
  int32_t top = w->qs[0];
  w->nq--;
  w->qk[0] = w->qk[w->nq]; w->qs[0] = w->qs[w->nq];
  int64_t i = 0;
  for (;;) {
    int64_t l = 2 * i + 1, r = l + 1, m = i;
    if (l < w->nq && w->qk[l] < w->qk[m]) m = l;
    if (r < w->nq && w->qk[r] < w->qk[m]) m = r;
    if (m == i) break;
    double tk = w->qk[m]; w->qk[m] = w->qk[i]; w->qk[i] = tk;
    int32_t ts = w->qs[m]; w->qs[m] = w->qs[i]; w->qs[i] = ts;
    i = m;
  }
  return top;
}

static int cand_cmp(const void *x, const void *y) {
  const Cand *p = x, *q = y;
  if (p->word != q->word) return p->word < q->word ? -1 : 1;
  if (p->vert != q->vert) return p->vert < q->vert ? -1 : 1;
  return 0;
}

static int build_word_lattice(const RawLat *L, float beam, WL *w) {
  memset(w, 0, sizeof(*w));
  w->stamp = calloc((size_t)L->n + 1, sizeof(uint32_t));
  w->rg = malloc(((size_t)L->n + 1) * sizeof(float));
  w->ra = malloc(((size_t)L->n + 1) * sizeof(float));
  w->hmask = 1023;
  w->htab = malloc(1024 * sizeof(int32_t));
  if (!w->stamp || !w->rg || !w->ra || !w->htab) return -1;
  for (int k = 0; k < 1024; k++) w->htab[k] = -1;
  double best = L->beta[L->start];
  if (best == INFINITY) return -2;
  if (lgrow((void **)&w->work, &w->work_cap, 1, sizeof(Elem))) return -1;
  w->work[0].vert = L->start; w->work[0].g = 0.0f; w->work[0].a = 0.0f;
  int64_t m = closure(L, w, 1);
  if (m < 0) return -1;
  int is_new;
  int32_t s0 = find_or_add_state(w, m, L, &is_new);
  if (s0 < 0) return -1;
  w->st[s0].F = 0.0;
  if (q_push(w, w->st[s0].B, s0)) return -1;
  while (w->nq > 0) {
    int32_t s = q_pop(w);
    if (w->st[s].done) continue;
    w->st[s].done = 1;
    double F = w->st[s].F;
    /* gather the word arcs leaving this subset */
    int64_t nc = 0;
    int32_t off = w->st[s].off, len = w->st[s].len;
    for (int32_t k = 0; k < len; k++) {
      Elem e = w->elems[off + k];
      for (int32_t j = L->arc_start[e.vert]; j < L->arc_start[e.vert + 1]; j++) {
        if (L->ol[j] == 0) continue;
        if (L->beta[L->dst[j]] == INFINITY) continue;
        if (lgrow((void **)&w->cand, &w->cand_cap, nc + 1, sizeof(Cand))) return -1;
        Cand *c = &w->cand[nc++];
        c->word = L->ol[j]; c->vert = L->dst[j];
        c->g = e.g + L->g[j]; c->a = e.a + L->a[j];
      }
    }
    qsort(w->cand, (size_t)nc, sizeof(Cand), cand_cmp);
    for (int64_t i = 0; i < nc;) {
      int64_t j = i;
      while (j < nc && w->cand[j].word == w->cand[i].word) j++;
      /* divisor: the smallest weight in this word's group */
      float dg = w->cand[i].g, da = w->cand[i].a;
      for (int64_t k = i + 1; k < j; k++)
        if (w_less(w->cand[k].g, w->cand[k].a, dg, da)) dg = w->cand[k].g, da = w->cand[k].a;
      int64_t n = 0;
      for (int64_t k = i; k < j; k++) {
        if (lgrow((void **)&w->work, &w->work_cap, n + 1, sizeof(Elem))) return -1;
        w->work[n].vert = w->cand[k].vert;
        w->work[n].g = w->cand[k].g - dg;
        w->work[n].a = w->cand[k].a - da;
        n++;
      }
      int64_t mm = closure(L, w, n);
      if (mm < 0) return -1;
      if (mm > 0) {
        int32_t t = find_or_add_state(w, mm, L, &is_new);
        if (t < 0) return -1;
        double through = F + (double)dg + (double)da + w->st[t].B;
        if (through <= best + beam) {
          if (lgrow((void **)&w->arcs, &w->na_cap, w->na + 1, sizeof(DArc))) return -1;
          DArc *ar = &w->arcs[w->na++];
          ar->src = s; ar->dst = t; ar->word = w->cand[i].word; ar->g = dg; ar->a = da;
          double nf = F + (double)dg + (double)da;
          if (nf < w->st[t].F) {
            w->st[t].F = nf;
            if (!w->st[t].done && q_push(w, nf + w->st[t].B, t)) return -1;
          }
        }
      }
      i = j;
    }
  }
  return 0;
}

/* ---------------------------------------------------------------------------
 * MBR posteriors (Xu, Povey, Mangu, Zhu 2011, Figures 4 and 5): the word
 * lattice is topologically sorted with one super-final state; R gets an
 * epsilon slot before, between and after its words; one forward-backward
 * pass over the edit-distance alignment to R gives gamma(q, word), and the
 * confidence of R's q-th word is gamma(q, R_q).
 * ------------------------------------------------------------------------- */

static inline double log_add(double x, double y) {
  if (x < y) { double t = x; x = y; y = t; }
  double diff = y - x;
  if (diff >= log(DBL_EPSILON)) return x + log1p(exp(diff));
  return x;
}

#define MBR_DELTA 1.0e-05
static inline double lcost(int32_t a, int32_t b, int penalize) {
  if (a == b) return 0.0;
  return penalize ? 1.0 + MBR_DELTA : 1.0;
}

typedef struct { int32_t start, end, word; float loglike; } MArc;

int ec_dec_word_posteriors(ec_dec *d, int32_t use_final, const int32_t *Rin, int32_t nR,
                           float graph_scale, float *conf, double *stats4) {
  if (use_final && !d->finalized && ec_dec_finalize(d)) return -1;
  if (build_raw(d, use_final)) return -1;
  RawLat *L = &d->lat;
  WL w;
  int rc = build_word_lattice(L, d->lattice_beam, &w);
  if (rc == -2) { wl_free(&w); return lfail(d, "no path reaches the end of the lattice"); }
  if (rc) { wl_free(&w); return lfail(d, "out of memory"); }
  int32_t S = (int32_t)w.ns;
  /* Keep the states that are reached (finite forward cost) and can reach an
   * end, number them in topological order from 1, and give the lattice one
   * final state (a super-final state, unless there is exactly one final state
   * with zero final weight and no arcs out). */
  int32_t *indeg = calloc((size_t)S + 1, sizeof(int32_t)), *order = malloc(((size_t)S + 1) * sizeof(int32_t));
  int32_t *num = malloc(((size_t)S + 1) * sizeof(int32_t));
  int32_t *astart = calloc((size_t)S + 2, sizeof(int32_t)), *aidx = malloc(((size_t)w.na + 1) * sizeof(int32_t));
  char *coacc = calloc((size_t)S + 1, 1);
  MArc *ma = NULL;
  int32_t *pstart = NULL, *pidx = NULL;
  if (!indeg || !order || !num || !astart || !aidx || !coacc) goto oom_graph;
  for (int64_t k = 0; k < w.na; k++) { indeg[w.arcs[k].dst]++; astart[w.arcs[k].src + 1]++; }
  for (int32_t s = 0; s < S; s++) astart[s + 1] += astart[s];
  {
    int32_t *fill = malloc(((size_t)S + 1) * sizeof(int32_t));
    if (!fill) goto oom_graph;
    memcpy(fill, astart, (size_t)S * sizeof(int32_t));
    for (int64_t k = 0; k < w.na; k++) aidx[fill[w.arcs[k].src]++] = (int32_t)k;
    free(fill);
  }
  int32_t head = 0, tail = 0;
  for (int32_t s = 0; s < S; s++)
    if (w.st[s].F != INFINITY && indeg[s] == 0) order[tail++] = s;
  while (head < tail) {
    int32_t s = order[head++];
    for (int32_t k = astart[s]; k < astart[s + 1]; k++) {
      int32_t t = w.arcs[aidx[k]].dst;
      if (--indeg[t] == 0) order[tail++] = t;
    }
  }
  int32_t alive = 0;
  for (int32_t s = 0; s < S; s++) alive += (w.st[s].F != INFINITY);
  if (tail != alive || alive == 0 || order[0] != 0) {
    free(indeg); free(order); free(num); free(astart); free(aidx); free(coacc); wl_free(&w);
    return lfail(d, "word lattice is not acyclic from its start");
  }
  for (int32_t i = tail - 1; i >= 0; i--) {
    int32_t s = order[i];
    char c = (char)w.st[s].final_ok;
    for (int32_t k = astart[s]; k < astart[s + 1] && !c; k++) c = coacc[w.arcs[aidx[k]].dst];
    coacc[s] = c;
  }
  if (!coacc[0]) {
    free(indeg); free(order); free(num); free(astart); free(aidx); free(coacc); wl_free(&w);
    return lfail(d, "word lattice has no complete path");
  }
  int32_t kept = 0;
  for (int32_t i = 0; i < tail; i++) {
    int32_t s = order[i];
    num[s] = coacc[s] ? ++kept : 0;
  }
  for (int32_t s = 0; s < S; s++) if (w.st[s].F == INFINITY) num[s] = 0;
  int64_t kept_arcs = 0;
  for (int64_t k = 0; k < w.na; k++) kept_arcs += (num[w.arcs[k].src] && num[w.arcs[k].dst]);
  int32_t nfinal = 0, the_final = -1;
  for (int32_t s = 0; s < S; s++) if (num[s] && w.st[s].final_ok) { nfinal++; the_final = s; }
  int has_out = 0;
  if (nfinal == 1)
    for (int32_t k = astart[the_final]; k < astart[the_final + 1]; k++)
      if (num[w.arcs[aidx[k]].dst]) has_out = 1;
  int use_existing = (nfinal == 1 && w.st[the_final].fg == 0.0f && w.st[the_final].fa == 0.0f &&
                      !has_out && num[the_final] == kept);
  int32_t N = use_existing ? kept : kept + 1;
  int64_t nm = kept_arcs + (use_existing ? 0 : nfinal);
  ma = malloc(((size_t)nm + 1) * sizeof(MArc));
  pstart = calloc((size_t)N + 2, sizeof(int32_t));
  pidx = malloc(((size_t)nm + 1) * sizeof(int32_t));
  if (!ma || !pstart || !pidx) goto oom_graph;
  int64_t k2 = 0;
  for (int64_t k = 0; k < w.na; k++) {
    const DArc *ar = &w.arcs[k];
    if (!num[ar->src] || !num[ar->dst]) continue;
    ma[k2].start = num[ar->src]; ma[k2].end = num[ar->dst]; ma[k2].word = ar->word;
    float g = ar->g * graph_scale;
    ma[k2].loglike = -(g + ar->a);
    k2++;
  }
  if (!use_existing)
    for (int32_t s = 0; s < S; s++)
      if (num[s] && w.st[s].final_ok) {
        float g = w.st[s].fg * graph_scale;
        ma[k2].start = num[s]; ma[k2].end = N; ma[k2].word = 0; ma[k2].loglike = -(g + w.st[s].fa);
        k2++;
      }
  for (int64_t k = 0; k < nm; k++) pstart[ma[k].end + 1]++;
  for (int32_t n = 0; n <= N; n++) pstart[n + 1] += pstart[n];
  {
    int32_t *fill = malloc(((size_t)N + 2) * sizeof(int32_t));
    if (!fill) goto oom_graph;
    memcpy(fill, pstart, ((size_t)N + 1) * sizeof(int32_t));
    for (int64_t k = 0; k < nm; k++) pidx[fill[ma[k].end]++] = (int32_t)k;
    free(fill);
  }
  free(coacc);
  coacc = NULL;
  /* R with epsilon slots */
  int32_t Q = 2 * nR + 1;
  int32_t *R = malloc((size_t)Q * sizeof(int32_t));
  double *alpha = malloc(((size_t)N + 1) * sizeof(double));
  double *alpha_dash = calloc(((size_t)N + 1) * (size_t)(Q + 1), sizeof(double));
  double *beta_dash = calloc(((size_t)N + 1) * (size_t)(Q + 1), sizeof(double));
  double *ad_arc = malloc(((size_t)Q + 1) * sizeof(double)), *bd_arc = malloc(((size_t)Q + 1) * sizeof(double));
  char *b_arc = malloc((size_t)Q + 1);
  double *gword = calloc((size_t)Q + 1, sizeof(double)), *gsum = calloc((size_t)Q + 1, sizeof(double));
  if (!R || !alpha || !alpha_dash || !beta_dash || !ad_arc || !bd_arc || !b_arc || !gword || !gsum) {
    free(R); free(alpha); free(alpha_dash); free(beta_dash); free(ad_arc); free(bd_arc); free(b_arc);
    free(gword); free(gsum); free(ma); free(pstart); free(pidx); free(indeg); free(order); free(num);
    free(astart); free(aidx); wl_free(&w);
    return lfail(d, "out of memory");
  }
  for (int32_t q = 0; q < Q; q++) R[q] = (q % 2 == 1) ? Rin[q / 2] : 0;
#define AD(n, q) alpha_dash[(size_t)(n) * (size_t)(Q + 1) + (size_t)(q)]
#define BD(n, q) beta_dash[(size_t)(n) * (size_t)(Q + 1) + (size_t)(q)]
#define RQ(q) R[(q) - 1]
  /* forward (Figure 4) */
  alpha[1] = 0.0;
  AD(1, 0) = 0.0;
  for (int32_t q = 1; q <= Q; q++) AD(1, q) = AD(1, q - 1) + lcost(0, RQ(q), 0);
  for (int32_t n = 2; n <= N; n++) {
    double an = -INFINITY;
    for (int32_t i = pstart[n]; i < pstart[n + 1]; i++) {
      const MArc *ar = &ma[pidx[i]];
      an = log_add(an, alpha[ar->start] + ar->loglike);
    }
    alpha[n] = an;
    for (int32_t i = pstart[n]; i < pstart[n + 1]; i++) {
      const MArc *ar = &ma[pidx[i]];
      int32_t sa = ar->start, wa = ar->word;
      double pa = ar->loglike;
      double occ = exp(alpha[sa] + pa - alpha[n]);
      for (int32_t q = 0; q <= Q; q++) {
        if (q == 0) {
          ad_arc[q] = AD(sa, q) + lcost(wa, 0, 1);
        } else {
          int32_t rq = RQ(q);
          double a1 = AD(sa, q - 1) + lcost(wa, rq, 0), a2 = AD(sa, q) + lcost(wa, 0, 1),
                 a3 = ad_arc[q - 1] + lcost(0, rq, 0);
          ad_arc[q] = a1 < a2 ? (a1 < a3 ? a1 : a3) : (a2 < a3 ? a2 : a3);
        }
        AD(n, q) += occ * ad_arc[q];
      }
    }
  }
  /* backward (Figure 5) */
  BD(N, Q) = 1.0;
  for (int32_t n = N; n >= 2; n--) {
    for (int32_t i = pstart[n]; i < pstart[n + 1]; i++) {
      const MArc *ar = &ma[pidx[i]];
      int32_t sa = ar->start, wa = ar->word;
      double pa = ar->loglike;
      double occ = exp(alpha[sa] + pa - alpha[n]);
      ad_arc[0] = AD(sa, 0) + lcost(wa, 0, 1);
      for (int32_t q = 1; q <= Q; q++) {
        int32_t rq = RQ(q);
        double a1 = AD(sa, q - 1) + lcost(wa, rq, 0), a2 = AD(sa, q) + lcost(wa, 0, 1),
               a3 = ad_arc[q - 1] + lcost(0, rq, 0);
        if (a1 <= a2) {
          if (a1 <= a3) { b_arc[q] = 1; ad_arc[q] = a1; } else { b_arc[q] = 3; ad_arc[q] = a3; }
        } else {
          if (a2 <= a3) { b_arc[q] = 2; ad_arc[q] = a2; } else { b_arc[q] = 3; ad_arc[q] = a3; }
        }
      }
      for (int32_t q = 0; q <= Q; q++) bd_arc[q] = 0.0;
      for (int32_t q = Q; q >= 1; q--) {
        bd_arc[q] += occ * BD(n, q);
        switch (b_arc[q]) {
          case 1:
            BD(sa, q - 1) += bd_arc[q];
            gsum[q] += bd_arc[q];
            if (wa == RQ(q)) gword[q] += bd_arc[q];
            break;
          case 2:
            BD(sa, q) += bd_arc[q];
            break;
          default:
            bd_arc[q - 1] += bd_arc[q];
            gsum[q] += bd_arc[q];
            if (RQ(q) == 0) gword[q] += bd_arc[q];
            break;
        }
      }
      bd_arc[0] += occ * BD(n, 0);
      BD(sa, 0) += bd_arc[0];
    }
  }
  for (int32_t q = 0; q <= Q; q++) bd_arc[q] = 0.0;
  for (int32_t q = Q; q >= 1; q--) {
    bd_arc[q] += BD(1, q);
    bd_arc[q - 1] += bd_arc[q];
    gsum[q] += bd_arc[q];
    if (RQ(q) == 0) gword[q] += bd_arc[q];
  }
  double maxdev = 0.0;
  for (int32_t q = 1; q <= Q; q++) {
    double dv = fabs(gsum[q] - 1.0);
    if (dv > maxdev) maxdev = dv;
  }
  for (int32_t k = 0; k < nR; k++) conf[k] = (float)gword[2 * k + 2];
  stats4[0] = (double)kept;
  stats4[1] = (double)kept_arcs;
  stats4[2] = (double)L->narcs;
  stats4[3] = maxdev;
#undef AD
#undef BD
#undef RQ
  free(R); free(alpha); free(alpha_dash); free(beta_dash); free(ad_arc); free(bd_arc); free(b_arc);
  free(gword); free(gsum); free(ma); free(pstart); free(pidx); free(indeg); free(order); free(num);
  free(astart); free(aidx); wl_free(&w);
  return 0;
oom_graph:
  free(indeg); free(order); free(num); free(astart); free(aidx); free(coacc);
  free(ma); free(pstart); free(pidx); wl_free(&w);
  return lfail(d, "out of memory");
}
