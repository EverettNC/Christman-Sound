/* Lattice-faster token passing over the memory-mapped HCLG, in tropical costs.
 *
 * Per frame: prune the token lattice every 25 frames (at a tenth of the
 * lattice beam), take the cutoff from the previous frame's tokens (beam,
 * max_active, min_active, adaptive beam), cross every emitting arc (a
 * transition id that is not 0 and not a disambiguation id), then close over
 * the epsilon arcs. Tokens keep forward links (input label, output label,
 * graph cost, acoustic cost), so the lattice can be pruned at lattice_beam
 * and read back. Acoustic costs on links carry a per-frame offset (minus the
 * best token's cost) that keeps the float sums small; it is taken back out
 * whenever a cost is read from the lattice.
 */
#include "decoder_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRUNE_INTERVAL 25
#define PRUNE_SCALE 0.1f
#define BEAM_DELTA 0.5f
#define FINAL_DELTA 1.0e-05f

static int fail(ec_dec *d, const char *msg) {
  snprintf(d->err, sizeof(d->err), "%s", msg);
  return -1;
}

const char *ec_dec_error(const ec_dec *d) { return d->err; }

static int grow(void **p, int64_t *cap, int64_t need, size_t elem) {
  if (need <= *cap) return 0;
  int64_t nc = *cap ? *cap : 1024;
  while (nc < need) nc *= 2;
  void *q = realloc(*p, (size_t)nc * elem);
  if (!q) return -1;
  *p = q;
  *cap = nc;
  return 0;
}

/* ---- pools ---- */
static int32_t tok_new(ec_dec *d) {
  int32_t t;
  if (d->tok_free >= 0) {
    t = d->tok_free;
    d->tok_free = d->toks[t].next;
  } else {
    if (grow((void **)&d->toks, &d->tok_cap, d->tok_used + 1, sizeof(Tok))) return -1;
    t = (int32_t)d->tok_used++;
  }
  d->num_toks++;
  return t;
}
static void tok_del(ec_dec *d, int32_t t) {
  d->toks[t].state = -1;
  d->toks[t].next = d->tok_free;
  d->tok_free = t;
  d->num_toks--;
}
static int32_t link_new(ec_dec *d) {
  int32_t l;
  if (d->link_free >= 0) {
    l = d->link_free;
    d->link_free = d->links[l].next;
  } else {
    if (grow((void **)&d->links, &d->link_cap, d->link_used + 1, sizeof(Link))) return -1;
    l = (int32_t)d->link_used++;
  }
  return l;
}
static void link_del(ec_dec *d, int32_t l) {
  d->links[l].next = d->link_free;
  d->link_free = l;
}
static void delete_links(ec_dec *d, int32_t t) {
  int32_t l = d->toks[t].links;
  while (l >= 0) {
    int32_t n = d->links[l].next;
    link_del(d, l);
    l = n;
  }
  d->toks[t].links = -1;
}
static int add_link(ec_dec *d, int32_t from, int32_t to, int32_t il, int32_t ol, float gc,
                    float ac) {
  int32_t l = link_new(d);
  if (l < 0) return -1;
  Link *k = &d->links[l];
  k->next_tok = to;
  k->ilabel = il;
  k->olabel = ol;
  k->graph_cost = gc;
  k->ac_cost = ac;
  k->next = d->toks[from].links;
  d->toks[from].links = l;
  return 0;
}

/* ---- the current frame's state -> token table ---- */
static inline uint32_t hfun(int32_t s) { return (uint32_t)s * 2654435761u; }

static int hash_reset(ec_dec *d) {
  d->hstamp++;
  if (d->hstamp == 0) {
    memset(d->hslots, 0, (size_t)(d->hmask + 1) * sizeof(HSlot));
    d->hstamp = 1;
  }
  d->ncur = 0;
  return 0;
}

static int hash_rehash(ec_dec *d, uint32_t newcap) {
  HSlot *ns = calloc(newcap, sizeof(HSlot));
  if (!ns) return -1;
  free(d->hslots);
  d->hslots = ns;
  d->hmask = newcap - 1;
  d->hstamp = 1;
  for (int64_t i = 0; i < d->ncur; i++) {
    int32_t t = d->cur[i];
    int32_t s = d->toks[t].state;
    uint32_t h = hfun(s) & d->hmask;
    while (d->hslots[h].stamp == d->hstamp) h = (h + 1) & d->hmask;
    d->hslots[h].stamp = d->hstamp;
    d->hslots[h].state = s;
    d->hslots[h].tok = t;
  }
  return 0;
}

static inline int32_t hash_find(const ec_dec *d, int32_t s) {
  uint32_t h = hfun(s) & d->hmask;
  while (d->hslots[h].stamp == d->hstamp) {
    if (d->hslots[h].state == s) return d->hslots[h].tok;
    h = (h + 1) & d->hmask;
  }
  return -1;
}

/* Returns the token for state s on the newest frame, adding it if absent and
 * lowering its cost if tot_cost is better. *changed says whether it moved. */
static int32_t find_or_add(ec_dec *d, int32_t s, int32_t frame, float tot_cost, int *changed) {
  uint32_t h = hfun(s) & d->hmask;
  while (d->hslots[h].stamp == d->hstamp) {
    if (d->hslots[h].state == s) {
      int32_t t = d->hslots[h].tok;
      if (d->toks[t].tot_cost > tot_cost) {
        d->toks[t].tot_cost = tot_cost;
        if (changed) *changed = 1;
      } else if (changed) {
        *changed = 0;
      }
      return t;
    }
    h = (h + 1) & d->hmask;
  }
  int32_t t = tok_new(d);
  if (t < 0) return -1;
  Tok *k = &d->toks[t];
  k->tot_cost = tot_cost;
  k->extra_cost = 0.0f;
  k->state = s;
  k->links = -1;
  k->next = d->frames[frame].toks;
  d->frames[frame].toks = t;
  d->hslots[h].stamp = d->hstamp;
  d->hslots[h].state = s;
  d->hslots[h].tok = t;
  if (grow((void **)&d->cur, &d->cur_cap, d->ncur + 1, sizeof(int32_t))) return -1;
  d->cur[d->ncur++] = t;
  if ((uint64_t)d->ncur * 2 > (uint64_t)d->hmask + 1) {
    if (hash_rehash(d, (d->hmask + 1) * 2)) return -1;
  }
  if (changed) *changed = 1;
  return t;
}

/* ---- labels ---- */
static inline int is_eps_label(const ec_dec *d, int32_t il) {
  if (il == 0) return 1;
  return il >= 0 && il < d->dis_len && d->dis[il];
}
static int state_needs_eps(const ec_dec *d, int32_t s) {
  ec_state st;
  ec_get_state(d->g, s, &st);
  if (st.nieps > 0) return 1;
  if (!d->num_disambig) return 0;
  for (uint32_t i = 0; i < st.narcs; i++) {
    ec_arc a;
    ec_get_arc(d->g, st.pos + i, &a);
    if (a.ilabel != 0 && is_eps_label(d, a.ilabel)) return 1;
  }
  return 0;
}

/* ---- construction ---- */
ec_dec *ec_dec_new(const ec_graph *g, const int32_t *tid2pdf, int32_t num_tids_plus_one,
                   int32_t num_pdfs, const int32_t *disambig, int32_t num_disambig,
                   float beam, int32_t max_active, int32_t min_active, float lattice_beam,
                   float acoustic_scale) {
  ec_dec *d = calloc(1, sizeof(*d));
  if (!d) return NULL;
  d->g = g;
  d->beam = beam;
  d->max_active = max_active;
  d->min_active = min_active;
  d->lattice_beam = lattice_beam;
  d->acoustic_scale = acoustic_scale;
  d->num_pdfs = num_pdfs;
  d->ntid = num_tids_plus_one;
  d->tok_free = d->link_free = -1;
  d->tid2pdf = malloc((size_t)(num_tids_plus_one > 0 ? num_tids_plus_one : 1) * sizeof(int32_t));
  if (!d->tid2pdf) goto oom;
  for (int32_t i = 0; i < num_tids_plus_one; i++) {
    int32_t p = tid2pdf[i];
    if (i > 0 && (p < 0 || p >= num_pdfs)) {
      fail(d, "a transition id maps to a pdf outside 0..num_pdfs-1");
      d->bad = 1;
    }
    d->tid2pdf[i] = p;
  }
  int32_t mx = 0;
  for (int32_t i = 0; i < num_disambig; i++)
    if (disambig[i] > mx) mx = disambig[i];
  d->dis_len = mx + 1;
  d->dis = calloc((size_t)d->dis_len, 1);
  if (!d->dis) goto oom;
  for (int32_t i = 0; i < num_disambig; i++)
    if (disambig[i] > 0) d->dis[disambig[i]] = 1, d->num_disambig++;
  d->hslots = calloc(1 << 14, sizeof(HSlot));
  if (!d->hslots) goto oom;
  d->hmask = (1 << 14) - 1;
  d->hstamp = 1;
  d->ll = malloc((size_t)(num_pdfs > 0 ? num_pdfs : 1) * sizeof(float));
  if (!d->ll) goto oom;
  return d;
oom:
  ec_dec_free(d);
  return NULL;
}

void ec_dec_free(ec_dec *d) {
  if (!d) return;
  free(d->tid2pdf);
  free(d->dis);
  free(d->toks);
  free(d->links);
  free(d->hslots);
  free(d->cur);
  free(d->prev);
  free(d->frames);
  free(d->cost_offsets);
  free(d->tmp);
  free(d->queue);
  free(d->ll);
  ec_lat_free(&d->lat);
  free(d);
}

static void clear_all(ec_dec *d) {
  d->tok_used = d->link_used = 0;
  d->tok_free = d->link_free = -1;
  d->num_toks = 0;
  d->nframes = 0;
  d->finalized = 0;
  d->ncur = 0;
  hash_reset(d);
}

static int new_frame(ec_dec *d) {
  if (grow((void **)&d->frames, &d->frames_cap, d->nframes + 2, sizeof(FrameToks))) return -1;
  if (grow((void **)&d->cost_offsets, &d->co_cap, d->nframes + 2, sizeof(float))) return -1;
  FrameToks *f = &d->frames[d->nframes + 1];
  f->toks = -1;
  f->must_prune_forward_links = 1;
  f->must_prune_tokens = 1;
  return 0;
}

/* ---- epsilon closure on the newest frame ---- */
static int process_nonemitting(ec_dec *d, float cutoff) {
  int32_t frame = d->nframes; /* newest token list */
  int64_t nq = 0;
  for (int64_t i = 0; i < d->ncur; i++) {
    int32_t s = d->toks[d->cur[i]].state;
    if (state_needs_eps(d, s)) {
      if (grow((void **)&d->queue, &d->queue_cap, nq + 1, sizeof(int32_t))) return -1;
      d->queue[nq++] = s;
    }
  }
  while (nq > 0) {
    int32_t s = d->queue[--nq];
    int32_t t = hash_find(d, s);
    float cur_cost = d->toks[t].tot_cost;
    if (cur_cost >= cutoff) continue;
    delete_links(d, t);
    ec_state st;
    ec_get_state(d->g, s, &st);
    for (uint32_t i = 0; i < st.narcs; i++) {
      ec_arc a;
      ec_get_arc(d->g, st.pos + i, &a);
      if (!is_eps_label(d, a.ilabel)) continue;
      float graph_cost = a.weight, tot_cost = cur_cost + graph_cost;
      if (tot_cost < cutoff) {
        int changed = 0;
        int32_t nt = find_or_add(d, a.nextstate, frame, tot_cost, &changed);
        if (nt < 0) return fail(d, "out of memory");
        if (add_link(d, t, nt, 0, a.olabel, graph_cost, 0.0f)) return fail(d, "out of memory");
        if (changed && state_needs_eps(d, a.nextstate)) {
          if (grow((void **)&d->queue, &d->queue_cap, nq + 1, sizeof(int32_t)))
            return fail(d, "out of memory");
          d->queue[nq++] = a.nextstate;
        }
      }
    }
  }
  return 0;
}

/* nth_element: afterwards a[k] holds the value that sorted order puts there,
 * everything before it is <= and everything after is >=. */
static void select_nth(float *a, int64_t n, int64_t k) {
  int64_t lo = 0, hi = n - 1;
  while (hi > lo) {
    int64_t mid = lo + (hi - lo) / 2;
    if (a[mid] < a[lo]) { float t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
    if (a[hi] < a[lo]) { float t = a[hi]; a[hi] = a[lo]; a[lo] = t; }
    if (a[hi] < a[mid]) { float t = a[hi]; a[hi] = a[mid]; a[mid] = t; }
    float pivot = a[mid];
    int64_t i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        float t = a[i]; a[i] = a[j]; a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return;
  }
}

/* Cutoff for the tokens in d->prev: beam, max_active, min_active. */
static float get_cutoff(ec_dec *d, float *adaptive_beam, int32_t *best_tok) {
  float best = INFINITY;
  *best_tok = -1;
  int64_t n = d->nprev;
  if (d->max_active == INT32_MAX && d->min_active == 0) {
    for (int64_t i = 0; i < n; i++) {
      float w = d->toks[d->prev[i]].tot_cost;
      if (w < best) { best = w; *best_tok = d->prev[i]; }
    }
    *adaptive_beam = d->beam;
    return best + d->beam;
  }
  if (grow((void **)&d->tmp, &d->tmp_cap, n + 1, sizeof(float))) return -INFINITY;
  for (int64_t i = 0; i < n; i++) {
    float w = d->toks[d->prev[i]].tot_cost;
    d->tmp[i] = w;
    if (w < best) { best = w; *best_tok = d->prev[i]; }
  }
  float beam_cutoff = best + d->beam, min_active_cutoff = INFINITY, max_active_cutoff = INFINITY;
  if (n > d->max_active) {
    select_nth(d->tmp, n, d->max_active);
    max_active_cutoff = d->tmp[d->max_active];
  }
  if (max_active_cutoff < beam_cutoff) {
    *adaptive_beam = max_active_cutoff - best + BEAM_DELTA;
    return max_active_cutoff;
  }
  if (n > d->min_active) {
    if (d->min_active == 0) {
      min_active_cutoff = best;
    } else {
      select_nth(d->tmp, n > d->max_active ? d->max_active : n, d->min_active);
      min_active_cutoff = d->tmp[d->min_active];
    }
  }
  if (min_active_cutoff > beam_cutoff) {
    *adaptive_beam = min_active_cutoff - best + BEAM_DELTA;
    return min_active_cutoff;
  }
  *adaptive_beam = d->beam;
  return beam_cutoff;
}

static float process_emitting(ec_dec *d, const float *loglikes_row) {
  int32_t frame = d->nframes; /* tokens on frame f, arcs consume frame f */
  for (int32_t p = 0; p < d->num_pdfs; p++) d->ll[p] = d->acoustic_scale * loglikes_row[p];
  if (new_frame(d)) {
    fail(d, "out of memory");
    return NAN;
  }
  /* hand the newest frame's tokens over as the previous list */
  int32_t *tp = d->prev; d->prev = d->cur; d->cur = tp;
  int64_t tc = d->prev_cap; d->prev_cap = d->cur_cap; d->cur_cap = tc;
  d->nprev = d->ncur;
  hash_reset(d);
  float adaptive_beam;
  int32_t best_tok;
  float cost_cutoff = get_cutoff(d, &adaptive_beam, &best_tok);
  if (cost_cutoff == -INFINITY) {
    fail(d, "out of memory");
    return NAN;
  }
  d->last_cutoff = cost_cutoff;
  d->last_adaptive_beam = adaptive_beam;
  d->last_considered = (float)d->nprev;
  float next_cutoff = INFINITY, cost_offset = 0.0f;
  if (best_tok >= 0) {
    cost_offset = -d->toks[best_tok].tot_cost;
    ec_state st;
    ec_get_state(d->g, d->toks[best_tok].state, &st);
    for (uint32_t i = 0; i < st.narcs; i++) {
      ec_arc a;
      ec_get_arc(d->g, st.pos + i, &a);
      if (is_eps_label(d, a.ilabel)) continue;
      if (a.ilabel >= d->ntid || a.ilabel < 0) {
        fail(d, "graph input label is not a transition id of this model");
        return NAN;
      }
      float new_weight = a.weight + cost_offset - d->ll[d->tid2pdf[a.ilabel]] +
                         d->toks[best_tok].tot_cost;
      if (new_weight + adaptive_beam < next_cutoff) next_cutoff = new_weight + adaptive_beam;
    }
  }
  d->cost_offsets[frame] = cost_offset;
  for (int64_t i = 0; i < d->nprev; i++) {
    int32_t t = d->prev[i];
    float cur_cost = d->toks[t].tot_cost;
    if (!(cur_cost <= cost_cutoff)) continue;
    ec_state st;
    ec_get_state(d->g, d->toks[t].state, &st);
    for (uint32_t j = 0; j < st.narcs; j++) {
      ec_arc a;
      ec_get_arc(d->g, st.pos + j, &a);
      if (is_eps_label(d, a.ilabel)) continue;
      if (a.ilabel >= d->ntid || a.ilabel < 0) {
        fail(d, "graph input label is not a transition id of this model");
        return NAN;
      }
      float ac_cost = cost_offset - d->ll[d->tid2pdf[a.ilabel]], graph_cost = a.weight,
            tot_cost = cur_cost + ac_cost + graph_cost;
      if (tot_cost >= next_cutoff) continue;
      if (tot_cost + adaptive_beam < next_cutoff) next_cutoff = tot_cost + adaptive_beam;
      int32_t nt = find_or_add(d, a.nextstate, frame + 1, tot_cost, NULL);
      if (nt < 0 || add_link(d, t, nt, a.ilabel, a.olabel, graph_cost, ac_cost)) {
        fail(d, "out of memory");
        return NAN;
      }
    }
  }
  d->nframes++;
  return next_cutoff;
}

/* ---- lattice pruning ---- */
static void prune_forward_links(ec_dec *d, int32_t f, int *extra_costs_changed,
                                int *links_pruned, float delta) {
  *extra_costs_changed = 0;
  *links_pruned = 0;
  int changed = 1;
  while (changed) {
    changed = 0;
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) {
      Tok *tok = &d->toks[t];
      int32_t prev_link = -1;
      float tok_extra_cost = INFINITY;
      for (int32_t l = tok->links; l >= 0;) {
        Link *k = &d->links[l];
        Tok *nt = &d->toks[k->next_tok];
        float link_extra_cost =
            nt->extra_cost + ((tok->tot_cost + k->ac_cost + k->graph_cost) - nt->tot_cost);
        if (link_extra_cost > d->lattice_beam) {
          int32_t nl = k->next;
          if (prev_link >= 0) d->links[prev_link].next = nl;
          else tok->links = nl;
          link_del(d, l);
          l = nl;
          *links_pruned = 1;
        } else {
          if (link_extra_cost < 0.0f) link_extra_cost = 0.0f;
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
          prev_link = l;
          l = k->next;
        }
      }
      if (fabsf(tok_extra_cost - tok->extra_cost) > delta) changed = 1;
      tok->extra_cost = tok_extra_cost;
    }
    if (changed) *extra_costs_changed = 1;
  }
}

static void prune_tokens_for_frame(ec_dec *d, int32_t f) {
  int32_t prev = -1;
  for (int32_t t = d->frames[f].toks; t >= 0;) {
    int32_t nt = d->toks[t].next;
    if (d->toks[t].extra_cost == INFINITY) {
      if (prev >= 0) d->toks[prev].next = nt;
      else d->frames[f].toks = nt;
      delete_links(d, t);
      tok_del(d, t);
    } else {
      prev = t;
    }
    t = nt;
  }
}

static void prune_active_tokens(ec_dec *d, float delta) {
  int32_t cur_frame_plus_one = d->nframes;
  for (int32_t f = cur_frame_plus_one - 1; f >= 0; f--) {
    if (d->frames[f].must_prune_forward_links) {
      int ecc = 0, lp = 0;
      prune_forward_links(d, f, &ecc, &lp, delta);
      if (ecc && f > 0) d->frames[f - 1].must_prune_forward_links = 1;
      if (lp) d->frames[f].must_prune_tokens = 1;
      d->frames[f].must_prune_forward_links = 0;
    }
    if (f + 1 < cur_frame_plus_one && d->frames[f + 1].must_prune_tokens) {
      prune_tokens_for_frame(d, f + 1);
      d->frames[f + 1].must_prune_tokens = 0;
    }
  }
}

/* Final costs on the newest frame. any_final says whether any token sits on a
 * state with a finite final weight; best_with_final is the best tot_cost plus
 * final cost (or the best tot_cost when nothing is final); relative is the
 * best cost with final costs minus the best cost without them, infinity when
 * no token has a final cost. */
void ec_final_costs(const ec_dec *d, int *any_final, float *best_with_final, float *relative) {
  float best = INFINITY, bwf = INFINITY;
  int anyf = 0;
  for (int32_t t = d->frames[d->nframes].toks; t >= 0; t = d->toks[t].next) {
    float fc = ec_graph_final(d->g, d->toks[t].state);
    float c = d->toks[t].tot_cost;
    if (c < best) best = c;
    if (c + fc < bwf) bwf = c + fc;
    if (fc != INFINITY) anyf = 1;
  }
  *any_final = anyf;
  *best_with_final = bwf != INFINITY ? bwf : best;
  if (relative) *relative = (best == INFINITY && bwf == INFINITY) ? INFINITY : bwf - best;
}

float ec_tok_final_cost(const ec_dec *d, int32_t t, int any_final) {
  if (!any_final) return 0.0f;
  return ec_graph_final(d->g, d->toks[t].state);
}

static void prune_forward_links_final(ec_dec *d) {
  int32_t f = d->nframes;
  int any_final;
  float best;
  float relative;
  ec_final_costs(d, &any_final, &best, &relative);
  d->final_any = any_final;
  d->final_best = best;
  d->final_relative = relative;
  int changed = 1;
  while (changed) {
    changed = 0;
    for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) {
      Tok *tok = &d->toks[t];
      int32_t prev_link = -1;
      float final_cost = ec_tok_final_cost(d, t, any_final);
      float tok_extra_cost = tok->tot_cost + final_cost - best;
      for (int32_t l = tok->links; l >= 0;) {
        Link *k = &d->links[l];
        Tok *nt = &d->toks[k->next_tok];
        float link_extra_cost =
            nt->extra_cost + ((tok->tot_cost + k->ac_cost + k->graph_cost) - nt->tot_cost);
        if (link_extra_cost > d->lattice_beam) {
          int32_t nl = k->next;
          if (prev_link >= 0) d->links[prev_link].next = nl;
          else tok->links = nl;
          link_del(d, l);
          l = nl;
        } else {
          if (link_extra_cost < 0.0f) link_extra_cost = 0.0f;
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
          prev_link = l;
          l = k->next;
        }
      }
      if (tok_extra_cost > d->lattice_beam) tok_extra_cost = INFINITY;
      float a = tok->extra_cost, b = tok_extra_cost;
      int approx_equal = (a == b);
      if (!approx_equal) {
        float diff = fabsf(a - b);
        approx_equal = !(diff == INFINITY || diff != diff) &&
                       diff <= FINAL_DELTA * (fabsf(a) + fabsf(b));
      }
      if (!approx_equal) changed = 1;
      tok->extra_cost = tok_extra_cost;
    }
  }
}

/* ---- public search calls ---- */
int ec_dec_init(ec_dec *d) {
  if (d->bad) return -1;
  clear_all(d);
  if (d->g->num_states == 0 || d->g->start < 0) return fail(d, "graph has no start state");
  if (grow((void **)&d->frames, &d->frames_cap, 1, sizeof(FrameToks))) return fail(d, "out of memory");
  d->frames[0].toks = -1;
  d->frames[0].must_prune_forward_links = 1;
  d->frames[0].must_prune_tokens = 1;
  if (find_or_add(d, (int32_t)d->g->start, 0, 0.0f, NULL) < 0) return fail(d, "out of memory");
  d->initialized = 1;
  return process_nonemitting(d, d->beam);
}

int ec_dec_advance(ec_dec *d, const float *loglikes, int32_t num_frames, int32_t num_pdfs) {
  if (d->bad) return -1;
  if (!d->initialized) return fail(d, "advance before init");
  if (d->finalized) return fail(d, "advance after the decoding was finalized");
  if (num_pdfs != d->num_pdfs) return fail(d, "loglike width is not num_pdfs");
  for (int32_t i = 0; i < num_frames; i++) {
    if (d->nframes % PRUNE_INTERVAL == 0) prune_active_tokens(d, d->lattice_beam * PRUNE_SCALE);
    float cutoff = process_emitting(d, loglikes + (size_t)i * (size_t)num_pdfs);
    if (cutoff != cutoff) return -1;
    if (process_nonemitting(d, cutoff)) return -1;
  }
  return 0;
}

int32_t ec_dec_frames_decoded(const ec_dec *d) { return d->nframes; }
int32_t ec_dec_is_finalized(const ec_dec *d) { return d->finalized; }

int ec_dec_finalize(ec_dec *d) {
  if (!d->initialized) return fail(d, "finalize before init");
  if (d->finalized) return 0;
  int32_t final_frame_plus_one = d->nframes;
  prune_forward_links_final(d);
  d->finalized = 1;
  for (int32_t f = final_frame_plus_one - 1; f >= 0; f--) {
    int b1, b2;
    prune_forward_links(d, f, &b1, &b2, 0.0f);
    prune_tokens_for_frame(d, f + 1);
  }
  prune_tokens_for_frame(d, 0);
  return 0;
}

int32_t ec_dec_frame_tokens(const ec_dec *d, int32_t f) {
  if (f < 0 || f > d->nframes || !d->initialized) return -1;
  int32_t n = 0;
  for (int32_t t = d->frames[f].toks; t >= 0; t = d->toks[t].next) n++;
  return n;
}

float ec_dec_final_relative_cost(const ec_dec *d) {
  if (!d->initialized) return INFINITY;
  if (d->finalized) return d->final_relative;
  int any_final;
  float best, relative;
  ec_final_costs(d, &any_final, &best, &relative);
  return relative;
}

void ec_dec_last_cutoff(const ec_dec *d, float *out3) {
  out3[0] = d->last_cutoff;
  out3[1] = d->last_adaptive_beam;
  out3[2] = d->last_considered;
}

int64_t ec_dec_bytes(const ec_dec *d) {
  return d->tok_cap * (int64_t)sizeof(Tok) + d->link_cap * (int64_t)sizeof(Link) +
         (int64_t)(d->hmask + 1) * (int64_t)sizeof(HSlot) + d->lat.bytes;
}
