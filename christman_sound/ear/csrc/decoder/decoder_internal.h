/* Private decoder state, shared by decoder.c (search) and lattice.c (lattice
 * read-back, best path, word lattice, posteriors). */
#ifndef EARCORE_DECODER_INTERNAL_H
#define EARCORE_DECODER_INTERNAL_H

#include "graph_internal.h"

/* 20 bytes. tokens on one frame form a list through next; freed tokens chain
 * through next on the free list. */
typedef struct {
  float tot_cost, extra_cost;
  int32_t state, links, next;
} Tok;

/* 24 bytes. ac_cost carries the frame's cost offset. */
typedef struct {
  int32_t next_tok, ilabel, olabel;
  float graph_cost, ac_cost;
  int32_t next;
} Link;

typedef struct {
  int32_t toks;
  uint8_t must_prune_forward_links, must_prune_tokens;
} FrameToks;

typedef struct {
  int32_t state, tok;
  uint32_t stamp;
} HSlot;

/* The token lattice read back as an acyclic graph in topological order. */
typedef struct {
  int32_t n, narcs, start;
  int64_t n_cap, a_cap;
  int32_t *vert_tok, *vert_frame, *arc_start; /* arc_start has n+1 entries */
  int32_t *dst, *il, *ol;
  float *g, *a, *fin; /* fin: final graph cost, INFINITY if not final */
  double *beta;       /* best total cost from a vert to the end */
  /* best path */
  int32_t path_len;
  int64_t path_cap;
  int32_t *p_il, *p_ol;
  float *p_g, *p_a;
  int64_t bytes;
} RawLat;

struct ec_dec {
  const ec_graph *g;
  float beam, lattice_beam, acoustic_scale;
  int32_t max_active, min_active, num_pdfs, ntid;
  int32_t *tid2pdf;
  uint8_t *dis;
  int32_t dis_len, num_disambig;
  float *ll;
  int bad, initialized, finalized;
  char err[256];

  Tok *toks;
  int64_t tok_cap, tok_used, num_toks;
  int32_t tok_free;
  Link *links;
  int64_t link_cap, link_used;
  int32_t link_free;

  HSlot *hslots;
  uint32_t hmask, hstamp;
  int32_t *cur, *prev;
  int64_t ncur, cur_cap, nprev, prev_cap;

  FrameToks *frames;
  int64_t frames_cap;
  int32_t nframes;
  float *cost_offsets;
  int64_t co_cap;

  float *tmp;
  int64_t tmp_cap;
  int32_t *queue;
  int64_t queue_cap;

  float last_cutoff, last_adaptive_beam, last_considered;
  int final_any;
  float final_best, final_relative;

  RawLat lat;
};

void ec_final_costs(const ec_dec *d, int *any_final, float *best_with_final, float *relative);
float ec_tok_final_cost(const ec_dec *d, int32_t t, int any_final);
void ec_lat_free(RawLat *L);

#endif
