/* Private layout of the memory-mapped ConstFst, shared by graph.c and decoder.c. */
#ifndef EARCORE_GRAPH_INTERNAL_H
#define EARCORE_GRAPH_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "decoder.h"

#define EC_STATE_BYTES 20
#define EC_ARC_BYTES 16

struct ec_graph {
  int fd;
  void *map;
  size_t maplen;
  const unsigned char *states, *arcs;
  int32_t magic, version, flags, aligned;
  uint64_t properties;
  int64_t start, num_states, num_arcs, header_bytes;
  char fst_type[64], arc_type[64];
};

typedef struct {
  float final_w;
  uint32_t pos, narcs, nieps, noeps;
} ec_state;

typedef struct {
  int32_t ilabel, olabel;
  float weight;
  int32_t nextstate;
} ec_arc;

static inline void ec_get_state(const ec_graph *g, int32_t s, ec_state *out) {
  memcpy(out, g->states + (size_t)s * EC_STATE_BYTES, EC_STATE_BYTES);
}

static inline void ec_get_arc(const ec_graph *g, uint32_t i, ec_arc *out) {
  memcpy(out, g->arcs + (size_t)i * EC_ARC_BYTES, EC_ARC_BYTES);
}

#endif
