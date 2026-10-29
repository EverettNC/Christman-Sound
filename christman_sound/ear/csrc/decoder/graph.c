/* HCLG.fst as an OpenFst ConstFst, read from a memory map.
 *
 * File layout: header (magic, fst type, arc type, version, flags, properties,
 * start, num states, num arcs), optional input/output symbol tables, then the
 * state array (final weight, first arc, num arcs, num input epsilons, num
 * output epsilons: 20 bytes each) and the arc array (ilabel, olabel, weight,
 * nextstate: 16 bytes each). Version 1, or the aligned flag, puts each array
 * on a 16-byte boundary. The arrays need not be 4-byte aligned in the file,
 * so every field is read with memcpy.
 */
#include "graph_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define FST_MAGIC 2125659606
#define SYMTAB_MAGIC 2125658996
#define CONST_FILE_VERSION 2
#define CONST_ALIGNED_FILE_VERSION 1
#define HAS_ISYMBOLS 0x1
#define HAS_OSYMBOLS 0x2
#define IS_ALIGNED 0x4

typedef struct {
  const unsigned char *p;
  size_t n, pos;
  int bad;
} Cursor;

static void need(Cursor *c, size_t k) {
  if (c->pos + k > c->n) c->bad = 1;
}
static int32_t rd_i32(Cursor *c) {
  int32_t v = 0;
  need(c, 4);
  if (!c->bad) memcpy(&v, c->p + c->pos, 4);
  c->pos += 4;
  return v;
}
static int64_t rd_i64(Cursor *c) {
  int64_t v = 0;
  need(c, 8);
  if (!c->bad) memcpy(&v, c->p + c->pos, 8);
  c->pos += 8;
  return v;
}
static void rd_str(Cursor *c, char *out, int len) {
  int32_t n = rd_i32(c);
  if (c->bad || n < 0) {
    c->bad = 1;
    return;
  }
  need(c, (size_t)n);
  if (c->bad) return;
  if (out) {
    int k = n < len - 1 ? n : len - 1;
    memcpy(out, c->p + c->pos, (size_t)k);
    out[k] = 0;
  }
  c->pos += (size_t)n;
}
static void skip_symbol_table(Cursor *c) {
  if (rd_i32(c) != SYMTAB_MAGIC) {
    c->bad = 1;
    return;
  }
  rd_str(c, NULL, 0);           /* name */
  (void)rd_i64(c);              /* available key */
  int64_t n = rd_i64(c);        /* size */
  for (int64_t i = 0; i < n && !c->bad; i++) {
    rd_str(c, NULL, 0);
    (void)rd_i64(c);
  }
}

static void set_err(char *err, int errlen, const char *msg, const char *path) {
  if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s: %s", path, msg);
}

ec_graph *ec_graph_open(const char *path, char *err, int errlen) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    set_err(err, errlen, strerror(errno), path);
    return NULL;
  }
  struct stat sb;
  if (fstat(fd, &sb) != 0 || sb.st_size < 16) {
    set_err(err, errlen, "cannot stat, or too small", path);
    close(fd);
    return NULL;
  }
  size_t len = (size_t)sb.st_size;
  void *map = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    set_err(err, errlen, strerror(errno), path);
    close(fd);
    return NULL;
  }
  ec_graph *g = calloc(1, sizeof(*g));
  if (!g) {
    munmap(map, len);
    close(fd);
    set_err(err, errlen, "out of memory", path);
    return NULL;
  }
  g->fd = fd;
  g->map = map;
  g->maplen = len;
  Cursor c = {map, len, 0, 0};
  g->magic = rd_i32(&c);
  rd_str(&c, g->fst_type, sizeof(g->fst_type));
  rd_str(&c, g->arc_type, sizeof(g->arc_type));
  g->version = rd_i32(&c);
  g->flags = rd_i32(&c);
  g->properties = (uint64_t)rd_i64(&c);
  g->start = rd_i64(&c);
  g->num_states = rd_i64(&c);
  g->num_arcs = rd_i64(&c);
  const char *why = NULL;
  if (c.bad) why = "header runs past the end of the file";
  else if (g->magic != FST_MAGIC) why = "not an FST (bad magic number)";
  else if (strcmp(g->fst_type, "const") != 0) why = "fst type is not const";
  else if (strcmp(g->arc_type, "standard") != 0) why = "arc type is not standard";
  else if (g->version != CONST_FILE_VERSION && g->version != CONST_ALIGNED_FILE_VERSION)
    why = "unknown const fst version";
  else if (g->num_states < 0 || g->num_arcs < 0) why = "negative state or arc count";
  if (!why) {
    if (g->flags & HAS_ISYMBOLS) skip_symbol_table(&c);
    if (g->flags & HAS_OSYMBOLS) skip_symbol_table(&c);
    if (c.bad) why = "symbol table runs past the end of the file";
  }
  if (!why) {
    g->header_bytes = (int64_t)c.pos;
    g->aligned = (g->version == CONST_ALIGNED_FILE_VERSION) || (g->flags & IS_ALIGNED);
    size_t pos = c.pos;
    if (g->aligned) pos = (pos + 15) & ~(size_t)15;
    size_t sbytes = (size_t)g->num_states * EC_STATE_BYTES;
    if (pos + sbytes > len) why = "state array runs past the end of the file";
    else {
      g->states = (const unsigned char *)map + pos;
      pos += sbytes;
      if (g->aligned) pos = (pos + 15) & ~(size_t)15;
      size_t abytes = (size_t)g->num_arcs * EC_ARC_BYTES;
      if (pos + abytes > len) why = "arc array runs past the end of the file";
      else g->arcs = (const unsigned char *)map + pos;
    }
  }
  if (!why && g->num_states > 0 && (g->start < 0 || g->start >= g->num_states))
    why = "start state out of range";
  if (why) {
    set_err(err, errlen, why, path);
    ec_graph_close(g);
    return NULL;
  }
  return g;
}

void ec_graph_close(ec_graph *g) {
  if (!g) return;
  if (g->map) munmap(g->map, g->maplen);
  if (g->fd >= 0) close(g->fd);
  free(g);
}

void ec_graph_info(const ec_graph *g, int64_t *out) {
  out[0] = g->magic;
  out[1] = g->version;
  out[2] = g->flags;
  out[3] = (int64_t)g->properties;
  out[4] = g->start;
  out[5] = g->num_states;
  out[6] = g->num_arcs;
  out[7] = g->aligned;
  out[8] = g->header_bytes;
}

void ec_graph_types(const ec_graph *g, char *fst_type, char *arc_type, int len) {
  snprintf(fst_type, (size_t)len, "%s", g->fst_type);
  snprintf(arc_type, (size_t)len, "%s", g->arc_type);
}

float ec_graph_final(const ec_graph *g, int64_t s) {
  ec_state st;
  ec_get_state(g, (int32_t)s, &st);
  return st.final_w;
}

int64_t ec_graph_num_arcs(const ec_graph *g, int64_t s) {
  ec_state st;
  ec_get_state(g, (int32_t)s, &st);
  return st.narcs;
}

int64_t ec_graph_num_input_eps(const ec_graph *g, int64_t s) {
  ec_state st;
  ec_get_state(g, (int32_t)s, &st);
  return st.nieps;
}

int64_t ec_graph_arcs(const ec_graph *g, int64_t s, int32_t *ilabel, int32_t *olabel,
                      float *weight, int32_t *nextstate, int64_t cap) {
  if (s < 0 || s >= g->num_states) return -1;
  ec_state st;
  ec_get_state(g, (int32_t)s, &st);
  if ((int64_t)st.pos + st.narcs > g->num_arcs) return -1;
  int64_t n = st.narcs < cap ? st.narcs : cap;
  for (int64_t i = 0; i < n; i++) {
    ec_arc a;
    ec_get_arc(g, st.pos + (uint32_t)i, &a);
    ilabel[i] = a.ilabel;
    olabel[i] = a.olabel;
    weight[i] = a.weight;
    nextstate[i] = a.nextstate;
  }
  return st.narcs;
}
