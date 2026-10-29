/* ear-core decoder: reads HCLG.fst (an OpenFst ConstFst, standard arcs) straight
 * from a memory map and runs lattice-faster token passing over it, in tropical
 * costs. Transition ids are the graph's input labels; words are its output
 * labels. Also builds the word lattice and its MBR word posteriors.
 *
 * Functions return a negative number or NULL on failure and leave a message in
 * the err buffer they were handed (or in ec_dec_error for a decoder).
 */
#ifndef EARCORE_DECODER_H
#define EARCORE_DECODER_H

#include <stdint.h>

typedef struct ec_graph ec_graph;
typedef struct ec_dec ec_dec;

/* ---- graph ---- */
ec_graph *ec_graph_open(const char *path, char *err, int errlen);
void ec_graph_close(ec_graph *g);
/* out[0..8]: magic, version, flags, properties, start, num_states, num_arcs,
 * aligned (0/1), header_bytes */
void ec_graph_info(const ec_graph *g, int64_t *out);
void ec_graph_types(const ec_graph *g, char *fst_type, char *arc_type, int len);
float ec_graph_final(const ec_graph *g, int64_t s);
int64_t ec_graph_num_arcs(const ec_graph *g, int64_t s);
int64_t ec_graph_num_input_eps(const ec_graph *g, int64_t s);
int64_t ec_graph_arcs(const ec_graph *g, int64_t s, int32_t *ilabel, int32_t *olabel,
                      float *weight, int32_t *nextstate, int64_t cap);

/* ---- decoder ---- */
ec_dec *ec_dec_new(const ec_graph *g, const int32_t *tid2pdf, int32_t num_tids_plus_one,
                   int32_t num_pdfs, const int32_t *disambig, int32_t num_disambig,
                   float beam, int32_t max_active, int32_t min_active, float lattice_beam,
                   float acoustic_scale);
void ec_dec_free(ec_dec *d);
const char *ec_dec_error(const ec_dec *d);
int ec_dec_init(ec_dec *d);
int ec_dec_advance(ec_dec *d, const float *loglikes, int32_t num_frames, int32_t num_pdfs);
int32_t ec_dec_frames_decoded(const ec_dec *d);
int ec_dec_finalize(ec_dec *d);
int32_t ec_dec_is_finalized(const ec_dec *d);
/* Best cost with final costs minus best cost without, on the newest frame;
 * infinity when no active token has a final cost. Kept from finalize on. */
float ec_dec_final_relative_cost(const ec_dec *d);
/* tokens alive in the list of frame f (0..frames_decoded) */
int32_t ec_dec_frame_tokens(const ec_dec *d, int32_t f);
/* the last emitting step: [cost_cutoff, adaptive_beam, tokens considered] */
void ec_dec_last_cutoff(const ec_dec *d, float *out3);
/* bytes held by token, link and hash storage */
int64_t ec_dec_bytes(const ec_dec *d);

/* Best path through the token lattice. Returns the number of arcs on it, or
 * negative. costs3 gets [graph, acoustic, final]: acoustic without the
 * per-frame offsets; graph includes the final cost, which is also given alone. */
int64_t ec_dec_best_path(ec_dec *d, int32_t use_final, double *costs3);
void ec_dec_path_arcs(const ec_dec *d, int32_t *ilabels, int32_t *olabels, float *graph,
                      float *acoustic);

/* Word lattice (determinized on words, pruned at lattice_beam) and the MBR
 * edit-distance posteriors of the word sequence R against it.
 * conf[k] for each of the nR words; stats4 gets [word-lattice states,
 * word-lattice arcs, raw lattice arcs, max |sum over a bin of gamma - 1|]. */
int ec_dec_word_posteriors(ec_dec *d, int32_t use_final, const int32_t *R, int32_t nR,
                           float graph_scale, float *conf, double *stats4);

#endif
