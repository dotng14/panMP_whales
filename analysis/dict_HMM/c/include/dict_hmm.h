#ifndef DICT_HMM_H
#define DICT_HMM_H

#include <stddef.h>

/* Defaults that produced the 35-motif dictionary and the wav 2 / wav 3 placements. */
#define DICT_HMM_K 1
#define DICT_HMM_EPS 0.01
#define DICT_HMM_W_MIN 10
#define DICT_HMM_W_MAX 40
#define DICT_HMM_SEEDS_PER_PASS 10
#define DICT_HMM_N_MIN_LEARN 10
#define DICT_HMM_N_MIN_PLACE 1
#define DICT_HMM_REFINE_ITERS 2
#define DICT_HMM_TAU 0.4
#define DICT_HMM_MAX_OUTER 50
#define DICT_HMM_MIN_COVERAGE 0.7
#define DICT_HMM_N_COMPONENTS 5

#define DICT_HMM_N_THEMES 4
#define DICT_HMM_N_THEMES_BEFORE_MERGE 5
#define DICT_HMM_BIN_MINUTES 0.5
#define DICT_HMM_ALPHA 0.3
#define DICT_HMM_DURATION_MIN_BINS 2
#define DICT_HMM_DURATION_MAX_BINS 60
#define DICT_HMM_DURATION_SIGMA_FLOOR 0.25
#define DICT_HMM_LIKELIHOOD_GAP 1.920729410347062
#define DICT_HMM_RECORDING_MINUTES 75.0
#define DICT_HMM_WAV2_UNIX 1682557875.0

typedef struct DhWindow {
    int start_sample;
    int end_sample;
    int sample_freq;
    int overlap;
    int n_freq;
    int n_time;
    double *data; /* row-major, n_freq * n_time, not owned */
    char id[128];
} DhWindow;

typedef struct DhSegment {
    int window_index;
    int t_start; /* inclusive column */
    int t_end;   /* exclusive column */
} DhSegment;

typedef struct DhMotif {
    int motif_id;
    int n_components;
    int n_freq;
    int width;
    double *components; /* n_components * n_freq * width, owned */
    double *mean;       /* n_freq * width, owned, or NULL */
} DhMotif;

typedef struct DhPlacement {
    int motif_id;
    int window_index;
    int t_start;
    double amplitude;
    double unexplained;
} DhPlacement;

typedef struct DhLearnParams {
    int k;
    double eps;
    int w_min;
    int w_max;
    int seeds_per_pass;
    int n_min;
    int refine_iters;
    double tau;
    int max_outer;
    double min_coverage;
    int n_components;
} DhLearnParams;

typedef struct DhPlaceParams {
    int k;
    double eps;
    double tau;
    int n_min;
} DhPlaceParams;

typedef struct DhInterval {
    int index; /* motif id, theme 0..3, or song number */
    double start_sec;
    double end_sec;
} DhInterval;

typedef struct DhThemes {
    DhInterval *motifs;
    int n_motifs;
    DhInterval *themes;
    int n_themes;
    DhInterval *songs;
    int n_songs;
} DhThemes;

double dict_hmm_hop_seconds(int overlap, int sample_freq);
double dict_hmm_placement_seconds(int start_sample, int sample_freq, int overlap, int t_start);

void dict_hmm_learn_defaults(DhLearnParams *params);
void dict_hmm_place_defaults(DhPlaceParams *params);

/* Explained fraction at every in-range start. fraction and explained have length n_time - width + 1. */
void dict_hmm_explain(const double *spectrogram, int n_freq, int n_time,
                      const DhMotif *motif, double *fraction, double *explained, double *first_dots);

int dict_hmm_pursuit(const DhWindow *windows, int n_windows,
                     const DhMotif *motifs, int n_motifs,
                     int *const *counts, int k, double eps,
                     DhPlacement **out, int *n_out);

void dict_hmm_principal_components(const double *patches, int n_patches, int n_freq, int width,
                                   int n_components, double **basis_out, int *n_basis, double **mean_out);

void dict_hmm_rescore(const DhWindow *windows, DhMotif *motifs, int n_motifs,
                      DhPlacement *placements, int n_placements);

void dict_hmm_apply_tau(DhMotif **motifs, int *n_motifs,
                        DhPlacement **placements, int *n_placements,
                        double tau, int n_min);

/* seed_indices lists segment indexes, seeds_per_pass of them per outer pass.
   Pass NULL to draw with an internal generator (not NumPy PCG64). */
int dict_hmm_learn(const DhWindow *windows, int n_windows,
                   const DhSegment *segments, int n_segments,
                   const DhLearnParams *params,
                   const int *seed_indices, int n_seed_indices,
                   DhMotif **motifs_out, int *n_motifs,
                   DhPlacement **placements_out, int *n_placements,
                   char *stop_reason, int stop_reason_cap);

int dict_hmm_place(const DhWindow *windows, int n_windows,
                   DhMotif *motifs, int n_motifs,
                   const DhPlaceParams *params,
                   DhPlacement **out, int *n_out);

/* motif_ids[i], start_sec[i] are one kept placement. end_sec may be NULL. */
int dict_hmm_themes(const int *motif_ids, const double *start_sec, const double *end_sec,
                    int n_placements, DhThemes *out);

void dict_hmm_free_motifs(DhMotif *motifs, int n);
void dict_hmm_free_placements(DhPlacement *placements);
void dict_hmm_free_themes(DhThemes *themes);

#endif
