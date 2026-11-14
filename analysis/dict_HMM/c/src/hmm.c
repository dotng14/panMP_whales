#include "dict_hmm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "dict_hmm: out of memory\n");
        exit(1);
    }
    return p;
}

typedef struct {
    int a, b, k;
} Seg;

static double logsumexp_slice(const double *x, int begin, int end) {
    double m = x[begin];
    for (int i = begin + 1; i < end; i++) if (x[i] > m) m = x[i];
    double s = 0.0;
    for (int i = begin; i < end; i++) s += exp(x[i] - m);
    return m + log(s);
}

static void lognormal_row(double mu, double sigma, int dmin, int dmax, double *out) {
    double *raw = xcalloc((size_t)dmax, sizeof(double));
    for (int d = 1; d <= dmax; d++) {
        double ld = log((double)d);
        raw[d - 1] = -0.5 * ((ld - mu) / sigma) * ((ld - mu) / sigma) - log((double)d * sigma);
    }
    for (int i = 0; i < dmax; i++) out[i] = -1e12;
    double norm = logsumexp_slice(raw, dmin - 1, dmax);
    for (int i = dmin - 1; i < dmax; i++) out[i] = raw[i] - norm;
    free(raw);
}

static int viterbi(const double *log_emit, int T, int K, const double *log_dur, const double *logpi,
                   int dmin, int dmax, Seg **out, int *n_out) {
    const double NEG = -1e18;
    double *delta = xcalloc((size_t)(T + 1) * (size_t)K, sizeof(double));
    int *prev_d = xcalloc((size_t)(T + 1) * (size_t)K, sizeof(int));
    for (int i = 0; i < (T + 1) * K; i++) {
        delta[i] = NEG;
        prev_d[i] = -1;
    }
    double *prefix = xcalloc((size_t)T * (size_t)K, sizeof(double));
    for (int k = 0; k < K; k++) {
        double run = 0.0;
        for (int t = 0; t < T; t++) {
            run += log_emit[(size_t)t * (size_t)K + (size_t)k];
            prefix[(size_t)t * (size_t)K + (size_t)k] = run;
        }
    }
    for (int k = 0; k < K; k++) {
        int d_hi = dmax < T ? dmax : T;
        for (int d = dmin; d <= d_hi; d++) {
            double emit = prefix[(size_t)(d - 1) * (size_t)K + (size_t)k];
            double score = logpi[k] + emit + log_dur[(size_t)k * (size_t)dmax + (size_t)(d - 1)];
            if (score > delta[(size_t)d * (size_t)K + (size_t)k]) {
                delta[(size_t)d * (size_t)K + (size_t)k] = score;
                prev_d[(size_t)d * (size_t)K + (size_t)k] = d;
            }
        }
    }
    for (int t = 1; t <= T; t++) {
        for (int k = 0; k < K; k++) {
            int prev = (k - 1 + K) % K;
            double best = delta[(size_t)t * (size_t)K + (size_t)k];
            int bd = best > NEG / 2.0 ? prev_d[(size_t)t * (size_t)K + (size_t)k] : -1;
            int d_hi = dmax < t ? dmax : t;
            for (int d = dmin; d <= d_hi; d++) {
                if (delta[(size_t)(t - d) * (size_t)K + (size_t)prev] <= NEG / 2.0) continue;
                double end = prefix[(size_t)(t - 1) * (size_t)K + (size_t)k];
                double before = (t - d > 0) ? prefix[(size_t)(t - d - 1) * (size_t)K + (size_t)k] : 0.0;
                double score = delta[(size_t)(t - d) * (size_t)K + (size_t)prev] + (end - before)
                    + log_dur[(size_t)k * (size_t)dmax + (size_t)(d - 1)];
                if (score > best) {
                    best = score;
                    bd = d;
                }
            }
            delta[(size_t)t * (size_t)K + (size_t)k] = best;
            prev_d[(size_t)t * (size_t)K + (size_t)k] = bd;
        }
    }
    int k = 0;
    for (int i = 1; i < K; i++) {
        if (delta[(size_t)T * (size_t)K + (size_t)i] > delta[(size_t)T * (size_t)K + (size_t)k]) k = i;
    }
    int cap = 64, n = 0;
    Seg *rev = xcalloc((size_t)cap, sizeof(Seg));
    int t = T;
    while (t > 0) {
        int d = prev_d[(size_t)t * (size_t)K + (size_t)k];
        if (d < 1) {
            free(rev);
            free(delta);
            free(prev_d);
            free(prefix);
            return -1;
        }
        if (n == cap) {
            cap *= 2;
            rev = realloc(rev, (size_t)cap * sizeof(Seg));
        }
        rev[n].a = t - d;
        rev[n].b = t;
        rev[n].k = k;
        n++;
        t -= d;
        k = (k - 1 + K) % K;
    }
    Seg *segs = xcalloc((size_t)n, sizeof(Seg));
    for (int i = 0; i < n; i++) segs[i] = rev[n - 1 - i];
    free(rev);
    free(delta);
    free(prev_d);
    free(prefix);
    *out = segs;
    *n_out = n;
    return 0;
}

static void labels_to_segs(const int *labels, int T, Seg **out, int *n_out) {
    int cap = 16, n = 0;
    Seg *segs = xcalloc((size_t)cap, sizeof(Seg));
    int a = 0;
    for (int t = 1; t <= T; t++) {
        if (t == T || labels[t] != labels[a]) {
            if (n == cap) {
                cap *= 2;
                segs = realloc(segs, (size_t)cap * sizeof(Seg));
            }
            segs[n].a = a;
            segs[n].b = t;
            segs[n].k = labels[a];
            n++;
            a = t;
        }
    }
    *out = segs;
    *n_out = n;
}

static void mstep(const double *counts, int T, int M, const Seg *segs, int n_segs, int K,
                  double alpha, int dmin, int dmax, double *p, double *log_dur, double *logpi) {
    (void)T;
    for (int k = 0; k < K; k++) for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] = alpha;
    int *nlen = xcalloc((size_t)K, sizeof(int));
    int *cap = xcalloc((size_t)K, sizeof(int));
    int **lens = xcalloc((size_t)K, sizeof(int *));
    for (int i = 0; i < n_segs; i++) {
        int k = segs[i].k;
        for (int t = segs[i].a; t < segs[i].b; t++) {
            for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] += counts[(size_t)t * (size_t)M + (size_t)m];
        }
        if (nlen[k] == cap[k]) {
            cap[k] = cap[k] ? cap[k] * 2 : 4;
            lens[k] = realloc(lens[k], (size_t)cap[k] * sizeof(int));
        }
        lens[k][nlen[k]++] = segs[i].b - segs[i].a;
    }
    for (int k = 0; k < K; k++) {
        double sum = 0.0;
        for (int m = 0; m < M; m++) sum += p[(size_t)k * (size_t)M + (size_t)m];
        for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] /= sum;
        double mu, sigma;
        if (nlen[k] == 0) {
            mu = log(6.0);
            sigma = 0.5;
        } else {
            double mean = 0.0, var = 0.0;
            for (int i = 0; i < nlen[k]; i++) mean += log((double)lens[k][i]);
            mean /= (double)nlen[k];
            for (int i = 0; i < nlen[k]; i++) {
                double d = log((double)lens[k][i]) - mean;
                var += d * d;
            }
            var /= (double)nlen[k];
            mu = mean;
            sigma = sqrt(var);
            if (sigma < DICT_HMM_DURATION_SIGMA_FLOOR) sigma = DICT_HMM_DURATION_SIGMA_FLOOR;
        }
        lognormal_row(mu, sigma, dmin, dmax, log_dur + (size_t)k * (size_t)dmax);
    }
    for (int k = 0; k < K; k++) logpi[k] = -log((double)K);
    for (int k = 0; k < K; k++) free(lens[k]);
    free(lens);
    free(nlen);
    free(cap);
}

static void emit_from_p(const double *counts, int T, int M, const double *p, int K, double *log_emit) {
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int m = 0; m < M; m++) {
                double c = counts[(size_t)t * (size_t)M + (size_t)m];
                if (c != 0.0) s += c * log(p[(size_t)k * (size_t)M + (size_t)m]);
            }
            log_emit[(size_t)t * (size_t)K + (size_t)k] = s;
        }
    }
}

static void train(const double *counts, int T, int M, Seg **segs, int *n_segs, int K,
                  int iters, int dmin, int dmax) {
    double *p = xcalloc((size_t)K * (size_t)M, sizeof(double));
    double *log_dur = xcalloc((size_t)K * (size_t)dmax, sizeof(double));
    double *logpi = xcalloc((size_t)K, sizeof(double));
    double *log_emit = xcalloc((size_t)T * (size_t)K, sizeof(double));
    for (int it = 0; it < iters; it++) {
        mstep(counts, T, M, *segs, *n_segs, K, DICT_HMM_ALPHA, dmin, dmax, p, log_dur, logpi);
        emit_from_p(counts, T, M, p, K, log_emit);
        Seg *next = NULL;
        int n_next = 0;
        if (viterbi(log_emit, T, K, log_dur, logpi, dmin, dmax, &next, &n_next) != 0) break;
        free(*segs);
        *segs = next;
        *n_segs = n_next;
    }
    free(p);
    free(log_dur);
    free(logpi);
    free(log_emit);
}

static int lead_motif(const double *p, int K, int k, const int *vocab, int M) {
    int best = 0;
    for (int m = 1; m < M; m++) {
        if (p[(size_t)k * (size_t)M + (size_t)m] > p[(size_t)k * (size_t)M + (size_t)best]) best = m;
    }
    (void)K;
    return vocab[best];
}

static void fill_p(const double *counts, int M, const Seg *segs, int n_segs, int K, double *p) {
    for (int k = 0; k < K; k++) for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] = DICT_HMM_ALPHA;
    for (int i = 0; i < n_segs; i++) {
        int k = segs[i].k;
        for (int t = segs[i].a; t < segs[i].b; t++) {
            for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] += counts[(size_t)t * (size_t)M + (size_t)m];
        }
    }
    for (int k = 0; k < K; k++) {
        double sum = 0.0;
        for (int m = 0; m < M; m++) sum += p[(size_t)k * (size_t)M + (size_t)m];
        for (int m = 0; m < M; m++) p[(size_t)k * (size_t)M + (size_t)m] /= sum;
    }
}

static double segment_ll(const double *counts, int M, int a, int b, const double *prob) {
    double s = 0.0;
    for (int t = a; t < b; t++) {
        for (int m = 0; m < M; m++) {
            double c = counts[(size_t)t * (size_t)M + (size_t)m];
            if (c != 0.0) s += c * log(prob[m]);
        }
    }
    return s;
}

int dict_hmm_themes(const int *motif_ids, const double *start_sec, const double *end_sec,
                    int n_placements, DhThemes *out) {
    int T = (int)ceil(DICT_HMM_RECORDING_MINUTES / DICT_HMM_BIN_MINUTES);
    int *ids = xcalloc((size_t)(n_placements > 0 ? n_placements : 1), sizeof(int));
    int M = 0;
    for (int i = 0; i < n_placements; i++) {
        int id = motif_ids[i];
        int seen = 0;
        for (int j = 0; j < M; j++) if (ids[j] == id) seen = 1;
        if (!seen) ids[M++] = id;
    }
    for (int i = 0; i < M; i++) {
        for (int j = i + 1; j < M; j++) {
            if (ids[j] < ids[i]) {
                int tmp = ids[i];
                ids[i] = ids[j];
                ids[j] = tmp;
            }
        }
    }
    int index_of[256];
    for (int i = 0; i < 256; i++) index_of[i] = -1;
    for (int i = 0; i < M; i++) {
        if (ids[i] < 0 || ids[i] >= 256) {
            fprintf(stderr, "dict_hmm: motif id %d is outside 0..255\n", ids[i]);
            free(ids);
            return -1;
        }
        index_of[ids[i]] = i;
    }
    double *counts = xcalloc((size_t)T * (size_t)M, sizeof(double));
    for (int i = 0; i < n_placements; i++) {
        int bin = (int)((start_sec[i] / 60.0) / DICT_HMM_BIN_MINUTES);
        if (bin < 0) bin = 0;
        if (bin >= T) bin = T - 1;
        counts[(size_t)bin * (size_t)M + (size_t)index_of[motif_ids[i]]] += 1.0;
    }

    const int K5 = DICT_HMM_N_THEMES_BEFORE_MERGE;
    int *labels = xcalloc((size_t)T, sizeof(int));
    int period = (int)llround(15.0 / DICT_HMM_BIN_MINUTES);
    for (int t = 0; t < T; t++) labels[t] = ((t * K5) / period) % K5;
    Seg *segs = NULL;
    int n_segs = 0;
    labels_to_segs(labels, T, &segs, &n_segs);
    free(labels);
    train(counts, T, M, &segs, &n_segs, K5, 15, DICT_HMM_DURATION_MIN_BINS, DICT_HMM_DURATION_MAX_BINS);

    int shift = segs[0].k;
    for (int i = 0; i < n_segs; i++) segs[i].k = (segs[i].k - shift + K5) % K5;

    double *p5 = xcalloc((size_t)K5 * (size_t)M, sizeof(double));
    fill_p(counts, M, segs, n_segs, K5, p5);
    int leads[8];
    for (int k = 0; k < K5; k++) leads[k] = lead_motif(p5, K5, k, ids, M);
    int merge_a = -1;
    for (int k = 0; k < K5; k++) {
        if (leads[k] == 21 && leads[(k + 1) % K5] == 21) merge_a = k;
    }
    if (merge_a < 0) {
        fprintf(stderr, "dict_hmm: no neighboring motif-21 themes to merge (leads");
        for (int k = 0; k < K5; k++) fprintf(stderr, " %d", leads[k]);
        fprintf(stderr, ")\n");
        free(p5);
        free(segs);
        free(counts);
        free(ids);
        return -1;
    }
    int merge_b = (merge_a + 1) % K5;
    int map[8];
    for (int k = 0; k < K5; k++) map[k] = -1;
    map[merge_a] = 0;
    map[merge_b] = 0;
    int next = 1;
    for (int step = 1; step < K5; step++) {
        int k = (merge_b + step) % K5;
        if (k == merge_a) continue;
        map[k] = next++;
    }
    for (int i = 0; i < n_segs; i++) segs[i].k = map[segs[i].k];
    int w = 0;
    for (int i = 0; i < n_segs; i++) {
        if (w > 0 && segs[w - 1].k == segs[i].k && segs[w - 1].b == segs[i].a) {
            segs[w - 1].b = segs[i].b;
        } else {
            segs[w++] = segs[i];
        }
    }
    n_segs = w;
    free(p5);
    const int K = DICT_HMM_N_THEMES;
    train(counts, T, M, &segs, &n_segs, K, 20, DICT_HMM_DURATION_MIN_BINS, DICT_HMM_DURATION_MAX_BINS);

    double *p = xcalloc((size_t)K * (size_t)M, sizeof(double));
    fill_p(counts, M, segs, n_segs, K, p);
    double *edges = xcalloc((size_t)(n_segs + 1), sizeof(double));
    edges[0] = 0.0;
    edges[n_segs] = DICT_HMM_RECORDING_MINUTES;
    for (int i = 0; i < n_segs - 1; i++) {
        int L = segs[i].a;
        int R = segs[i + 1].b;
        int kL = segs[i].k;
        int kR = segs[i + 1].k;
        double *ownL = xcalloc((size_t)M, sizeof(double));
        double *ownR = xcalloc((size_t)M, sizeof(double));
        for (int m = 0; m < M; m++) {
            ownL[m] = DICT_HMM_ALPHA;
            ownR[m] = DICT_HMM_ALPHA;
        }
        for (int j = 0; j < n_segs; j++) {
            if (j == i || j == i + 1) continue;
            double *dest = segs[j].k == kL ? ownL : (segs[j].k == kR ? ownR : NULL);
            if (!dest) continue;
            for (int t = segs[j].a; t < segs[j].b; t++) {
                for (int m = 0; m < M; m++) dest[m] += counts[(size_t)t * (size_t)M + (size_t)m];
            }
        }
        double sumL = 0.0, sumR = 0.0;
        for (int m = 0; m < M; m++) {
            sumL += ownL[m];
            sumR += ownR[m];
        }
        for (int m = 0; m < M; m++) {
            ownL[m] /= sumL;
            ownR[m] /= sumR;
        }
        double best = -INFINITY;
        int best_c = segs[i].b;
        for (int c = L + 1; c < R; c++) {
            double score = segment_ll(counts, M, L, c, ownL) + segment_ll(counts, M, c, R, ownR);
            if (score > best) {
                best = score;
                best_c = c;
            }
        }
        edges[i + 1] = (double)best_c * DICT_HMM_BIN_MINUTES;
        free(ownL);
        free(ownR);
    }

    out->n_themes = n_segs;
    out->themes = xcalloc((size_t)n_segs, sizeof(DhInterval));
    for (int i = 0; i < n_segs; i++) {
        out->themes[i].index = segs[i].k;
        out->themes[i].start_sec = edges[i] * 60.0;
        out->themes[i].end_sec = edges[i + 1] * 60.0;
    }
    int n_songs = 0;
    for (int i = 0; i < n_segs; i++) if (segs[i].k == 0) n_songs++;
    out->n_songs = n_songs;
    out->songs = xcalloc((size_t)(n_songs > 0 ? n_songs : 1), sizeof(DhInterval));
    int song = 0;
    for (int i = 0; i < n_segs; i++) {
        if (segs[i].k != 0) continue;
        int end_i = n_segs;
        for (int j = i + 1; j < n_segs; j++) if (segs[j].k == 0) {
            end_i = j;
            break;
        }
        out->songs[song].index = song + 1;
        out->songs[song].start_sec = edges[i] * 60.0;
        out->songs[song].end_sec = (end_i == n_segs ? DICT_HMM_RECORDING_MINUTES : edges[end_i]) * 60.0;
        song++;
    }
    out->n_motifs = n_placements;
    out->motifs = xcalloc((size_t)(n_placements > 0 ? n_placements : 1), sizeof(DhInterval));
    for (int i = 0; i < n_placements; i++) {
        double tmin = start_sec[i] / 60.0;
        int which = 0;
        for (int s = 0; s < n_segs; s++) {
            if (edges[s] <= tmin && tmin < edges[s + 1]) which = s;
        }
        out->motifs[i].index = motif_ids[i];
        out->motifs[i].start_sec = start_sec[i];
        out->motifs[i].end_sec = end_sec ? end_sec[i] : start_sec[i];
        (void)which;
    }
    /* Stash the theme index of each motif in a side channel by encoding is not available.
       Callers that need the theme use the theme intervals. */
    free(edges);
    free(p);
    free(segs);
    free(counts);
    free(ids);
    return 0;
}

void dict_hmm_free_themes(DhThemes *themes) {
    free(themes->motifs);
    free(themes->themes);
    free(themes->songs);
    memset(themes, 0, sizeof(*themes));
}
