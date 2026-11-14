#include "dict_hmm.h"

#include <math.h>
#include <stdint.h>
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

double dict_hmm_hop_seconds(int overlap, int sample_freq) {
    return (double)overlap / (double)sample_freq;
}

double dict_hmm_placement_seconds(int start_sample, int sample_freq, int overlap, int t_start) {
    return ((double)(start_sample - 1) / (double)sample_freq)
        + (double)t_start * dict_hmm_hop_seconds(overlap, sample_freq);
}

void dict_hmm_learn_defaults(DhLearnParams *params) {
    params->k = DICT_HMM_K;
    params->eps = DICT_HMM_EPS;
    params->w_min = DICT_HMM_W_MIN;
    params->w_max = DICT_HMM_W_MAX;
    params->seeds_per_pass = DICT_HMM_SEEDS_PER_PASS;
    params->n_min = DICT_HMM_N_MIN_LEARN;
    params->refine_iters = DICT_HMM_REFINE_ITERS;
    params->tau = DICT_HMM_TAU;
    params->max_outer = DICT_HMM_MAX_OUTER;
    params->min_coverage = DICT_HMM_MIN_COVERAGE;
    params->n_components = DICT_HMM_N_COMPONENTS;
}

void dict_hmm_place_defaults(DhPlaceParams *params) {
    params->k = DICT_HMM_K;
    params->eps = DICT_HMM_EPS;
    params->tau = DICT_HMM_TAU;
    params->n_min = DICT_HMM_N_MIN_PLACE;
}

static double *patch_energy(const double *spec, int n_freq, int n_time, int width) {
    int n_valid = n_time - width + 1;
    double *column = xcalloc((size_t)n_time, sizeof(double));
    for (int t = 0; t < n_time; t++) {
        double s = 0.0;
        for (int f = 0; f < n_freq; f++) {
            double v = spec[(size_t)f * (size_t)n_time + (size_t)t];
            s += v * v;
        }
        column[t] = s;
    }
    double *cum = xcalloc((size_t)n_time, sizeof(double));
    cum[0] = column[0];
    for (int t = 1; t < n_time; t++) cum[t] = cum[t - 1] + column[t];
    double *energy = xcalloc((size_t)n_valid, sizeof(double));
    energy[0] = cum[width - 1];
    for (int t = 1; t < n_valid; t++) energy[t] = cum[t + width - 1] - cum[t - 1];
    free(column);
    free(cum);
    return energy;
}

static void dots_at_starts(const double *spec, int n_freq, int n_time,
                           const double *pattern, int width, double *out) {
    int n_valid = n_time - width + 1;
    for (int s = 0; s < n_valid; s++) {
        double sum = 0.0;
        for (int f = 0; f < n_freq; f++) {
            const double *row = spec + (size_t)f * (size_t)n_time + (size_t)s;
            const double *pat = pattern + (size_t)f * (size_t)width;
            for (int w = 0; w < width; w++) sum += row[w] * pat[w];
        }
        out[s] = sum;
    }
}

void dict_hmm_explain(const double *spectrogram, int n_freq, int n_time,
                      const DhMotif *motif, double *fraction, double *explained, double *first_dots) {
    int width = motif->width;
    int n_valid = n_time - width + 1;
    double *energy = patch_energy(spectrogram, n_freq, n_time, width);
    const double *comp0 = motif->components;
    dots_at_starts(spectrogram, n_freq, n_time, comp0, width, first_dots);
    if (!motif->mean) {
        memset(explained, 0, (size_t)n_valid * sizeof(double));
        for (int c = 0; c < motif->n_components; c++) {
            double *dots = (c == 0) ? first_dots : xcalloc((size_t)n_valid, sizeof(double));
            if (c != 0) {
                dots_at_starts(spectrogram, n_freq, n_time,
                               motif->components + (size_t)c * (size_t)n_freq * (size_t)width,
                               width, dots);
            }
            for (int s = 0; s < n_valid; s++) explained[s] += dots[s] * dots[s];
            if (c != 0) free(dots);
        }
    } else {
        int cells = n_freq * width;
        double mean_energy = 0.0;
        for (int i = 0; i < cells; i++) mean_energy += motif->mean[i] * motif->mean[i];
        double *mean_dots = xcalloc((size_t)n_valid, sizeof(double));
        dots_at_starts(spectrogram, n_freq, n_time, motif->mean, width, mean_dots);
        double *projected = xcalloc((size_t)n_valid, sizeof(double));
        for (int c = 0; c < motif->n_components; c++) {
            const double *comp = motif->components + (size_t)c * (size_t)cells;
            double mean_coeff = 0.0;
            for (int i = 0; i < cells; i++) mean_coeff += motif->mean[i] * comp[i];
            double *dots = (c == 0) ? first_dots : xcalloc((size_t)n_valid, sizeof(double));
            if (c != 0) dots_at_starts(spectrogram, n_freq, n_time, comp, width, dots);
            for (int s = 0; s < n_valid; s++) {
                double d = dots[s] - mean_coeff;
                projected[s] += d * d;
            }
            if (c != 0) free(dots);
        }
        for (int s = 0; s < n_valid; s++) {
            double centered = energy[s] + mean_energy - 2.0 * mean_dots[s];
            double residual = centered - projected[s];
            if (residual < 0.0) residual = 0.0;
            explained[s] = energy[s] - residual;
        }
        free(mean_dots);
        free(projected);
    }
    for (int s = 0; s < n_valid; s++) {
        if (energy[s] > 0.0) fraction[s] = explained[s] / energy[s];
        else fraction[s] = -INFINITY;
    }
    free(energy);
}

static int admissible(const int *counts, int n_time, int start, int width, int k) {
    if (start < 0 || start + width > n_time) return 0;
    int limit = k - 1;
    for (int t = start; t < start + width; t++) {
        if (counts[t] > limit) return 0;
    }
    return 1;
}

int dict_hmm_pursuit(const DhWindow *windows, int n_windows,
                     const DhMotif *motifs, int n_motifs,
                     int *const *counts, int k, double eps,
                     DhPlacement **out, int *n_out) {
    typedef struct {
        int motif;
        int window;
        int n_valid;
        double *fraction;
        double *explained;
        double *dots;
    } Prep;
    int n_prep = 0;
    int cap = n_motifs * n_windows;
    if (cap < 1) cap = 1;
    Prep *prep = xcalloc((size_t)cap, sizeof(Prep));
    for (int m = 0; m < n_motifs; m++) {
        for (int w = 0; w < n_windows; w++) {
            if (windows[w].n_freq != motifs[m].n_freq) {
                fprintf(stderr, "dict_hmm: motif %d frequency %d does not match window %d\n",
                        motifs[m].motif_id, motifs[m].n_freq, windows[w].n_freq);
                free(prep);
                return -1;
            }
            if (windows[w].n_time < motifs[m].width) continue;
            int n_valid = windows[w].n_time - motifs[m].width + 1;
            Prep *p = &prep[n_prep++];
            p->motif = m;
            p->window = w;
            p->n_valid = n_valid;
            p->fraction = xcalloc((size_t)n_valid, sizeof(double));
            p->explained = xcalloc((size_t)n_valid, sizeof(double));
            p->dots = xcalloc((size_t)n_valid, sizeof(double));
            dict_hmm_explain(windows[w].data, windows[w].n_freq, windows[w].n_time,
                             &motifs[m], p->fraction, p->explained, p->dots);
        }
    }
    int n_acc = 0, acc_cap = 16;
    DhPlacement *acc = xcalloc((size_t)acc_cap, sizeof(DhPlacement));
    int have_first = 0;
    double first_energy = 0.0;
    for (;;) {
        int found = 0;
        double best_score = -INFINITY;
        int best_m = 0, best_w = 0, best_s = 0;
        double best_energy = 0.0, best_dot = 0.0;
        for (int i = 0; i < n_prep; i++) {
            Prep *p = &prep[i];
            const DhMotif *motif = &motifs[p->motif];
            const int *cover = counts[p->window];
            double local = -INFINITY;
            int local_s = -1;
            for (int s = 0; s < p->n_valid; s++) {
                if (!admissible(cover, windows[p->window].n_time, s, motif->width, k)) continue;
                if (p->fraction[s] > local) {
                    local = p->fraction[s];
                    local_s = s;
                }
            }
            if (local_s < 0 || !isfinite(local)) continue;
            if (!found || local > best_score) {
                found = 1;
                best_score = local;
                best_m = p->motif;
                best_w = p->window;
                best_s = local_s;
                best_energy = p->explained[local_s];
                best_dot = p->dots[local_s];
            }
        }
        if (!found) break;
        if (!have_first) {
            have_first = 1;
            first_energy = best_energy;
        } else if (best_energy < eps * first_energy) {
            break;
        }
        int width = motifs[best_m].width;
        for (int t = best_s; t < best_s + width; t++) counts[best_w][t] += 1;
        if (n_acc == acc_cap) {
            acc_cap *= 2;
            acc = realloc(acc, (size_t)acc_cap * sizeof(DhPlacement));
            if (!acc) exit(1);
        }
        acc[n_acc].motif_id = motifs[best_m].motif_id;
        acc[n_acc].window_index = best_w;
        acc[n_acc].t_start = best_s;
        acc[n_acc].amplitude = best_dot;
        acc[n_acc].unexplained = 1.0 - best_score;
        n_acc++;
    }
    for (int i = 0; i < n_prep; i++) {
        free(prep[i].fraction);
        free(prep[i].explained);
        free(prep[i].dots);
    }
    free(prep);
    *out = acc;
    *n_out = n_acc;
    return 0;
}

/* Symmetric Jacobi. A is n*n and is overwritten. V receives eigenvectors in columns. */
static void jacobi(int n, double *A, double *V) {
    for (int i = 0; i < n * n; i++) V[i] = 0.0;
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double a = fabs(A[i * n + j]);
                if (a > off) {
                    off = a;
                    p = i;
                    q = j;
                }
            }
        }
        if (off < 1e-15) break;
        double app = A[p * n + p];
        double aqq = A[q * n + q];
        double apq = A[p * n + q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;
        A[p * n + p] = app - t * apq;
        A[q * n + q] = aqq + t * apq;
        A[p * n + q] = A[q * n + p] = 0.0;
        for (int r = 0; r < n; r++) {
            if (r == p || r == q) continue;
            double arp = A[r * n + p];
            double arq = A[r * n + q];
            A[r * n + p] = A[p * n + r] = c * arp - s * arq;
            A[r * n + q] = A[q * n + r] = s * arp + c * arq;
        }
        for (int r = 0; r < n; r++) {
            double vip = V[r * n + p];
            double viq = V[r * n + q];
            V[r * n + p] = c * vip - s * viq;
            V[r * n + q] = s * vip + c * viq;
        }
    }
}

void dict_hmm_principal_components(const double *patches, int n_patches, int n_freq, int width,
                                   int n_components, double **basis_out, int *n_basis, double **mean_out) {
    int cells = n_freq * width;
    int k = n_components < n_patches ? n_components : n_patches;
    double *mean = xcalloc((size_t)cells, sizeof(double));
    for (int p = 0; p < n_patches; p++) {
        const double *patch = patches + (size_t)p * (size_t)cells;
        for (int i = 0; i < cells; i++) mean[i] += patch[i];
    }
    for (int i = 0; i < cells; i++) mean[i] /= (double)n_patches;
    double *centered = xcalloc((size_t)n_patches * (size_t)cells, sizeof(double));
    double centered_norm = 0.0;
    for (int p = 0; p < n_patches; p++) {
        const double *patch = patches + (size_t)p * (size_t)cells;
        double *row = centered + (size_t)p * (size_t)cells;
        for (int i = 0; i < cells; i++) {
            row[i] = patch[i] - mean[i];
            centered_norm += row[i] * row[i];
        }
    }
    double *basis = xcalloc((size_t)k * (size_t)cells, sizeof(double));
    int filled = 0;
    if (n_patches == 1 || centered_norm == 0.0) {
        double norm = 0.0;
        for (int i = 0; i < cells; i++) norm += mean[i] * mean[i];
        norm = sqrt(norm);
        if (norm == 0.0) {
            fprintf(stderr, "dict_hmm: principal component has zero norm\n");
            exit(1);
        }
        for (int i = 0; i < cells; i++) basis[i] = mean[i] / norm;
        filled = 1;
    } else {
        double *gram = xcalloc((size_t)n_patches * (size_t)n_patches, sizeof(double));
        for (int a = 0; a < n_patches; a++) {
            for (int b = a; b < n_patches; b++) {
                double dot = 0.0;
                const double *ra = centered + (size_t)a * (size_t)cells;
                const double *rb = centered + (size_t)b * (size_t)cells;
                for (int i = 0; i < cells; i++) dot += ra[i] * rb[i];
                gram[a * n_patches + b] = gram[b * n_patches + a] = dot;
            }
        }
        double *V = xcalloc((size_t)n_patches * (size_t)n_patches, sizeof(double));
        jacobi(n_patches, gram, V);
        int *order = xcalloc((size_t)n_patches, sizeof(int));
        for (int i = 0; i < n_patches; i++) order[i] = i;
        for (int i = 0; i < n_patches; i++) {
            for (int j = i + 1; j < n_patches; j++) {
                if (gram[order[j] * n_patches + order[j]] > gram[order[i] * n_patches + order[i]]) {
                    int tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
            }
        }
        for (int oi = 0; oi < n_patches && filled < k; oi++) {
            int idx = order[oi];
            double lam = gram[idx * n_patches + idx];
            if (lam <= 0.0) break;
            double *comp = basis + (size_t)filled * (size_t)cells;
            for (int i = 0; i < cells; i++) {
                double s = 0.0;
                for (int p = 0; p < n_patches; p++) {
                    s += centered[(size_t)p * (size_t)cells + (size_t)i] * V[p * n_patches + idx];
                }
                comp[i] = s / sqrt(lam);
            }
            double align = 0.0;
            for (int i = 0; i < cells; i++) align += comp[i] * mean[i];
            if (align < 0.0) {
                for (int i = 0; i < cells; i++) comp[i] = -comp[i];
            }
            filled++;
        }
        free(gram);
        free(V);
        free(order);
        if (filled == 0) {
            double norm = 0.0;
            for (int i = 0; i < cells; i++) norm += mean[i] * mean[i];
            norm = sqrt(norm);
            if (norm == 0.0) {
                fprintf(stderr, "dict_hmm: principal component has zero norm\n");
                exit(1);
            }
            for (int i = 0; i < cells; i++) basis[i] = mean[i] / norm;
            filled = 1;
        }
    }
    free(centered);
    *basis_out = basis;
    *n_basis = filled;
    *mean_out = mean;
}

static const DhMotif *motif_by_id(const DhMotif *motifs, int n, int id) {
    for (int i = 0; i < n; i++) if (motifs[i].motif_id == id) return &motifs[i];
    return NULL;
}

static DhMotif *motif_by_id_mut(DhMotif *motifs, int n, int id) {
    for (int i = 0; i < n; i++) if (motifs[i].motif_id == id) return &motifs[i];
    return NULL;
}

void dict_hmm_rescore(const DhWindow *windows, DhMotif *motifs, int n_motifs,
                      DhPlacement *placements, int n_placements) {
    for (int i = 0; i < n_placements; i++) {
        DhMotif *motif = motif_by_id_mut(motifs, n_motifs, placements[i].motif_id);
        if (!motif) continue;
        const DhWindow *w = &windows[placements[i].window_index];
        int cells = motif->n_freq * motif->width;
        double *patch = xcalloc((size_t)cells, sizeof(double));
        for (int f = 0; f < motif->n_freq; f++) {
            memcpy(patch + (size_t)f * (size_t)motif->width,
                   w->data + (size_t)f * (size_t)w->n_time + (size_t)placements[i].t_start,
                   (size_t)motif->width * sizeof(double));
        }
        double energy = 0.0;
        for (int c = 0; c < cells; c++) energy += patch[c] * patch[c];
        double *coeffs = xcalloc((size_t)motif->n_components, sizeof(double));
        double unexplained;
        if (energy <= 0.0) {
            unexplained = 1.0;
        } else if (!motif->mean) {
            for (int k = 0; k < motif->n_components; k++) {
                const double *comp = motif->components + (size_t)k * (size_t)cells;
                double dot = 0.0;
                for (int c = 0; c < cells; c++) dot += patch[c] * comp[c];
                coeffs[k] = dot;
            }
            double proj = 0.0;
            for (int k = 0; k < motif->n_components; k++) proj += coeffs[k] * coeffs[k];
            unexplained = 1.0 - proj / energy;
        } else {
            for (int k = 0; k < motif->n_components; k++) {
                const double *comp = motif->components + (size_t)k * (size_t)cells;
                double dot = 0.0;
                for (int c = 0; c < cells; c++) dot += (patch[c] - motif->mean[c]) * comp[c];
                coeffs[k] = dot;
            }
            double *recon = xcalloc((size_t)cells, sizeof(double));
            memcpy(recon, motif->mean, (size_t)cells * sizeof(double));
            for (int k = 0; k < motif->n_components; k++) {
                const double *comp = motif->components + (size_t)k * (size_t)cells;
                for (int c = 0; c < cells; c++) recon[c] += coeffs[k] * comp[c];
            }
            double residual = 0.0;
            for (int c = 0; c < cells; c++) {
                double d = patch[c] - recon[c];
                residual += d * d;
            }
            unexplained = residual / energy;
            free(recon);
        }
        double amp;
        if (motif->n_components <= 1) amp = coeffs[0];
        else {
            amp = 0.0;
            for (int k = 0; k < motif->n_components; k++) amp += coeffs[k] * coeffs[k];
            amp = sqrt(amp);
        }
        placements[i].amplitude = amp;
        placements[i].unexplained = unexplained;
        free(coeffs);
        free(patch);
    }
}

static void drop_small(DhMotif **motifs, int *n_motifs, DhPlacement **placements, int *n_placements, int n_min) {
    int n = *n_placements;
    int *keep_m = xcalloc((size_t)(*n_motifs), sizeof(int));
    for (int i = 0; i < *n_motifs; i++) {
        int count = 0;
        for (int p = 0; p < n; p++) if ((*placements)[p].motif_id == (*motifs)[i].motif_id) count++;
        keep_m[i] = count >= n_min;
    }
    int nw = 0;
    DhPlacement *next = xcalloc((size_t)(n > 0 ? n : 1), sizeof(DhPlacement));
    for (int p = 0; p < n; p++) {
        int ok = 0;
        for (int i = 0; i < *n_motifs; i++) {
            if (keep_m[i] && (*motifs)[i].motif_id == (*placements)[p].motif_id) ok = 1;
        }
        if (ok) next[nw++] = (*placements)[p];
    }
    free(*placements);
    *placements = next;
    *n_placements = nw;
    int nm = 0;
    DhMotif *nmot = xcalloc((size_t)(*n_motifs > 0 ? *n_motifs : 1), sizeof(DhMotif));
    for (int i = 0; i < *n_motifs; i++) {
        if (keep_m[i]) nmot[nm++] = (*motifs)[i];
        else {
            free((*motifs)[i].components);
            free((*motifs)[i].mean);
        }
    }
    free(*motifs);
    *motifs = nmot;
    *n_motifs = nm;
    free(keep_m);
}

void dict_hmm_apply_tau(DhMotif **motifs, int *n_motifs,
                        DhPlacement **placements, int *n_placements,
                        double tau, int n_min) {
    int n = *n_placements;
    int nw = 0;
    DhPlacement *next = xcalloc((size_t)(n > 0 ? n : 1), sizeof(DhPlacement));
    for (int i = 0; i < n; i++) {
        if ((*placements)[i].unexplained <= tau) next[nw++] = (*placements)[i];
    }
    free(*placements);
    *placements = next;
    *n_placements = nw;
    drop_small(motifs, n_motifs, placements, n_placements, n_min);
}

static void rebuild_motifs(const DhWindow *windows, DhMotif *motifs, int n_motifs,
                           const DhPlacement *placements, int n_placements, int n_components) {
    for (int m = 0; m < n_motifs; m++) {
        int count = 0;
        for (int p = 0; p < n_placements; p++) if (placements[p].motif_id == motifs[m].motif_id) count++;
        if (count == 0) continue;
        int cells = motifs[m].n_freq * motifs[m].width;
        double *patches = xcalloc((size_t)count * (size_t)cells, sizeof(double));
        int at = 0;
        for (int p = 0; p < n_placements; p++) {
            if (placements[p].motif_id != motifs[m].motif_id) continue;
            const DhWindow *w = &windows[placements[p].window_index];
            double *dest = patches + (size_t)at * (size_t)cells;
            for (int f = 0; f < motifs[m].n_freq; f++) {
                memcpy(dest + (size_t)f * (size_t)motifs[m].width,
                       w->data + (size_t)f * (size_t)w->n_time + (size_t)placements[p].t_start,
                       (size_t)motifs[m].width * sizeof(double));
            }
            at++;
        }
        double *basis = NULL;
        double *mean = NULL;
        int n_basis = 0;
        dict_hmm_principal_components(patches, count, motifs[m].n_freq, motifs[m].width,
                                      n_components, &basis, &n_basis, &mean);
        free(motifs[m].components);
        free(motifs[m].mean);
        motifs[m].components = basis;
        motifs[m].n_components = n_basis;
        if (n_components == 1) {
            free(mean);
            motifs[m].mean = NULL;
        } else {
            motifs[m].mean = mean;
        }
        free(patches);
    }
}

static uint64_t rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1ULL;
    return *state;
}

int dict_hmm_learn(const DhWindow *windows, int n_windows,
                   const DhSegment *segments, int n_segments,
                   const DhLearnParams *params,
                   const int *seed_indices, int n_seed_indices,
                   DhMotif **motifs_out, int *n_motifs,
                   DhPlacement **placements_out, int *n_placements,
                   char *stop_reason, int stop_reason_cap) {
    int *eligible = xcalloc((size_t)(n_segments > 0 ? n_segments : 1), sizeof(int));
    int **counts = xcalloc((size_t)n_windows, sizeof(int *));
    for (int w = 0; w < n_windows; w++) counts[w] = xcalloc((size_t)windows[w].n_time, sizeof(int));
    DhMotif *dictionary = NULL;
    int n_dict = 0;
    DhPlacement *kept = NULL;
    int n_kept = 0;
    int next_id = 0;
    int seed_cursor = 0;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    const char *reason = "max_outer";
    for (int outer = 1; outer <= params->max_outer; outer++) {
        int n_el = 0;
        for (int s = 0; s < n_segments; s++) {
            int width = segments[s].t_end - segments[s].t_start;
            if (width < params->w_min || width > params->w_max) continue;
            int w = segments[s].window_index;
            int ok = 1;
            for (int t = segments[s].t_start; t < segments[s].t_end; t++) {
                if (counts[w][t] > params->k - 1) ok = 0;
            }
            if (ok) eligible[n_el++] = s;
        }
        if (n_el == 0) {
            reason = "no_seeds";
            break;
        }
        int take = params->seeds_per_pass < n_el ? params->seeds_per_pass : n_el;
        int *picked = xcalloc((size_t)take, sizeof(int));
        if (seed_indices) {
            int got = 0;
            while (got < take && seed_cursor < n_seed_indices) {
                int idx = seed_indices[seed_cursor++];
                int ok = 0;
                for (int e = 0; e < n_el; e++) if (eligible[e] == idx) ok = 1;
                if (ok) picked[got++] = idx;
            }
            take = got;
            if (take == 0) {
                free(picked);
                reason = "no_seeds";
                break;
            }
        } else {
            int *pool = xcalloc((size_t)n_el, sizeof(int));
            memcpy(pool, eligible, (size_t)n_el * sizeof(int));
            for (int i = 0; i < take; i++) {
                int j = i + (int)(rng_next(&rng) % (uint64_t)(n_el - i));
                int tmp = pool[i];
                pool[i] = pool[j];
                pool[j] = tmp;
                picked[i] = pool[i];
            }
            free(pool);
        }
        DhMotif *seeds = xcalloc((size_t)take, sizeof(DhMotif));
        for (int i = 0; i < take; i++) {
            const DhSegment *seg = &segments[picked[i]];
            const DhWindow *w = &windows[seg->window_index];
            int width = seg->t_end - seg->t_start;
            int cells = w->n_freq * width;
            double *comp = xcalloc((size_t)cells, sizeof(double));
            double norm = 0.0;
            for (int f = 0; f < w->n_freq; f++) {
                for (int t = 0; t < width; t++) {
                    double v = w->data[(size_t)f * (size_t)w->n_time + (size_t)(seg->t_start + t)];
                    comp[(size_t)f * (size_t)width + (size_t)t] = v;
                    norm += v * v;
                }
            }
            norm = sqrt(norm);
            if (norm > 0.0) for (int c = 0; c < cells; c++) comp[c] /= norm;
            seeds[i].motif_id = next_id++;
            seeds[i].n_components = 1;
            seeds[i].n_freq = w->n_freq;
            seeds[i].width = width;
            seeds[i].components = comp;
            seeds[i].mean = NULL;
        }
        DhPlacement *pass = NULL;
        int n_pass = 0;
        for (int round = 0; round < params->refine_iters; round++) {
            int **working = xcalloc((size_t)n_windows, sizeof(int *));
            for (int w = 0; w < n_windows; w++) {
                working[w] = xcalloc((size_t)windows[w].n_time, sizeof(int));
                memcpy(working[w], counts[w], (size_t)windows[w].n_time * sizeof(int));
            }
            DhPlacement *fresh = NULL;
            int n_fresh = 0;
            if (dict_hmm_pursuit(windows, n_windows, seeds, take, working, params->k, params->eps, &fresh, &n_fresh) != 0) {
                return -1;
            }
            for (int w = 0; w < n_windows; w++) free(working[w]);
            free(working);
            DhPlacement *merged = xcalloc((size_t)(n_pass + n_fresh > 0 ? n_pass + n_fresh : 1), sizeof(DhPlacement));
            memcpy(merged, pass, (size_t)n_pass * sizeof(DhPlacement));
            memcpy(merged + n_pass, fresh, (size_t)n_fresh * sizeof(DhPlacement));
            free(pass);
            free(fresh);
            pass = merged;
            n_pass += n_fresh;
            drop_small(&seeds, &take, &pass, &n_pass, params->n_min);
            if (take > 0) {
                rebuild_motifs(windows, seeds, take, pass, n_pass, params->n_components);
                dict_hmm_rescore(windows, seeds, take, pass, n_pass);
                dict_hmm_apply_tau(&seeds, &take, &pass, &n_pass, params->tau, params->n_min);
            }
            for (int w = 0; w < n_windows; w++) memset(counts[w], 0, (size_t)windows[w].n_time * sizeof(int));
            for (int p = 0; p < n_kept; p++) {
                const DhMotif *m = motif_by_id(dictionary, n_dict, kept[p].motif_id);
                if (!m) continue;
                for (int t = kept[p].t_start; t < kept[p].t_start + m->width; t++) counts[kept[p].window_index][t] += 1;
            }
            for (int p = 0; p < n_pass; p++) {
                const DhMotif *m = motif_by_id(seeds, take, pass[p].motif_id);
                if (!m) continue;
                for (int t = pass[p].t_start; t < pass[p].t_start + m->width; t++) counts[pass[p].window_index][t] += 1;
            }
        }
        free(picked);
        if (take == 0) {
            free(pass);
            reason = "no_survivors";
            break;
        }
        dictionary = realloc(dictionary, (size_t)(n_dict + take) * sizeof(DhMotif));
        memcpy(dictionary + n_dict, seeds, (size_t)take * sizeof(DhMotif));
        free(seeds);
        n_dict += take;
        kept = realloc(kept, (size_t)(n_kept + n_pass > 0 ? n_kept + n_pass : 1) * sizeof(DhPlacement));
        if (n_pass) memcpy(kept + n_kept, pass, (size_t)n_pass * sizeof(DhPlacement));
        free(pass);
        n_kept += n_pass;
        int covered = 0, total = 0;
        for (int w = 0; w < n_windows; w++) {
            total += windows[w].n_time;
            for (int t = 0; t < windows[w].n_time; t++) if (counts[w][t]) covered++;
        }
        if (total > 0 && (double)covered / (double)total > params->min_coverage) {
            reason = "coverage";
            break;
        }
    }
    if (stop_reason && stop_reason_cap > 0) {
        snprintf(stop_reason, (size_t)stop_reason_cap, "%s", reason);
    }
    for (int w = 0; w < n_windows; w++) free(counts[w]);
    free(counts);
    free(eligible);
    *motifs_out = dictionary;
    *n_motifs = n_dict;
    *placements_out = kept ? kept : xcalloc(1, sizeof(DhPlacement));
    *n_placements = n_kept;
    return 0;
}

int dict_hmm_place(const DhWindow *windows, int n_windows,
                   DhMotif *motifs, int n_motifs,
                   const DhPlaceParams *params,
                   DhPlacement **out, int *n_out) {
    int **counts = xcalloc((size_t)n_windows, sizeof(int *));
    for (int w = 0; w < n_windows; w++) counts[w] = xcalloc((size_t)windows[w].n_time, sizeof(int));
    DhPlacement *placements = NULL;
    int n = 0;
    int rc = dict_hmm_pursuit(windows, n_windows, motifs, n_motifs, counts, params->k, params->eps, &placements, &n);
    for (int w = 0; w < n_windows; w++) free(counts[w]);
    free(counts);
    if (rc != 0) return rc;
    dict_hmm_rescore(windows, motifs, n_motifs, placements, n);
    int kept_n = 0;
    DhPlacement *kept = xcalloc((size_t)(n > 0 ? n : 1), sizeof(DhPlacement));
    for (int i = 0; i < n; i++) {
        if (placements[i].unexplained <= params->tau) kept[kept_n++] = placements[i];
    }
    int *occ = xcalloc((size_t)(n_motifs > 0 ? n_motifs : 1), sizeof(int));
    for (int i = 0; i < kept_n; i++) {
        for (int m = 0; m < n_motifs; m++) {
            if (motifs[m].motif_id == kept[i].motif_id) occ[m]++;
        }
    }
    int out_n = 0;
    DhPlacement *filtered = xcalloc((size_t)(kept_n > 0 ? kept_n : 1), sizeof(DhPlacement));
    for (int i = 0; i < kept_n; i++) {
        int ok = 0;
        for (int m = 0; m < n_motifs; m++) {
            if (motifs[m].motif_id == kept[i].motif_id && occ[m] >= params->n_min) ok = 1;
        }
        if (ok) filtered[out_n++] = kept[i];
    }
    free(occ);
    free(kept);
    free(placements);
    *out = filtered;
    *n_out = out_n;
    return 0;
}

void dict_hmm_free_motifs(DhMotif *motifs, int n) {
    for (int i = 0; i < n; i++) {
        free(motifs[i].components);
        free(motifs[i].mean);
    }
    free(motifs);
}

void dict_hmm_free_placements(DhPlacement *placements) { free(placements); }
