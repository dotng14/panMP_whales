#include "dict_hmm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", msg);
        g_failed++;
    }
}

static double *read_bin(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    double *data = calloc(n, sizeof(double));
    if (fread(data, sizeof(double), n, f) != n) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(f);
    return data;
}

static void test_time(void) {
    double hop = dict_hmm_hop_seconds(24576, 200000);
    expect(fabs(hop - 0.12288) < 1e-12, "hop");
    double t = dict_hmm_placement_seconds(1, 200000, 24576, 0);
    expect(fabs(t) < 1e-12, "time at origin");
    t = dict_hmm_placement_seconds(15000001, 200000, 24576, 10);
    expect(fabs(t - (75.0 + 10.0 * 0.12288)) < 1e-9, "time in second window");
}

static void test_defaults(void) {
    DhLearnParams learn;
    DhPlaceParams place;
    dict_hmm_learn_defaults(&learn);
    dict_hmm_place_defaults(&place);
    expect(learn.k == 1 && learn.eps == 0.01 && learn.tau == 0.4, "learn defaults");
    expect(learn.refine_iters == 2 && learn.n_components == 5, "learn refine and components");
    expect(learn.n_min == 10 && place.n_min == 1, "n_min differs for place");
    expect(place.tau == 0.4 && place.k == 1, "place defaults");
}

static void test_oracle(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/pursuit.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    int n_freq, n_time, width, n_frac, n_place;
    if (fscanf(f, "%d %d %d %d", &n_freq, &n_time, &width, &n_frac) != 4) exit(1);
    if (fscanf(f, "%d", &n_place) != 1) exit(1);
    int *starts = calloc((size_t)n_place, sizeof(int));
    int *ids = calloc((size_t)n_place, sizeof(int));
    for (int i = 0; i < n_place; i++) {
        double unexplained, amplitude;
        if (fscanf(f, "%d %d %lf %lf", &ids[i], &starts[i], &unexplained, &amplitude) != 4) exit(1);
    }
    fclose(f);

    snprintf(path, sizeof path, "%s/spec.bin", dir);
    double *spec = read_bin(path, (size_t)n_freq * (size_t)n_time);
    snprintf(path, sizeof path, "%s/pattern.bin", dir);
    double *pattern = read_bin(path, (size_t)n_freq * (size_t)width);
    snprintf(path, sizeof path, "%s/fraction.bin", dir);
    double *fraction_py = read_bin(path, (size_t)n_frac);

    DhMotif motif = {0};
    motif.motif_id = 7;
    motif.n_components = 1;
    motif.n_freq = n_freq;
    motif.width = width;
    motif.components = pattern;
    double *fraction = calloc((size_t)n_frac, sizeof(double));
    double *explained = calloc((size_t)n_frac, sizeof(double));
    double *dots = calloc((size_t)n_frac, sizeof(double));
    dict_hmm_explain(spec, n_freq, n_time, &motif, fraction, explained, dots);
    double max_diff = 0.0;
    for (int i = 0; i < n_frac; i++) {
        double d = fabs(fraction[i] - fraction_py[i]);
        if (d > max_diff) max_diff = d;
    }
    expect(max_diff < 1e-8, "explained fraction matches Python");

    DhWindow window = {0};
    window.n_freq = n_freq;
    window.n_time = n_time;
    window.data = spec;
    window.sample_freq = 200000;
    window.overlap = 24576;
    window.start_sample = 1;
    snprintf(window.id, sizeof window.id, "w0");
    int *counts = calloc((size_t)n_time, sizeof(int));
    int *count_rows[1] = {counts};
    DhPlacement *got = NULL;
    int n_got = 0;
    expect(dict_hmm_pursuit(&window, 1, &motif, 1, count_rows, 1, 0.01, &got, &n_got) == 0, "pursuit runs");
    expect(n_got == n_place, "pursuit count");
    int same = n_got == n_place;
    for (int i = 0; same && i < n_got; i++) {
        if (got[i].motif_id != ids[i] || got[i].t_start != starts[i]) same = 0;
    }
    expect(same, "pursuit starts match Python");

    snprintf(path, sizeof path, "%s/basis.txt", dir);
    f = fopen(path, "r");
    int npatch, pf, pw, nb;
    if (fscanf(f, "%d %d %d %d", &npatch, &pf, &pw, &nb) != 4) exit(1);
    fclose(f);
    snprintf(path, sizeof path, "%s/patches.bin", dir);
    double *patches = read_bin(path, (size_t)npatch * (size_t)pf * (size_t)pw);
    snprintf(path, sizeof path, "%s/basis.bin", dir);
    double *basis_py = read_bin(path, (size_t)nb * (size_t)pf * (size_t)pw);
    double *basis = NULL;
    double *mean = NULL;
    int n_basis = 0;
    dict_hmm_principal_components(patches, npatch, pf, pw, 2, &basis, &n_basis, &mean);
    expect(n_basis == nb, "component count");
    double basis_diff = 0.0;
    int cells = pf * pw;
    for (int c = 0; c < n_basis && c < nb; c++) {
        for (int i = 0; i < cells; i++) {
            double d = fabs(basis[(size_t)c * (size_t)cells + (size_t)i]
                            - basis_py[(size_t)c * (size_t)cells + (size_t)i]);
            if (d > basis_diff) basis_diff = d;
        }
    }
    expect(basis_diff < 1e-6, "principal components match Python");

    free(spec);
    free(pattern);
    free(fraction_py);
    free(fraction);
    free(explained);
    free(dots);
    free(counts);
    free(got);
    free(starts);
    free(ids);
    free(patches);
    free(basis_py);
    free(basis);
    free(mean);
}

static void test_learn_and_place(void) {
    int n_freq = 3, n_time = 24, width = 4;
    double *data = calloc((size_t)n_freq * (size_t)n_time, sizeof(double));
    for (int f = 0; f < n_freq; f++) {
        for (int t = 0; t < width; t++) data[(size_t)f * (size_t)n_time + (size_t)t] = (double)((f + 1) * (t + 1));
        for (int t = 0; t < width; t++) data[(size_t)f * (size_t)n_time + 12u + (size_t)t] = data[(size_t)f * (size_t)n_time + (size_t)t];
    }
    DhWindow window = {0};
    window.n_freq = n_freq;
    window.n_time = n_time;
    window.data = data;
    window.sample_freq = 200000;
    window.overlap = 24576;
    window.start_sample = 1;
    DhSegment segment = {.window_index = 0, .t_start = 0, .t_end = width};
    DhLearnParams learn;
    dict_hmm_learn_defaults(&learn);
    learn.w_min = 2;
    learn.w_max = 8;
    learn.n_min = 1;
    learn.refine_iters = 1;
    learn.n_components = 2;
    learn.seeds_per_pass = 1;
    learn.max_outer = 1;
    learn.min_coverage = 2.0;
    learn.tau = 0.95;
    int seed_index = 0;
    DhMotif *motifs = NULL;
    DhPlacement *placements = NULL;
    int n_motifs = 0, n_placements = 0;
    char reason[32];
    expect(dict_hmm_learn(&window, 1, &segment, 1, &learn, &seed_index, 1,
                          &motifs, &n_motifs, &placements, &n_placements, reason, 32) == 0,
           "learn runs");
    expect(n_motifs >= 1 && n_placements >= 1, "learn keeps a motif and a placement");
    expect(motifs && motifs[0].mean != NULL, "five-component default path stores a mean when asked for 2");
    DhPlaceParams place;
    dict_hmm_place_defaults(&place);
    place.n_min = 1;
    place.tau = 0.95;
    DhPlacement *placed = NULL;
    int n_placed = 0;
    expect(dict_hmm_place(&window, 1, motifs, n_motifs, &place, &placed, &n_placed) == 0, "place runs");
    expect(n_placed >= 1, "place keeps a placement");
    dict_hmm_free_motifs(motifs, n_motifs);
    dict_hmm_free_placements(placements);
    dict_hmm_free_placements(placed);
    free(data);
}

int main(int argc, char **argv) {
    test_time();
    test_defaults();
    test_learn_and_place();
    if (argc > 1) test_oracle(argv[1]);
    if (g_failed) {
        fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    printf("core checks passed\n");
    return 0;
}
