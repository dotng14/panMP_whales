/* Whale-song detector for 10 kHz Site H recordings.

   The stages run at the same time and hand chunks of 30 windows (37.5
   minutes) to each other:
     1. One reader thread reads the recordings in order.
     2. CPU threads resample each 75-second window to 12.5 kHz, take the
        spectrogram (2048-sample frames every 1536 samples), and whiten it.
     3. One thread per GPU correlates every window of a chunk with every
        motif template (cuFFT and cuBLAS, single precision unless built with
        -DGPU_DOUBLE).
     4. CPU threads turn the correlations into explained fractions (double
        precision) and run the greedy pursuit and the stop rules.
   When all chunks are placed, the placements of each recording are decoded
   with the fixed four-theme cyclic model and each cycle is tested (own-theme
   margins, shuffled-order test).

   The numbers follow spectro.py, place.py, themes.py, and songs.py. The
   shuffles use NumPy's default generator with seed 0, drawn in the order the
   recordings are listed, as songs.py does.

   usage: song_detector MODEL OUT_DIR WAV [WAV ...]
*/
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    SOURCE_RATE = 10000,
    FFTL = 2048,
    STEP = 1536,
    WINDOW_SEC = 75,
    WINDOW_SAMPLES = WINDOW_SEC * SOURCE_RATE,
    RESAMPLED = WINDOW_SAMPLES / 4 * 5,
    FIRST_BIN = 2,
    N_TIME = (RESAMPLED - FFTL) / STEP,
    CHUNK_WINDOWS = 30,
    NCORR = 640, /* correlation length: at least N_TIME + widest motif - 1 */
    NFREQ = NCORR / 2 + 1,
    HALF_LEN = 50,
    N_TAPS = 21,
    TAPS_PADDED = 24,
    FRAME_BLOCK = 8,
    FINE_SHIFT = 37, /* sorting digits: bits 37-45, then everything from bit 46 up */
    FINE_BUCKETS = 512,
    COARSE_SHIFT = 46,
    MAX_BUCKETS = 2048,
    N_THEMES = 4,
    MIN_BINS = 2,
    MAX_BINS = 60,
    N_SHUFFLES = 500,
    MAX_CAND = 128,
    MAX_GPUS = 8,
    MAX_CPU_THREADS = 256,
    POOL_CHUNKS = 8,
};
static const double HOP_SEC = (double)STEP / 12500.0;
static const double BIN_SEC = 30.0;
static const double EPS = 0.01;
static const double TAU = 0.4;
static const double ALMOST_INF = 1e300;

/* ---- library declarations (the headers are not installed) ---- */
typedef double fftw_complex[2];
typedef struct fftw_plan_s *fftw_plan;
fftw_plan fftw_plan_dft_r2c_1d(int, double *, fftw_complex *, unsigned);
void fftw_execute_dft_r2c(const fftw_plan, double *, fftw_complex *);
void *fftw_malloc(size_t);
void fftw_free(void *);

typedef int cufftHandle;
typedef struct cublasContext *cublasHandle_t;
enum { COPY_H2D = 1, COPY_D2H = 2, CUBLAS_OP_N = 0, CUBLAS_OP_C = 2 };

/* The GPUs work in single precision, which these cards run many times faster than double. Building with
   -DGPU_DOUBLE keeps double precision there, which reproduces the Python detector's placements exactly. */
#ifdef GPU_DOUBLE
typedef double gpu_real;
enum { GPU_FORWARD = 0x6a, GPU_INVERSE = 0x6c }; /* CUFFT_D2Z, CUFFT_Z2D */
#define gpu_fft_forward cufftExecD2Z
#define gpu_fft_inverse cufftExecZ2D
#define gpu_gemm_batched cublasZgemmStridedBatched
#define GPU_PRECISION "double"
#else
typedef float gpu_real;
enum { GPU_FORWARD = 0x2a, GPU_INVERSE = 0x2c }; /* CUFFT_R2C, CUFFT_C2R */
#define gpu_fft_forward cufftExecR2C
#define gpu_fft_inverse cufftExecC2R
#define gpu_gemm_batched cublasCgemmStridedBatched
#define GPU_PRECISION "single"
#endif
typedef struct { gpu_real x, y; } gpu_complex;
int cudaGetDeviceCount(int *);
int cudaSetDevice(int);
int cudaMalloc(void **, size_t);
int cudaFree(void *);
int cudaHostAlloc(void **, size_t, unsigned); /* flag 1: pinned for every device */
int cudaMemset(void *, int, size_t);
int cudaMemcpy(void *, const void *, size_t, int);
int cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, int);
int cudaDeviceSynchronize(void);
const char *cudaGetErrorString(int);
int cufftPlanMany(cufftHandle *, int, int *, int *, int, int, int *, int, int, int, int);
int gpu_fft_forward(cufftHandle, gpu_real *, gpu_complex *);
int gpu_fft_inverse(cufftHandle, gpu_complex *, gpu_real *);
int cufftDestroy(cufftHandle);
int cublasCreate_v2(cublasHandle_t *);
int cublasDestroy_v2(cublasHandle_t);
int gpu_gemm_batched(cublasHandle_t, int, int, int, int, int, const gpu_complex *,
                     const gpu_complex *, int, long long, const gpu_complex *, int, long long,
                     const gpu_complex *, gpu_complex *, int, long long, int);

static void die(const char *what, const char *detail) {
    fprintf(stderr, "song_detector: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory", NULL);
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size);
    if (!p) die("out of memory", NULL);
    return p;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void gpu_check(int rc, const char *what) {
    if (rc != 0) die(what, cudaGetErrorString(rc));
}

static void lib_check(int rc, const char *what) {
    if (rc != 0) {
        char code[32];
        snprintf(code, sizeof code, "status %d", rc);
        die(what, code);
    }
}

/* ---- model ---- */
typedef struct {
    int n_motifs, n_comp, n_rows, n_templates;
    int *ids, *widths;
    double **mean;  /* [motif] n_rows * width */
    double **basis; /* [motif] n_comp * n_rows * width */
    double *mean_energy;
    double *mean_coeff; /* n_motifs * n_comp */
    double *log_p;      /* N_THEMES * n_motifs */
    double log_duration[N_THEMES][MAX_BINS + 1];
} Model;

static void read_exact(FILE *f, void *dst, size_t n) {
    if (fread(dst, 1, n, f) != n) die("model file is truncated", NULL);
}

static Model load_model(const char *path) {
    Model m;
    memset(&m, 0, sizeof m);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open model", path);
    char magic[8];
    int32_t head[4];
    read_exact(f, magic, 8);
    if (memcmp(magic, "SONGMDL1", 8) != 0) die("not a song model", path);
    read_exact(f, head, sizeof head);
    m.n_motifs = head[0];
    m.n_comp = head[1];
    m.n_rows = head[2];
    if (head[3] != N_THEMES) die("model has a different number of themes", NULL);
    m.n_templates = m.n_motifs * (1 + m.n_comp);
    m.ids = xmalloc((size_t)m.n_motifs * sizeof(int));
    m.widths = xmalloc((size_t)m.n_motifs * sizeof(int));
    read_exact(f, m.ids, (size_t)m.n_motifs * sizeof(int32_t));
    read_exact(f, m.widths, (size_t)m.n_motifs * sizeof(int32_t));
    m.mean = xmalloc((size_t)m.n_motifs * sizeof(double *));
    m.basis = xmalloc((size_t)m.n_motifs * sizeof(double *));
    m.mean_energy = xcalloc((size_t)m.n_motifs, sizeof(double));
    m.mean_coeff = xcalloc((size_t)m.n_motifs * (size_t)m.n_comp, sizeof(double));
    for (int i = 0; i < m.n_motifs; i++) {
        size_t cells = (size_t)m.n_rows * (size_t)m.widths[i];
        if (N_TIME + m.widths[i] - 1 > NCORR) die("a motif is too wide for the correlation length", NULL);
        m.mean[i] = xmalloc(cells * sizeof(double));
        m.basis[i] = xmalloc(cells * (size_t)m.n_comp * sizeof(double));
        read_exact(f, m.mean[i], cells * sizeof(double));
        read_exact(f, m.basis[i], cells * (size_t)m.n_comp * sizeof(double));
        double e = 0;
        for (size_t c = 0; c < cells; c++) e += m.mean[i][c] * m.mean[i][c];
        m.mean_energy[i] = e;
        for (int k = 0; k < m.n_comp; k++) {
            double dot = 0;
            const double *b = m.basis[i] + (size_t)k * cells;
            for (size_t c = 0; c < cells; c++) dot += m.mean[i][c] * b[c];
            m.mean_coeff[(size_t)i * m.n_comp + k] = dot;
        }
    }
    m.log_p = xmalloc((size_t)N_THEMES * (size_t)m.n_motifs * sizeof(double));
    double mu[N_THEMES], sigma[N_THEMES];
    read_exact(f, m.log_p, (size_t)N_THEMES * (size_t)m.n_motifs * sizeof(double));
    read_exact(f, mu, sizeof mu);
    read_exact(f, sigma, sizeof sigma);
    fclose(f);
    for (int k = 0; k < N_THEMES; k++) {
        double raw[MAX_BINS + 1], total = 0;
        for (int d = 1; d <= MAX_BINS; d++) {
            double z = (log((double)d) - mu[k]) / sigma[k];
            raw[d] = -0.5 * z * z - log((double)d * sigma[k]);
            if (d >= MIN_BINS) total += exp(raw[d]);
        }
        m.log_duration[k][0] = -INFINITY;
        for (int d = 1; d <= MAX_BINS; d++)
            m.log_duration[k][d] = d < MIN_BINS ? -INFINITY : raw[d] - log(total);
    }
    return m;
}

/* ---- front end: resample, spectrogram, whitening ---- */
static double g_taps[5][TAPS_PADDED]; /* polyphase taps, stored so the dot product runs forward; zeros after N_TAPS */
static double g_hamming[FFTL];
static fftw_plan g_plan;

static double bessel_i0(double x) {
    double term = 1.0, sum = 1.0, half = 0.5 * x;
    for (int k = 1; k < 200; k++) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

static void init_front_end(void) {
    /* scipy.signal.resample_poly(x, 5, 4): 101-tap Kaiser(5.0) low-pass at 1/5, unit gain, times 5. */
    double h[2 * HALF_LEN + 1], total = 0;
    for (int i = 0; i <= 2 * HALF_LEN; i++) {
        double m = (double)(i - HALF_LEN);
        double arg = M_PI * 0.2 * m;
        double sinc = m == 0.0 ? 1.0 : sin(arg) / arg;
        double r = m / (double)HALF_LEN;
        h[i] = 0.2 * sinc * bessel_i0(5.0 * sqrt(1.0 - r * r)) / bessel_i0(5.0);
        total += h[i];
    }
    for (int i = 0; i <= 2 * HALF_LEN; i++) h[i] = h[i] / total * 5.0;
    for (int p = 0; p < 5; p++) {
        for (int j = 0; j < N_TAPS; j++) {
            int index = p + 5 * j;
            g_taps[p][N_TAPS - 1 - j] = index <= 2 * HALF_LEN ? h[index] : 0.0;
        }
    }
    for (int i = 0; i < FFTL; i++) g_hamming[i] = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(FFTL - 1));
    double *in = fftw_malloc(FFTL * sizeof(double));
    fftw_complex *out = fftw_malloc((FFTL / 2 + 1) * sizeof(fftw_complex));
    g_plan = fftw_plan_dft_r2c_1d(FFTL, in, out, 0 /* FFTW_MEASURE */);
    if (!g_plan) die("fftw plan failed", NULL);
    fftw_free(in);
    fftw_free(out);
}

typedef struct {
    double *input;     /* the samples one hop of resampling needs, as doubles */
    double *audio;     /* FFTL resampled samples: the current frame */
    double *frame;     /* FFTL, fftw-aligned */
    fftw_complex *spectrum;
    double *block;     /* FRAME_BLOCK frames of (n_rows + 1) magnitudes */
    double *magnitude; /* (n_rows + 1) * N_TIME, whitened in place */
    double *zeros;     /* N_TIME */
    double *smooth;    /* N_TIME: one smoothed row */
    uint64_t *keys, *spare;
    unsigned *counts;  /* FINE_BUCKETS + MAX_BUCKETS */
    double *score, *explained; /* n_motifs * N_TIME */
} Scratch;

static Scratch scratch_new(const Model *m) {
    Scratch s;
    s.input = xmalloc(FFTL * sizeof(double));
    s.audio = xmalloc(FFTL * sizeof(double));
    s.frame = fftw_malloc(FFTL * sizeof(double));
    s.spectrum = fftw_malloc((FFTL / 2 + 1) * sizeof(fftw_complex));
    s.block = xmalloc((size_t)FRAME_BLOCK * (size_t)(m->n_rows + 1) * sizeof(double));
    s.magnitude = xmalloc((size_t)(m->n_rows + 1) * N_TIME * sizeof(double));
    s.zeros = xcalloc(N_TIME, sizeof(double));
    s.smooth = xmalloc(N_TIME * sizeof(double));
    s.keys = xmalloc(N_TIME * sizeof(uint64_t));
    s.spare = xmalloc(N_TIME * sizeof(uint64_t));
    s.counts = xmalloc((FINE_BUCKETS + MAX_BUCKETS) * sizeof(unsigned));
    s.score = xmalloc((size_t)m->n_motifs * N_TIME * sizeof(double));
    s.explained = xmalloc((size_t)m->n_motifs * N_TIME * sizeof(double));
    return s;
}

/* Resampled samples [first_out, end_out) of one window; samples outside the window count as zero. */
static void resample_span(const int16_t *samples, int first_out, int end_out, double *out, double *input) {
    int first = (4 * first_out + HALF_LEN) / 5 - (N_TAPS - 1);
    int end = (4 * (end_out - 1) + HALF_LEN) / 5 - (N_TAPS - 1) + TAPS_PADDED;
    int lo = first < 0 ? 0 : first, hi = end > WINDOW_SAMPLES ? WINDOW_SAMPLES : end;
    for (int i = first; i < lo; i++) input[i - first] = 0.0;
    for (int i = lo; i < hi; i++) input[i - first] = (double)samples[i] / 32768.0;
    for (int i = hi > lo ? hi : lo; i < end; i++) input[i - first] = 0.0;
    for (int o = first_out; o < end_out; o++) {
        int base = 4 * o + HALF_LEN;
        const double *taps = g_taps[base % 5];
        const double *from = input + (base / 5 - (N_TAPS - 1) - first);
        double acc = 0;
#pragma omp simd reduction(+ : acc)
        for (int j = 0; j < TAPS_PADDED; j++) acc += taps[j] * from[j];
        out[o - first_out] = acc;
    }
}

/* Noise level of one row: the mean of the values inside its densest half. */
static double shortest_half_mean(const double *row, uint64_t *keys, uint64_t *spare, unsigned *counts) {
    /* Sort the bit patterns; magnitudes are non-negative, so they sort like the values. Two counting passes
       order the exponent and the leading 15 bits of the fraction, and an insertion pass settles the rest. */
    double lo4[4] = {row[0], row[0], row[0], row[0]}, hi4[4] = {row[0], row[0], row[0], row[0]};
    for (int i = 0; i + 4 <= N_TIME; i += 4) {
        for (int lane = 0; lane < 4; lane++) {
            lo4[lane] = row[i + lane] < lo4[lane] ? row[i + lane] : lo4[lane];
            hi4[lane] = row[i + lane] > hi4[lane] ? row[i + lane] : hi4[lane];
        }
    }
    double lo_value = lo4[0], hi_value = hi4[0];
    for (int lane = 1; lane < 4; lane++) {
        lo_value = lo4[lane] < lo_value ? lo4[lane] : lo_value;
        hi_value = hi4[lane] > hi_value ? hi4[lane] : hi_value;
    }
    for (int i = N_TIME - N_TIME % 4; i < N_TIME; i++) {
        lo_value = row[i] < lo_value ? row[i] : lo_value;
        hi_value = row[i] > hi_value ? row[i] : hi_value;
    }
    uint64_t lo_bits, hi_bits;
    memcpy(&lo_bits, &lo_value, 8);
    memcpy(&hi_bits, &hi_value, 8);
    uint64_t base = lo_bits >> COARSE_SHIFT, span = (hi_bits >> COARSE_SHIFT) - base + 1;
    memcpy(keys, row, N_TIME * sizeof(uint64_t));
    if (span <= MAX_BUCKETS) {
        unsigned *fine = counts, *coarse = counts + FINE_BUCKETS;
        memset(counts, 0, (FINE_BUCKETS + (size_t)span) * sizeof(unsigned));
        for (int i = 0; i < N_TIME; i++) {
            fine[(keys[i] >> FINE_SHIFT) & (FINE_BUCKETS - 1)]++;
            coarse[(keys[i] >> COARSE_SHIFT) - base]++;
        }
        unsigned at = 0;
        for (int b = 0; b < FINE_BUCKETS; b++) {
            unsigned count = fine[b];
            fine[b] = at;
            at += count;
        }
        at = 0;
        for (uint64_t b = 0; b < span; b++) {
            unsigned count = coarse[b];
            coarse[b] = at;
            at += count;
        }
        for (int i = 0; i < N_TIME; i++) spare[fine[(keys[i] >> FINE_SHIFT) & (FINE_BUCKETS - 1)]++] = keys[i];
        for (int i = 0; i < N_TIME; i++) keys[coarse[(spare[i] >> COARSE_SHIFT) - base]++] = spare[i];
    } else {
        /* A very wide range of values (a row with zeros in it): byte passes over the high 32 bits. */
        for (int pass = 0; pass < 4; pass++) {
            int shift = 32 + 8 * pass;
            unsigned at = 0;
            memset(counts, 0, 256 * sizeof(unsigned));
            for (int i = 0; i < N_TIME; i++) counts[(keys[i] >> shift) & 255]++;
            for (int b = 0; b < 256; b++) {
                unsigned count = counts[b];
                counts[b] = at;
                at += count;
            }
            for (int i = 0; i < N_TIME; i++) spare[counts[(keys[i] >> shift) & 255]++] = keys[i];
            uint64_t *swap = keys;
            keys = spare;
            spare = swap;
        }
    }
    for (int i = 1; i < N_TIME; i++) {
        uint64_t k = keys[i];
        int j = i;
        for (; j > 0 && keys[j - 1] > k; j--) keys[j] = keys[j - 1];
        keys[j] = k;
    }
    int n = N_TIME - (N_TIME % 2), half = n / 2, pick = 0;
    double best = INFINITY;
    for (int i = 0; i < half; i++) {
        double lo, hi;
        memcpy(&lo, &keys[i], 8);
        memcpy(&hi, &keys[i + half], 8);
        if (hi - lo < best) {
            best = hi - lo;
            pick = i;
        }
    }
    double low, high, sum = 0;
    memcpy(&low, &keys[pick], 8);
    memcpy(&high, &keys[pick + half], 8);
    int count = 0;
    for (int i = 0; i < N_TIME; i++) {
        /* Add the value when it is inside and zero otherwise, without a branch. */
        int inside = (row[i] >= low) & (row[i] <= high);
        uint64_t bits;
        double value;
        memcpy(&bits, &row[i], 8);
        bits &= (uint64_t)0 - (uint64_t)inside;
        memcpy(&value, &bits, 8);
        sum += value;
        count += inside;
    }
    return sum / (double)count;
}

/* One window: samples -> whitened spectrogram (n_rows * N_TIME, in the GPUs' precision) and cumulative
   column energy (N_TIME + 1, double precision).
   The work moves through small buffers that stay in the cache: one frame of audio at a time, a few frames
   of magnitudes at a time, and three rows at a time for the smoothing. */
static void front_end(const Model *m, const int16_t *samples, Scratch *s, gpu_real *restrict spec, double *restrict cumulative) {
    int rows = m->n_rows + 1; /* one extra row so the smoothing of the top kept row sees its neighbor */
    for (int t0 = 0; t0 < N_TIME; t0 += FRAME_BLOCK) {
        int n = N_TIME - t0 < FRAME_BLOCK ? N_TIME - t0 : FRAME_BLOCK;
        for (int b = 0; b < n; b++) {
            int t = t0 + b;
            if (t == 0) {
                resample_span(samples, 0, FFTL, s->audio, s->input);
            } else {
                memmove(s->audio, s->audio + STEP, (FFTL - STEP) * sizeof(double));
                resample_span(samples, t * STEP + FFTL - STEP, t * STEP + FFTL, s->audio + FFTL - STEP, s->input);
            }
            for (int i = 0; i < FFTL; i++) s->frame[i] = s->audio[i] * g_hamming[i];
            fftw_execute_dft_r2c(g_plan, s->frame, s->spectrum);
            double *restrict mag = s->block + (size_t)b * rows;
            for (int r = 0; r < rows; r++) {
                const double *z = s->spectrum[FIRST_BIN + r];
                mag[r] = sqrt(z[0] * z[0] + z[1] * z[1]);
            }
        }
        for (int r = 0; r < rows; r++) {
            double *to = s->magnitude + (size_t)r * N_TIME + t0;
            for (int b = 0; b < n; b++) to[b] = s->block[(size_t)b * rows + r];
        }
    }
    memset(cumulative, 0, (N_TIME + 1) * sizeof(double));
    for (int r = 0; r < rows; r++) {
        double *restrict row = s->magnitude + (size_t)r * N_TIME;
        double mu = shortest_half_mean(row, s->keys, s->spare, s->counts);
        for (int t = 0; t < N_TIME; t++) row[t] = fabs((row[t] - mu) / mu);
        if (r == 0) continue;
        /* Smooth the row below, now that its upper neighbor is whitened. Missing neighbors count as zero. */
        const double *mid = row - N_TIME, *above = row;
        const double *below = r > 1 ? mid - N_TIME : s->zeros;
        double *restrict smooth = s->smooth;
        gpu_real *restrict out = spec + (size_t)(r - 1) * N_TIME;
        smooth[0] = (4.0 * mid[0] + below[0] + above[0] + mid[1]) / 8.0;
        for (int t = 1; t < N_TIME - 1; t++) smooth[t] = (4.0 * mid[t] + below[t] + above[t] + mid[t - 1] + mid[t + 1]) / 8.0;
        int last = N_TIME - 1;
        smooth[last] = (4.0 * mid[last] + below[last] + above[last] + mid[last - 1]) / 8.0;
        for (int t = 0; t < N_TIME; t++) {
            cumulative[t + 1] += smooth[t] * smooth[t];
            out[t] = (gpu_real)smooth[t];
        }
    }
    for (int t = 0; t < N_TIME; t++) cumulative[t + 1] += cumulative[t];
}

/* ---- GPU correlation ---- */
typedef struct {
    int device;
    gpu_complex *templates; /* NFREQ matrices, each n_rows x n_templates */
    gpu_real *in;            /* CHUNK_WINDOWS * n_rows rows of NCORR */
    gpu_complex *spectra;   /* NFREQ matrices, each n_rows x CHUNK_WINDOWS */
    gpu_complex *product;   /* NFREQ matrices, each n_templates x CHUNK_WINDOWS */
    gpu_real *out;           /* CHUNK_WINDOWS * n_templates rows of NCORR */
    cufftHandle forward, inverse;
    cublasHandle_t blas;
} Gpu;

static gpu_real *g_template_host; /* [template][row][NCORR], zero-padded in time */

static void build_template_host(const Model *m) {
    size_t rows = (size_t)m->n_templates * (size_t)m->n_rows;
    g_template_host = xcalloc(rows * NCORR, sizeof(gpu_real));
    for (int i = 0; i < m->n_motifs; i++) {
        int width = m->widths[i];
        for (int part = 0; part <= m->n_comp; part++) {
            const double *src = part == 0 ? m->mean[i] : m->basis[i] + (size_t)(part - 1) * m->n_rows * width;
            size_t t = (size_t)i * (1 + m->n_comp) + part;
            for (int r = 0; r < m->n_rows; r++) {
                gpu_real *to = g_template_host + (t * m->n_rows + r) * NCORR;
                for (int j = 0; j < width; j++) to[j] = (gpu_real)src[(size_t)r * width + j];
            }
        }
    }
}

static Gpu gpu_new(const Model *m, int device) {
    Gpu g;
    memset(&g, 0, sizeof g);
    g.device = device;
    gpu_check(cudaSetDevice(device), "cudaSetDevice");
    int n[1] = {NCORR}, real_embed[1] = {NCORR}, freq_embed[1] = {NFREQ};
    int template_rows = m->n_templates * m->n_rows;
    int window_rows = CHUNK_WINDOWS * m->n_rows;
    int product_rows = CHUNK_WINDOWS * m->n_templates;

    /* Template spectra, laid out so frequency k holds an n_rows x n_templates matrix. */
    void *staging;
    cufftHandle plan;
    gpu_check(cudaMalloc(&staging, (size_t)template_rows * NCORR * sizeof(gpu_real)), "cudaMalloc templates");
    gpu_check(cudaMalloc((void **)&g.templates, (size_t)template_rows * NFREQ * sizeof(gpu_complex)), "cudaMalloc spectra");
    gpu_check(cudaMemcpy(staging, g_template_host, (size_t)template_rows * NCORR * sizeof(gpu_real), COPY_H2D), "upload templates");
    lib_check(cufftPlanMany(&plan, 1, n, real_embed, 1, NCORR, freq_embed, template_rows, 1, GPU_FORWARD, template_rows), "cufft template plan");
    lib_check(gpu_fft_forward(plan, staging, g.templates), "cufft template transform");
    gpu_check(cudaDeviceSynchronize(), "template transform");
    cufftDestroy(plan);
    cudaFree(staging);

    gpu_check(cudaMalloc((void **)&g.in, (size_t)window_rows * NCORR * sizeof(gpu_real)), "cudaMalloc in");
    gpu_check(cudaMemset(g.in, 0, (size_t)window_rows * NCORR * sizeof(gpu_real)), "cudaMemset");
    gpu_check(cudaMalloc((void **)&g.spectra, (size_t)window_rows * NFREQ * sizeof(gpu_complex)), "cudaMalloc spectra");
    gpu_check(cudaMalloc((void **)&g.product, (size_t)product_rows * NFREQ * sizeof(gpu_complex)), "cudaMalloc product");
    gpu_check(cudaMalloc((void **)&g.out, (size_t)product_rows * NCORR * sizeof(gpu_real)), "cudaMalloc out");
    lib_check(cufftPlanMany(&g.forward, 1, n, real_embed, 1, NCORR, freq_embed, window_rows, 1, GPU_FORWARD, window_rows), "cufft forward plan");
    lib_check(cufftPlanMany(&g.inverse, 1, n, freq_embed, product_rows, 1, real_embed, 1, NCORR, GPU_INVERSE, product_rows), "cufft inverse plan");
    lib_check(cublasCreate_v2(&g.blas), "cublasCreate");
    return g;
}

/* spec: CHUNK_WINDOWS windows of n_rows * N_TIME. dots: [window][template][N_TIME], not yet divided by NCORR. */
static void gpu_correlate(const Model *m, Gpu *g, const gpu_real *spec, gpu_real *dots) {
    int window_rows = CHUNK_WINDOWS * m->n_rows;
    int product_rows = CHUNK_WINDOWS * m->n_templates;
    const gpu_complex one = {1, 0}, zero = {0, 0};
    gpu_check(cudaMemcpy2D(g->in, NCORR * sizeof(gpu_real), spec, N_TIME * sizeof(gpu_real),
                           N_TIME * sizeof(gpu_real), (size_t)window_rows, COPY_H2D), "upload windows");
    lib_check(gpu_fft_forward(g->forward, g->in, g->spectra), "cufft forward");
    lib_check(gpu_gemm_batched(g->blas, CUBLAS_OP_C, CUBLAS_OP_N, m->n_templates, CHUNK_WINDOWS, m->n_rows, &one,
                                        g->templates, m->n_rows, (long long)m->n_rows * m->n_templates,
                                        g->spectra, m->n_rows, (long long)window_rows, &zero,
                                        g->product, m->n_templates, (long long)product_rows, NFREQ), "cublas gemm");
    lib_check(gpu_fft_inverse(g->inverse, g->product, g->out), "cufft inverse");
    gpu_check(cudaMemcpy2D(dots, N_TIME * sizeof(gpu_real), g->out, NCORR * sizeof(gpu_real),
                           N_TIME * sizeof(gpu_real), (size_t)product_rows, COPY_D2H), "download dots");
}

/* ---- pursuit ---- */
typedef struct {
    double score, energy;
    int window, motif, start, order;
} Cand;

typedef struct {
    int motif_id;
    double begin, end, unexplained;
} Row;

/* Explained fraction of every motif at every start of one window, then the greedy pass. */
static int pursue_window(const Model *m, const gpu_real *dots, const double *cumulative, Scratch *s, int window, Cand *out) {
    const double inv = 1.0 / (double)NCORR;
    int per = 1 + m->n_comp;
    for (int i = 0; i < m->n_motifs; i++) {
        int width = m->widths[i], n_valid = N_TIME - width + 1;
        const gpu_real *d = dots + (size_t)i * per * N_TIME;
        const double *coeff = m->mean_coeff + (size_t)i * m->n_comp;
        double *score = s->score + (size_t)i * N_TIME, *explained = s->explained + (size_t)i * N_TIME;
        for (int t = 0; t < n_valid; t++) {
            double energy = cumulative[t + width] - cumulative[t];
            double centered = energy + m->mean_energy[i] - 2.0 * ((double)d[t] * inv);
            double projected = 0;
            for (int k = 0; k < m->n_comp; k++) {
                double v = (double)d[(size_t)(k + 1) * N_TIME + t] * inv - coeff[k];
                projected += v * v;
            }
            double residual = centered - projected;
            if (residual < 0.0) residual = 0.0;
            double gain = energy - residual;
            explained[t] = gain;
            score[t] = energy > 0.0 ? gain / energy : -INFINITY;
        }
        for (int t = n_valid; t < N_TIME; t++) {
            explained[t] = 0.0;
            score[t] = -INFINITY;
        }
    }
    int n = 0, cells = m->n_motifs * N_TIME;
    for (;;) {
        double best = -INFINITY;
        int at = -1;
        for (int i = 0; i < cells; i++) {
            if (s->score[i] > best) {
                best = s->score[i];
                at = i;
            }
        }
        if (at < 0 || !(best < ALMOST_INF)) break;
        if (n == MAX_CAND) die("too many placements in one window", NULL);
        int motif = at / N_TIME, start = at % N_TIME;
        out[n].score = best;
        out[n].energy = s->explained[at];
        out[n].window = window;
        out[n].motif = motif;
        out[n].start = start;
        n++;
        int end = start + m->widths[motif];
        if (end > N_TIME) end = N_TIME;
        for (int other = 0; other < m->n_motifs; other++) {
            int lo = start - m->widths[other] + 1;
            if (lo < 0) lo = 0;
            double *row = s->score + (size_t)other * N_TIME;
            for (int t = lo; t < end; t++) row[t] = -INFINITY;
        }
    }
    return n;
}

static int by_score(const void *a, const void *b) {
    const Cand *x = a, *y = b;
    if (x->score > y->score) return -1;
    if (x->score < y->score) return 1;
    return x->order - y->order;
}

/* ---- recordings and jobs ---- */
typedef struct {
    const char *path;
    char stem[256];
    int fd;
    off_t data_offset;
    int64_t n_samples;
    int n_windows, n_chunks;
    time_t start_time;
    Row **chunk_rows;
    int *chunk_count;
} Recording;

static Recording open_recording(const char *path) {
    Recording r;
    memset(&r, 0, sizeof r);
    r.path = path;
    r.fd = open(path, O_RDONLY);
    if (r.fd < 0) die("cannot open recording", path);
    unsigned char head[12], chunk[8], fmt[16];
    if (pread(r.fd, head, 12, 0) != 12 || memcmp(head, "RIFF", 4) != 0 || memcmp(head + 8, "WAVE", 4) != 0)
        die("not a wav file", path);
    off_t at = 12;
    int rate = 0;
    for (;;) {
        if (pread(r.fd, chunk, 8, at) != 8) die("wav file has no data chunk", path);
        uint32_t size;
        memcpy(&size, chunk + 4, 4);
        at += 8;
        if (memcmp(chunk, "fmt ", 4) == 0) {
            if (pread(r.fd, fmt, 16, at) != 16) die("short fmt chunk", path);
            uint16_t audio, channels, bits;
            memcpy(&audio, fmt, 2);
            memcpy(&channels, fmt + 2, 2);
            memcpy(&rate, fmt + 4, 4);
            memcpy(&bits, fmt + 14, 2);
            if (audio != 1 || channels != 1 || bits != 16) die("not 16-bit mono PCM", path);
        } else if (memcmp(chunk, "data", 4) == 0) {
            r.data_offset = at;
            r.n_samples = size / 2;
            break;
        }
        at += size + (size & 1);
    }
    if (rate != SOURCE_RATE) die("recording is not 10 kHz", path);
    r.n_windows = (int)(r.n_samples / WINDOW_SAMPLES);
    r.n_chunks = (r.n_windows + CHUNK_WINDOWS - 1) / CHUNK_WINDOWS;
    r.chunk_rows = xcalloc((size_t)r.n_chunks, sizeof(Row *));
    r.chunk_count = xcalloc((size_t)r.n_chunks, sizeof(int));
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    size_t len = strcspn(base, ".");
    if (len >= sizeof r.stem) len = sizeof r.stem - 1;
    memcpy(r.stem, base, len);
    /* start time: the first _YYMMDD_HHMMSS in the name */
    r.start_time = (time_t)-1;
    size_t name_len = strlen(base);
    for (size_t at_char = 0; at_char + 14 <= name_len; at_char++) {
        const char *p = base + at_char;
        int ok = p[0] == '_' && p[7] == '_';
        for (int i = 1; ok && i <= 13; i++)
            if (i != 7 && (p[i] < '0' || p[i] > '9')) ok = 0;
        if (!ok) continue;
        struct tm tm;
        memset(&tm, 0, sizeof tm);
#define TWO(i) ((p[i] - '0') * 10 + (p[(i) + 1] - '0'))
        tm.tm_year = 100 + TWO(1);
        tm.tm_mon = TWO(3) - 1;
        tm.tm_mday = TWO(5);
        tm.tm_hour = TWO(8);
        tm.tm_min = TWO(10);
        tm.tm_sec = TWO(12);
#undef TWO
        r.start_time = timegm(&tm);
        break;
    }
    if (r.start_time == (time_t)-1) die("file name has no _YYMMDD_HHMMSS stamp", path);
    posix_fadvise(r.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    return r;
}

/* ---- pipeline: reader -> CPU threads (spectrograms) -> GPU threads (matching) -> CPU threads (pursuit) ---- */
typedef struct Chunk {
    int recording, chunk, n_windows;
    int next_window, done_windows; /* progress of the CPU stage the chunk is in */
    int16_t *samples;   /* CHUNK_WINDOWS * WINDOW_SAMPLES */
    gpu_real *spec;     /* CHUNK_WINDOWS windows of n_rows * N_TIME, pinned for the GPUs */
    double *cumulative; /* CHUNK_WINDOWS * (N_TIME + 1) */
    gpu_real *dots;     /* CHUNK_WINDOWS windows of n_templates * N_TIME, pinned for the GPUs */
    Cand *cands;        /* CHUNK_WINDOWS * MAX_CAND */
    int cand_count[CHUNK_WINDOWS];
    struct Chunk *next;
} Chunk;

typedef struct {
    Chunk *head, *tail;
} Fifo;

static void fifo_push(Fifo *q, Chunk *c) {
    c->next = NULL;
    if (q->tail) q->tail->next = c;
    else q->head = c;
    q->tail = c;
}

static Chunk *fifo_pop(Fifo *q) {
    Chunk *c = q->head;
    if (c) {
        q->head = c->next;
        if (!q->head) q->tail = NULL;
    }
    return c;
}

static const Model *g_model;
static Recording *g_recordings;
static int g_n_recordings;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_free_cv = PTHREAD_COND_INITIALIZER; /* a chunk buffer was returned */
static pthread_cond_t g_cpu_cv = PTHREAD_COND_INITIALIZER;  /* windows are waiting for a CPU thread */
static pthread_cond_t g_gpu_cv = PTHREAD_COND_INITIALIZER;  /* a chunk is waiting for a GPU */
static Fifo g_free, g_front, g_match, g_pursue;
static int g_reader_done, g_outstanding;
static double g_t_read, g_t_front, g_t_gpu, g_t_pursuit;

static void *reader_main(void *arg) {
    (void)arg;
    const Model *m = g_model;
    size_t window_cells = (size_t)m->n_rows * N_TIME;
    double spent = 0;
    for (int r = 0; r < g_n_recordings; r++) {
        Recording *rec = &g_recordings[r];
        for (int chunk = 0; chunk < rec->n_chunks; chunk++) {
            pthread_mutex_lock(&g_mu);
            Chunk *c;
            while (!(c = fifo_pop(&g_free))) pthread_cond_wait(&g_free_cv, &g_mu);
            pthread_mutex_unlock(&g_mu);
            int first = chunk * CHUNK_WINDOWS;
            c->recording = r;
            c->chunk = chunk;
            c->n_windows = rec->n_windows - first < CHUNK_WINDOWS ? rec->n_windows - first : CHUNK_WINDOWS;
            c->next_window = c->done_windows = 0;
            double t0 = now();
            size_t want = (size_t)c->n_windows * WINDOW_SAMPLES * sizeof(int16_t), got = 0;
            off_t offset = rec->data_offset + (off_t)first * WINDOW_SAMPLES * (off_t)sizeof(int16_t);
            while (got < want) {
                ssize_t n = pread(rec->fd, (char *)c->samples + got, want - got, offset + (off_t)got);
                if (n <= 0) die("read failed", rec->path);
                got += (size_t)n;
            }
            spent += now() - t0;
            if (c->n_windows < CHUNK_WINDOWS)
                memset(c->spec + (size_t)c->n_windows * window_cells, 0,
                       (size_t)(CHUNK_WINDOWS - c->n_windows) * window_cells * sizeof(gpu_real));
            pthread_mutex_lock(&g_mu);
            g_outstanding++;
            fifo_push(&g_front, c);
            pthread_cond_broadcast(&g_cpu_cv);
            pthread_mutex_unlock(&g_mu);
        }
    }
    pthread_mutex_lock(&g_mu);
    g_reader_done = 1;
    g_t_read = spent;
    pthread_cond_broadcast(&g_cpu_cv);
    pthread_cond_broadcast(&g_gpu_cv);
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

/* One pursuit over the chunk: best fraction first, stop at the energy rule, keep fits within tau. */
static void select_chunk(const Model *m, Chunk *c) {
    int n = 0;
    for (int i = 0; i < c->n_windows; i++) {
        memmove(c->cands + n, c->cands + (size_t)i * MAX_CAND, (size_t)c->cand_count[i] * sizeof(Cand));
        n += c->cand_count[i];
    }
    for (int i = 0; i < n; i++) c->cands[i].order = i;
    qsort(c->cands, (size_t)n, sizeof(Cand), by_score);
    Row *rows = xmalloc((size_t)(n > 0 ? n : 1) * sizeof(Row));
    int kept = 0;
    double chunk_start = (double)(c->chunk * CHUNK_WINDOWS) * WINDOW_SEC, first_energy = 0;
    for (int i = 0; i < n; i++) {
        const Cand *cand = &c->cands[i];
        if (i == 0) first_energy = cand->energy;
        else if (cand->energy < EPS * first_energy) break;
        double unexplained = 1.0 - cand->score;
        if (unexplained <= TAU) {
            double begin = chunk_start + (double)(cand->window * WINDOW_SEC) + (double)cand->start * HOP_SEC;
            rows[kept].motif_id = m->ids[cand->motif];
            rows[kept].begin = begin;
            rows[kept].end = begin + (double)m->widths[cand->motif] * HOP_SEC;
            rows[kept].unexplained = unexplained;
            kept++;
        }
    }
    g_recordings[c->recording].chunk_rows[c->chunk] = rows;
    g_recordings[c->recording].chunk_count[c->chunk] = kept;
}

/* Takes one window at a time: pursuit of a matched chunk first, otherwise the spectrogram of a new one. */
static void *cpu_main(void *arg) {
    (void)arg;
    const Model *m = g_model;
    size_t window_cells = (size_t)m->n_rows * N_TIME, window_dots = (size_t)m->n_templates * N_TIME;
    Scratch scratch = scratch_new(m);
    double t_front = 0, t_pursuit = 0;
    pthread_mutex_lock(&g_mu);
    for (;;) {
        Chunk *c;
        if ((c = g_pursue.head)) {
            int window = c->next_window++;
            if (c->next_window == c->n_windows) fifo_pop(&g_pursue);
            pthread_mutex_unlock(&g_mu);
            double t0 = now();
            c->cand_count[window] = pursue_window(m, c->dots + (size_t)window * window_dots,
                                                  c->cumulative + (size_t)window * (N_TIME + 1), &scratch, window,
                                                  c->cands + (size_t)window * MAX_CAND);
            t_pursuit += now() - t0;
            pthread_mutex_lock(&g_mu);
            if (++c->done_windows == c->n_windows) {
                pthread_mutex_unlock(&g_mu);
                t0 = now();
                select_chunk(m, c);
                t_pursuit += now() - t0;
                pthread_mutex_lock(&g_mu);
                fifo_push(&g_free, c);
                pthread_cond_signal(&g_free_cv);
                if (--g_outstanding == 0 && g_reader_done) {
                    pthread_cond_broadcast(&g_cpu_cv);
                    pthread_cond_broadcast(&g_gpu_cv);
                }
            }
        } else if ((c = g_front.head)) {
            int window = c->next_window++;
            if (c->next_window == c->n_windows) fifo_pop(&g_front);
            pthread_mutex_unlock(&g_mu);
            double t0 = now();
            front_end(m, c->samples + (size_t)window * WINDOW_SAMPLES, &scratch, c->spec + (size_t)window * window_cells,
                      c->cumulative + (size_t)window * (N_TIME + 1));
            t_front += now() - t0;
            pthread_mutex_lock(&g_mu);
            if (++c->done_windows == c->n_windows) {
                c->next_window = c->done_windows = 0;
                fifo_push(&g_match, c);
                pthread_cond_signal(&g_gpu_cv);
            }
        } else if (g_reader_done && g_outstanding == 0) {
            break;
        } else {
            pthread_cond_wait(&g_cpu_cv, &g_mu);
        }
    }
    g_t_front += t_front;
    g_t_pursuit += t_pursuit;
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

static void *gpu_main(void *arg) {
    int device = (int)(intptr_t)arg;
    const Model *m = g_model;
    Gpu gpu = gpu_new(m, device);
    double spent = 0;
    pthread_mutex_lock(&g_mu);
    for (;;) {
        Chunk *c = fifo_pop(&g_match);
        if (!c) {
            if (g_reader_done && g_outstanding == 0) break;
            pthread_cond_wait(&g_gpu_cv, &g_mu);
            continue;
        }
        pthread_mutex_unlock(&g_mu);
        double t0 = now();
        gpu_correlate(m, &gpu, c->spec, c->dots);
        spent += now() - t0;
        pthread_mutex_lock(&g_mu);
        fifo_push(&g_pursue, c);
        pthread_cond_broadcast(&g_cpu_cv);
    }
    g_t_gpu += spent;
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

/* ---- themes and songs ---- */
typedef struct {
    int theme, first, end;
} Visit;

static int by_begin(const void *a, const void *b) {
    const Row *x = a, *y = b;
    return (x->begin > y->begin) - (x->begin < y->begin);
}

/* Best cyclic path through the bins. Returns the visits in time order. */
static Visit *decode(const Model *m, const double *counts, int n_bins, int *n_visits) {
    int nm = m->n_motifs;
    double *cumulative = xcalloc((size_t)(n_bins + 1) * N_THEMES, sizeof(double));
    for (int t = 0; t < n_bins; t++) {
        for (int k = 0; k < N_THEMES; k++) {
            double e = 0;
            for (int j = 0; j < nm; j++) e += counts[(size_t)t * nm + j] * m->log_p[(size_t)k * nm + j];
            cumulative[(size_t)(t + 1) * N_THEMES + k] = cumulative[(size_t)t * N_THEMES + k] + e;
        }
    }
    double *best = xmalloc((size_t)(n_bins + 1) * N_THEMES * sizeof(double));
    int *back = xcalloc((size_t)(n_bins + 1) * N_THEMES, sizeof(int));
    for (size_t i = 0; i < (size_t)(n_bins + 1) * N_THEMES; i++) best[i] = -INFINITY;
    for (int t = 1; t <= n_bins; t++) {
        for (int k = 0; k < N_THEMES; k++) {
            int previous = (k + N_THEMES - 1) % N_THEMES;
            int longest = t < MAX_BINS ? t : MAX_BINS;
            for (int d = MIN_BINS; d <= longest; d++) {
                int start = t - d;
                double lead = start == 0 ? -log((double)N_THEMES) : best[(size_t)start * N_THEMES + previous];
                if (!(lead > -ALMOST_INF)) continue;
                double score = lead + m->log_duration[k][d] + cumulative[(size_t)t * N_THEMES + k] - cumulative[(size_t)start * N_THEMES + k];
                if (score > best[(size_t)t * N_THEMES + k]) {
                    best[(size_t)t * N_THEMES + k] = score;
                    back[(size_t)t * N_THEMES + k] = d;
                }
            }
        }
    }
    int theme = 0;
    for (int k = 1; k < N_THEMES; k++)
        if (best[(size_t)n_bins * N_THEMES + k] > best[(size_t)n_bins * N_THEMES + theme]) theme = k;
    Visit *visits = xmalloc((size_t)(n_bins + 1) * sizeof(Visit));
    int n = 0, t = n_bins;
    while (t > 0) {
        int d = back[(size_t)t * N_THEMES + theme];
        if (d <= 0) die("theme decoding found no path", NULL);
        visits[n].theme = theme;
        visits[n].first = t - d;
        visits[n].end = t;
        n++;
        t -= d;
        theme = (theme + N_THEMES - 1) % N_THEMES;
    }
    for (int i = 0; i < n / 2; i++) {
        Visit swap = visits[i];
        visits[i] = visits[n - 1 - i];
        visits[n - 1 - i] = swap;
    }
    free(cumulative);
    free(best);
    free(back);
    *n_visits = n;
    return visits;
}

/* Best score of the bins as one visit of each theme, in the order 0, 1, 2, 3. order[i] is the bin used at position i. */
static double one_cycle_score(const Model *m, const double *emission, const int *order, int n) {
    double cumulative[N_THEMES][2 * MAX_BINS * N_THEMES + 1];
    double best[2 * MAX_BINS * N_THEMES + 1], next[2 * MAX_BINS * N_THEMES + 1];
    for (int k = 0; k < N_THEMES; k++) {
        cumulative[k][0] = 0;
        for (int i = 0; i < n; i++) cumulative[k][i + 1] = cumulative[k][i] + emission[(size_t)order[i] * N_THEMES + k];
    }
    for (int i = 0; i <= n; i++) best[i] = -INFINITY;
    best[0] = 0.0;
    for (int k = 0; k < N_THEMES; k++) {
        for (int i = 0; i <= n; i++) next[i] = -INFINITY;
        for (int end = 1; end <= n; end++) {
            int longest = end < MAX_BINS ? end : MAX_BINS;
            for (int d = MIN_BINS; d <= longest; d++) {
                int start = end - d;
                if (!(best[start] > -ALMOST_INF)) continue;
                double score = best[start] + m->log_duration[k][d] + cumulative[k][end] - cumulative[k][start];
                if (score > next[end]) next[end] = score;
            }
        }
        memcpy(best, next, (size_t)(n + 1) * sizeof(double));
    }
    return best[n];
}

/* NumPy's default generator (PCG64 seeded through SeedSequence), so the shuffles are the ones songs.py draws. */
typedef struct {
    unsigned __int128 state, inc;
    int has_half;
    uint32_t half;
} Pcg64;

static uint32_t seed_hashmix(uint32_t value, uint32_t *hash_const) {
    value ^= *hash_const;
    *hash_const *= 0x931e8875u;
    value *= *hash_const;
    value ^= value >> 16;
    return value;
}

static uint32_t seed_mix(uint32_t x, uint32_t y) {
    uint32_t result = 0xca01f9ddu * x - 0x4973f715u * y;
    result ^= result >> 16;
    return result;
}

static void pcg64_step(Pcg64 *g) {
    const unsigned __int128 mult = ((unsigned __int128)2549297995355413924ULL << 64) | 4865540595714422341ULL;
    g->state = g->state * mult + g->inc;
}

/* np.random.default_rng(0) */
static Pcg64 pcg64_seed0(void) {
    uint32_t pool[4], hash_const = 0x43b0d7e5u, entropy[1] = {0};
    for (int i = 0; i < 4; i++) pool[i] = seed_hashmix(i < 1 ? entropy[i] : 0, &hash_const);
    for (int src = 0; src < 4; src++)
        for (int dst = 0; dst < 4; dst++)
            if (src != dst) pool[dst] = seed_mix(pool[dst], seed_hashmix(pool[src], &hash_const));
    uint32_t words[8];
    hash_const = 0x8b51f9ddu;
    for (int i = 0; i < 8; i++) {
        uint32_t value = pool[i % 4];
        value ^= hash_const;
        hash_const *= 0x58f38dedu;
        value *= hash_const;
        value ^= value >> 16;
        words[i] = value;
    }
    uint64_t v[4];
    for (int i = 0; i < 4; i++) v[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    Pcg64 g;
    unsigned __int128 init_state = ((unsigned __int128)v[0] << 64) | v[1];
    unsigned __int128 init_seq = ((unsigned __int128)v[2] << 64) | v[3];
    g.state = 0;
    g.inc = (init_seq << 1) | 1;
    pcg64_step(&g);
    g.state += init_state;
    pcg64_step(&g);
    g.has_half = 0;
    g.half = 0;
    return g;
}

static uint64_t pcg64_next64(Pcg64 *g) {
    pcg64_step(g);
    uint64_t hi = (uint64_t)(g->state >> 64), lo = (uint64_t)g->state;
    uint64_t value = hi ^ lo;
    unsigned rot = (unsigned)(hi >> 58);
    return (value >> rot) | (value << ((64 - rot) & 63));
}

static uint32_t pcg64_next32(Pcg64 *g) {
    if (g->has_half) {
        g->has_half = 0;
        return g->half;
    }
    uint64_t next = pcg64_next64(g);
    g->has_half = 1;
    g->half = (uint32_t)(next >> 32);
    return (uint32_t)next;
}

/* Generator.permutation(n): start from 0..n-1 and swap from the top down. */
static void numpy_permutation(Pcg64 *g, int *order, int n) {
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i >= 1; i--) {
        uint32_t mask = (uint32_t)i, value;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        while ((value = pcg64_next32(g) & mask) > (uint32_t)i) {
        }
        int swap = order[value];
        order[value] = order[i];
        order[i] = swap;
    }
}

static void format_time(time_t base, double seconds, char *out, size_t cap) {
    time_t t = base + (time_t)llround(seconds);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(out, cap, "%Y-%m-%d %H:%M:%S", &tm);
}

/* Write the placements of one recording, decode them, and append its cycles to the song table. */
static void finish_recording(const Model *m, Recording *rec, Pcg64 *rng, const char *out_dir, FILE *songs,
                             int *n_cycles, int *n_conforming) {
    int total = 0, nm = m->n_motifs;
    for (int c = 0; c < rec->n_chunks; c++) total += rec->chunk_count[c];
    Row *rows = xmalloc((size_t)total * sizeof(Row));
    int n = 0;
    for (int c = 0; c < rec->n_chunks; c++) {
        memcpy(rows + n, rec->chunk_rows[c], (size_t)rec->chunk_count[c] * sizeof(Row));
        n += rec->chunk_count[c];
        free(rec->chunk_rows[c]);
    }
    qsort(rows, (size_t)n, sizeof(Row), by_begin);
    char path[1024], text[64];
    snprintf(path, sizeof path, "%s/placements_%s.csv", out_dir, rec->stem);
    FILE *f = fopen(path, "w");
    if (!f) die("cannot write", path);
    fprintf(f, "motif,start_sec,end_sec,unexplained\n");
    double last_end = 0;
    for (int i = 0; i < n; i++) {
        fprintf(f, "%d,%.5f,%.5f,%.6f\n", rows[i].motif_id, rows[i].begin, rows[i].end, rows[i].unexplained);
        /* The decoder reads the times as written, to five decimals. */
        snprintf(text, sizeof text, "%.5f", rows[i].begin);
        rows[i].begin = strtod(text, NULL);
        snprintf(text, sizeof text, "%.5f", rows[i].end);
        rows[i].end = strtod(text, NULL);
        if (rows[i].end > last_end) last_end = rows[i].end;
    }
    fclose(f);
    fprintf(stderr, "%s: %d windows, %d placements\n", rec->stem, rec->n_windows, n);
    if (n == 0) {
        free(rows);
        return;
    }

    int n_bins = (int)ceil(last_end / BIN_SEC);
    double *counts = xcalloc((size_t)n_bins * nm, sizeof(double));
    int *column = xmalloc(1024 * sizeof(int));
    for (int j = 0; j < 1024; j++) column[j] = -1;
    for (int j = 0; j < nm; j++) column[m->ids[j]] = j;
    for (int i = 0; i < n; i++) {
        int b = (int)floor(rows[i].begin / BIN_SEC);
        if (b >= 0 && b < n_bins) counts[(size_t)b * nm + column[rows[i].motif_id]] += 1.0;
    }
    double *emission = xmalloc((size_t)n_bins * N_THEMES * sizeof(double));
    for (int t = 0; t < n_bins; t++) {
        for (int k = 0; k < N_THEMES; k++) {
            double e = 0;
            for (int j = 0; j < nm; j++) e += counts[(size_t)t * nm + j] * m->log_p[(size_t)k * nm + j];
            emission[(size_t)t * N_THEMES + k] = e;
        }
    }
    int n_visits;
    Visit *visits = decode(m, counts, n_bins, &n_visits);
    int order[2 * MAX_BINS * N_THEMES];
    for (int a = 0; a < n_visits; a++) {
        if (visits[a].theme != 0 || a + N_THEMES >= n_visits) continue;
        const Visit *cycle = visits + a; /* themes 0, 1, 2, 3; the next visit is theme 0 again */
        double margins[N_THEMES], min_margin = INFINITY;
        for (int v = 0; v < N_THEMES; v++) {
            double score[N_THEMES] = {0};
            for (int t = cycle[v].first; t < cycle[v].end; t++)
                for (int k = 0; k < N_THEMES; k++) score[k] += emission[(size_t)t * N_THEMES + k];
            double other = -INFINITY;
            for (int k = 0; k < N_THEMES; k++)
                if (k != cycle[v].theme && score[k] > other) other = score[k];
            margins[v] = score[cycle[v].theme] - other;
            if (margins[v] < min_margin) min_margin = margins[v];
        }
        int first = cycle[0].first, end = cycle[N_THEMES - 1].end, len = end - first;
        double placements = 0;
        for (int t = first; t < end; t++)
            for (int j = 0; j < nm; j++) placements += counts[(size_t)t * nm + j];
        double z = NAN;
        int at_or_above = -1;
        if (min_margin > 0) {
            for (int i = 0; i < len; i++) order[i] = first + i;
            double actual = one_cycle_score(m, emission, order, len), sum = 0, sum2 = 0;
            at_or_above = 0;
            for (int s = 0; s < N_SHUFFLES; s++) {
                numpy_permutation(rng, order, len);
                for (int i = 0; i < len; i++) order[i] += first;
                double score = one_cycle_score(m, emission, order, len);
                sum += score;
                sum2 += score * score;
                if (score >= actual) at_or_above++;
            }
            double mean = sum / N_SHUFFLES, var = sum2 / N_SHUFFLES - mean * mean;
            z = (actual - mean) / sqrt(var > 0 ? var : 0);
        }
        char start_text[32], end_text[32];
        format_time(rec->start_time, first * BIN_SEC, start_text, sizeof start_text);
        format_time(rec->start_time, end * BIN_SEC, end_text, sizeof end_text);
        fprintf(songs, "%s,%s,%s,%.1f,%.1f,%g,%d,", rec->stem, start_text, end_text, first * BIN_SEC, end * BIN_SEC,
                len * BIN_SEC / 60.0, (int)placements);
        for (int v = 0; v < N_THEMES; v++) fprintf(songs, "%s%g", v ? "/" : "", (cycle[v].end - cycle[v].first) / 2.0);
        fputc(',', songs);
        for (int v = 0; v < N_THEMES; v++) fprintf(songs, "%s%g", v ? "/" : "", cycle[v].first * BIN_SEC);
        fputc(',', songs);
        for (int v = 0; v < N_THEMES; v++) fprintf(songs, "%s%.1f", v ? "/" : "", margins[v]);
        fprintf(songs, ",%.6f,%.6f,%d\n", min_margin, z, at_or_above);
        (*n_cycles)++;
        if (min_margin > 0 && at_or_above == 0) {
            (*n_conforming)++;
            printf("%s  %s  %4.1f min  themes ", rec->stem, start_text, len * BIN_SEC / 60.0);
            for (int v = 0; v < N_THEMES; v++) printf("%s%g", v ? "/" : "", (cycle[v].end - cycle[v].first) / 2.0);
            printf("  smallest margin %.1f  order score %.1f\n", min_margin, z);
        }
    }
    free(visits);
    free(emission);
    free(counts);
    free(column);
    free(rows);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL OUT_DIR WAV [WAV ...]\n", argv[0]);
        return 2;
    }
    double began = now();
    Model model = load_model(argv[1]);
    g_model = &model;
    const char *out_dir = argv[2];
    g_n_recordings = argc - 3;
    g_recordings = xmalloc((size_t)g_n_recordings * sizeof(Recording));
    for (int i = 0; i < g_n_recordings; i++) g_recordings[i] = open_recording(argv[3 + i]);

    init_front_end();
    build_template_host(&model);
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != 0 || n_gpus < 1) die("no GPU found", NULL);
    if (n_gpus > MAX_GPUS) n_gpus = MAX_GPUS;
    int cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int n_cpu = cpus - n_gpus - 1 > 1 ? cpus - n_gpus - 1 : 1; /* the reader and the GPU threads keep a CPU each */
    if (n_cpu > MAX_CPU_THREADS) n_cpu = MAX_CPU_THREADS;

    /* Chunk buffers. What goes to and from the GPUs is pinned, which makes those copies fast. */
    gpu_check(cudaSetDevice(0), "cudaSetDevice");
    size_t window_cells = (size_t)model.n_rows * N_TIME, window_dots = (size_t)model.n_templates * N_TIME;
    for (int i = 0; i < POOL_CHUNKS; i++) {
        Chunk *c = xcalloc(1, sizeof(Chunk));
        c->samples = xmalloc((size_t)CHUNK_WINDOWS * WINDOW_SAMPLES * sizeof(int16_t));
        gpu_check(cudaHostAlloc((void **)&c->spec, CHUNK_WINDOWS * window_cells * sizeof(gpu_real), 1), "cudaHostAlloc");
        gpu_check(cudaHostAlloc((void **)&c->dots, CHUNK_WINDOWS * window_dots * sizeof(gpu_real), 1), "cudaHostAlloc");
        c->cands = xmalloc((size_t)CHUNK_WINDOWS * MAX_CAND * sizeof(Cand));
        c->cumulative = xmalloc((size_t)CHUNK_WINDOWS * (N_TIME + 1) * sizeof(double));
        fifo_push(&g_free, c);
    }

    pthread_t reader, cpu[MAX_CPU_THREADS], gpu[MAX_GPUS];
    double started = now();
    for (int d = 0; d < n_gpus; d++)
        if (pthread_create(&gpu[d], NULL, gpu_main, (void *)(intptr_t)d) != 0) die("cannot start a GPU thread", NULL);
    for (int i = 0; i < n_cpu; i++)
        if (pthread_create(&cpu[i], NULL, cpu_main, NULL) != 0) die("cannot start a CPU thread", NULL);
    if (pthread_create(&reader, NULL, reader_main, NULL) != 0) die("cannot start the reader", NULL);
    pthread_join(reader, NULL);
    for (int i = 0; i < n_cpu; i++) pthread_join(cpu[i], NULL);
    for (int d = 0; d < n_gpus; d++) pthread_join(gpu[d], NULL);
    double placed = now();

    char path[1024];
    snprintf(path, sizeof path, "%s/decoded_songs.csv", out_dir);
    FILE *songs = fopen(path, "w");
    if (!songs) die("cannot write", path);
    fprintf(songs, "file,start_utc,end_utc,start_sec,end_sec,minutes,placements,theme_minutes,theme_starts_sec,"
                   "margins,min_margin,order_z,shuffles_at_or_above\n");
    int n_cycles = 0, n_conforming = 0;
    Pcg64 rng = pcg64_seed0();
    for (int i = 0; i < g_n_recordings; i++)
        finish_recording(&model, &g_recordings[i], &rng, out_dir, songs, &n_cycles, &n_conforming);
    fclose(songs);
    double done = now();

    fprintf(stderr, "cycles decoded %d, conforming %d\n", n_cycles, n_conforming);
    fprintf(stderr, "%d recordings, %d GPUs (%s precision), %d CPU threads\n", g_n_recordings, n_gpus, GPU_PRECISION, n_cpu);
    fprintf(stderr, "wall seconds: setup %.2f, placement %.2f, themes and songs %.2f, total %.2f\n",
            started - began, placed - started, done - placed, done - began);
    fprintf(stderr, "seconds summed over threads: read %.2f, spectrogram %.2f, GPU match %.2f, pursuit %.2f\n",
            g_t_read, g_t_front, g_t_gpu, g_t_pursuit);
    return 0;
}
