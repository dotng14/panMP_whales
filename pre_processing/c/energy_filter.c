/* Band energy ratio of Site H recordings from the GPL whitened spectrogram.

   Each recording is cut into 75-second GPL windows. In every window the code
   takes a Hamming STFT (fftl 1024, hop 768 at 10 kHz), crops rows from 10 Hz
   to Nyquist, whitens each row with the densest-half noise level, and smooths
   with the GPL cross kernel. For each 600-second segment, the reported ratio
   is the sum of squares of the whitened cells in 50–500 Hz divided by the sum
   of squares over all kept rows.

   CPU threads share the work of the STFT and the whitening. Each GPU, when
   used, transforms one window's frames with a batched cuFFT plan.
*/
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    LO_HZ = 50,
    HI_HZ = 500,
    FREQ_LO_HZ = 10,
    FREQ_RES_HZ = 10,
    WINDOW_SEC = 75,
    SKIP_PERCENT = 75,
    MAX_GPUS = 2,
};

typedef float fftwf_complex[2];
typedef struct fftwf_plan_s *fftwf_plan;
typedef int cufftHandle;

fftwf_plan fftwf_plan_dft_r2c_1d(int, float *, fftwf_complex *, unsigned);
void fftwf_execute_dft_r2c(const fftwf_plan, float *, fftwf_complex *);
void fftwf_destroy_plan(fftwf_plan);
void *fftwf_malloc(size_t);
void fftwf_free(void *);

int cudaGetDeviceCount(int *);
int cudaSetDevice(int);
int cudaMalloc(void **, size_t);
int cudaFree(void *);
int cudaMemcpy(void *, const void *, size_t, int);
int cudaDeviceSynchronize(void);
const char *cudaGetErrorString(int);
int cufftPlanMany(cufftHandle *, int, int *, int *, int, int, int *, int, int, int, int);
int cufftExecR2C(cufftHandle, float *, fftwf_complex *);
int cufftDestroy(cufftHandle);

enum { COPY_H2D = 1, COPY_D2H = 2, CUFFT_R2C = 0x2a, FFTW_MEASURE = 0 };

static int g_count = 12;
static int g_winter = 0;
static int g_segment_sec = 600;
static uint64_t g_seed = 0;
static const char *g_root = "/mnt/sda/SiteH";
static const char *g_out_path = NULL;
static int g_backend = 2; /* 0 cpu, 1 gpu, 2 gpu if it initializes */
static char **g_explicit = NULL;
static int g_n_explicit = 0;

typedef struct {
    char *path;
    int year;
} Item;

static Item *g_found = NULL;
static int g_nfound = 0, g_capfound = 0;
static uint64_t g_rng;

typedef struct {
    int rate, fftl, hop, window_samples, max_frames, nfreq;
    int row0, nrows, band0, band1;
    float *hamming;
} Layout;

static void die(const char *msg) {
    fprintf(stderr, "energy_filter: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size);
    if (!p) die("out of memory");
    return p;
}

static uint64_t rng_next(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1ULL;
    return g_rng;
}

static int parse_stamp(const char *name, int *year, int *month) {
    const char *best = NULL;
    for (const char *p = name; *p; p++) {
        if (p > name && isdigit((unsigned char)p[-1])) continue;
        int ok = 1;
        for (int i = 0; i < 6; i++) if (!isdigit((unsigned char)p[i])) ok = 0;
        if (!ok || p[6] != '_') continue;
        for (int i = 7; i < 13; i++) if (!isdigit((unsigned char)p[i])) ok = 0;
        if (!ok || isdigit((unsigned char)p[13])) continue;
        best = p;
    }
    if (!best) return 0;
    int date = 0;
    for (int i = 0; i < 6; i++) date = date * 10 + (best[i] - '0');
    *year = 2000 + date / 10000;
    *month = (date / 100) % 100;
    return *month >= 1 && *month <= 12;
}

static int ends_with(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && memcmp(s + n - m, suf, m) == 0;
}

static int cmp_item(const void *va, const void *vb) {
    const Item *a = va, *b = vb;
    if (a->year != b->year) return a->year < b->year ? -1 : 1;
    return strcmp(a->path, b->path);
}

static int visit(const char *path, const struct stat *st, int type, struct FTW *ftwbuf) {
    (void)st;
    (void)ftwbuf;
    if (type != FTW_F) return 0;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!strstr(base, "df20") || !ends_with(base, ".wav")) return 0;
    int year, month;
    if (!parse_stamp(base, &year, &month)) return 0;
    int keep = g_winter ? (month == 12 || month <= 2) : (month >= 3 && month <= 5);
    if (!keep) return 0;
    if (g_nfound == g_capfound) {
        g_capfound = g_capfound ? g_capfound * 2 : 256;
        g_found = realloc(g_found, (size_t)g_capfound * sizeof(Item));
        if (!g_found) die("out of memory");
    }
    g_found[g_nfound].path = strdup(path);
    g_found[g_nfound].year = year;
    if (!g_found[g_nfound].path) die("out of memory");
    g_nfound++;
    return 0;
}

static void shuffle_pairs(int *a, int *b, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next() % (uint64_t)(i + 1));
        int ta = a[i], tb = b[i];
        a[i] = a[j];
        b[i] = b[j];
        a[j] = ta;
        b[j] = tb;
    }
}

static Item *select_files(int *n_out) {
    if (nftw(g_root, visit, 32, FTW_PHYS) != 0) die("cannot walk the recording root");
    if (g_nfound == 0) die(g_winter ? "no winter df20 recordings" : "no spring df20 recordings");
    qsort(g_found, (size_t)g_nfound, sizeof(Item), cmp_item);
    int nyears = 0;
    for (int i = 0; i < g_nfound; i++)
        if (i == 0 || g_found[i].year != g_found[i - 1].year) nyears++;
    int *start = xmalloc((size_t)nyears * sizeof(int));
    int *len = xmalloc((size_t)nyears * sizeof(int));
    int y = -1;
    for (int i = 0; i < g_nfound; i++) {
        if (i == 0 || g_found[i].year != g_found[i - 1].year) {
            y++;
            start[y] = i;
            len[y] = 0;
        }
        len[y]++;
    }
    g_rng = g_seed;
    shuffle_pairs(start, len, nyears);
    int take = g_count < nyears ? g_count : nyears;
    Item *chosen = xmalloc((size_t)take * sizeof(Item));
    for (int i = 0; i < take; i++) {
        int j = start[i] + (int)(rng_next() % (uint64_t)len[i]);
        chosen[i].path = g_found[j].path;
        chosen[i].year = g_found[j].year;
    }
    qsort(chosen, (size_t)take, sizeof(Item), cmp_item);
    free(start);
    free(len);
    fprintf(stderr, "%s df20 files %d, years %d, selected %d, seed %llu\n",
            g_winter ? "winter" : "spring", g_nfound, nyears, take, (unsigned long long)g_seed);
    *n_out = take;
    return chosen;
}

/* ---- frequency layout (GPL_parameter defaults for the file's sample rate) ---- */
static int hz_to_bin(const Layout *L, double hz) {
    return (int)llround(hz * (double)L->fftl / (double)L->rate);
}

static int ceil_log2_int(int n) {
    int e = 0, v = 1;
    while (v < n) {
        v <<= 1;
        e++;
    }
    return e;
}

static Layout layout_for(int rate) {
    Layout L;
    memset(&L, 0, sizeof L);
    L.rate = rate;
    L.fftl = 1 << ceil_log2_int((rate + FREQ_RES_HZ - 1) / FREQ_RES_HZ);
    if (L.fftl < 16) L.fftl = 16;
    L.hop = (int)llround(L.fftl - (1.0 - SKIP_PERCENT * 1e-2) * L.fftl);
    if (L.hop < 1) L.hop = 1;
    L.window_samples = WINDOW_SEC * rate;
    L.max_frames = 1 + (L.window_samples - L.fftl) / L.hop;
    L.nfreq = L.fftl / 2 + 1;
    int nyquist_bin = L.fftl / 2;
    L.row0 = hz_to_bin(&L, (double)FREQ_LO_HZ);
    if (L.row0 < 1) L.row0 = 1;
    if (L.row0 > nyquist_bin) L.row0 = nyquist_bin;
    L.nrows = nyquist_bin - L.row0 + 1;
    int blo = hz_to_bin(&L, (double)LO_HZ);
    int bhi = hz_to_bin(&L, (double)HI_HZ);
    if (blo < L.row0) blo = L.row0;
    if (bhi > nyquist_bin) bhi = nyquist_bin;
    L.band0 = blo - L.row0;
    L.band1 = bhi - L.row0;
    L.hamming = xmalloc((size_t)L.fftl * sizeof(float));
    for (int i = 0; i < L.fftl; i++)
        L.hamming[i] = (float)(0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(L.fftl - 1)));
    return L;
}

static void layout_free(Layout *L) {
    free(L->hamming);
    L->hamming = NULL;
}

/* ---- wav ---- */
typedef struct {
    int sample_rate;
    int64_t data_off;
    int64_t n_samples;
} Wav;

static Wav open_wav(const char *path, int fd) {
    unsigned char hdr[65536];
    ssize_t n = pread(fd, hdr, sizeof hdr, 0);
    if (n < 44) die("wav header is short");
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) die("not a wav file");
    int rate = 0;
    int64_t data_off = 0, data_bytes = 0;
    int i = 12;
    while (i + 8 <= n) {
        unsigned char *c = hdr + i;
        uint32_t sz;
        memcpy(&sz, c + 4, 4);
        if (memcmp(c, "fmt ", 4) == 0 && i + 24 <= n) {
            uint16_t audio, ch, bps;
            uint32_t r;
            memcpy(&audio, c + 8, 2);
            memcpy(&ch, c + 10, 2);
            memcpy(&r, c + 12, 4);
            memcpy(&bps, c + 22, 2);
            if (audio != 1 || ch != 1 || bps != 16) die("expected mono 16-bit PCM");
            rate = (int)r;
        } else if (memcmp(c, "data", 4) == 0) {
            data_off = i + 8;
            data_bytes = sz;
            break;
        }
        int64_t step = 8 + (int64_t)sz + (sz & 1);
        if (step <= 0) die("bad wav chunk");
        i += (int)step;
    }
    if (!rate || !data_off) die("wav is missing fmt or data");
    if (rate < 2 * HI_HZ) {
        fprintf(stderr, "energy_filter: %s is %d Hz and cannot represent %d Hz\n", path, rate, HI_HZ);
        exit(1);
    }
    Wav w = {rate, data_off, data_bytes / 2};
    return w;
}

static void read_samples(int fd, const Wav *w, int64_t first, int n, float *out) {
    memset(out, 0, (size_t)n * sizeof(float));
    if (first >= w->n_samples || n <= 0) return;
    int64_t lo = first < 0 ? 0 : first;
    int64_t hi = first + n;
    if (hi > w->n_samples) hi = w->n_samples;
    if (hi <= lo) return;
    size_t bytes = (size_t)(hi - lo) * sizeof(int16_t);
    int16_t *raw = xmalloc(bytes);
    size_t got = 0;
    while (got < bytes) {
        ssize_t r = pread(fd, (char *)raw + got, bytes - got, w->data_off + lo * 2 + (int64_t)got);
        if (r < 0) die("read failed");
        if (r == 0) break;
        got += (size_t)r;
    }
    int64_t nread = (int64_t)(got / sizeof(int16_t));
    for (int64_t s = 0; s < nread; s++) {
        int i = (int)(lo + s - first);
        if (i >= 0 && i < n) out[i] = (float)raw[s] / 32768.f;
    }
    free(raw);
}

/* ---- whitening ---- */
static double shortest_half_mean(const float *row, int n, uint64_t *keys, uint64_t *spare) {
    unsigned hist[4][256];
    memset(hist, 0, sizeof hist);
    for (int i = 0; i < n; i++) {
        double v = row[i];
        uint64_t k;
        memcpy(&k, &v, 8);
        keys[i] = k;
        for (int p = 0; p < 4; p++) hist[p][(k >> (32 + 8 * p)) & 255]++;
    }
    for (int p = 0; p < 4; p++) {
        int shift = 32 + 8 * p;
        if (hist[p][(keys[0] >> shift) & 255] == (unsigned)n) continue;
        unsigned at = 0;
        for (int b = 0; b < 256; b++) {
            unsigned count = hist[p][b];
            hist[p][b] = at;
            at += count;
        }
        for (int i = 0; i < n; i++) spare[hist[p][(keys[i] >> shift) & 255]++] = keys[i];
        uint64_t *swap = keys;
        keys = spare;
        spare = swap;
    }
    for (int i = 1; i < n; i++) {
        uint64_t k = keys[i];
        int j = i;
        for (; j > 0 && keys[j - 1] > k; j--) keys[j] = keys[j - 1];
        keys[j] = k;
    }
    int even = n - (n % 2), half = even / 2, pick = 0;
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
    for (int i = 0; i < n; i++) {
        if (row[i] >= low && row[i] <= high) {
            sum += row[i];
            count++;
        }
    }
    return count > 0 ? sum / (double)count : 0.0;
}

typedef struct {
    Layout *L;
    float *samples;
    float *frame;
    fftwf_complex *spectrum;
    float *magnitude;
    float *white;
    float *smooth;
    uint64_t *keys, *spare;
    float *gpu_in;
    fftwf_complex *gpu_out;
    float *d_in;
    fftwf_complex *d_out;
    fftwf_plan cpu_plan;
    cufftHandle plan;
    int use_gpu;
    int device;
} Scratch;

static pthread_mutex_t g_fftw_mu = PTHREAD_MUTEX_INITIALIZER;

static void gpu_check(int rc, const char *what) {
    if (rc != 0) {
        fprintf(stderr, "energy_filter: %s (%s)\n", what, cudaGetErrorString(rc));
        exit(1);
    }
}

static Scratch scratch_for(Layout *L, int use_gpu, int device) {
    Scratch s;
    memset(&s, 0, sizeof s);
    s.L = L;
    s.use_gpu = use_gpu;
    s.device = device;
    s.samples = xmalloc((size_t)L->window_samples * sizeof(float));
    s.frame = fftwf_malloc((size_t)L->fftl * sizeof(float));
    s.spectrum = fftwf_malloc((size_t)L->nfreq * sizeof(fftwf_complex));
    s.magnitude = xmalloc((size_t)L->nrows * L->max_frames * sizeof(float));
    s.white = xmalloc((size_t)L->nrows * L->max_frames * sizeof(float));
    s.smooth = xmalloc((size_t)L->nrows * L->max_frames * sizeof(float));
    s.keys = xmalloc((size_t)L->max_frames * sizeof(uint64_t));
    s.spare = xmalloc((size_t)L->max_frames * sizeof(uint64_t));
    if (use_gpu) {
        gpu_check(cudaSetDevice(device), "cudaSetDevice");
        s.gpu_in = xmalloc((size_t)L->max_frames * L->fftl * sizeof(float));
        s.gpu_out = xmalloc((size_t)L->max_frames * L->nfreq * sizeof(fftwf_complex));
        gpu_check(cudaMalloc((void **)&s.d_in, (size_t)L->max_frames * L->fftl * sizeof(float)), "cudaMalloc in");
        gpu_check(cudaMalloc((void **)&s.d_out, (size_t)L->max_frames * L->nfreq * sizeof(fftwf_complex)), "cudaMalloc out");
        int n[1] = {L->fftl}, inembed[1] = {L->fftl}, onembed[1] = {L->nfreq};
        if (cufftPlanMany(&s.plan, 1, n, inembed, 1, L->fftl, onembed, 1, L->nfreq, CUFFT_R2C, L->max_frames) != 0)
            die("cufftPlanMany failed");
    } else {
        pthread_mutex_lock(&g_fftw_mu);
        s.cpu_plan = fftwf_plan_dft_r2c_1d(L->fftl, s.frame, s.spectrum, 64 /* FFTW_ESTIMATE */);
        pthread_mutex_unlock(&g_fftw_mu);
        if (!s.cpu_plan) die("fftw plan failed");
    }
    return s;
}

static void scratch_free(Scratch *s) {
    if (!s) return;
    free(s->samples);
    fftwf_free(s->frame);
    fftwf_free(s->spectrum);
    free(s->magnitude);
    free(s->white);
    free(s->smooth);
    free(s->keys);
    free(s->spare);
    if (s->use_gpu) {
        cudaSetDevice(s->device);
        cufftDestroy(s->plan);
        cudaFree(s->d_in);
        cudaFree(s->d_out);
        free(s->gpu_in);
        free(s->gpu_out);
    } else if (s->cpu_plan) {
        fftwf_destroy_plan(s->cpu_plan);
    }
}

static int n_frames_for(const Layout *L, int n_samples) {
    if (n_samples < L->fftl) return 0;
    return (n_samples - L->fftl) / L->hop;
}

static void stft_magnitude(Scratch *s, int n_frames) {
    const Layout *L = s->L;
    if (s->use_gpu) {
        gpu_check(cudaSetDevice(s->device), "cudaSetDevice");
        memset(s->gpu_in, 0, (size_t)L->max_frames * L->fftl * sizeof(float));
        for (int t = 0; t < n_frames; t++) {
            const float *from = s->samples + (size_t)t * L->hop;
            float *to = s->gpu_in + (size_t)t * L->fftl;
            for (int i = 0; i < L->fftl; i++) to[i] = from[i] * L->hamming[i];
        }
        gpu_check(cudaMemcpy(s->d_in, s->gpu_in, (size_t)L->max_frames * L->fftl * sizeof(float), COPY_H2D), "H2D");
        if (cufftExecR2C(s->plan, s->d_in, s->d_out) != 0) die("cufft R2C failed");
        gpu_check(cudaDeviceSynchronize(), "sync");
        gpu_check(cudaMemcpy(s->gpu_out, s->d_out, (size_t)L->max_frames * L->nfreq * sizeof(fftwf_complex), COPY_D2H), "D2H");
        for (int t = 0; t < n_frames; t++) {
            const fftwf_complex *z = s->gpu_out + (size_t)t * L->nfreq;
            for (int r = 0; r < L->nrows; r++) {
                const float *c = z[L->row0 + r];
                s->magnitude[(size_t)r * n_frames + t] = sqrtf(c[0] * c[0] + c[1] * c[1]);
            }
        }
        return;
    }
    for (int t = 0; t < n_frames; t++) {
        const float *from = s->samples + (size_t)t * L->hop;
        for (int i = 0; i < L->fftl; i++) s->frame[i] = from[i] * L->hamming[i];
        fftwf_execute_dft_r2c(s->cpu_plan, s->frame, s->spectrum);
        for (int r = 0; r < L->nrows; r++) {
            const float *c = s->spectrum[L->row0 + r];
            s->magnitude[(size_t)r * n_frames + t] = sqrtf(c[0] * c[0] + c[1] * c[1]);
        }
    }
}

static void whiten_window(Scratch *s, int n_frames) {
    const Layout *L = s->L;
    for (int r = 0; r < L->nrows; r++) {
        float *row = s->magnitude + (size_t)r * n_frames;
        float *dst = s->white + (size_t)r * n_frames;
        double mu = shortest_half_mean(row, n_frames, s->keys, s->spare);
        if (!(mu > 0.0)) mu = 1e-20;
        for (int t = 0; t < n_frames; t++) dst[t] = (float)fabs((row[t] - mu) / mu);
    }
    /* GPL cross kernel [[0,1,0],[1,4,1],[0,1,0]] / 8, missing neighbors count as zero. */
    for (int r = 0; r < L->nrows; r++) {
        const float *mid = s->white + (size_t)r * n_frames;
        const float *below = r > 0 ? mid - n_frames : NULL;
        const float *above = r + 1 < L->nrows ? mid + n_frames : NULL;
        float *out = s->smooth + (size_t)r * n_frames;
        for (int t = 0; t < n_frames; t++) {
            double v = 4.0 * mid[t];
            if (below) v += below[t];
            if (above) v += above[t];
            if (t > 0) v += mid[t - 1];
            if (t + 1 < n_frames) v += mid[t + 1];
            out[t] = (float)(v / 8.0);
        }
    }
}

typedef struct {
    double *band;
    double *overall;
    int nseg;
} Segments;

static Segments segments_new(int64_t n_samples, int rate) {
    Segments s;
    int64_t seg_len = (int64_t)g_segment_sec * rate;
    s.nseg = (int)((n_samples + seg_len - 1) / seg_len);
    s.band = xcalloc((size_t)s.nseg, sizeof(double));
    s.overall = xcalloc((size_t)s.nseg, sizeof(double));
    return s;
}

static void accumulate_window(Scratch *s, int n_frames, int64_t window_start, Segments *seg) {
    const Layout *L = s->L;
    int64_t seg_len = (int64_t)g_segment_sec * L->rate;
    for (int t = 0; t < n_frames; t++) {
        int64_t sample = window_start + (int64_t)t * L->hop + L->fftl / 2;
        int b = (int)(sample / seg_len);
        if (b < 0) b = 0;
        if (b >= seg->nseg) b = seg->nseg - 1;
        double band = 0, overall = 0;
        for (int r = 0; r < L->nrows; r++) {
            double v = s->smooth[(size_t)r * n_frames + t];
            double e = v * v;
            overall += e;
            if (r >= L->band0 && r <= L->band1) band += e;
        }
        seg->band[b] += band;
        seg->overall[b] += overall;
    }
}

static void process_window(Scratch *s, const float *samples, int n_samples, int64_t window_start, Segments *seg) {
    const Layout *L = s->L;
    int n_frames = n_frames_for(L, n_samples);
    if (n_frames <= 0) return;
    memcpy(s->samples, samples, (size_t)n_samples * sizeof(float));
    if (n_samples < L->window_samples)
        memset(s->samples + n_samples, 0, (size_t)(L->window_samples - n_samples) * sizeof(float));
    stft_magnitude(s, n_frames);
    whiten_window(s, n_frames);
    accumulate_window(s, n_frames, window_start, seg);
}

static pthread_mutex_t g_csv_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_csv;

static void site_of(const char *path, char *site, size_t nsite) {
    size_t n = strlen(g_root);
    const char *p = path;
    if (strncmp(p, g_root, n) == 0) {
        p += n;
        if (*p == '/') p++;
    }
    size_t i = 0;
    while (p[i] && p[i] != '/' && i + 1 < nsite) {
        site[i] = p[i];
        i++;
    }
    site[i] = 0;
}

static void write_segments(const Item *item, const Segments *seg) {
    const char *base = strrchr(item->path, '/');
    base = base ? base + 1 : item->path;
    char site[256];
    site_of(item->path, site, sizeof site);
    pthread_mutex_lock(&g_csv_mu);
    for (int i = 0; i < seg->nseg; i++) {
        double ratio = seg->overall[i] > 0.0 ? seg->band[i] / seg->overall[i] : 0.0;
        fprintf(g_csv, "%s,%s,%d,%d,%.3f,%.8e,%.8e,%.8e\n",
                site, base, item->year, i, (double)i * g_segment_sec,
                seg->band[i], seg->overall[i], ratio);
    }
    fflush(g_csv);
    pthread_mutex_unlock(&g_csv_mu);
}

static void process_file(int use_gpu, int device, const Item *item) {
    int fd = open(item->path, O_RDONLY);
    if (fd < 0) die("cannot open recording");
    Wav w = open_wav(item->path, fd);
    Layout L = layout_for(w.sample_rate);
    fprintf(stderr, "%s  %s  %d Hz  %.1f h  fftl %d hop %d\n", use_gpu ? "gpu" : "cpu", item->path,
            w.sample_rate, (double)w.n_samples / w.sample_rate / 3600.0, L.fftl, L.hop);
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    Scratch s = scratch_for(&L, use_gpu, device);
    Segments seg = segments_new(w.n_samples, w.sample_rate);
    float *buf = xmalloc((size_t)L.window_samples * sizeof(float));
    for (int64_t start = 0; start < w.n_samples; start += L.window_samples) {
        int n = (int)(w.n_samples - start);
        if (n > L.window_samples) n = L.window_samples;
        read_samples(fd, &w, start, n, buf);
        process_window(&s, buf, n, start, &seg);
    }
    write_segments(item, &seg);
    free(buf);
    free(seg.band);
    free(seg.overall);
    scratch_free(&s);
    layout_free(&L);
    close(fd);
}

typedef struct {
    Item *chosen;
    int n;
    int next;
    pthread_mutex_t mu;
} Work;

static Work g_work;

static void *cpu_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_work.mu);
        int i = g_work.next < g_work.n ? g_work.next++ : -1;
        pthread_mutex_unlock(&g_work.mu);
        if (i < 0) break;
        process_file(0, -1, &g_work.chosen[i]);
    }
    return NULL;
}

static void run_cpu(Item *chosen, int n) {
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (threads < 1) threads = 1;
    if (threads > n) threads = n;
    if (threads > 32) threads = 32;
    g_work.chosen = chosen;
    g_work.n = n;
    g_work.next = 0;
    if (pthread_mutex_init(&g_work.mu, NULL) != 0) die("mutex");
    fprintf(stderr, "cpu workers %d\n", threads);
    pthread_t *ths = xmalloc((size_t)threads * sizeof(pthread_t));
    for (int i = 0; i < threads; i++) {
        if (pthread_create(&ths[i], NULL, cpu_worker, NULL) != 0) die("thread");
    }
    for (int i = 0; i < threads; i++) pthread_join(ths[i], NULL);
    free(ths);
    pthread_mutex_destroy(&g_work.mu);
}

static void *gpu_worker(void *arg) {
    int device = (int)(intptr_t)arg;
    for (;;) {
        pthread_mutex_lock(&g_work.mu);
        int i = g_work.next < g_work.n ? g_work.next++ : -1;
        pthread_mutex_unlock(&g_work.mu);
        if (i < 0) break;
        process_file(1, device, &g_work.chosen[i]);
    }
    return NULL;
}

static void run_gpu(Item *chosen, int n) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != 0 || ndev < 1) die("no GPU");
    if (ndev > MAX_GPUS) ndev = MAX_GPUS;
    g_work.chosen = chosen;
    g_work.n = n;
    g_work.next = 0;
    if (pthread_mutex_init(&g_work.mu, NULL) != 0) die("mutex");
    fprintf(stderr, "gpu workers %d\n", ndev);
    pthread_t th[MAX_GPUS];
    for (int d = 0; d < ndev; d++) {
        if (pthread_create(&th[d], NULL, gpu_worker, (void *)(intptr_t)d) != 0) die("thread");
    }
    for (int d = 0; d < ndev; d++) pthread_join(th[d], NULL);
    pthread_mutex_destroy(&g_work.mu);
}

/* Whitening treats a steady tone as the noise floor, so the self-test uses short
   bursts on a quiet background: those stand above the densest-half level. */
static void burst_samples(float *buf, int n, int rate, double hz, double amp) {
    memset(buf, 0, (size_t)n * sizeof(float));
    int burst = rate / 10;
    for (int start = rate; start + burst < n; start += 5 * rate) {
        for (int i = 0; i < burst; i++)
            buf[start + i] = (float)(amp * sin(2.0 * M_PI * hz * (double)i / (double)rate));
    }
}

static double burst_ratio(int use_gpu, int device, double hz) {
    Layout L = layout_for(10000);
    Scratch s = scratch_for(&L, use_gpu, device);
    Segments seg = segments_new(L.window_samples, L.rate);
    float *buf = xmalloc((size_t)L.window_samples * sizeof(float));
    burst_samples(buf, L.window_samples, L.rate, hz, 0.5);
    process_window(&s, buf, L.window_samples, 0, &seg);
    double band = 0, overall = 0;
    for (int i = 0; i < seg.nseg; i++) {
        band += seg.band[i];
        overall += seg.overall[i];
    }
    free(buf);
    free(seg.band);
    free(seg.overall);
    scratch_free(&s);
    layout_free(&L);
    return overall > 0.0 ? band / overall : 0.0;
}

static void self_test(void) {
    double in_band = burst_ratio(0, -1, 200.0);
    double above = burst_ratio(0, -1, 2000.0);
    fprintf(stderr, "cpu ratio 200 Hz bursts %.4f  2000 Hz bursts %.4e\n", in_band, above);
    if (!(in_band > 0.5 && above < 0.1)) die("cpu ratio did not separate 200 Hz from 2000 Hz bursts");
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != 0 || ndev < 1) die("self-test found no GPU");
    double gin = burst_ratio(1, 0, 200.0);
    double gout = burst_ratio(1, 0, 2000.0);
    fprintf(stderr, "gpu ratio 200 Hz bursts %.4f  2000 Hz bursts %.4e\n", gin, gout);
    if (!(gin > 0.5 && gout < 0.1)) die("gpu ratio did not separate 200 Hz from 2000 Hz bursts");
    double rel = fabs(gin - in_band) / in_band;
    fprintf(stderr, "cpu/gpu relative difference %.3e\n", rel);
    if (rel > 1e-2) die("cpu and gpu ratios disagree");
    fprintf(stderr, "self-test passed\n");
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--root DIR] [--count N] [--seed N] [--season spring|winter] [--segment-sec N]\n"
            "          [--out CSV] [--cpu|--gpu] [--self-test] [WAV ...]\n"
            "If WAV paths are given, those files are processed and --season/--count/--seed are ignored.\n",
            argv0);
}

static int year_of_path(const char *path) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    int year = 0, month = 0;
    if (parse_stamp(base, &year, &month)) return year;
    return 0;
}

int main(int argc, char **argv) {
    int self_test_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--root") && i + 1 < argc) g_root = argv[++i];
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) g_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) g_seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--segment-sec") && i + 1 < argc) g_segment_sec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--season") && i + 1 < argc) {
            const char *season = argv[++i];
            if (!strcmp(season, "winter")) g_winter = 1;
            else if (!strcmp(season, "spring")) g_winter = 0;
            else die("season must be spring or winter");
        }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) g_out_path = argv[++i];
        else if (!strcmp(argv[i], "--cpu")) g_backend = 0;
        else if (!strcmp(argv[i], "--gpu")) g_backend = 1;
        else if (!strcmp(argv[i], "--self-test")) self_test_only = 1;
        else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else {
            g_explicit = &argv[i];
            g_n_explicit = argc - i;
            break;
        }
    }
    if (self_test_only) {
        self_test();
        return 0;
    }
    if (!g_out_path) die("missing --out");
    if (g_segment_sec < 1) die("segment-sec must be positive");
    int n = 0;
    Item *chosen;
    if (g_n_explicit > 0) {
        n = g_n_explicit;
        chosen = xmalloc((size_t)n * sizeof(Item));
        for (int i = 0; i < n; i++) {
            chosen[i].path = g_explicit[i];
            chosen[i].year = year_of_path(g_explicit[i]);
        }
        fprintf(stderr, "explicit files %d, segment %d s\n", n, g_segment_sec);
    } else {
        if (g_count < 1) die("count must be positive");
        chosen = select_files(&n);
    }
    g_csv = fopen(g_out_path, "w");
    if (!g_csv) die("cannot write the csv");
    fprintf(g_csv, "site,filename,year,segment_index,start_sec,band_energy,overall_energy,ratio\n");
    if (g_backend == 0) run_cpu(chosen, n);
    else {
        int ndev = 0;
        if (g_backend == 2 && (cudaGetDeviceCount(&ndev) != 0 || ndev < 1)) run_cpu(chosen, n);
        else run_gpu(chosen, n);
    }
    fclose(g_csv);
    fprintf(stderr, "wrote %s\n", g_out_path);
    return 0;
}
