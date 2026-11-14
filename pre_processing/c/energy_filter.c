/* Band energy ratio of Site H recordings.

   For each 600-second segment, the ratio is the energy in 50–500 Hz divided
   by the energy of the unfiltered segment. The CPU path uses threaded FFTW.
   The GPU path uses cuFFT, one recording per GPU.
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

enum { LO_HZ = 50, HI_HZ = 500, SEGMENT_SEC = 600, EDGE_SEC = 2 };
enum { NFFT = 1 << 20 };
enum { FFTW_ESTIMATE = 1u << 6 };
enum { COPY_H2D = 1, COPY_D2H = 2 };
enum { CUFFT_R2C = 0x2a, CUFFT_C2R = 0x2c };

typedef float fftwf_complex[2];
typedef struct fftwf_plan_s *fftwf_plan;
typedef int cufftHandle;

int fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int);
fftwf_plan fftwf_plan_dft_r2c_1d(int, float *, fftwf_complex *, unsigned);
fftwf_plan fftwf_plan_dft_c2r_1d(int, fftwf_complex *, float *, unsigned);
void fftwf_execute(fftwf_plan);
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
int cufftPlan1d(cufftHandle *, int, int, int);
int cufftExecR2C(cufftHandle, float *, fftwf_complex *);
int cufftExecC2R(cufftHandle, fftwf_complex *, float *);
int cufftDestroy(cufftHandle);

static int g_count = 12;
static int g_winter = 0;
static uint64_t g_seed = 0;
static const char *g_root = "/mnt/sda/SiteH";
static const char *g_out_path = NULL;
static int g_backend = 2; /* 0 cpu, 1 gpu, 2 gpu if it initializes */

typedef struct {
    char *path;
    int year;
} Item;

static Item *g_found = NULL;
static int g_nfound = 0, g_capfound = 0;
static uint64_t g_rng;

static void die(const char *msg) {
    fprintf(stderr, "energy_filter: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
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

typedef struct {
    int gpu;
    int device;
    float *time;
    fftwf_complex *freq;
    fftwf_plan r2c, c2r;
    float *d_time;
    fftwf_complex *d_freq;
    cufftHandle gr2c, gc2r;
} Engine;

static void bins_for(int sample_rate, int *kcut, int *khi) {
    *kcut = (LO_HZ * NFFT + sample_rate - 1) / sample_rate;
    if (sample_rate / 2 <= HI_HZ) *khi = NFFT / 2;
    else *khi = (int)((int64_t)HI_HZ * NFFT / sample_rate);
}

static void zero_bins(fftwf_complex *freq, int kcut, int khi) {
    int nfreq = NFFT / 2 + 1;
    for (int k = 0; k < nfreq; k++) {
        if (k < kcut || k > khi) {
            freq[k][0] = 0.f;
            freq[k][1] = 0.f;
        }
    }
}

static void scale_time(float *time) {
    float inv = 1.f / (float)NFFT;
    for (int i = 0; i < NFFT; i++) time[i] *= inv;
}

static Engine *engine_cpu(int threads) {
    Engine *e = calloc(1, sizeof(Engine));
    if (!e) die("out of memory");
    if (!fftwf_init_threads()) die("fftw threads failed to start");
    fftwf_plan_with_nthreads(threads);
    e->time = fftwf_malloc((size_t)NFFT * sizeof(float));
    e->freq = fftwf_malloc((size_t)(NFFT / 2 + 1) * sizeof(fftwf_complex));
    if (!e->time || !e->freq) die("fftw allocation failed");
    e->r2c = fftwf_plan_dft_r2c_1d(NFFT, e->time, e->freq, FFTW_ESTIMATE);
    e->c2r = fftwf_plan_dft_c2r_1d(NFFT, e->freq, e->time, FFTW_ESTIMATE);
    if (!e->r2c || !e->c2r) die("fftw plan failed");
    return e;
}

static void gpu_check(int rc, const char *what) {
    if (rc == 0) return;
    fprintf(stderr, "energy_filter: %s (%s)\n", what, cudaGetErrorString(rc));
    exit(1);
}

static Engine *engine_gpu(int device) {
    Engine *e = calloc(1, sizeof(Engine));
    if (!e) die("out of memory");
    e->gpu = 1;
    e->device = device;
    gpu_check(cudaSetDevice(device), "cudaSetDevice");
    e->time = xmalloc((size_t)NFFT * sizeof(float));
    e->freq = xmalloc((size_t)(NFFT / 2 + 1) * sizeof(fftwf_complex));
    gpu_check(cudaMalloc((void **)&e->d_time, (size_t)NFFT * sizeof(float)), "cudaMalloc time");
    gpu_check(cudaMalloc((void **)&e->d_freq, (size_t)(NFFT / 2 + 1) * sizeof(fftwf_complex)), "cudaMalloc freq");
    int rc = cufftPlan1d(&e->gr2c, NFFT, CUFFT_R2C, 1);
    if (rc != 0) {
        fprintf(stderr, "energy_filter: cufftPlan R2C failed (%d)\n", rc);
        exit(1);
    }
    rc = cufftPlan1d(&e->gc2r, NFFT, CUFFT_C2R, 1);
    if (rc != 0) {
        fprintf(stderr, "energy_filter: cufftPlan C2R failed (%d)\n", rc);
        exit(1);
    }
    return e;
}

static void engine_filter(Engine *e, int kcut, int khi) {
    if (!e->gpu) {
        fftwf_execute(e->r2c);
        zero_bins(e->freq, kcut, khi);
        fftwf_execute(e->c2r);
        scale_time(e->time);
        return;
    }
    gpu_check(cudaSetDevice(e->device), "cudaSetDevice");
    gpu_check(cudaMemcpy(e->d_time, e->time, (size_t)NFFT * sizeof(float), COPY_H2D), "copy to device");
    if (cufftExecR2C(e->gr2c, e->d_time, e->d_freq) != 0) die("cufft R2C failed");
    gpu_check(cudaMemcpy(e->freq, e->d_freq, (size_t)(NFFT / 2 + 1) * sizeof(fftwf_complex), COPY_D2H), "copy spectrum");
    zero_bins(e->freq, kcut, khi);
    gpu_check(cudaMemcpy(e->d_freq, e->freq, (size_t)(NFFT / 2 + 1) * sizeof(fftwf_complex), COPY_H2D), "copy spectrum back");
    if (cufftExecC2R(e->gc2r, e->d_freq, e->d_time) != 0) die("cufft C2R failed");
    gpu_check(cudaDeviceSynchronize(), "synchronize");
    gpu_check(cudaMemcpy(e->time, e->d_time, (size_t)NFFT * sizeof(float), COPY_D2H), "copy from device");
    scale_time(e->time);
}

static void engine_free(Engine *e) {
    if (!e) return;
    if (e->gpu) {
        cudaSetDevice(e->device);
        cufftDestroy(e->gr2c);
        cufftDestroy(e->gc2r);
        cudaFree(e->d_time);
        cudaFree(e->d_freq);
        free(e->time);
        free(e->freq);
    } else {
        fftwf_destroy_plan(e->r2c);
        fftwf_destroy_plan(e->c2r);
        fftwf_free(e->time);
        fftwf_free(e->freq);
    }
    free(e);
}

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
    int rate = 0, channels = 0, bits = 0;
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
            channels = ch;
            rate = (int)r;
            bits = bps;
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
    (void)channels;
    (void)bits;
    if (rate < 2 * HI_HZ) {
        fprintf(stderr, "energy_filter: %s is %d Hz and cannot represent %d Hz\n", path, rate, HI_HZ);
        exit(1);
    }
    Wav w = {rate, data_off, data_bytes / 2};
    return w;
}

static void load_block(int fd, const Wav *w, int64_t out0, int discard, float *time) {
    int64_t begin = out0 - discard;
    int64_t lo = begin < 0 ? 0 : begin;
    int64_t hi = begin + NFFT;
    if (hi > w->n_samples) hi = w->n_samples;
    memset(time, 0, (size_t)NFFT * sizeof(float));
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
    for (int64_t s = lo; s < lo + nread; s++) {
        int i = (int)(s - begin);
        if (i >= 0 && i < NFFT) time[i] = (float)raw[s - lo] / 32768.f;
    }
    free(raw);
}

typedef struct {
    double *band;
    double *overall;
    int nseg;
} Segments;

static Segments segments_new(int64_t n_samples, int seg_len) {
    Segments s;
    s.nseg = (int)((n_samples + seg_len - 1) / seg_len);
    s.band = calloc((size_t)s.nseg, sizeof(double));
    s.overall = calloc((size_t)s.nseg, sizeof(double));
    if (!s.band || !s.overall) die("out of memory");
    return s;
}

static void add_squares(const float *time, int64_t out0, int discard, int64_t n_samples, int seg_len, double *dest) {
    for (int i = discard; i < NFFT - discard; i++) {
        int64_t s = out0 + (i - discard);
        if (s >= n_samples) break;
        int b = (int)(s / seg_len);
        double x = time[i];
        dest[b] += x * x;
    }
}

static void accumulate(Engine *e, int fd, const Wav *w, Segments *seg) {
    int discard = EDGE_SEC * w->sample_rate;
    if (discard > NFFT / 4) discard = NFFT / 4;
    int hop = NFFT - 2 * discard;
    int seg_len = SEGMENT_SEC * w->sample_rate;
    int kcut, khi;
    bins_for(w->sample_rate, &kcut, &khi);
    for (int64_t out0 = 0; out0 < w->n_samples; out0 += hop) {
        load_block(fd, w, out0, discard, e->time);
        add_squares(e->time, out0, discard, w->n_samples, seg_len, seg->overall);
        engine_filter(e, kcut, khi);
        add_squares(e->time, out0, discard, w->n_samples, seg_len, seg->band);
    }
}

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

static pthread_mutex_t g_csv_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_csv;

static void write_segments(const Item *item, const Segments *seg) {
    const char *base = strrchr(item->path, '/');
    base = base ? base + 1 : item->path;
    char site[256];
    site_of(item->path, site, sizeof site);
    pthread_mutex_lock(&g_csv_mu);
    for (int i = 0; i < seg->nseg; i++) {
        double ratio = seg->overall[i] > 0.0 ? seg->band[i] / seg->overall[i] : 0.0;
        fprintf(g_csv, "%s,%s,%d,%d,%.3f,%.8e,%.8e,%.8e\n",
                site, base, item->year, i, (double)i * SEGMENT_SEC,
                seg->band[i], seg->overall[i], ratio);
    }
    fflush(g_csv);
    pthread_mutex_unlock(&g_csv_mu);
}

static void process_file(Engine *e, const Item *item) {
    int fd = open(item->path, O_RDONLY);
    if (fd < 0) die("cannot open recording");
    Wav w = open_wav(item->path, fd);
    fprintf(stderr, "device %d  %s  %.1f h\n", e->gpu ? e->device : -1, item->path,
            (double)w.n_samples / w.sample_rate / 3600.0);
    Segments seg = segments_new(w.n_samples, SEGMENT_SEC * w.sample_rate);
    accumulate(e, fd, &w, &seg);
    write_segments(item, &seg);
    free(seg.band);
    free(seg.overall);
    close(fd);
}

typedef struct {
    Item *chosen;
    int n;
    int next;
    pthread_mutex_t mu;
} Work;

static Work g_work;

static void run_cpu(Item *chosen, int n) {
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (threads < 1) threads = 1;
    Engine *e = engine_cpu(threads);
    fprintf(stderr, "cpu threads %d\n", threads);
    for (int i = 0; i < n; i++) process_file(e, &chosen[i]);
    engine_free(e);
}

static void *gpu_worker(void *arg) {
    int device = (int)(intptr_t)arg;
    Engine *e = engine_gpu(device);
    for (;;) {
        pthread_mutex_lock(&g_work.mu);
        int i = g_work.next < g_work.n ? g_work.next++ : -1;
        pthread_mutex_unlock(&g_work.mu);
        if (i < 0) break;
        process_file(e, &g_work.chosen[i]);
    }
    engine_free(e);
    return NULL;
}

static void run_gpu(Item *chosen, int n) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != 0 || ndev < 1) die("no GPU");
    if (ndev > 2) ndev = 2;
    g_work.chosen = chosen;
    g_work.n = n;
    g_work.next = 0;
    if (pthread_mutex_init(&g_work.mu, NULL) != 0) die("mutex");
    pthread_t th[2];
    for (int d = 0; d < ndev; d++) {
        if (pthread_create(&th[d], NULL, gpu_worker, (void *)(intptr_t)d) != 0) die("thread");
    }
    for (int d = 0; d < ndev; d++) pthread_join(th[d], NULL);
    pthread_mutex_destroy(&g_work.mu);
}

static double tone_ratio(Engine *e, int sample_rate, double hz, double amp, double seconds) {
    int64_t n = (int64_t)(seconds * sample_rate);
    int discard = EDGE_SEC * sample_rate;
    int hop = NFFT - 2 * discard;
    int kcut, khi;
    bins_for(sample_rate, &kcut, &khi);
    double band = 0.0, overall = 0.0;
    for (int64_t out0 = 0; out0 < n; out0 += hop) {
        memset(e->time, 0, (size_t)NFFT * sizeof(float));
        for (int i = 0; i < NFFT; i++) {
            int64_t s = out0 - discard + i;
            if (s < 0 || s >= n) continue;
            e->time[i] = (float)(amp * sin(2.0 * 3.141592653589793 * hz * (double)s / sample_rate));
        }
        for (int i = discard; i < NFFT - discard; i++) {
            int64_t s = out0 + (i - discard);
            if (s >= n) break;
            overall += (double)e->time[i] * (double)e->time[i];
        }
        engine_filter(e, kcut, khi);
        for (int i = discard; i < NFFT - discard; i++) {
            int64_t s = out0 + (i - discard);
            if (s >= n) break;
            band += (double)e->time[i] * (double)e->time[i];
        }
    }
    return overall > 0.0 ? band / overall : 0.0;
}

static void self_test(void) {
    int rate = 10000;
    Engine *cpu = engine_cpu(4);
    double in_band = tone_ratio(cpu, rate, 200.0, 0.5, 30.0);
    double above = tone_ratio(cpu, rate, 2000.0, 0.5, 30.0);
    fprintf(stderr, "cpu ratio 200 Hz %.4f  2000 Hz %.4e\n", in_band, above);
    if (!(in_band > 0.9 && above < 0.05)) die("cpu ratio did not separate 200 Hz from 2000 Hz");
    engine_free(cpu);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != 0 || ndev < 1) die("self-test found no GPU");
    Engine *gpu = engine_gpu(0);
    double gin = tone_ratio(gpu, rate, 200.0, 0.5, 30.0);
    double gout = tone_ratio(gpu, rate, 2000.0, 0.5, 30.0);
    fprintf(stderr, "gpu ratio 200 Hz %.4f  2000 Hz %.4e\n", gin, gout);
    if (!(gin > 0.9 && gout < 0.05)) die("gpu ratio did not separate 200 Hz from 2000 Hz");
    double rel = fabs(gin - in_band) / in_band;
    fprintf(stderr, "cpu/gpu relative difference %.3e\n", rel);
    if (rel > 1e-3) die("cpu and gpu ratios disagree");
    engine_free(gpu);
    fprintf(stderr, "self-test passed\n");
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--root DIR] [--count N] [--seed N] [--season spring|winter] [--out CSV] [--cpu|--gpu] [--self-test]\n",
            argv0);
}

int main(int argc, char **argv) {
    int self_test_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--root") && i + 1 < argc) g_root = argv[++i];
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) g_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) g_seed = strtoull(argv[++i], NULL, 10);
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
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (self_test_only) {
        self_test();
        return 0;
    }
    if (!g_out_path) die("missing --out");
    if (g_count < 1) die("count must be positive");
    int n = 0;
    Item *chosen = select_files(&n);
    g_csv = fopen(g_out_path, "w");
    if (!g_csv) die("cannot write the csv");
    fprintf(g_csv, "site,filename,year,segment_index,start_sec,band_energy,overall_energy,ratio\n");
    if (g_backend == 0) run_cpu(chosen, n);
    else run_gpu(chosen, n);
    fclose(g_csv);
    fprintf(stderr, "wrote %s\n", g_out_path);
    return 0;
}
