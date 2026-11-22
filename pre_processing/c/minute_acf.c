/* Per-minute ACF peak of the whitened 50–500 Hz energy ratio.

   The whitening matches energy_filter.c / minute_energy.c: 75 s GPL windows,
   Hamming STFT (freq_resolution_hz 10, skip_percent 75), rows from 10 Hz to
   Nyquist, densest-half whitening, then the GPL cross kernel. A 0.2 s bin's
   energy ratio is the sum of squares of whitened cells in 50–500 Hz divided
   by the sum of squares over every kept row. Frames are assigned to a bin by
   the sample at the center of the FFT window.

   Each complete minute (300 bins of 0.2 s; a short trailing minute is kept)
   stores two numbers from the demeaned, lag-0-normalized autocorrelation:

     1. acf_peak — height of the strongest peak with lag in [1, 20] s.
     2. acf_lag_s — that lag in seconds. If no local peak exists in range,
        the strongest lag in [1, 20] s is used.

   STFT frames run on the GPUs through batched cuFFT when a device is
   present, and on the CPUs through FFTW otherwise. Whitening of frequency
   rows runs in an OpenMP team. Windows of one file are split across the
   CPU workers or across both GPUs. Per-minute ACF stays on the CPU.

   One CSV per recording is written under
   /home/yfreund/whale_processing/12TB_results/minute_acf unless --out-dir
   says otherwise.
*/
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
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
    BINS_PER_SEC = 5,          /* 0.2 s bins */
    BINS_PER_MINUTE = 300,     /* 300 * 0.2 s */
    LAG_LO_BIN = 5,            /* 1.0 s */
    LAG_HI_BIN = 100,          /* 20.0 s */
    WHITEN_THREADS = 8
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

enum { COPY_H2D = 1, COPY_D2H = 2, CUFFT_R2C = 0x2a };

static const char *g_out_dir = "/home/yfreund/whale_processing/12TB_results/minute_acf";
static int g_backend = 2; /* 0 cpu, 1 gpu, 2 gpu if it initializes */
static double g_duration_sec = 0; /* 0 means the whole file */
static pthread_mutex_t g_fftw_mu = PTHREAD_MUTEX_INITIALIZER;
static const double BIN_DT = 0.2;

typedef struct {
    int rate, fftl, hop, window_samples, max_frames, nfreq;
    int row0, nrows, band0, band1;
    float *hamming;
} Layout;

typedef struct {
    int sample_rate;
    int64_t data_off;
    int64_t n_samples;
} Wav;

typedef struct {
    double *band;
    double *overall;
    int n_bins;
} Acc;

typedef struct {
    const Layout *L;
    float *samples;
    float *frame;
    fftwf_complex *spectrum;
    float *magnitude;
    float *white;
    float *smooth;
    float *gpu_in;
    fftwf_complex *gpu_out;
    float *d_in;
    fftwf_complex *d_out;
    fftwf_plan cpu_plan;
    cufftHandle plan;
    int use_gpu;
    int device;
} Scratch;

static void die(const char *msg) {
    fprintf(stderr, "minute_acf: %s\n", msg);
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

static int llround_int(double x) {
    return x >= 0 ? (int)floor(x + 0.5) : (int)ceil(x - 0.5);
}

static int ceil_log2_int(int n) {
    int e = 0, v = 1;
    while (v < n) {
        v <<= 1;
        e++;
    }
    return e;
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

static int hz_to_bin(const Layout *L, double hz) {
    return llround_int(hz * (double)L->fftl / (double)L->rate);
}

static Layout layout_for(int rate) {
    Layout L;
    memset(&L, 0, sizeof L);
    L.rate = rate;
    L.fftl = 1 << ceil_log2_int((rate + FREQ_RES_HZ - 1) / FREQ_RES_HZ);
    if (L.fftl < 16) L.fftl = 16;
    L.hop = llround_int(L.fftl - (1.0 - SKIP_PERCENT * 1e-2) * L.fftl);
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

static void gpu_check(int rc, const char *what) {
    if (rc != 0) {
        fprintf(stderr, "minute_acf: %s (%s)\n", what, cudaGetErrorString(rc));
        exit(1);
    }
}

static Scratch scratch_for(const Layout *L, int use_gpu, int device) {
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
        s.cpu_plan = fftwf_plan_dft_r2c_1d(L->fftl, s.frame, s.spectrum, 64);
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

static Wav open_wav(const char *path, int fd) {
    unsigned char hdr[65536];
    size_t got = 0;
    int empty_tries = 0;
    while (got < sizeof hdr) {
        ssize_t n = pread(fd, hdr + got, sizeof hdr - got, (off_t)got);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == EIO) {
                usleep(50000);
                continue;
            }
            die("cannot read wav header");
        }
        if (n == 0) {
            /* Transient EOF / empty read under heavy disk load; retry briefly. */
            if (got < 44 && empty_tries++ < 40) {
                usleep(50000);
                continue;
            }
            break;
        }
        empty_tries = 0;
        got += (size_t)n;
        if (got >= 44 && memcmp(hdr, "RIFF", 4) == 0) {
            /* Stop early once the data chunk is visible in what we have. */
            int i = 12;
            int saw_data = 0;
            while (i + 8 <= (int)got) {
                uint32_t sz;
                memcpy(&sz, hdr + i + 4, 4);
                if (memcmp(hdr + i, "data", 4) == 0) {
                    saw_data = 1;
                    break;
                }
                int64_t step = 8 + (int64_t)sz + (sz & 1);
                if (step <= 0) break;
                if ((size_t)i + (size_t)step > got) break;
                i += (int)step;
            }
            if (saw_data) break;
        }
    }
    if (got < 44) die("wav header is short");
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) die("not a wav file");
    int rate = 0;
    int64_t data_off = 0, data_bytes = 0;
    int i = 12;
    while (i + 8 <= (int)got) {
        unsigned char *c = hdr + i;
        uint32_t sz;
        memcpy(&sz, c + 4, 4);
        if (memcmp(c, "fmt ", 4) == 0 && i + 24 <= (int)got) {
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
        if ((size_t)i + (size_t)step > got) break;
        i += (int)step;
    }
    if (!rate || !data_off) die("wav is missing fmt or data");
    if (rate < 2 * HI_HZ) {
        fprintf(stderr, "minute_acf: %s is %d Hz and cannot represent %d Hz\n", path, rate, HI_HZ);
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
    int empty_tries = 0;
    while (got < bytes) {
        ssize_t r = pread(fd, (char *)raw + got, bytes - got, w->data_off + lo * 2 + (int64_t)got);
        if (r < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == EIO) {
                usleep(50000);
                continue;
            }
            die("read failed");
        }
        if (r == 0) {
            if (empty_tries++ < 40) {
                usleep(50000);
                continue;
            }
            break;
        }
        empty_tries = 0;
        got += (size_t)r;
    }
    int64_t nread = (int64_t)(got / sizeof(int16_t));
    for (int64_t s = 0; s < nread; s++) {
        int i = (int)(lo + s - first);
        if (i >= 0 && i < n) out[i] = (float)raw[s] / 32768.f;
    }
    free(raw);
}

/* Densest-half mean. The radix order matches energy_filter.c, which sorts the
   IEEE bits of the non-negative magnitudes. */
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
    const int nrows = L->nrows;
#pragma omp parallel num_threads(WHITEN_THREADS)
    {
        uint64_t *keys = xmalloc((size_t)n_frames * sizeof(uint64_t));
        uint64_t *spare = xmalloc((size_t)n_frames * sizeof(uint64_t));
#pragma omp for schedule(static)
        for (int r = 0; r < nrows; r++) {
            float *row = s->magnitude + (size_t)r * n_frames;
            float *dst = s->white + (size_t)r * n_frames;
            double mu = shortest_half_mean(row, n_frames, keys, spare);
            if (!(mu > 0.0)) mu = 1e-20;
            for (int t = 0; t < n_frames; t++) dst[t] = (float)fabs((row[t] - mu) / mu);
        }
        free(keys);
        free(spare);
    }
#pragma omp parallel for num_threads(WHITEN_THREADS) schedule(static)
    for (int r = 0; r < nrows; r++) {
        const float *mid = s->white + (size_t)r * n_frames;
        const float *below = r > 0 ? mid - n_frames : NULL;
        const float *above = r + 1 < nrows ? mid + n_frames : NULL;
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

static int64_t bin_samples(int rate) {
    int64_t seg = (int64_t)llround_int((double)rate * BIN_DT);
    return seg < 1 ? 1 : seg;
}

static void accumulate_window(Scratch *s, int n_frames, int64_t window_start, Acc *acc) {
    const Layout *L = s->L;
    int64_t seg_len = bin_samples(L->rate);
    for (int t = 0; t < n_frames; t++) {
        int64_t sample = window_start + (int64_t)t * L->hop + L->fftl / 2;
        int b = (int)(sample / seg_len);
        if (b < 0) b = 0;
        if (b >= acc->n_bins) b = acc->n_bins - 1;
        double band = 0, overall = 0;
        for (int r = 0; r < L->nrows; r++) {
            double v = s->smooth[(size_t)r * n_frames + t];
            double e = v * v;
            overall += e;
            if (r >= L->band0 && r <= L->band1) band += e;
        }
        acc->band[b] += band;
        acc->overall[b] += overall;
    }
}

static void process_window(Scratch *s, const float *samples, int n_samples, int64_t window_start, Acc *acc) {
    const Layout *L = s->L;
    int n_frames = n_frames_for(L, n_samples);
    if (n_frames <= 0) return;
    memcpy(s->samples, samples, (size_t)n_samples * sizeof(float));
    if (n_samples < L->window_samples)
        memset(s->samples + n_samples, 0, (size_t)(L->window_samples - n_samples) * sizeof(float));
    stft_magnitude(s, n_frames);
    whiten_window(s, n_frames);
    accumulate_window(s, n_frames, window_start, acc);
}

static Acc acc_new(int n_bins) {
    Acc a;
    a.n_bins = n_bins;
    a.band = xcalloc((size_t)n_bins, sizeof(double));
    a.overall = xcalloc((size_t)n_bins, sizeof(double));
    return a;
}

static void acc_free(Acc *a) {
    free(a->band);
    free(a->overall);
    a->band = a->overall = NULL;
}

static void acc_add(Acc *dst, const Acc *src) {
    for (int i = 0; i < dst->n_bins; i++) {
        dst->band[i] += src->band[i];
        dst->overall[i] += src->overall[i];
    }
}

/* Peak height and lag (seconds) of ACF in [LAG_LO_BIN, LAG_HI_BIN]. */
static void acf_peak(const double *x, int n, double *peak_out, double *lag_s_out) {
    *peak_out = 0.0;
    *lag_s_out = 0.0;
    if (n < LAG_LO_BIN + 2) return;
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    double *z = xmalloc((size_t)n * sizeof(double));
    double power = 0.0;
    for (int i = 0; i < n; i++) {
        z[i] = x[i] - mean;
        power += z[i] * z[i];
    }
    if (!(power > 0.0)) {
        free(z);
        return;
    }
    int hi = LAG_HI_BIN;
    if (hi > n - 1) hi = n - 1;
    int lo = LAG_LO_BIN;
    if (lo > hi) {
        free(z);
        return;
    }
    /* acf[0] unused; store values for lags 0..hi so neighbors are available. */
    double *acf = xmalloc((size_t)(hi + 2) * sizeof(double));
    for (int lag = 0; lag <= hi + 1 && lag < n; lag++) {
        double c = 0.0;
        for (int i = 0; i + lag < n; i++) c += z[i] * z[i + lag];
        acf[lag] = c / power;
    }
    double best_peak = -1.0;
    int best_j = lo;
    int found_peak = 0;
    double best_any = acf[lo];
    int best_any_j = lo;
    for (int lag = lo; lag <= hi; lag++) {
        double c = acf[lag];
        if (c > best_any) {
            best_any = c;
            best_any_j = lag;
        }
        double left = acf[lag - 1];
        double right = (lag + 1 < n) ? acf[lag + 1] : c - 1.0;
        if (c > left && c >= right && c > 0.0) {
            if (!found_peak || c > best_peak) {
                best_peak = c;
                best_j = lag;
                found_peak = 1;
            }
        }
    }
    if (found_peak) {
        *peak_out = best_peak;
        *lag_s_out = (double)best_j * BIN_DT;
    } else {
        *peak_out = best_any;
        *lag_s_out = (double)best_any_j * BIN_DT;
    }
    free(acf);
    free(z);
}

static void write_minutes(FILE *out, const char *filename, int year, const Acc *acc) {
    double *ratio = xmalloc((size_t)acc->n_bins * sizeof(double));
    for (int i = 0; i < acc->n_bins; i++)
        ratio[i] = acc->overall[i] > 0.0 ? acc->band[i] / acc->overall[i] : 0.0;
    int nmin = (acc->n_bins + BINS_PER_MINUTE - 1) / BINS_PER_MINUTE;
    for (int m = 0; m < nmin; m++) {
        int i0 = m * BINS_PER_MINUTE;
        int n = acc->n_bins - i0;
        if (n > BINS_PER_MINUTE) n = BINS_PER_MINUTE;
        double peak = 0.0, lag_s = 0.0;
        acf_peak(ratio + i0, n, &peak, &lag_s);
        fprintf(out, "%s,%d,%d,%.3f,%d,%.8e,%.8e\n",
                filename, year, m, (double)m * 60.0, n, peak, lag_s);
    }
    free(ratio);
}

typedef struct {
    int fd;
    Wav wav;
    const Layout *L;
    int64_t n_limit;
    int worker;
    int stride;
    int use_gpu;
    int device;
    Acc acc;
} Job;

static void *window_worker(void *arg) {
    Job *job = arg;
    const Layout *L = job->L;
    Scratch s = scratch_for(L, job->use_gpu, job->device);
    float *buf = xmalloc((size_t)L->window_samples * sizeof(float));
    int64_t win = L->window_samples;
    for (int64_t start = (int64_t)job->worker * win; start < job->n_limit; start += (int64_t)job->stride * win) {
        int n = (int)(job->n_limit - start);
        if (n > L->window_samples) n = L->window_samples;
        read_samples(job->fd, &job->wav, start, n, buf);
        process_window(&s, buf, n, start, &job->acc);
    }
    free(buf);
    scratch_free(&s);
    return NULL;
}

static int year_of_name(const char *base) {
    int year = 0, month = 0;
    if (parse_stamp(base, &year, &month)) return year;
    return 0;
}

static void ensure_out_dir(void) {
    char buf[4096];
    size_t n = strlen(g_out_dir);
    if (n >= sizeof buf) die("out-dir path is too long");
    memcpy(buf, g_out_dir, n + 1);
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(buf, 0775) != 0 && errno != EEXIST) {
            fprintf(stderr, "minute_acf: cannot create %s (%s)\n", buf, strerror(errno));
            exit(1);
        }
        *p = '/';
    }
    if (mkdir(buf, 0775) != 0 && errno != EEXIST) {
        fprintf(stderr, "minute_acf: cannot create %s (%s)\n", buf, strerror(errno));
        exit(1);
    }
}

static void process_file(const char *path, int use_gpu, int nworkers) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open recording");
    Wav w = open_wav(path, fd);
    int64_t n_limit = w.n_samples;
    if (g_duration_sec > 0) {
        int64_t cap = (int64_t)(g_duration_sec * (double)w.sample_rate);
        if (cap < n_limit) n_limit = cap;
    }
    Layout L = layout_for(w.sample_rate);
    int64_t seg_len = bin_samples(w.sample_rate);
    int n_bins = (int)((n_limit + seg_len - 1) / seg_len);
    if (n_bins < 1) n_bins = 1;
    if (nworkers < 1) nworkers = 1;
    int64_t nwin = (n_limit + L.window_samples - 1) / L.window_samples;
    if (nworkers > nwin) nworkers = (int)nwin;
    if (nworkers < 1) nworkers = 1;

    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    fprintf(stderr, "%s  %s  %d Hz  %.2f h  %d workers  fftl %d hop %d\n",
            use_gpu ? "gpu" : "cpu", path, w.sample_rate,
            (double)n_limit / w.sample_rate / 3600.0, nworkers, L.fftl, L.hop);
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    Job *jobs = xcalloc((size_t)nworkers, sizeof(Job));
    pthread_t *ths = xmalloc((size_t)nworkers * sizeof(pthread_t));
    for (int i = 0; i < nworkers; i++) {
        jobs[i].fd = fd;
        jobs[i].wav = w;
        jobs[i].L = &L;
        jobs[i].n_limit = n_limit;
        jobs[i].worker = i;
        jobs[i].stride = nworkers;
        jobs[i].use_gpu = use_gpu;
        jobs[i].device = use_gpu ? i : -1;
        jobs[i].acc = acc_new(n_bins);
        if (pthread_create(&ths[i], NULL, window_worker, &jobs[i]) != 0) die("thread");
    }
    for (int i = 0; i < nworkers; i++) pthread_join(ths[i], NULL);

    Acc total = acc_new(n_bins);
    for (int i = 0; i < nworkers; i++) acc_add(&total, &jobs[i].acc);

    ensure_out_dir();
    char out_path[4096];
    char tmp_path[4100];
    int nw = snprintf(out_path, sizeof out_path, "%s/%s.minute_acf.csv", g_out_dir, base);
    if (nw < 0 || nw >= (int)sizeof out_path) die("output path is too long");
    snprintf(tmp_path, sizeof tmp_path, "%s.tmp", out_path);
    FILE *out = fopen(tmp_path, "w");
    if (!out) die("cannot write the csv");
    fprintf(out, "filename,year,minute_index,start_sec,n_bins,acf_peak,acf_lag_s\n");
    write_minutes(out, base, year_of_name(base), &total);
    if (fclose(out) != 0) die("cannot close the csv");
    if (rename(tmp_path, out_path) != 0) die("cannot replace the csv");
    int nmin = (n_bins + BINS_PER_MINUTE - 1) / BINS_PER_MINUTE;
    fprintf(stderr, "wrote %s  (%d minutes)\n", out_path, nmin);

    for (int i = 0; i < nworkers; i++) acc_free(&jobs[i].acc);
    acc_free(&total);
    free(jobs);
    free(ths);
    layout_free(&L);
    close(fd);
}

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
    int64_t seg = bin_samples(L.rate);
    Acc acc = acc_new((int)((L.window_samples + seg - 1) / seg));
    float *buf = xmalloc((size_t)L.window_samples * sizeof(float));
    burst_samples(buf, L.window_samples, L.rate, hz, 0.5);
    process_window(&s, buf, L.window_samples, 0, &acc);
    double band = 0, overall = 0;
    for (int i = 0; i < acc.n_bins; i++) {
        band += acc.band[i];
        overall += acc.overall[i];
    }
    free(buf);
    acc_free(&acc);
    scratch_free(&s);
    layout_free(&L);
    return overall > 0.0 ? band / overall : 0.0;
}

static void self_test(void) {
    double tone[BINS_PER_MINUTE];
    /* Period 5.0 s → lag bin 25 at 0.2 s spacing. */
    for (int i = 0; i < BINS_PER_MINUTE; i++)
        tone[i] = cos(2.0 * M_PI * (double)i / 25.0);
    double peak = 0.0, lag_s = 0.0;
    acf_peak(tone, BINS_PER_MINUTE, &peak, &lag_s);
    fprintf(stderr, "5.0 s cosine ACF peak %.4f at %.2f s\n", peak, lag_s);
    if (fabs(lag_s - 5.0) > 0.21) die("expected ACF lag near 5.0 s");
    if (peak < 0.8) die("expected a strong ACF peak for a pure cosine");

    for (int i = 0; i < BINS_PER_MINUTE; i++) tone[i] = 1.0;
    acf_peak(tone, BINS_PER_MINUTE, &peak, &lag_s);
    fprintf(stderr, "constant series ACF peak %.4e at %.2f s\n", peak, lag_s);
    if (fabs(peak) > 1e-9) die("a constant series must have zero ACF after demeaning");

    double cpu_in = burst_ratio(0, -1, 200.0);
    double cpu_out = burst_ratio(0, -1, 2000.0);
    fprintf(stderr, "cpu ratio 200 Hz bursts %.4f  2000 Hz bursts %.4e\n", cpu_in, cpu_out);
    if (!(cpu_in > 0.5 && cpu_out < 0.1)) die("cpu ratio did not separate 200 Hz from 2000 Hz bursts");

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != 0 || ndev < 1) die("self-test found no GPU");
    double gpu_in = burst_ratio(1, 0, 200.0);
    double gpu_out = burst_ratio(1, 0, 2000.0);
    fprintf(stderr, "gpu ratio 200 Hz bursts %.4f  2000 Hz bursts %.4e\n", gpu_in, gpu_out);
    if (!(gpu_in > 0.5 && gpu_out < 0.1)) die("gpu ratio did not separate 200 Hz from 2000 Hz bursts");
    double rel = fabs(gpu_in - cpu_in) / cpu_in;
    fprintf(stderr, "cpu/gpu relative difference %.3e\n", rel);
    if (rel > 1e-2) die("cpu and gpu ratios disagree");
    fprintf(stderr, "self-test passed\n");
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--out-dir DIR] [--duration-sec N] [--cpu|--gpu] [--self-test] WAV [WAV ...]\n"
            "Writes <WAV basename>.minute_acf.csv in DIR (default %s).\n"
            "Each row is one minute: the peak height and lag (1–20 s) of the\n"
            "autocorrelation of the 0.2 s whitened 50–500 Hz energy ratio.\n",
            argv0, g_out_dir);
}

int main(int argc, char **argv) {
    int self_test_only = 0;
    char **wavs = NULL;
    int nwav = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) g_out_dir = argv[++i];
        else if (!strcmp(argv[i], "--duration-sec") && i + 1 < argc) g_duration_sec = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) g_backend = 0;
        else if (!strcmp(argv[i], "--gpu")) g_backend = 1;
        else if (!strcmp(argv[i], "--self-test")) self_test_only = 1;
        else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else {
            wavs = &argv[i];
            nwav = argc - i;
            break;
        }
    }
    if (self_test_only) {
        self_test();
        return 0;
    }
    if (nwav < 1) {
        usage(argv[0]);
        return 2;
    }
    int use_gpu = 0;
    int nworkers = 1;
    if (g_backend == 0) {
        use_gpu = 0;
        nworkers = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nworkers < 1) nworkers = 1;
        if (nworkers > 8) nworkers = 8;
    } else {
        int ndev = 0;
        int ok = cudaGetDeviceCount(&ndev) == 0 && ndev > 0;
        if (!ok) {
            if (g_backend == 1) die("no GPU");
            nworkers = (int)sysconf(_SC_NPROCESSORS_ONLN);
            if (nworkers < 1) nworkers = 1;
            if (nworkers > 8) nworkers = 8;
            fprintf(stderr, "no GPU, using %d cpu workers\n", nworkers);
        } else {
            use_gpu = 1;
            if (ndev > MAX_GPUS) ndev = MAX_GPUS;
            nworkers = ndev;
        }
    }
    for (int i = 0; i < nwav; i++) process_file(wavs[i], use_gpu, nworkers);
    return 0;
}
