/*-----------------------------------------------------------------------*/
/* stream_bench.c -- a STREAM variant that accounts for write-allocate      */
/*                                                                       */
/* The classic STREAM benchmark charges each kernel only the bytes the   */
/* algorithm logically needs:                                            */
/*                                                                       */
/*     read   : 1N   (read a)                                            */
/*     write  : 1N   (write c)                                           */
/*     copy   : 2N   (read a, write c)                                   */
/*     triad  : 3N   (read b, read c, write a)                           */
/*                                                                       */
/* On a write-back, write-allocate cache a store that misses does not    */
/* go straight to DRAM: the line is first fetched (read-for-ownership,   */
/* RFO), then dirtied, then written back when evicted.  So the real      */
/* traffic through the memory controller is:                             */
/*                                                                       */
/*     read   : 1N                        (no store)                     */
/*     write  : 2N   = 1N RFO + 1N WB                                    */
/*     copy   : 3N   = 1N read + 1N RFO + 1N WB                          */
/*     triad  : 4N   = 2N read + 1N RFO + 1N WB                          */
/*                                                                       */
/* i.e. DRAM = useful + one extra N per array stored to.                 */
/*                                                                       */
/* This program reports both numbers for every kernel, and it also runs  */
/* each store kernel again with non-temporal (streaming) stores.  Those  */
/* bypass the cache, so there is no RFO and DRAM == useful -- which is    */
/* the direct experimental evidence for the write-allocate model above.  */
/*                                                                       */
/* Build:                                                                */
/*   gcc -O2 -march=native -fopenmp -fno-tree-loop-distribute-patterns \ */
/*       -o stream_bench stream_bench.c                                        */
/*                                                                       */
/* -fno-tree-loop-distribute-patterns matters: without it GCC may turn   */
/* the Copy loop into memcpy(), and glibc's memcpy uses non-temporal     */
/* stores for large sizes, which would silently remove the RFO from the  */
/* "regular" Copy row.                                                   */
/*                                                                       */
/* Knobs:                                                                */
/*   -DSTREAM_ARRAY_SIZE=<elements>   default 16M (128 MiB per array)    */
/*   -DNTIMES=<n>                     default 10                         */
/*                                                                       */
/* The model assumes every access misses, stores are full lines, the     */
/* arrays are much larger than LLC, and no other workload is running.    */
/* Verify against hardware with, e.g.:                                   */
/*   perf stat -a -e uncore_imc_0/cas_count_read/,                      */
/*                  uncore_imc_0/cas_count_write/ ...                    */
/* (uncore counters are socket-wide; see the notes -- they cannot        */
/*  attribute traffic to a PID.)                                         */
/*-----------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define PLATFORM_X86 1
#else
#  define PLATFORM_X86 0
#endif

#ifndef STREAM_ARRAY_SIZE
#  define STREAM_ARRAY_SIZE (16u * 1024u * 1024u)
#endif
#ifndef NTIMES
#  define NTIMES 10
#endif
#if NTIMES < 2
#  undef NTIMES
#  define NTIMES 10
#endif

#define ALIGNMENT   64u
#define SCALAR      3.0
#define MIB         (1024.0 * 1024.0)

static const size_t N = STREAM_ARRAY_SIZE;

static double *a, *b, *c;

/* Read at run time so the compiler cannot fold a constant and turn the
 * write loop into memset(). */
static double g_scalar = SCALAR;

/* Expected element values after each stage, for the self-check. */
static double exp_a = 1.0, exp_b = 2.0, exp_c = 0.0;
static double read_check;

/*-----------------------------------------------------------------------*/
/* timing                                                                */
/*-----------------------------------------------------------------------*/

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/*-----------------------------------------------------------------------*/
/* regular kernels (write-allocate / RFO path)                           */
/*-----------------------------------------------------------------------*/

/* 1N of DRAM traffic: read-only, no stores.  8 independent chains so the
 * FP add dependency does not become the bottleneck. */
static double read_sum(const double *restrict x)
{
    double sum = 0.0;
    const size_t n8 = N & ~(size_t)7;
    size_t j;

#pragma omp parallel for schedule(static) reduction(+:sum)
    for (j = 0; j < n8; j += 8) {
        double t0 = (x[j + 0] + x[j + 1]) + (x[j + 2] + x[j + 3]);
        double t1 = (x[j + 4] + x[j + 5]) + (x[j + 6] + x[j + 7]);
        sum += t0 + t1;
    }
    for (j = n8; j < N; j++)
        sum += x[j];

    return sum;
}

/* useful 1N, actual 2N (RFO read + write-back). */
static void write_seq(double *restrict x, double v)
{
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++)
        x[j] = v;
}

/* useful 2N, actual 3N (read src + RFO read dst + write-back dst). */
static void copy_seq(double *restrict dst, const double *restrict src)
{
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++)
        dst[j] = src[j];
}

/* useful 3N, actual 4N (2 reads + RFO read dst + write-back dst). */
static void triad_seq(double *restrict dst, const double *restrict x,
                      const double *restrict y, double s)
{
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++)
        dst[j] = x[j] + s * y[j];
}

/*-----------------------------------------------------------------------*/
/* non-temporal (streaming store) kernels -- no RFO, no write-back of a  */
/* line the program is going to overwrite anyway.                        */
/*-----------------------------------------------------------------------*/

#if PLATFORM_X86 && defined(__AVX__)
#  define NT_VW 4
typedef __m256d ntvec;
static inline ntvec nt_load(const double *p)      { return _mm256_load_pd(p); }
static inline void  nt_store(double *p, ntvec v)  { _mm256_stream_pd(p, v); }
static inline ntvec nt_set1(double v)             { return _mm256_set1_pd(v); }
static inline ntvec nt_add(ntvec x, ntvec y)      { return _mm256_add_pd(x, y); }
static inline ntvec nt_mul(ntvec x, ntvec y)      { return _mm256_mul_pd(x, y); }
#  define NT_FENCE() _mm_sfence()
#  define HAVE_NT 1
#elif PLATFORM_X86 && defined(__SSE2__)
#  define NT_VW 2
typedef __m128d ntvec;
static inline ntvec nt_load(const double *p)      { return _mm_load_pd(p); }
static inline void  nt_store(double *p, ntvec v)  { _mm_stream_pd(p, v); }
static inline ntvec nt_set1(double v)             { return _mm_set1_pd(v); }
static inline ntvec nt_add(ntvec x, ntvec y)      { return _mm_add_pd(x, y); }
static inline ntvec nt_mul(ntvec x, ntvec y)      { return _mm_mul_pd(x, y); }
#  define NT_FENCE() _mm_sfence()
#  define HAVE_NT 1
#else
#  define HAVE_NT 0
#endif

#if HAVE_NT

static void write_nt(double *restrict x, double v)
{
    const size_t nv = N & ~(size_t)(NT_VW - 1);
    const ntvec vv = nt_set1(v);
    size_t j;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (j = 0; j < nv; j += NT_VW)
            nt_store(x + j, vv);
        NT_FENCE();
    }
    for (j = nv; j < N; j++)
        x[j] = v;
}

static void copy_nt(double *restrict dst, const double *restrict src)
{
    const size_t nv = N & ~(size_t)(NT_VW - 1);
    size_t j;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (j = 0; j < nv; j += NT_VW)
            nt_store(dst + j, nt_load(src + j));
        NT_FENCE();
    }
    for (j = nv; j < N; j++)
        dst[j] = src[j];
}

static void triad_nt(double *restrict dst, const double *restrict x,
                     const double *restrict y, double s)
{
    const size_t nv = N & ~(size_t)(NT_VW - 1);
    const ntvec sv = nt_set1(s);
    size_t j;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (j = 0; j < nv; j += NT_VW)
            nt_store(dst + j, nt_add(nt_load(x + j), nt_mul(sv, nt_load(y + j))));
        NT_FENCE();
    }
    for (j = nv; j < N; j++)
        dst[j] = x[j] + s * y[j];
}

#endif /* HAVE_NT */

/*-----------------------------------------------------------------------*/
/* test wrappers + self-check                                            */
/*-----------------------------------------------------------------------*/

static void run_read(void)      { read_check = read_sum(a); }
static void run_write(void)     { write_seq(c, g_scalar); }
static void run_copy(void)      { copy_seq(c, a); }
static void run_triad(void)     { triad_seq(a, b, c, g_scalar); }
#if HAVE_NT
static void run_write_nt(void)  { write_nt(c, g_scalar); }
static void run_copy_nt(void)   { copy_nt(c, a); }
static void run_triad_nt(void)  { triad_nt(a, b, c, g_scalar); }
#endif

static size_t count_ne(const double *x, double v)
{
    size_t bad = 0;
    for (size_t j = 0; j < N; j++)
        if (x[j] != v)
            bad++;
    return bad;
}

/* Exact comparison is fine: every value here is exactly representable. */
static int verify_read(void)
{
    double mean = read_check / (double)N;
    return fabs(mean - exp_a) <= 1e-9 * fabs(exp_a) + 1e-12;
}

static int verify_write(void)
{
    exp_c = g_scalar;
    return count_ne(c, exp_c) == 0;
}

static int verify_copy(void)
{
    size_t bad = 0;
    for (size_t j = 0; j < N; j++)
        if (c[j] != a[j])
            bad++;
    exp_c = exp_a;
    return bad == 0;
}

static int verify_triad(void)
{
    exp_a = exp_b + g_scalar * exp_c;
    return count_ne(a, exp_a) == 0;
}

/*-----------------------------------------------------------------------*/
/* driver                                                                */
/*-----------------------------------------------------------------------*/

typedef struct {
    const char *name;
    double useful_arrays;   /* algorithmic bytes, in units of N*8          */
    int    stored_arrays;   /* arrays written to (RFO applies to each)     */
    int    nt;              /* streaming stores used                       */
    void (*run)(void);
    int  (*verify)(void);
} test_t;

static void bench(const test_t *t)
{
    double best = DBL_MAX, total = 0.0;

    for (int k = 0; k < NTIMES; k++) {
        double t0 = now_sec();
        t->run();
        double dt = now_sec() - t0;
        if (k > 0) {                 /* first pass is warm-up, like STREAM */
            if (dt < best) best = dt;
            total += dt;
        }
    }

    const double avg = total / (double)(NTIMES - 1);
    const double elem = (double)sizeof(double);
    const double useful = t->useful_arrays * (double)N * elem;
    const double dram = (t->useful_arrays +
                         (t->nt ? 0.0 : (double)t->stored_arrays))
                        * (double)N * elem;

    const int ok = t->verify();

    printf("%-10s %10.1f %10.1f %9.3f %9.3f %10.2f %10.2f\n",
           t->name,
           useful / MIB,
           dram / MIB,
           1e3 * best,
           1e3 * avg,
           useful / best / 1e9,
           dram / best / 1e9);

    if (!ok)
        fprintf(stderr, "WARNING: %s failed its result check\n", t->name);
}

static void usage(const char *prog)
{
    printf("usage: %s [-k kernel]\n", prog);
    printf("  -k read|write|write_nt|copy|copy_nt|triad|triad_nt\n");
    printf("     run a single kernel (bracket it with perf stat to check the\n");
    printf("     write-allocate model against the memory controller)\n");
}

/* "write_nt" and "Write NT" name the same test. */
static int name_matches(const char *want, const char *name)
{
    while (*want && *name) {
        char w = (*want == '_') ? ' ' : *want;
        char c = *name;
        if (w >= 'A' && w <= 'Z') w = (char)(w - 'A' + 'a');
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (w != c) return 0;
        want++;
        name++;
    }
    return *want == '\0' && *name == '\0';
}

int main(int argc, char **argv)
{
    const char *only = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            only = argv[++i];
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    printf("-------------------------------------------------------------\n");
    printf("STREAM-WA: write-allocate aware memory bandwidth\n");
    printf("-------------------------------------------------------------\n");

    if (posix_memalign((void **)&a, ALIGNMENT, N * sizeof(double)) ||
        posix_memalign((void **)&b, ALIGNMENT, N * sizeof(double)) ||
        posix_memalign((void **)&c, ALIGNMENT, N * sizeof(double))) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    printf("Array size per array = %zu elements (%.1f MiB)\n",
           N, (double)N * sizeof(double) / MIB);
    printf("Total memory         = %.1f MiB\n", 3.0 * (double)N * sizeof(double) / MIB);
    printf("Iterations           = %d (first discarded, best time reported)\n", NTIMES);
#ifdef _OPENMP
    printf("Threads              = %d\n", omp_get_max_threads());
#else
    printf("Threads              = 1 (built without OpenMP)\n");
#endif
#if HAVE_NT
    printf("Streaming stores     = %d doubles per instruction (%s)\n",
           NT_VW, NT_VW == 4 ? "AVX" : "SSE2");
#else
    printf("Streaming stores     = unavailable on this target\n");
#endif
    printf("-------------------------------------------------------------\n");
    printf("%-10s %10s %10s %9s %9s %10s %10s\n",
           "Kernel", "UsefulMiB", "DRAMMiB", "Best(ms)", "Avg(ms)",
           "UsefGB/s", "DRAMGB/s");
    printf("-------------------------------------------------------------\n");

    static const test_t tests[] = {
        { "Read",     1.0, 0, 0, run_read,     verify_read  },
        { "Write",    1.0, 1, 0, run_write,    verify_write },
#if HAVE_NT
        { "Write NT", 1.0, 1, 1, run_write_nt, verify_write },
#endif
        { "Copy",     2.0, 1, 0, run_copy,     verify_copy  },
#if HAVE_NT
        { "Copy NT",  2.0, 1, 1, run_copy_nt,  verify_copy  },
#endif
        { "Triad",    3.0, 1, 0, run_triad,    verify_triad },
#if HAVE_NT
        { "Triad NT", 3.0, 1, 1, run_triad_nt, verify_triad },
#endif
    };

    int ran = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (only && !name_matches(only, tests[i].name))
            continue;
        bench(&tests[i]);
        ran++;
    }
    if (only && !ran) {
        fprintf(stderr, "unknown kernel '%s'\n", only);
        return 2;
    }

    printf("-------------------------------------------------------------\n");
    printf("Useful = minimum algorithmic bytes required by the kernel.\n");
    printf("DRAM   = useful + one RFO read per array written, for the\n");
    printf("         regular (non-NT) rows: write 2N, copy 3N, triad 4N.\n");
    printf("         NT rows use MOVNT* stores: the cache is bypassed, so\n");
    printf("         DRAM == useful there.\n");
    printf("-------------------------------------------------------------\n");
    if (!only) {
        printf("Tip: run one kernel, e.g. ./stream_bench -k Copy, under\n");
        printf("     perf stat -a -e uncore_imc_free_running/data_read/,\n");
        printf("                    uncore_imc_free_running/data_write/,\n");
        printf("                    uncore_imc_free_running/data_total/\n");
        printf("     (add _0/_1 yourself if your perf only lists the instances)\n");
        printf("     and compare the controller bytes with that row's DRAMMiB.\n");
        printf("-------------------------------------------------------------\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
