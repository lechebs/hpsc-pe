#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "timer.h"
#include "write_vtk.h"

#ifndef NUM_POINTS_PER_DIR
#define NUM_POINTS_PER_DIR 260
#endif

#ifndef NUM_TIMESTEPS
#define NUM_TIMESTEPS 20
#endif

#define DX (1.0 / NUM_POINTS_PER_DIR)
#define DT (1.0 / NUM_TIMESTEPS)

#ifndef NU
#define NU 0.1
#endif

#define NUM_POINTS \
    (NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR)

#define ALLOC_SIZE (NUM_POINTS * sizeof(double))

#ifndef SCRATCH_PAD
#define SCRATCH_PAD 0
#endif

#ifndef DZZ_BLOCK
#define DZZ_BLOCK 8
#endif

#define DXX_VLEN 4

#if (NUM_POINTS_PER_DIR % DXX_VLEN) != 0
#error "NUM_POINTS_PER_DIR must be a multiple of DXX_VLEN"
#endif

#define DYY_SCRATCH_SIZE \
    (NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR + NUM_POINTS_PER_DIR)

#define DZZ_SCRATCH_SIZE \
    (DZZ_BLOCK * NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR + \
     NUM_POINTS_PER_DIR)

#define SOLVER_SCRATCH_SIZE \
    (DYY_SCRATCH_SIZE > DZZ_SCRATCH_SIZE ? \
     DYY_SCRATCH_SIZE : DZZ_SCRATCH_SIZE)

#define SCRATCH_SIZE \
    ((NUM_POINTS + SCRATCH_PAD + \
      SOLVER_SCRATCH_SIZE) * sizeof(double))

#ifndef OUTPUT_VTK_FREQ
#define OUTPUT_VTK_FREQ 0
#endif

#define ROW_MAJ(x, y, z) \
   ((z) * NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR + \
    (y) * NUM_POINTS_PER_DIR + x)

static double forcing(double x, double y, double z, double t)
{
    double rx = (x - 0.5) * (x - 0.5);
    double ry = (y - 0.5) * (y - 0.5);
    double rz = (z - 0.5) * (z - 0.5);

    if (rx + ry + rz < 0.01) {
        return 0.5;
    } else {
        return 0;
    }
}

__attribute__((target("avx2")))
static inline void transpose4x4(__m256d *r0, __m256d *r1,
                                __m256d *r2, __m256d *r3)
{
    __m256d t0 = _mm256_unpacklo_pd(*r0, *r1);
    __m256d t1 = _mm256_unpackhi_pd(*r0, *r1);
    __m256d t2 = _mm256_unpacklo_pd(*r2, *r3);
    __m256d t3 = _mm256_unpackhi_pd(*r2, *r3);

    *r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

__attribute__((target("avx2")))
static inline void solve_Dxx_block(const double *restrict eps,
                                   double *restrict eta,
                                   const double *restrict upp_redu,
                                   double *restrict rhs_redu,
                                   double mid_i, double low_i,
                                   int y, int z)
{
    const __m256d low_vec = _mm256_set1_pd(low_i);
    const __m256d zero = _mm256_setzero_pd();

    __m256d rhs_prev = zero;
    for (int tk = 0; tk < NUM_POINTS_PER_DIR; tk += DXX_VLEN) {

        const int base = ROW_MAJ(tk, y, z);

        __m256d e0 = _mm256_loadu_pd(
            eta + base + 0 * NUM_POINTS_PER_DIR);
        __m256d e1 = _mm256_loadu_pd(
            eta + base + 1 * NUM_POINTS_PER_DIR);
        __m256d e2 = _mm256_loadu_pd(
            eta + base + 2 * NUM_POINTS_PER_DIR);
        __m256d e3 = _mm256_loadu_pd(
            eta + base + 3 * NUM_POINTS_PER_DIR);

        __m256d p0 = _mm256_loadu_pd(
            eps + base + 0 * NUM_POINTS_PER_DIR);
        __m256d p1 = _mm256_loadu_pd(
            eps + base + 1 * NUM_POINTS_PER_DIR);
        __m256d p2 = _mm256_loadu_pd(
            eps + base + 2 * NUM_POINTS_PER_DIR);
        __m256d p3 = _mm256_loadu_pd(
            eps + base + 3 * NUM_POINTS_PER_DIR);

        __m256d d0 = _mm256_sub_pd(p0, e0);
        __m256d d1 = _mm256_sub_pd(p1, e1);
        __m256d d2 = _mm256_sub_pd(p2, e2);
        __m256d d3 = _mm256_sub_pd(p3, e3);
        transpose4x4(&d0, &d1, &d2, &d3);

        __m256d diff_col[DXX_VLEN] = {d0, d1, d2, d3};

        for (int k = 0; k < DXX_VLEN; ++k) {
            const int x = tk + k;

            if (x == 0 || x == NUM_POINTS_PER_DIR - 1) {
                _mm256_storeu_pd(rhs_redu + x * DXX_VLEN, zero);
                rhs_prev = zero;
            } else {
                const double norm_coef =
                    mid_i - low_i * upp_redu[x - 1];

                __m256d rhs = diff_col[k];
                rhs = _mm256_sub_pd(
                    rhs, _mm256_mul_pd(low_vec, rhs_prev));
                rhs = _mm256_div_pd(
                    rhs, _mm256_set1_pd(norm_coef));

                _mm256_storeu_pd(rhs_redu + x * DXX_VLEN, rhs);
                rhs_prev = rhs;
            }
        }
    }

    // homogeneous dirichlet
    for (int lane = 0; lane < DXX_VLEN; ++lane) {
        eta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y + lane, z)] = 0;
    }
    _mm256_storeu_pd(rhs_redu +
                     (NUM_POINTS_PER_DIR - 1) * DXX_VLEN, zero);

    // Backward substitution
    __m256d delta_prev = zero;
    for (int tk = NUM_POINTS_PER_DIR - DXX_VLEN; tk >= 0;
         tk -= DXX_VLEN) {

        __m256d delta_col[DXX_VLEN];

        for (int k = DXX_VLEN - 1; k >= 0; --k) {
            const int x = tk + k;

            if (x == NUM_POINTS_PER_DIR - 1) {
                delta_col[k] = zero;
                delta_prev = zero;
            } else {
                __m256d rhs = _mm256_loadu_pd(
                    rhs_redu + x * DXX_VLEN);
                __m256d upp_vec = _mm256_set1_pd(upp_redu[x]);
                __m256d delta = _mm256_sub_pd(
                    rhs, _mm256_mul_pd(upp_vec, delta_prev));

                delta_col[k] = delta;
                delta_prev = delta;
            }
        }

        transpose4x4(&delta_col[0], &delta_col[1],
                     &delta_col[2], &delta_col[3]);

        const int base = ROW_MAJ(tk, y, z);
        for (int lane = 0; lane < DXX_VLEN; ++lane) {
            double *row = eta + base + lane * NUM_POINTS_PER_DIR;
            __m256d e = _mm256_loadu_pd(row);
            _mm256_storeu_pd(row, _mm256_add_pd(e, delta_col[lane]));
        }
    }
}

// Solves (I - Dxx)(eta_n+1 - eta_n) = eps_n+1 - eta_n
__attribute__((target("avx2")))
static void solve_Dxx_tridiag_fused(const double *restrict eps,
                                    double *restrict eta,
                                    double *restrict upp_redu,
                                    double *restrict rhs_redu)
{
    TIMER_SCOPE_START(solve_Dxx_tridiag_fused);

    const double w = DT * NU / (2 * DX * DX);
    const double upp_i = -w;
    const double mid_i = 1.0 + 2 * w;
    const double low_i = -w;

    upp_redu[0] = 0;
    for (int x = 1; x < NUM_POINTS_PER_DIR - 1; ++x) {
        const double norm_coef = mid_i - low_i * upp_redu[x - 1];
        upp_redu[x] = upp_i / norm_coef;
    }

    const __m256d low_vec = _mm256_set1_pd(low_i);
    const __m256d zero = _mm256_setzero_pd();

    double *rhs0 = rhs_redu;
    double *rhs1 = rhs_redu + NUM_POINTS_PER_DIR * DXX_VLEN;

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        int y = 0;

        // Two independent y-tiles at a time.
        for (; y + 2 * DXX_VLEN <= NUM_POINTS_PER_DIR;
             y += 2 * DXX_VLEN) {

            const int y0 = y;
            const int y1 = y + DXX_VLEN;

            __m256d rhs_prev0 = zero;
            __m256d rhs_prev1 = zero;

            for (int tk = 0; tk < NUM_POINTS_PER_DIR; tk += DXX_VLEN) {

                const int base0 = ROW_MAJ(tk, y0, z);
                const int base1 = ROW_MAJ(tk, y1, z);

                __m256d e00 = _mm256_loadu_pd(
                    eta + base0 + 0 * NUM_POINTS_PER_DIR);
                __m256d e01 = _mm256_loadu_pd(
                    eta + base0 + 1 * NUM_POINTS_PER_DIR);
                __m256d e02 = _mm256_loadu_pd(
                    eta + base0 + 2 * NUM_POINTS_PER_DIR);
                __m256d e03 = _mm256_loadu_pd(
                    eta + base0 + 3 * NUM_POINTS_PER_DIR);

                __m256d p00 = _mm256_loadu_pd(
                    eps + base0 + 0 * NUM_POINTS_PER_DIR);
                __m256d p01 = _mm256_loadu_pd(
                    eps + base0 + 1 * NUM_POINTS_PER_DIR);
                __m256d p02 = _mm256_loadu_pd(
                    eps + base0 + 2 * NUM_POINTS_PER_DIR);
                __m256d p03 = _mm256_loadu_pd(
                    eps + base0 + 3 * NUM_POINTS_PER_DIR);

                __m256d e10 = _mm256_loadu_pd(
                    eta + base1 + 0 * NUM_POINTS_PER_DIR);
                __m256d e11 = _mm256_loadu_pd(
                    eta + base1 + 1 * NUM_POINTS_PER_DIR);
                __m256d e12 = _mm256_loadu_pd(
                    eta + base1 + 2 * NUM_POINTS_PER_DIR);
                __m256d e13 = _mm256_loadu_pd(
                    eta + base1 + 3 * NUM_POINTS_PER_DIR);

                __m256d p10 = _mm256_loadu_pd(
                    eps + base1 + 0 * NUM_POINTS_PER_DIR);
                __m256d p11 = _mm256_loadu_pd(
                    eps + base1 + 1 * NUM_POINTS_PER_DIR);
                __m256d p12 = _mm256_loadu_pd(
                    eps + base1 + 2 * NUM_POINTS_PER_DIR);
                __m256d p13 = _mm256_loadu_pd(
                    eps + base1 + 3 * NUM_POINTS_PER_DIR);

                __m256d d00 = _mm256_sub_pd(p00, e00);
                __m256d d01 = _mm256_sub_pd(p01, e01);
                __m256d d02 = _mm256_sub_pd(p02, e02);
                __m256d d03 = _mm256_sub_pd(p03, e03);

                __m256d d10 = _mm256_sub_pd(p10, e10);
                __m256d d11 = _mm256_sub_pd(p11, e11);
                __m256d d12 = _mm256_sub_pd(p12, e12);
                __m256d d13 = _mm256_sub_pd(p13, e13);

                transpose4x4(&d00, &d01, &d02, &d03);
                transpose4x4(&d10, &d11, &d12, &d13);

                __m256d diff0[DXX_VLEN] = {d00, d01, d02, d03};
                __m256d diff1[DXX_VLEN] = {d10, d11, d12, d13};

                for (int k = 0; k < DXX_VLEN; ++k) {
                    const int x = tk + k;

                    if (x == 0 || x == NUM_POINTS_PER_DIR - 1) {
                        _mm256_storeu_pd(rhs0 + x * DXX_VLEN, zero);
                        _mm256_storeu_pd(rhs1 + x * DXX_VLEN, zero);
                        rhs_prev0 = zero;
                        rhs_prev1 = zero;
                    } else {
                        const double norm_coef =
                            mid_i - low_i * upp_redu[x - 1];

                        __m256d rhs_a = diff0[k];
                        rhs_a = _mm256_sub_pd(
                            rhs_a, _mm256_mul_pd(low_vec, rhs_prev0));
                        rhs_a = _mm256_div_pd(
                            rhs_a, _mm256_set1_pd(norm_coef));

                        __m256d rhs_b = diff1[k];
                        rhs_b = _mm256_sub_pd(
                            rhs_b, _mm256_mul_pd(low_vec, rhs_prev1));
                        rhs_b = _mm256_div_pd(
                            rhs_b, _mm256_set1_pd(norm_coef));

                        _mm256_storeu_pd(rhs0 + x * DXX_VLEN, rhs_a);
                        _mm256_storeu_pd(rhs1 + x * DXX_VLEN, rhs_b);
                        rhs_prev0 = rhs_a;
                        rhs_prev1 = rhs_b;
                    }
                }
            }

            // homogeneous dirichlet
            for (int lane = 0; lane < DXX_VLEN; ++lane) {
                eta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y0 + lane, z)] = 0;
                eta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y1 + lane, z)] = 0;
            }
            _mm256_storeu_pd(rhs0 +
                             (NUM_POINTS_PER_DIR - 1) * DXX_VLEN, zero);
            _mm256_storeu_pd(rhs1 +
                             (NUM_POINTS_PER_DIR - 1) * DXX_VLEN, zero);

            // Backward substitution
            __m256d delta_prev0 = zero;
            __m256d delta_prev1 = zero;

            for (int tk = NUM_POINTS_PER_DIR - DXX_VLEN; tk >= 0;
                 tk -= DXX_VLEN) {

                __m256d delta0_col[DXX_VLEN];
                __m256d delta1_col[DXX_VLEN];

                for (int k = DXX_VLEN - 1; k >= 0; --k) {
                    const int x = tk + k;

                    if (x == NUM_POINTS_PER_DIR - 1) {
                        delta0_col[k] = zero;
                        delta1_col[k] = zero;
                        delta_prev0 = zero;
                        delta_prev1 = zero;
                    } else {
                        __m256d rhs_a = _mm256_loadu_pd(
                            rhs0 + x * DXX_VLEN);
                        __m256d rhs_b = _mm256_loadu_pd(
                            rhs1 + x * DXX_VLEN);
                        __m256d upp_vec = _mm256_set1_pd(upp_redu[x]);

                        __m256d delta_a = _mm256_sub_pd(
                            rhs_a, _mm256_mul_pd(upp_vec, delta_prev0));
                        __m256d delta_b = _mm256_sub_pd(
                            rhs_b, _mm256_mul_pd(upp_vec, delta_prev1));

                        delta0_col[k] = delta_a;
                        delta1_col[k] = delta_b;
                        delta_prev0 = delta_a;
                        delta_prev1 = delta_b;
                    }
                }

                transpose4x4(&delta0_col[0], &delta0_col[1],
                             &delta0_col[2], &delta0_col[3]);
                transpose4x4(&delta1_col[0], &delta1_col[1],
                             &delta1_col[2], &delta1_col[3]);

                const int base0 = ROW_MAJ(tk, y0, z);
                const int base1 = ROW_MAJ(tk, y1, z);

                for (int lane = 0; lane < DXX_VLEN; ++lane) {
                    double *row0 = eta + base0 +
                                   lane * NUM_POINTS_PER_DIR;
                    double *row1 = eta + base1 +
                                   lane * NUM_POINTS_PER_DIR;

                    __m256d e0 = _mm256_loadu_pd(row0);
                    __m256d e1 = _mm256_loadu_pd(row1);
                    _mm256_storeu_pd(row0,
                        _mm256_add_pd(e0, delta0_col[lane]));
                    _mm256_storeu_pd(row1,
                        _mm256_add_pd(e1, delta1_col[lane]));
                }
            }
        }

        // Remaining single y-tile.
        for (; y < NUM_POINTS_PER_DIR; y += DXX_VLEN) {
            solve_Dxx_block(eps, eta, upp_redu, rhs_redu,
                            mid_i, low_i, y, z);
        }
    }

    TIMER_SCOPE_BW_END(solve_Dxx_tridiag_fused, 3 * ALLOC_SIZE);
}

// Solves (I - Dyy)(zeta_n+1 - zeta_n) = eta_n+1 - zeta_n
static void solve_Dyy_tridiag_fused(const double *restrict eta,
                                    double *restrict zeta,
                                    double *restrict upp_redu,
                                    double *restrict rhs_redu)
{
    TIMER_SCOPE_START(solve_Dyy_tridiag_fused);

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {

        upp_redu[0] = 0;

        // homogeneous dirichlet
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
            rhs_redu[ROW_MAJ(x, 0, 0)] = 0;
        }

        // Forward reduction
        for (int y = 1; y < NUM_POINTS_PER_DIR - 1; ++y) {

            const double w = DT * NU / (2 * DX * DX);
            const double upp_i = -w;
            const double mid_i = 1.0 + 2 * w;
            const double low_i = -w;

            const double norm_coef = mid_i - low_i * upp_redu[y - 1];

            // upp_i = upp_i / (mid_i - low_i upp_i-1)
            upp_redu[y] = upp_i / norm_coef;

            for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs_redu[ROW_MAJ(x, y, 0)] =
                    (eta[ROW_MAJ(x, y, z)] -
                     zeta[ROW_MAJ(x, y, z)] -
                     low_i * rhs_redu[ROW_MAJ(x, y - 1, 0)]) /
                    norm_coef;
            }
        }

        // homogeneous dirichlet
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
            zeta[ROW_MAJ(x, NUM_POINTS_PER_DIR - 1, z)] = 0;
            rhs_redu[ROW_MAJ(x, NUM_POINTS_PER_DIR - 1, 0)] = 0;
        }

        // Backward substitution
        for (int y = NUM_POINTS_PER_DIR - 2; y >= 0; --y) {
            for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                // delta_i = rhs_i - upp_i delta_i+1
                double delta =
                    rhs_redu[ROW_MAJ(x, y, 0)] -
                    upp_redu[y] * rhs_redu[ROW_MAJ(x, y + 1, 0)];
                // zeta_n+1 = zeta_n + delta
                zeta[ROW_MAJ(x, y, z)] += delta;
                rhs_redu[ROW_MAJ(x, y, 0)] = delta;
            }
        }
    }

    TIMER_SCOPE_BW_END(solve_Dyy_tridiag_fused, 3 * ALLOC_SIZE);
}

// Solves (I - Dzz)(u_n+1 - u_n) = zeta_n+1 - u_n
static void solve_Dzz_tridiag_fused(const double *restrict zeta,
                                    double *restrict u,
                                    double *restrict upp_redu,
                                    double *restrict rhs_redu)
{
    TIMER_SCOPE_START(solve_Dzz_tridiag_fused);

    const double w = DT * NU / (2 * DX * DX);
    const double upp_i = -w;
    const double mid_i = 1.0 + 2 * w;
    const double low_i = -w;

    upp_redu[0] = 0;
    for (int z = 1; z < NUM_POINTS_PER_DIR - 1; ++z) {
        const double norm_coef = mid_i - low_i * upp_redu[z - 1];
        upp_redu[z] = upp_i / norm_coef;
    }

    const int plane = DZZ_BLOCK * NUM_POINTS_PER_DIR;

    for (int y0 = 0; y0 < NUM_POINTS_PER_DIR; y0 += DZZ_BLOCK) {

        int y_end = y0 + DZZ_BLOCK;
        if (y_end > NUM_POINTS_PER_DIR) {
            y_end = NUM_POINTS_PER_DIR;
        }
        const int height = y_end - y0;

        // homogeneous dirichlet
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                rhs_redu[y * NUM_POINTS_PER_DIR + x] = 0;
            }
        }

        // Forward reduction
        for (int z = 1; z < NUM_POINTS_PER_DIR - 1; ++z) {

            const double norm_coef = mid_i - low_i * upp_redu[z - 1];

            for (int y = 0; y < height; ++y) {
                const int row = ROW_MAJ(0, y0 + y, z);
                for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                    rhs_redu[z * plane + y * NUM_POINTS_PER_DIR + x] =
                        (zeta[row + x] - u[row + x] -
                         low_i * rhs_redu[(z - 1) * plane +
                                          y * NUM_POINTS_PER_DIR + x]) /
                        norm_coef;
                }
            }
        }

        // homogeneous dirichlet
        for (int y = 0; y < height; ++y) {
            const int row = ROW_MAJ(0, y0 + y, NUM_POINTS_PER_DIR - 1);
            for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                u[row + x] = 0;
                rhs_redu[(NUM_POINTS_PER_DIR - 1) * plane +
                         y * NUM_POINTS_PER_DIR + x] = 0;
            }
        }

        // Backward substitution
        for (int z = NUM_POINTS_PER_DIR - 2; z >= 0; --z) {
            for (int y = 0; y < height; ++y) {
                const int row = ROW_MAJ(0, y0 + y, z);
                for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {
                    double delta =
                        rhs_redu[z * plane + y * NUM_POINTS_PER_DIR + x] -
                        upp_redu[z] *
                        rhs_redu[(z + 1) * plane +
                                 y * NUM_POINTS_PER_DIR + x];
                    u[row + x] += delta;
                    rhs_redu[z * plane + y * NUM_POINTS_PER_DIR + x] = delta;
                }
            }
        }
    }

    TIMER_SCOPE_BW_END(solve_Dzz_tridiag_fused, 3 * ALLOC_SIZE);
}

static void compute_eps_naive(const double *restrict eta,
                              const double *restrict zeta,
                              const double *restrict u,
                              double *restrict eps, int t)
{
    TIMER_SCOPE_START(compute_eps_naive);

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int y = 0; y < NUM_POINTS_PER_DIR; ++y) {
            for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {

                // we don't care about the rhs value at boundaries
                if (x == 0 || x == NUM_POINTS_PER_DIR - 1 ||
                    y == 0 || y == NUM_POINTS_PER_DIR - 1 ||
                    z == 0 || z == NUM_POINTS_PER_DIR - 1) {
                    eps[ROW_MAJ(x, y, z)] = 0.0;
                    continue;
                }

                const double dxx = eta[ROW_MAJ(x - 1, y, z)] -
                    2.0 * eta[ROW_MAJ(x, y, z)] + eta[ROW_MAJ(x + 1, y, z)];

                const double dyy = zeta[ROW_MAJ(x, y - 1, z)] -
                    2.0 * zeta[ROW_MAJ(x, y, z)] + zeta[ROW_MAJ(x, y + 1, z)];

                const double dzz = u[ROW_MAJ(x, y, z - 1)] -
                    2.0 * u[ROW_MAJ(x, y, z)] + u[ROW_MAJ(x, y, z + 1)];

                eps[ROW_MAJ(x, y, z)] = eta[ROW_MAJ(x, y, z)] + DT * (
                    forcing(x * DX, y * DX, z * DX, (t - 0.5) * DT) +
                    NU * (dxx + dyy + dzz) / (DX * DX));
            }
        }
    }

    TIMER_SCOPE_END(compute_eps_naive);
}

static void solve_timestep(double *restrict eta, double *restrict zeta,
                           double *restrict u, double *restrict scratch,
                           int t)
{
    double *eps = scratch;
    double *redu = eps + NUM_POINTS + SCRATCH_PAD;
    double *upp_redu = redu;
    double *rhs_redu = redu + NUM_POINTS_PER_DIR;

    compute_eps_naive(eta, zeta, u, eps, t);

    // Solve for eta_n+1
    solve_Dxx_tridiag_fused(eps, eta, upp_redu, rhs_redu);
    // Solve for zeta_n+1
    solve_Dyy_tridiag_fused(eta, zeta, upp_redu, rhs_redu);
    // Solve for u_n+1
    solve_Dzz_tridiag_fused(zeta, u, upp_redu, rhs_redu);

    TIMER_LOG_NL_IF((t + 1) % TIMER_LOG_FREQ == 0);
}

int main(int argc, char *argv[])
{
    double *eta = malloc(ALLOC_SIZE);
    double *zeta = malloc(ALLOC_SIZE);
    double *u = malloc(ALLOC_SIZE);
    double *scratch = malloc(SCRATCH_SIZE);

    memset(eta, 0, ALLOC_SIZE);
    memset(zeta, 0, ALLOC_SIZE);
    memset(u, 0, ALLOC_SIZE);
    memset(scratch, 0, SCRATCH_SIZE);

    if (OUTPUT_VTK_FREQ > 0) {
        write_vtk("u-0000.vti", "u", u, NUM_POINTS_PER_DIR,
                  NUM_POINTS_PER_DIR, NUM_POINTS_PER_DIR, DX);
    }

    for (int t = 1; t < NUM_TIMESTEPS + 1; ++t) {
        TIMER_SCOPE_START(solve_timestep);
        solve_timestep(eta, zeta, u, scratch, t);
        TIMER_SCOPE_END(solve_timestep);

        if (OUTPUT_VTK_FREQ > 0 && t % OUTPUT_VTK_FREQ == 0) {
            char filename[32];
            sprintf(filename, "u-%04d.vti", t);
            write_vtk(filename, "u", u, NUM_POINTS_PER_DIR,
                      NUM_POINTS_PER_DIR, NUM_POINTS_PER_DIR, DX);
        }
    }

    free(eta);
    free(zeta);
    free(u);
    free(scratch);
}
