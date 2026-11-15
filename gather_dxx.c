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

    const __m128i gather_idx = _mm_setr_epi32(
        0,
        NUM_POINTS_PER_DIR,
        2 * NUM_POINTS_PER_DIR,
        3 * NUM_POINTS_PER_DIR);

    const __m256d low_vec = _mm256_set1_pd(low_i);
    const __m256d zero = _mm256_setzero_pd();

    const int nvec = (NUM_POINTS_PER_DIR / DXX_VLEN) * DXX_VLEN;

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int y = 0; y < nvec; y += DXX_VLEN) {

            // homogeneous dirichlet
            _mm256_storeu_pd(rhs_redu, zero);

            // Forward reduction
            for (int x = 1; x < NUM_POINTS_PER_DIR - 1; ++x) {

                const double norm_coef = mid_i - low_i * upp_redu[x - 1];
                const int i = ROW_MAJ(x, y, z);

                __m256d eta_vec = _mm256_i32gather_pd(eta + i, gather_idx, 8);
                __m256d eps_vec = _mm256_i32gather_pd(eps + i, gather_idx, 8);
                __m256d rhs_prev = _mm256_loadu_pd(
                    rhs_redu + (x - 1) * DXX_VLEN);

                __m256d rhs = _mm256_sub_pd(eps_vec, eta_vec);
                rhs = _mm256_sub_pd(rhs, _mm256_mul_pd(low_vec, rhs_prev));
                rhs = _mm256_div_pd(rhs, _mm256_set1_pd(norm_coef));

                _mm256_storeu_pd(rhs_redu + x * DXX_VLEN, rhs);
            }

            // homogeneous dirichlet
            for (int lane = 0; lane < DXX_VLEN; ++lane) {
                eta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y + lane, z)] = 0;
            }
            _mm256_storeu_pd(rhs_redu +
                             (NUM_POINTS_PER_DIR - 1) * DXX_VLEN, zero);

            // Backward substitution
            __m256d delta_prev = zero;
            for (int x = NUM_POINTS_PER_DIR - 2; x >= 0; --x) {

                __m256d rhs = _mm256_loadu_pd(rhs_redu + x * DXX_VLEN);
                __m256d upp_vec = _mm256_set1_pd(upp_redu[x]);
                __m256d delta = _mm256_sub_pd(
                    rhs, _mm256_mul_pd(upp_vec, delta_prev));

                double delta_lane[DXX_VLEN];
                _mm256_storeu_pd(delta_lane, delta);

                const int i = ROW_MAJ(x, y, z);
                for (int lane = 0; lane < DXX_VLEN; ++lane) {
                    eta[i + lane * NUM_POINTS_PER_DIR] += delta_lane[lane];
                }

                delta_prev = delta;
            }
        }

        // scalar tail
        for (int y = nvec; y < NUM_POINTS_PER_DIR; ++y) {

            rhs_redu[0] = 0;

            for (int x = 1; x < NUM_POINTS_PER_DIR - 1; ++x) {
                const double norm_coef = mid_i - low_i * upp_redu[x - 1];
                const int i = ROW_MAJ(x, y, z);
                rhs_redu[x] = (eps[i] - eta[i] -
                               low_i * rhs_redu[x - 1]) / norm_coef;
            }

            eta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y, z)] = 0;

            double delta_prev = 0;
            for (int x = NUM_POINTS_PER_DIR - 2; x >= 0; --x) {
                double delta = rhs_redu[x] - upp_redu[x] * delta_prev;
                eta[ROW_MAJ(x, y, z)] += delta;
                delta_prev = delta;
            }
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
