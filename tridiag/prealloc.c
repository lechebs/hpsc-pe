#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// could be useful to avoid cache set conflicts or intel 4k aliasing
#ifndef SCRATCH_PAD
#define SCRATCH_PAD 0
#endif

#define SCRATCH_SIZE \
    ((3 * (NUM_POINTS + SCRATCH_PAD) + \
      2 * NUM_POINTS_PER_DIR) * sizeof(double))

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

// Solves (I - Dxx) delta = rhs
static void solve_Dxx_tridiag(const double *rhs,
                              double *delta,
                              double *scratch)
{
    TIMER_SCOPE_START(solve_Dxx_tridiag);

    double *upp_redu = scratch;
    double *rhs_redu = scratch + NUM_POINTS_PER_DIR;

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int y = 0; y < NUM_POINTS_PER_DIR; ++y) {

            upp_redu[0] = 0;
            // homogeneous dirichlet
            rhs_redu[0] = 0;

            // Forward reduction
            for (int x = 1; x < NUM_POINTS_PER_DIR - 1; ++x) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp_redu[x - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp_redu[x] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs_redu[x] = (rhs[ROW_MAJ(x, y, z)] -
                               low_i * rhs_redu[x - 1]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y, z)] = 0;

            // Backward substitution
            for (int x = 1; x < NUM_POINTS_PER_DIR; ++x) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(NUM_POINTS_PER_DIR - x - 1, y, z)] =
                    rhs_redu[NUM_POINTS_PER_DIR - x - 1] -
                    upp_redu[NUM_POINTS_PER_DIR - x - 1] *
                    delta[ROW_MAJ(NUM_POINTS_PER_DIR - x, y, z)];
            }
        }
    }

    TIMER_SCOPE_BW_END(solve_Dxx_tridiag, 3 * ALLOC_SIZE);
}

// Solves (I - Dyy) delta = rhs
static void solve_Dyy_tridiag(const double *rhs,
                              double *delta,
                              double *scratch)
{
    TIMER_SCOPE_START(solve_Dyy_tridiag);

    double *upp_redu = scratch;
    double *rhs_redu = scratch + NUM_POINTS_PER_DIR;

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {

            upp_redu[0] = 0;
            // homogeneous dirichlet
            rhs_redu[0] = 0;

            // Forward reduction
            for (int y = 1; y < NUM_POINTS_PER_DIR - 1; ++y) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp_redu[y - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp_redu[y] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs_redu[y] = (rhs[ROW_MAJ(x, y, z)] -
                               low_i * rhs_redu[y - 1]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - 1, z)] = 0;

            // Backward substitution
            for (int y = 1; y < NUM_POINTS_PER_DIR; ++y) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - y - 1, z)] =
                    rhs_redu[NUM_POINTS_PER_DIR - y - 1] -
                    upp_redu[NUM_POINTS_PER_DIR - y - 1] *
                    delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - y, z)];
            }
        }
    }

    TIMER_SCOPE_BW_END(solve_Dyy_tridiag, 3 * ALLOC_SIZE);
}

// Solves (I - Dzz) delta = rhs
static void solve_Dzz_tridiag(const double *rhs,
                              double *delta,
                              double *scratch)
{
    TIMER_SCOPE_START(solve_Dzz_tridiag);

    double *upp_redu = scratch;
    double *rhs_redu = scratch + NUM_POINTS_PER_DIR;

    for (int y = 0; y < NUM_POINTS_PER_DIR; ++y) {
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {

            upp_redu[0] = 0;
            // homogeneous dirichlet
            rhs_redu[0] = 0;

            // Forward reduction
            for (int z = 1; z < NUM_POINTS_PER_DIR - 1; ++z) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp_redu[z - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp_redu[z] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs_redu[z] =
                    (rhs[ROW_MAJ(x, y, z)] -
                     low_i * rhs_redu[z - 1]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - 1)] = 0;

            // Backward substitution
            for (int z = 1; z < NUM_POINTS_PER_DIR; ++z) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - z - 1)] =
                    rhs_redu[NUM_POINTS_PER_DIR - z - 1] -
                    upp_redu[NUM_POINTS_PER_DIR - z - 1] *
                    delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - z)];
            }
        }
    }

    TIMER_SCOPE_BW_END(solve_Dzz_tridiag, 3 * ALLOC_SIZE);
}

static void compute_eps_naive(const double *eta, const double *zeta,
                              const double *u, double *eps, int t)
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

static void solve_timestep(double *eta, double *zeta,
                           double *u, double *scratch, int t)
{
    // Partition scratch space
    double *eps = scratch;
    double *rhs = eps + NUM_POINTS + SCRATCH_PAD;
    double *delta = rhs + NUM_POINTS + SCRATCH_PAD;
    double *redu = delta + NUM_POINTS + SCRATCH_PAD;

    compute_eps_naive(eta, zeta, u, eps, t);

    // Compute eps_n+1 - eta_n
    TIMER_SCOPE_START(compute_Dxx_rhs);
    for (int i = 0; i < NUM_POINTS; ++i) {
        rhs[i] = eps[i] - eta[i];
    }
    TIMER_SCOPE_BW_END(compute_Dxx_rhs, 4 * ALLOC_SIZE);
    // Solve for delta_eta
    solve_Dxx_tridiag(rhs, delta, redu);
    // Compute eta^n+1 = eta^n + delta_eta
    TIMER_SCOPE_START(update_eta);
    for (int i = 0; i < NUM_POINTS; ++i) {
        eta[i] += delta[i];
    }
    TIMER_SCOPE_BW_END(update_eta, 3 * ALLOC_SIZE);

    // Compute eta_n+1 - zeta_n
    TIMER_SCOPE_START(compute_Dyy_rhs);
    for (int i = 0; i < NUM_POINTS; ++i) {
        rhs[i] = eta[i] - zeta[i];
    }
    TIMER_SCOPE_BW_END(compute_Dyy_rhs, 4 * ALLOC_SIZE);
    // Solve for delta_zeta
    solve_Dyy_tridiag(rhs, delta, redu);
    // Compute eta^n+1 = eta^n + delta_eta
    TIMER_SCOPE_START(update_zeta);
    for (int i = 0; i < NUM_POINTS; ++i) {
        zeta[i] += delta[i];
    }
    TIMER_SCOPE_BW_END(update_zeta, 3 * ALLOC_SIZE);

    TIMER_SCOPE_START(compute_Dzz_rhs);
    // Compute zeta_n+1 - u_n
    for (int i = 0; i < NUM_POINTS; ++i) {
        rhs[i] = zeta[i] - u[i];
    }
    TIMER_SCOPE_BW_END(compute_Dzz_rhs, 4 * ALLOC_SIZE);
    // Solve for delta_zeta
    solve_Dzz_tridiag(rhs, delta, redu);
    // Compute eta^n+1 = eta^n + delta_eta
    TIMER_SCOPE_START(update_u);
    for (int i = 0; i < NUM_POINTS; ++i) {
        u[i] += delta[i];
    }
    TIMER_SCOPE_BW_END(update_u, 3 * ALLOC_SIZE);
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
        TIMER_LOG_NL_IF(t % TIMER_LOG_FREQ == 0);

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
