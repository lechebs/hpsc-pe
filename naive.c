#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "timer.h"
#include "write_vtk.h"

#define NUM_POINTS_PER_DIR 64
#define NUM_TIMESTEPS 100

#define DX (1.0 / NUM_POINTS_PER_DIR)
#define DT (1.0 / NUM_TIMESTEPS)

#define NU 0.1

#define NUM_POINTS \
    NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR

#define ALLOC_SIZE NUM_POINTS * sizeof(double)

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
        return 0.1;
    } else {
        return 0;
    }
}

// Solves (I - Dxx) delta = rhs
void solve_Dxx_tridiag(double *rhs, double *delta)
{
    TIMER_SCOPE_START(solve_Dxx_tridiag);

    double *upp = malloc((NUM_POINTS_PER_DIR - 1) * sizeof(double));

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int y = 0; y < NUM_POINTS_PER_DIR; ++y) {

            upp[0] = 0;
            // homogeneous dirichlet
            rhs[ROW_MAJ(0, y, z)] = 0;

            // Forward reduction
            for (int x = 1; x < NUM_POINTS_PER_DIR - 1; ++x) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp[x - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp[x] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs[ROW_MAJ(x, y, z)] =
                    (rhs[ROW_MAJ(x, y, z)] -
                     low_i * rhs[ROW_MAJ(x - 1, y, z)]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(NUM_POINTS_PER_DIR - 1, y, z)] = 0;

            // Backward substitution
            for (int x = 1; x < NUM_POINTS_PER_DIR; ++x) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(NUM_POINTS_PER_DIR - x - 1, y, z)] =
                    rhs[ROW_MAJ(NUM_POINTS_PER_DIR - x - 1, y, z)] -
                    upp[NUM_POINTS_PER_DIR - x - 1] *
                    delta[ROW_MAJ(NUM_POINTS_PER_DIR - x, y, z)];
            }
        }
    }

    free(upp);

    TIMER_SCOPE_BW_END(solve_Dxx_tridiag, 4 * ALLOC_SIZE);
}

// Solves (I - Dyy) delta = rhs
void solve_Dyy_tridiag(double *rhs, double *delta)
{
    TIMER_SCOPE_START(solve_Dyy_tridiag);

    double *upp = malloc(NUM_POINTS_PER_DIR * sizeof(double));

    for (int z = 0; z < NUM_POINTS_PER_DIR; ++z) {
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {

            upp[0] = 0;
            // homogeneous dirichlet
            rhs[ROW_MAJ(x, 0, z)] = 0;

            // Forward reduction
            for (int y = 1; y < NUM_POINTS_PER_DIR - 1; ++y) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp[y - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp[y] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs[ROW_MAJ(x, y, z)] =
                    (rhs[ROW_MAJ(x, y, z)] -
                     low_i * rhs[ROW_MAJ(x, y - 1, z)]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - 1, z)] = 0;

            // Backward substitution
            for (int y = 1; y < NUM_POINTS_PER_DIR; ++y) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - y - 1, z)] =
                    rhs[ROW_MAJ(x, NUM_POINTS_PER_DIR - y - 1, z)] -
                    upp[NUM_POINTS_PER_DIR - y - 1] *
                    delta[ROW_MAJ(x, NUM_POINTS_PER_DIR - y, z)];
            }
        }
    }

    free(upp);

    TIMER_SCOPE_BW_END(solve_Dyy_tridiag, 4 * ALLOC_SIZE);
}

// Solves (I - Dzz) delta = rhs
void solve_Dzz_tridiag(double *rhs, double *delta)
{
    TIMER_SCOPE_START(solve_Dzz_tridiag);

    double *upp = malloc(NUM_POINTS_PER_DIR * sizeof(double));

    for (int y = 0; y < NUM_POINTS_PER_DIR; ++y) {
        for (int x = 0; x < NUM_POINTS_PER_DIR; ++x) {

            upp[0] = 0;
            // homogeneous dirichlet
            rhs[ROW_MAJ(x, y, 0)] = 0;

            // Forward reduction
            for (int z = 1; z < NUM_POINTS_PER_DIR - 1; ++z) {

                const double w = DT * NU / (2 * DX * DX);
                const double upp_i = -w;
                const double mid_i = 1.0 + 2 * w;
                const double low_i = -w;

                const double norm_coef = mid_i - low_i * upp[z - 1];

                // upp_i = upp_i / (mid_i - low_i upp_i-1)
                upp[z] = upp_i / norm_coef;

                // rhs_i = (rhs_i - low_i rhs_i-1) / (mid_i - low_i upp_i-1)
                rhs[ROW_MAJ(x, y, z)] =
                    (rhs[ROW_MAJ(x, y, z)] -
                     low_i * rhs[ROW_MAJ(x, y, z - 1)]) / norm_coef;
            }

            // homogeneous dirichlet
            delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - 1)] = 0;

            // Backward substitution
            for (int z = 1; z < NUM_POINTS_PER_DIR; ++z) {
                // delta_i = rhs_i - upp_i delta_i+1
                delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - z - 1)] =
                    rhs[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - z - 1)] -
                    upp[NUM_POINTS_PER_DIR - z - 1] *
                    delta[ROW_MAJ(x, y, NUM_POINTS_PER_DIR - z)];
            }
        }
    }

    free(upp);

    TIMER_SCOPE_BW_END(solve_Dzz_tridiag, 4 * ALLOC_SIZE);
}

void compute_eps_naive(const double *eta, const double *zeta,
                       const double *u, double *eps, int t)
{
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
                    forcing(x * DX, y * DX, z * DX, (t + 0.5) * DT) +
                    NU * (dxx + dyy + dzz) / (DX * DX));
            }
        }
    }
}

void solve_timestep(double *eta, double *zeta, double *u, int t)
{
    double *eps = malloc(ALLOC_SIZE);
    double *rhs = malloc(ALLOC_SIZE);
    double *delta = malloc(ALLOC_SIZE);

    // -O3 optimizes away memset(delta, 0, ALLOC_SIZE)
    memset(delta, 1, ALLOC_SIZE);

    TIMER_SCOPE_START(compute_eps_naive);
    compute_eps_naive(eta, zeta, u, eps, t);
    TIMER_SCOPE_END(compute_eps_naive);

    // Compute eps_n+1 - eta_n
    TIMER_SCOPE_START(compute_Dxx_rhs);
    for (int i = 0; i < NUM_POINTS; ++i) {
        rhs[i] = eps[i] - eta[i];
    }
    TIMER_SCOPE_BW_END(compute_Dxx_rhs, 4 * ALLOC_SIZE);
    // Solve for delta_eta
    solve_Dxx_tridiag(rhs, delta);
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
    solve_Dyy_tridiag(rhs, delta);
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
    solve_Dzz_tridiag(rhs, delta);
    // Compute eta^n+1 = eta^n + delta_eta
    TIMER_SCOPE_START(update_u);
    for (int i = 0; i < NUM_POINTS; ++i) {
        u[i] += delta[i];
    }
    TIMER_SCOPE_BW_END(update_u, 3 * ALLOC_SIZE);

    TIMER_LOG_NL_IF((t + 1) % TIMER_LOG_FREQ == 0);

    free(eps);
    free(rhs);
    free(delta);
}

int main(int argc, char *argv[])
{
    double *eta = malloc(ALLOC_SIZE);
    double *zeta = malloc(ALLOC_SIZE);
    double *u = malloc(ALLOC_SIZE);

    memset(eta, 0, ALLOC_SIZE);
    memset(zeta, 0, ALLOC_SIZE);
    memset(u, 0, ALLOC_SIZE);

    if (OUTPUT_VTK_FREQ > 0) {
        write_vtk("u-0000.vti", "u", u, NUM_POINTS_PER_DIR,
                  NUM_POINTS_PER_DIR, NUM_POINTS_PER_DIR, DX);
    }

    for (int t = 1; t < NUM_TIMESTEPS + 1; ++t) {
        TIMER_SCOPE_START(solve_timestep);
        solve_timestep(eta, zeta, u, t);
        TIMER_SCOPE_END(solve_timestep);

        if (t % OUTPUT_VTK_FREQ == 0) {
            char filename[32];
            sprintf(filename, "u-%04d.vti", t);
            write_vtk(filename, "u", u, NUM_POINTS_PER_DIR,
                      NUM_POINTS_PER_DIR, NUM_POINTS_PER_DIR, DX);
        }
    }

    free(eta);
    free(zeta);
    free(u);
}
