#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

#define NUM_POINTS_PER_DIR 128
#define NUM_TIMESTEPS 100

#define DX 1.0 / NUM_POINTS_PER_DIR
#define DT 1.0 / NUM_TIMESTEPS

#define NU 1.0

#define NUM_POINTS \
    NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR * NUM_POINTS_PER_DIR

#define ALLOC_SIZE NUM_POINTS * sizeof(double)

// Solves (I - Dxx) delta = rhs
void solve_Dxx_tridiag(const double *rhs, double *delta, int n)
{
    return;
}

// Solves (I - Dyy) delta = rhs
void solve_Dyy_tridiag(const double *rhs, double *delta, int n)
{
    return;
}

// Solves (I - Dzz) delta = rhs
void solve_Dzz_tridiag(const double *rhs, double *delta, int n)
{
    return;
}

void compute_eps(const double *eta, const double *zeta,
                 const double *u, double *eps, int n)
{
    for (int i = 0; i < n; ++i) {
        eps[i] = 1;
    }
}

void solve_timestep(double *eta, double *zeta, double *u, int n, int t)
{
    TIMER_CREATE(compute_eps);
    TIMER_CREATE(solve_Dxx);
    TIMER_CREATE(solve_Dyy);
    TIMER_CREATE(solve_Dzz);

    double *eps = malloc(ALLOC_SIZE);
    double *rhs = malloc(ALLOC_SIZE);
    double *delta = malloc(ALLOC_SIZE);

    TIMER_RESTART(compute_eps);
    compute_eps(eta, zeta, u, eps, n);
    TIMER_STOP(compute_eps);

    TIMER_RESTART(solve_Dxx);

    // Compute eps_n+1 - eta_n
    for (int i = 0; i < n; ++i) {
        rhs[i] = eps[i] - eta[i];
    }
    // Solve for delta_eta
    solve_Dxx_tridiag(delta, rhs, n);
    // Compute eta^n+1 = eta^n + delta_eta
    for (int i = 0; i < n; ++i) {
        eta[i] += delta[i];
    }

    TIMER_STOP(solve_Dxx);

    TIMER_RESTART(solve_Dyy);

    // Compute eta_n+1 - zeta_n
    for (int i = 0; i < n; ++i) {
        rhs[i] = eta[i] - zeta[i];
    }
    // Solve for delta_zeta
    solve_Dyy_tridiag(delta, rhs, n);
    // Compute eta^n+1 = eta^n + delta_eta
    for (int i = 0; i < n; ++i) {
        zeta[i] += delta[i];
    }

    TIMER_STOP(solve_Dyy);

    TIMER_RESTART(solve_Dzz);

    // Compute zeta_n+1 - u_n
    for (int i = 0; i < n; ++i) {
        rhs[i] = zeta[i] - u[i];
    }
    // Solve for delta_zeta
    solve_Dzz_tridiag(delta, rhs, n);
    // Compute eta^n+1 = eta^n + delta_eta
    for (int i = 0; i < n; ++i) {
        u[i] += delta[i];
    }

    TIMER_STOP(solve_Dzz);

    TIMER_LOG_BW(compute_eps, 2 * ALLOC_SIZE);
    TIMER_LOG(solve_Dxx);
    TIMER_LOG(solve_Dyy);
    TIMER_LOG(solve_Dzz);
    TIMER_LOG_NL();

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

    for (int t = 0; t < NUM_TIMESTEPS; ++t) {
        solve_timestep(eta, zeta, u, NUM_POINTS, t);
    }

    free(eta);
    free(zeta);
    free(u);
}
