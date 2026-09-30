/*
 * _csmo.c — WSS3 SMO solver with kernel column callback.
 *
 * Pure C implementation of LIBSVM-style serial SMO (Fan et al. 2005)
 * with second-order working set selection, LRU kernel column cache,
 * and shrinking.  Kernel columns are provided by an external callback,
 * allowing Python/Numba/CuPy kernels to drive the solver.
 *
 * Built as a Python extension module (for setuptools install) but the
 * solver function is called via ctypes, not the Python C API.
 *
 * BSD-2-Clause — this file does not contain LIBSVM code; the algorithm
 * is reimplemented from the Fan et al. 2005 description.
 */

#include <Python.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

/* Callback: fill out[0..N-1] with normalized K(idx, :). */
typedef void (*column_callback_t)(int idx, int N, double *out, void *userdata);

/* ------------------------------------------------------------------ */
/* LRU kernel column cache                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int capacity;
    int N;
    int *col_index;      /* which training index is in each slot (-1 = empty) */
    double **data;        /* column data for each slot */
    long *last_used;      /* access counter for LRU eviction */
    long timer;
    column_callback_t callback;
    void *userdata;
    long hits;
    long misses;
} ColumnCache;

static ColumnCache *cache_create(int capacity, int N,
                                  column_callback_t cb, void *ud) {
    ColumnCache *c = (ColumnCache *)malloc(sizeof(ColumnCache));
    c->capacity = capacity;
    c->N = N;
    c->col_index = (int *)malloc(capacity * sizeof(int));
    c->data = (double **)malloc(capacity * sizeof(double *));
    c->last_used = (long *)calloc(capacity, sizeof(long));
    c->timer = 0;
    c->callback = cb;
    c->userdata = ud;
    c->hits = 0;
    c->misses = 0;
    for (int i = 0; i < capacity; i++) {
        c->col_index[i] = -1;
        c->data[i] = (double *)malloc(N * sizeof(double));
    }
    return c;
}

static void cache_destroy(ColumnCache *c) {
    for (int i = 0; i < c->capacity; i++)
        free(c->data[i]);
    free(c->data);
    free(c->col_index);
    free(c->last_used);
    free(c);
}

static double *cache_get(ColumnCache *c, int idx) {
    c->timer++;
    /* Search for cached column */
    for (int s = 0; s < c->capacity; s++) {
        if (c->col_index[s] == idx) {
            c->last_used[s] = c->timer;
            c->hits++;
            return c->data[s];
        }
    }
    /* Cache miss — find LRU slot */
    c->misses++;
    int lru_slot = 0;
    long lru_time = c->last_used[0];
    for (int s = 1; s < c->capacity; s++) {
        if (c->last_used[s] < lru_time) {
            lru_time = c->last_used[s];
            lru_slot = s;
        }
    }
    /* Fill slot via callback */
    c->col_index[lru_slot] = idx;
    c->last_used[lru_slot] = c->timer;
    c->callback(idx, c->N, c->data[lru_slot], c->userdata);
    return c->data[lru_slot];
}

/* ------------------------------------------------------------------ */
/* WSS3 SMO solver                                                    */
/* ------------------------------------------------------------------ */

/*
 * Returns: number of iterations performed.
 *   alpha_out[0..N-1] = y[i] * alpha[i]  (signed dual coefficients)
 *   *rho_out = bias  (score = K*coef + bias)
 */

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int csmo_solve(
    int N,
    const double *y,
    double C,
    double tol,
    int max_iter,
    int cache_columns,
    column_callback_t get_column,
    void *userdata,
    double *alpha_out,
    double *rho_out,
    int verbose
) {
    double *alpha = (double *)calloc(N, sizeof(double));
    double *G     = (double *)malloc(N * sizeof(double));
    int    *active = (int *)malloc(N * sizeof(int));
    int n_active = N;

    for (int i = 0; i < N; i++) {
        G[i] = -1.0;
        active[i] = 1;
    }

    ColumnCache *cache = cache_create(cache_columns, N, get_column, userdata);

    int shrink_interval = N > 1000 ? N : 1000;
    int unshrink_needed = 0;

    double gap = INFINITY;
    double m_val = INFINITY;
    double M_val = -INFINITY;

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int iteration;
    for (iteration = 0; iteration < max_iter; iteration++) {
        /* Compute m(alpha) and M(alpha) over active set */
        m_val = -INFINITY;
        M_val = INFINITY;
        int i_best = -1;

        for (int t = 0; t < N; t++) {
            if (!active[t]) continue;
            double neg_yG = -y[t] * G[t];
            int in_up  = (y[t] > 0 && alpha[t] < C) || (y[t] < 0 && alpha[t] > 0);
            int in_low = (y[t] > 0 && alpha[t] > 0) || (y[t] < 0 && alpha[t] < C);
            if (in_up && neg_yG > m_val) {
                m_val = neg_yG;
                i_best = t;
            }
            if (in_low && neg_yG < M_val) {
                M_val = neg_yG;
            }
        }

        if (i_best < 0) {
            if (unshrink_needed) {
                for (int t = 0; t < N; t++) active[t] = 1;
                n_active = N;
                unshrink_needed = 0;
                continue;
            }
            break;
        }

        gap = m_val - M_val;

        if (verbose && iteration % 1000 == 0) {
            int n_sv = 0;
            for (int t = 0; t < N; t++)
                if (alpha[t] > 1e-10) n_sv++;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec)
                           + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
            long total = cache->hits + cache->misses;
            double hr = total > 0 ? 100.0 * cache->hits / total : 0.0;
            fprintf(stderr,
                "  iter %8d  gap=%.4e  SVs=%d  active=%d/%d  "
                "cache hit=%.0f%%  %.1fs\n",
                iteration, gap, n_sv, n_active, N, hr, elapsed);
            fflush(stderr);
        }

        if (gap < tol) {
            if (unshrink_needed) {
                for (int t = 0; t < N; t++) active[t] = 1;
                n_active = N;
                unshrink_needed = 0;
                continue;
            }
            break;
        }

        /* Shrinking */
        if (iteration > 0 && iteration % shrink_interval == 0) {
            int shrunk = 0;
            for (int t = 0; t < N; t++) {
                if (!active[t]) continue;
                double neg_yG = -y[t] * G[t];
                int at_zero = alpha[t] < 1e-10;
                int at_C = alpha[t] > C - 1e-10;
                int do_shrink = 0;
                if (at_zero && y[t] > 0 && neg_yG < M_val) do_shrink = 1;
                if (at_zero && y[t] < 0 && neg_yG > m_val) do_shrink = 1;
                if (at_C && y[t] > 0 && neg_yG > m_val) do_shrink = 1;
                if (at_C && y[t] < 0 && neg_yG < M_val) do_shrink = 1;
                if (do_shrink) {
                    active[t] = 0;
                    shrunk++;
                }
            }
            if (shrunk > 0) {
                n_active -= shrunk;
                unshrink_needed = 1;
                if (verbose) {
                    fprintf(stderr, "  shrink: removed %d, active=%d/%d\n",
                            shrunk, n_active, N);
                    fflush(stderr);
                }
                continue;
            }
        }

        /* WSS3: select j by second-order gain */
        double *K_col_i = cache_get(cache, i_best);

        int j_best = -1;
        double best_gain = -INFINITY;

        for (int t = 0; t < N; t++) {
            if (!active[t]) continue;
            double neg_yG = -y[t] * G[t];
            int in_low = (y[t] > 0 && alpha[t] > 0) || (y[t] < 0 && alpha[t] < C);
            if (!in_low || neg_yG >= m_val) continue;

            double b = m_val - neg_yG;
            /* Q_ii = 1 for normalized, K_col_i[i_best] = 1,
               Q_jj = 1, a = 2 - 2*K(i,j) for normalized.
               General: a = Q_diag[i] + Q_diag[j] - 2*K(i,j). */
            double a = 1.0 + 1.0 - 2.0 * K_col_i[t];
            if (a < 1e-12) a = 1e-12;
            double gain = b * b / a;
            if (gain > best_gain) {
                best_gain = gain;
                j_best = t;
            }
        }

        if (j_best < 0) {
            if (unshrink_needed) {
                for (int t = 0; t < N; t++) active[t] = 1;
                n_active = N;
                unshrink_needed = 0;
                continue;
            }
            break;
        }

        double *K_col_j = cache_get(cache, j_best);

        /* Compute step */
        int i = i_best, j = j_best;
        double K_ij = K_col_i[j];
        double a = 1.0 + 1.0 - 2.0 * K_ij;
        if (a <= 0) a = 1e-12;

        double s = y[i] * y[j];
        double b = -s * G[i] + G[j];
        double d_j = -b / a;

        /* Box constraints */
        double lo, hi;
        if (s > 0) {
            lo = (alpha[i] - C > -alpha[j]) ? alpha[i] - C : -alpha[j];
            hi = (C - alpha[j] < alpha[i]) ? C - alpha[j] : alpha[i];
        } else {
            lo = (-alpha[i] > -alpha[j]) ? -alpha[i] : -alpha[j];
            hi = (C - alpha[i] < C - alpha[j]) ? C - alpha[i] : C - alpha[j];
        }
        if (d_j < lo) d_j = lo;
        if (d_j > hi) d_j = hi;
        double d_i = -s * d_j;

        /* Update */
        alpha[i] += d_i;
        alpha[j] += d_j;

        for (int t = 0; t < N; t++) {
            G[t] += d_i * y[i] * y[t] * K_col_i[t]
                  + d_j * y[j] * y[t] * K_col_j[t];
        }
    }

    if (verbose) {
        int n_sv = 0;
        for (int t = 0; t < N; t++)
            if (alpha[t] > 1e-10) n_sv++;
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_start.tv_sec)
                       + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
        long total = cache->hits + cache->misses;
        double hr = total > 0 ? 100.0 * cache->hits / total : 0.0;
        fprintf(stderr,
            "  SMO done: %d iters, %d SVs, gap=%.2e, "
            "cache hit=%.1f%%, %.1fs\n",
            iteration, n_sv, gap, hr, elapsed);
        fflush(stderr);
    }

    /* Compute bias */
    double rho;
    int n_free = 0;
    double sum_free = 0.0;
    for (int t = 0; t < N; t++) {
        if (alpha[t] > 1e-10 && alpha[t] < C - 1e-10) {
            sum_free += y[t] * G[t];
            n_free++;
        }
    }
    if (n_free > 0)
        rho = sum_free / n_free;
    else
        rho = -(m_val + M_val) / 2.0;

    /* Output: signed coefficients */
    for (int t = 0; t < N; t++)
        alpha_out[t] = alpha[t] * y[t];
    *rho_out = -rho;

    cache_destroy(cache);
    free(alpha);
    free(G);
    free(active);

    return iteration;
}

/* Python extension module — allows setuptools to compile and install
   this shared library.  The solver is called via ctypes. */

static PyModuleDef _csmo_module = {
    PyModuleDef_HEAD_INIT, "_csmo", NULL, -1, NULL,
};

PyMODINIT_FUNC PyInit__csmo(void) {
    return PyModule_Create(&_csmo_module);
}
