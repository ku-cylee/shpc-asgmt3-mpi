#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE   16

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE

  int i, j, k, t;
  int M_per_rank = M / mpi_world_size;
  int M_start = M_per_rank * mpi_rank;
  int VEC_K = K / TILE_SIZE;

  // MPI_Scatter(
  //   A, M * K / mpi_world_size, MPI_FLOAT,
  //   &A[M * K / mpi_world_size * mpi_rank], M * K / mpi_world_size, MPI_FLOAT,
  //   0, MPI_COMM_WORLD);
  MPI_Bcast(A, M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  __m512 *A_vecs = (__m512 *)aligned_alloc(32, M_per_rank * VEC_K * sizeof(__m512));
  for (i = 0; i < M_per_rank * VEC_K; i++) {
    A_vecs[i] = _mm512_load_ps(&A[M_start * K + i * TILE_SIZE]);
  }

  __m512 *B_vecs = (__m512 *)aligned_alloc(32, VEC_K * N * sizeof(__m512));

  #pragma omp parallel num_threads(threads_per_process) shared(C, j)
  {
    #pragma omp for private(i, k, t)
    for (j = 0; j < N; j++) {
      for (k = 0; k < VEC_K; k++) {
        float seg[TILE_SIZE];
        for (t = 0; t < TILE_SIZE; t++) seg[t] = B[(k * TILE_SIZE + t) * N + j];
        B_vecs[k * N + j] = _mm512_load_ps(seg);
      }

      for (i = 0; i < M_per_rank; i++) {
        __m512 C_vec = _mm512_setzero_ps();
        for (k = 0; k < VEC_K; k++) {
          __m512 A_vec = A_vecs[i * VEC_K + k];
          __m512 B_vec = B_vecs[k * N + j];
          C_vec = _mm512_fmadd_ps(A_vec, B_vec, C_vec);
        }
        C[(i + M_start) * N + j] = _mm512_reduce_add_ps(C_vec);
      }
    }
  }

  MPI_Gather(
    &C[M / mpi_world_size * mpi_rank * N], M * N / mpi_world_size, MPI_FLOAT,
    C, M * N / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}
