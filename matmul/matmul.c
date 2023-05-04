#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE

  int i, j;
  int M_per_rank = M / mpi_world_size;
  int M_start = M_per_rank * mpi_rank;

  // MPI_Scatter(
  //   A, M * K / mpi_world_size, MPI_FLOAT,
  //   &A[M * K / mpi_world_size * mpi_rank], M * K / mpi_world_size, MPI_FLOAT,
  //   0, MPI_COMM_WORLD);
  MPI_Bcast(A, M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  __m512 *C_vecs = (__m512 *)aligned_alloc(32, M_per_rank * N * sizeof(__m512));
  for (int i = 0; i < M_per_rank * N; i++) C_vecs[i] = _mm512_setzero_ps();
  
  #pragma omp parallel num_threads(threads_per_process) shared(C_vecs, j)
  {
    #pragma omp for private(i)
    for (j = 0; j < N; j++) {
      for (int k = 0; k < K; k += 16) {
        float B_seg[16];
        for (int kk = 0; kk < 16; kk++) B_seg[kk] = B[(k + kk) * N + j];
        __m512 B_vec = _mm512_load_ps(B_seg);
        for (int i = 0; i < M_per_rank; i++) {
          __m512 A_vec = _mm512_load_ps(&A[(i + M_start) * K + k]);
          int idx = i * N + j;
          C_vecs[idx] = _mm512_fmadd_ps(A_vec, B_vec, C_vecs[idx]);
        }
      }
    }
  }

  #pragma omp parallel num_threads(threads_per_process) shared(C, i)
  {
    #pragma omp for private(j)
    for (i = 0; i < M_per_rank; i++) {
      for (j = 0; j < N; j++) {
        float C_arr[16] __attribute__((aligned(32)));
        _mm512_store_ps(C_arr, C_vecs[i * N + j]);
        C[(i + M_start) * N + j] = _mm512_reduce_add_ps(C_vecs[i * N + j]);
      }
    }
  }

  MPI_Gather(
    &C[M / mpi_world_size * mpi_rank * N], M * N / mpi_world_size, MPI_FLOAT,
    C, M * N / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}
