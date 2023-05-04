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
  // MPI_Scatter(
  //   A, M * K / mpi_world_size, MPI_FLOAT,
  //   &A[M * K / mpi_world_size * mpi_rank], M * K / mpi_world_size, MPI_FLOAT,
  //   0, MPI_COMM_WORLD);
  MPI_Bcast(A, M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  int j;
  #pragma omp parallel num_threads(threads_per_process) shared(C, j)
  {
    #pragma omp for
    for (j = 0; j < N; j++) {
      for (int k = 0; k < K; k += 16) {
        float B_seg[16] = { 0.0f };
        for (int kk = 0; kk < 16; kk++) B_seg[kk] = B[(k + kk) * N + j];
        for (int i = M / mpi_world_size * mpi_rank; i < M / mpi_world_size * (mpi_rank + 1); i++) {
          float sum = 0.0f;
          for (int kk = 0; kk < 16; kk++) sum += A[i * K + k + kk] * B_seg[kk];
          C[i * N + j] += sum;
        }
      }
    }
  }

  MPI_Gather(
    &C[M / mpi_world_size * mpi_rank * N], M * N / mpi_world_size, MPI_FLOAT,
    C, M * N / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}
