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
  MPI_Scatter(
    A, M * K / mpi_world_size, MPI_FLOAT,
    &A[M * K / mpi_world_size * mpi_rank], M * K / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int i = M / mpi_world_size * mpi_rank; i < M / mpi_world_size * (mpi_rank + 1); i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }

  MPI_Gather(
    &C[M / mpi_world_size * mpi_rank * N], M * N / mpi_world_size, MPI_FLOAT,
    C, M * N / mpi_world_size, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}
