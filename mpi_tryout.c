#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

    int N = 1024;
    double *A = malloc(sizeof(double) * N);
    double *B = malloc(sizeof(double) * N);
    double *C = malloc(sizeof(double) * N);

    // Init
    for (int i = 0; i < N; ++i) {
        A[i] = 0.0; B[i] = 1.0; C[i] = 2.0;
    }

    MPI_Init(&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Allocate and init arrays
    int work = N / nprocs;
    int start = rank * work;
    int end = start + work;
    for (int i = start; i < end; ++i) {
        A[i] = B[i] + C[i];
        printf("%lf\n", A[i]);
    }
    MPI_Finalize();
}