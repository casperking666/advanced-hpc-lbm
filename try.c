#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10 // size of the array

int main(int argc, char** argv) {
    int rank, size;
    int array[N], sum[N], global_sum[N];

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialize array with some values
    for (int i = 0; i < N; i++) {
        array[i] = i + rank; // each rank initializes its own subset of the array
    }

    // compute local sum of array elements
    for (int i = 0; i < N; i++) {
        sum[i] = array[i];
    }

    // perform global sum using MPI_Reduce
    MPI_Reduce(array, global_sum, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // print the result of the global sum
        printf("Global sum: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", global_sum[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
