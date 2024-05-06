#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void sparseMatVec(int rank, int size, int n, int *row_ptr, int *col_idx, double *values, double *vec) {
    double *result = (double *)calloc(n, sizeof(double));

    int rows_per_proc = (n + size - 1) / size;  // Calculate rows per process with ceiling
    int start_row = rank * rows_per_proc;
    int end_row = (rank + 1) * rows_per_proc;
    if (end_row > n) end_row = n;  // Limit the end_row to n

    for (int i = start_row; i < end_row; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            result[i] += values[j] * vec[col_idx[j]];
        }
    }

    double *final_result = (double *)calloc(n, sizeof(double));
    MPI_Reduce(result, final_result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     printf("Resulting vector:\n");
    //     for (int i = 0; i < n; i++)
    //         printf("%lf ", final_result[i]);
    //     printf("\n");
    // }

    free(result);
    free(final_result);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) {
            printf("Invalid matrix size. Please provide a positive integer.\n");
        }
        MPI_Finalize();
        return -1;
    }

    // Sample sparse matrix data for testing
    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_idx = (int *)malloc(3 * n * sizeof(int)); // Approximate non-zero count
    double *values = (double *)malloc(3 * n * sizeof(double)); // Approximate non-zero count
    double *vec = (double *)malloc(n * sizeof(double));

    if (!row_ptr || !col_idx || !values || !vec) {
        printf("Memory allocation failed\n");
        MPI_Finalize();
        return -1;
    }

    // Initialize sample data for testing
    for (int i = 0; i < n + 1; i++) row_ptr[i] = 3 * i;
    for (int i = 0; i < 3 * n; i++) col_idx[i] = i % n;
    for (int i = 0; i < 3 * n; i++) values[i] = 1.0;
    for (int i = 0; i < n; i++) vec[i] = 1.0;

    sparseMatVec(rank, size, n, row_ptr, col_idx, values, vec);

    free(row_ptr);
    free(col_idx);
    free(values);
    free(vec);

    MPI_Finalize();
    return 0;
}


